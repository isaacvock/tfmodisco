"""User-facing progress reporting for RNA-MoDISco.

The numerical code in RNA-MoDISco contains several long-running Numba calls.
This module deliberately treats those calls as opaque stages while providing
real progress for the surrounding Python loops.  It also keeps presentation
details out of the motif-discovery implementation.
"""

from __future__ import print_function

import contextlib
import sys
import time

from tqdm.auto import tqdm


VALID_PROGRESS_MODES = ("auto", "bar", "log", "off")


def format_duration(seconds):
	"""Format a duration compactly while retaining useful precision."""
	seconds = max(float(seconds), 0.0)
	if seconds < 10:
		return "{:.1f}s".format(seconds)
	if seconds < 60:
		return "{:.0f}s".format(seconds)

	whole_seconds = int(round(seconds))
	hours, remainder = divmod(whole_seconds, 3600)
	minutes, secs = divmod(remainder, 60)
	if hours:
		return "{}h {:02d}m {:02d}s".format(hours, minutes, secs)
	return "{}m {:02d}s".format(minutes, secs)


class ProgressReporter(object):
	"""Render progress as terminal bars or append-only log messages.

	Parameters
	----------
	mode : {"auto", "bar", "log", "off"}
		``auto`` uses progress bars when the output stream is attached to a
		terminal and timestamped log messages otherwise.
	verbose : bool
		Whether diagnostic-only notes should be shown.
	stream : file-like
		Destination for all progress output. Defaults to stderr.
	"""

	def __init__(self, mode="off", verbose=False, stream=None,
		log_interval_seconds=30.0, log_initial_delay_seconds=5.0):
		mode = str(mode).lower()
		if mode not in VALID_PROGRESS_MODES:
			raise ValueError(
				"progress mode must be one of {}; got {!r}."
				.format(VALID_PROGRESS_MODES, mode)
			)

		self.stream = sys.stderr if stream is None else stream
		self.requested_mode = mode
		self.mode = self._resolve_mode(mode)
		self.verbose = bool(verbose)
		self.log_interval_seconds = float(log_interval_seconds)
		self.log_initial_delay_seconds = float(log_initial_delay_seconds)
		self.indent = 0
		self.sections = []
		self.started_at = time.monotonic()
		self.completed_stages = []

	def _resolve_mode(self, mode):
		if mode != "auto":
			return mode
		is_terminal = getattr(self.stream, "isatty", lambda: False)()
		return "bar" if is_terminal else "log"

	@property
	def enabled(self):
		return self.mode != "off"

	def _timestamp(self):
		return time.strftime("%Y-%m-%d %H:%M:%S")

	def _write(self, message, kind="INFO"):
		if not self.enabled:
			return

		prefix = "  " * self.indent
		if self.mode == "log":
			text = "{}  {:5s} {}".format(
				self._timestamp(), kind, prefix + str(message))
			print(text, file=self.stream, flush=True)
		else:
			tqdm.write(prefix + str(message), file=self.stream)

	def header(self, title):
		if not self.enabled:
			return
		if self.mode == "bar":
			self._write(title)
			self._write("=" * len(title))
		else:
			self._write(title)

	def note(self, message):
		self._write(message, kind="INFO")

	def verbose_note(self, message):
		if self.verbose:
			self._write(message, kind="INFO")

	def skipped(self, message):
		self._write("– " + message, kind="SKIP")

	@contextlib.contextmanager
	def section(self, title):
		if self.mode == "bar" and self.enabled:
			tqdm.write("", file=self.stream)
		self._write(title, kind="PHASE")
		self.sections.append(str(title))
		self.indent += 1
		try:
			yield
		finally:
			self.indent = max(0, self.indent - 1)
			self.sections.pop()

	def task(self, name, total=None, unit="it", detail=None,
		record_duration=True):
		"""Return a context-managed progress task."""
		return ProgressTask(
			reporter=self,
			name=name,
			total=total,
			unit=unit,
			detail=detail,
			record_duration=record_duration,
		)

	def _record_stage(self, name, elapsed):
		qualified_name = " / ".join(self.sections + [name])
		self.completed_stages.append((qualified_name, float(elapsed)))

	def finish(self, summary=None):
		if not self.enabled:
			return

		elapsed = time.monotonic() - self.started_at
		if self.mode == "bar":
			tqdm.write("", file=self.stream)
			self._write(
				"RNA-MoDISco completed in {}".format(
					format_duration(elapsed)),
				kind="DONE",
			)
		else:
			self._write(
				"RNA-MoDISco completed elapsed={}".format(
					format_duration(elapsed)),
				kind="DONE",
			)

		if summary:
			self.note(summary)

		if self.completed_stages:
			name, duration = max(self.completed_stages, key=lambda item: item[1])
			self.note(
				"Slowest stage: {} ({})".format(
					name, format_duration(duration))
			)


class ProgressTask(object):
	"""One progress stage managed by :class:`ProgressReporter`."""

	def __init__(self, reporter, name, total=None, unit="it", detail=None,
		record_duration=True):
		self.reporter = reporter
		self.name = str(name)
		self.total = None if total is None else int(total)
		self.unit = str(unit)
		self.detail = detail
		self.record_duration = bool(record_duration)
		self.completed = 0
		self.started_at = None
		self.last_log_at = None
		self.last_log_bucket = -1
		self.bar = None
		self.summary = None

	def __enter__(self):
		self.started_at = time.monotonic()
		self.last_log_at = self.started_at

		if not self.reporter.enabled:
			return self

		if self.reporter.mode == "bar" and self.total is not None:
			if self.detail:
				self.reporter._write(str(self.detail))
			self.bar = tqdm(
				total=self.total,
				desc=("  " * self.reporter.indent) + self.name,
				unit=self.unit,
				dynamic_ncols=True,
				leave=False,
				file=self.reporter.stream,
				mininterval=0.2,
			)
		else:
			self.reporter._write(
				self._start_message(),
				kind="START",
			)

		return self

	def _start_message(self):
		if self.detail:
			return "{} — {}".format(self.name, self.detail)
		return self.name

	def advance(self, amount=1, **metrics):
		amount = int(amount)
		self.completed += amount

		if not self.reporter.enabled:
			return

		if self.bar is not None:
			self.bar.update(amount)
			if metrics:
				self.bar.set_postfix(metrics, refresh=False)
			return

		if self.reporter.mode != "log" or self.total in (None, 0):
			return

		now = time.monotonic()
		if now - self.started_at < self.reporter.log_initial_delay_seconds:
			return

		percent = min(100.0, 100.0 * self.completed / self.total)
		bucket = int(percent // 10)
		time_due = now - self.last_log_at >= self.reporter.log_interval_seconds
		bucket_due = bucket > self.last_log_bucket
		if self.completed < self.total and (time_due or bucket_due):
			metric_text = self._format_metrics(metrics)
			self.reporter._write(
				"{} {}/{} ({:.0f}%){}".format(
					self.name,
					self.completed,
					self.total,
					percent,
					metric_text,
				),
				kind="PROG",
			)
			self.last_log_at = now
			self.last_log_bucket = bucket

	def set_postfix(self, **metrics):
		if self.bar is not None:
			self.bar.set_postfix(metrics, refresh=False)

	def set_summary(self, summary):
		self.summary = str(summary)

	def _format_metrics(self, metrics):
		if not metrics:
			return ""
		return " " + " ".join(
			"{}={}".format(key, value) for key, value in metrics.items())

	def __exit__(self, exc_type, exc_value, traceback):
		elapsed = time.monotonic() - self.started_at

		if self.bar is not None:
			self.bar.close()

		if not self.reporter.enabled:
			return False

		if exc_type is not None:
			self.reporter._write(
				"✗ {} failed after {}: {}".format(
					self.name, format_duration(elapsed), exc_value),
				kind="FAIL",
			)
			return False

		message = "✓ {} ({})".format(self.name, format_duration(elapsed))
		if self.summary:
			message += " — " + self.summary
		self.reporter._write(message, kind="DONE")
		if self.record_duration:
			self.reporter._record_stage(self.name, elapsed)
		return False


def ensure_progress(progress=None):
	"""Return a progress reporter while keeping library calls silent by default."""
	if progress is None or progress is False:
		return ProgressReporter(mode="off")
	if progress is True:
		return ProgressReporter(mode="auto")
	if isinstance(progress, str):
		return ProgressReporter(mode=progress)
	if hasattr(progress, "task") and hasattr(progress, "note"):
		return progress
	raise TypeError(
		"progress must be None, bool, a progress mode string, or a reporter."
	)
