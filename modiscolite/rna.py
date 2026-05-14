# rna.py
# Helpers for stranded RNA-MoDISco inputs and masks.

import warnings

import numpy as np


VALID_REGIONS = ("all", "5utr", "cds", "3utr")
VALID_MISSING_CDS = ("skip", "error")


def normalize_region(region):
	region = str(region).lower()
	if region not in VALID_REGIONS:
		raise ValueError(
			"region must be one of {}; got {!r}.".format(VALID_REGIONS, region)
		)
	return region


def normalize_missing_cds(missing_cds):
	missing_cds = str(missing_cds).lower()
	if missing_cds not in VALID_MISSING_CDS:
		raise ValueError(
			"missing_cds must be one of {}; got {!r}."
			.format(VALID_MISSING_CDS, missing_cds)
		)
	return missing_cds


def _squeeze_mask(mask):
	mask = np.asarray(mask)
	if mask.ndim == 3:
		if mask.shape[1] == 1:
			mask = mask[:, 0, :]
		elif mask.shape[2] == 1:
			mask = mask[:, :, 0]
		else:
			raise ValueError(
				"sequence_mask must have shape (N, L), (N, 1, L), or (N, L, 1)."
			)

	if mask.ndim != 2:
		raise ValueError(
			"sequence_mask must have shape (N, L), (N, 1, L), or (N, L, 1)."
		)

	return mask.astype(bool)


def infer_padding_mask(sequence, sequence_mask=None):
	"""Return an N x L mask for non-padding positions.

	Without an explicit mask, only trailing all-zero nucleotide columns are
	treated as padding. Interior all-zero nucleotide columns are ambiguous and
	raise an error instead of silently converting biological unknowns to padding.
	"""

	sequence = np.asarray(sequence)
	if sequence.ndim != 3:
		raise ValueError("sequence must have shape (N, L, C).")
	if sequence.shape[2] < 4:
		raise ValueError("sequence must have at least 4 nucleotide channels.")

	nucleotide_present = np.any(sequence[:, :, :4] != 0, axis=2)
	extra_present = (
		np.any(sequence[:, :, 4:] != 0, axis=2)
		if sequence.shape[2] > 4 else
		np.zeros(sequence.shape[:2], dtype=bool)
	)

	if sequence_mask is not None:
		mask = _squeeze_mask(sequence_mask)
		if mask.shape != sequence.shape[:2]:
			raise ValueError(
				"sequence_mask shape {} is incompatible with sequence shape {}."
				.format(mask.shape, sequence.shape[:2])
			)

		if np.any((nucleotide_present | extra_present) & ~mask):
			bad = np.argwhere((nucleotide_present | extra_present) & ~mask)[0]
			raise ValueError(
				"sequence_mask marks example {} position {} as padding, but "
				"the sequence has non-zero channels there."
				.format(int(bad[0]), int(bad[1]))
			)

		return mask

	padding_mask = np.zeros(sequence.shape[:2], dtype=bool)

	for example_idx, row in enumerate(nucleotide_present):
		nonzero_positions = np.flatnonzero(row)
		if len(nonzero_positions) == 0:
			if np.any(extra_present[example_idx]):
				bad_pos = np.flatnonzero(extra_present[example_idx])[0]
				raise ValueError(
					"example {} has extra-channel annotations at position {} "
					"but no nucleotide sequence."
					.format(example_idx, int(bad_pos))
				)
			continue

		last_valid = int(nonzero_positions[-1])
		interior_missing = np.flatnonzero(~row[:last_valid + 1])
		if len(interior_missing) > 0:
			raise ValueError(
				"example {} has all-zero nucleotide columns before the inferred "
				"3' padding starts, first at position {}. Provide an explicit "
				"sequence_mask if these positions are biologically meaningful."
				.format(example_idx, int(interior_missing[0]))
			)

		if np.any(extra_present[example_idx, last_valid + 1:]):
			bad_pos = last_valid + 1 + int(
				np.flatnonzero(extra_present[example_idx, last_valid + 1:])[0]
			)
			raise ValueError(
				"example {} has extra-channel annotations in inferred 3' padding "
				"at position {}."
				.format(example_idx, bad_pos)
			)

		padding_mask[example_idx, :last_valid + 1] = True

	return padding_mask


def infer_region_mask(sequence, padding_mask, region="all", missing_cds="skip"):
	"""Infer an N x L region mask from RNA sequence channels.

	For 6-channel inputs, channel 4 marks CDS codon starts and channel 5 marks
	5' splice-site annotations. CDS/UTR boundaries are inferred from phase-
	consistent codon starts; splice-site annotations are validated against
	padding and preserved rather than treated as nucleotide channels.
	"""

	region = normalize_region(region)
	missing_cds = normalize_missing_cds(missing_cds)
	padding_mask = np.asarray(padding_mask).astype(bool)

	if region == "all":
		return padding_mask.copy(), []

	sequence = np.asarray(sequence)
	if sequence.ndim != 3 or sequence.shape[2] < 6:
		raise ValueError(
			"region={!r} requires N x L x 6 sequence input with codon-start "
			"and 5' splice-site channels."
			.format(region)
		)

	if padding_mask.shape != sequence.shape[:2]:
		raise ValueError(
			"padding_mask shape {} is incompatible with sequence shape {}."
			.format(padding_mask.shape, sequence.shape[:2])
		)

	region_mask = np.zeros_like(padding_mask, dtype=bool)
	metadata = []
	skipped_missing_cds = []

	for example_idx in range(sequence.shape[0]):
		valid_positions = np.flatnonzero(padding_mask[example_idx])
		if len(valid_positions) == 0:
			warnings.warn(
				"Skipping example {} for region {!r}: no non-padding sequence."
				.format(example_idx, region)
			)
			metadata.append(None)
			continue

		valid_start = int(valid_positions[0])
		valid_end = int(valid_positions[-1]) + 1
		if valid_start != 0 or not np.all(padding_mask[example_idx, :valid_end]):
			raise ValueError(
				"example {} has a non-contiguous valid-position mask; RNA-MoDISco "
				"only infers 3' padding/regions from contiguous transcript prefixes."
				.format(example_idx)
			)

		codon_starts = np.flatnonzero(
			(sequence[example_idx, :, 4] > 0.5) & padding_mask[example_idx]
		)
		splice_sites = np.flatnonzero(
			(sequence[example_idx, :, 5] > 0.5) & padding_mask[example_idx]
		)

		if np.any((sequence[example_idx, :, 4] > 0.5) & ~padding_mask[example_idx]):
			raise ValueError(
				"example {} has codon-start annotations in inferred padding."
				.format(example_idx)
			)
		if np.any((sequence[example_idx, :, 5] > 0.5) & ~padding_mask[example_idx]):
			raise ValueError(
				"example {} has 5' splice-site annotations in inferred padding."
				.format(example_idx)
			)
		if len(codon_starts) == 0:
			message = (
				"example {} cannot infer {!r}: no CDS codon-start annotations."
				.format(example_idx, region)
			)
			if missing_cds == "error":
				raise ValueError(message)
			skipped_missing_cds.append(example_idx)
			metadata.append(None)
			continue

		cds_start = int(codon_starts[0])
		last_codon_start = int(codon_starts[-1])
		expected = np.arange(cds_start, last_codon_start + 1, 3)
		if not np.array_equal(codon_starts, expected):
			raise ValueError(
				"example {} has codon-start annotations that are not contiguous "
				"and phase-consistent every 3 nt."
				.format(example_idx)
			)

		cds_end = last_codon_start + 3
		if cds_end > valid_end:
			raise ValueError(
				"example {} has an incomplete terminal codon: last codon start {} "
				"extends beyond non-padding end {}."
				.format(example_idx, last_codon_start, valid_end)
			)

		bounds = {
			"valid_start": valid_start,
			"valid_end": valid_end,
			"cds_start": cds_start,
			"cds_end": cds_end,
			"splice_sites": [int(pos) for pos in splice_sites],
		}

		if region == "5utr":
			region_mask[example_idx, valid_start:cds_start] = True
		elif region == "cds":
			region_mask[example_idx, cds_start:cds_end] = True
		elif region == "3utr":
			region_mask[example_idx, cds_end:valid_end] = True

		if not np.any(region_mask[example_idx]):
			warnings.warn(
				"Skipping example {} for region {!r}: inferred region is empty."
				.format(example_idx, region)
			)

		metadata.append(bounds)

	if len(skipped_missing_cds) > 0:
		preview = ", ".join(str(x) for x in skipped_missing_cds[:5])
		if len(skipped_missing_cds) > 5:
			preview += ", ..."
		warnings.warn(
			"Skipping {} examples for region {!r} because they lack CDS "
			"codon-start annotations; first skipped examples: {}. Use "
			"missing_cds='error' for strict validation."
			.format(len(skipped_missing_cds), region, preview)
		)

	if not np.any(region_mask):
		raise ValueError(
			"region {!r} produced no valid positions across all examples. "
			"Examples without CDS codon-start annotations are skipped by default "
			"for region-specific analyses."
			.format(region)
		)

	return region_mask, metadata


def candidate_window_mask(position_mask, window_size, flank):
	"""Return valid smoothed-window starts for seqlet extraction."""

	position_mask = np.asarray(position_mask).astype(bool)
	if position_mask.ndim != 2:
		raise ValueError("position_mask must have shape (N, L).")

	n_examples, length = position_mask.shape
	if window_size > length:
		raise ValueError(
			"window_size ({}) cannot exceed sequence length ({})."
			.format(window_size, length)
		)

	n_windows = length - window_size + 1
	mask = np.zeros((n_examples, n_windows), dtype=bool)
	span = window_size + 2 * flank
	cumsum = np.pad(
		np.cumsum(position_mask.astype(np.int64), axis=1),
		((0, 0), (1, 0)),
		mode="constant",
	)

	for window_start in range(n_windows):
		seqlet_start = window_start - flank
		seqlet_end = window_start + window_size + flank
		if seqlet_start < 0 or seqlet_end > length:
			continue

		mask[:, window_start] = (
			cumsum[:, seqlet_end] - cumsum[:, seqlet_start]
		) == span

	return mask
