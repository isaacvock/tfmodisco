import io

import numpy as np
import scipy.sparse
from click.testing import CliRunner

import modiscolite
from modiscolite.cli import motifs
from modiscolite.progress import ProgressReporter
from modiscolite.progress import ensure_progress
from modiscolite.progress import format_duration
from modiscolite import tfmodisco
from modiscolite import util


class _TerminalBuffer(io.StringIO):
    def isatty(self):
        return True


def _write_cli_inputs(tmp_path):
    sequence = np.zeros((2, 4, 12), dtype="float32")
    sequence[:, 0, :] = 1
    attributions = np.ones_like(sequence)

    sequence_path = tmp_path / "sequence.npy"
    attribution_path = tmp_path / "attributions.npy"
    np.save(sequence_path, sequence)
    np.save(attribution_path, attributions)
    return sequence_path, attribution_path


def test_auto_mode_selects_terminal_bars_or_append_only_logs():
    terminal_reporter = ProgressReporter(
        mode="auto", stream=_TerminalBuffer())
    log_reporter = ProgressReporter(mode="auto", stream=io.StringIO())

    assert terminal_reporter.mode == "bar"
    assert log_reporter.mode == "log"


def test_log_reporter_emits_stages_progress_summary_and_slowest_stage():
    stream = io.StringIO()
    reporter = ProgressReporter(
        mode="log",
        stream=stream,
        log_interval_seconds=3600,
        log_initial_delay_seconds=0,
    )

    reporter.header("RNA-MoDISco")
    with reporter.section("Positive motif discovery"):
        with reporter.task("Leiden clustering", total=10, unit="seed") as task:
            task.advance()
            task.set_summary("best quality=0.75")
    reporter.finish("3 positive patterns")

    output = stream.getvalue()
    assert "INFO  RNA-MoDISco" in output
    assert "PHASE Positive motif discovery" in output
    assert "START   Leiden clustering" in output
    assert "PROG    Leiden clustering 1/10 (10%)" in output
    assert "DONE    ✓ Leiden clustering" in output
    assert "best quality=0.75" in output
    assert "RNA-MoDISco completed elapsed=" in output
    assert "3 positive patterns" in output
    assert (
        "Slowest stage: Positive motif discovery / Leiden clustering"
        in output
    )


def test_disabled_reporter_is_silent():
    stream = io.StringIO()
    reporter = ProgressReporter(mode="off", stream=stream)

    reporter.header("RNA-MoDISco")
    with reporter.section("Section"):
        with reporter.task("Task", total=2) as task:
            task.advance(2)
    reporter.finish("finished")

    assert stream.getvalue() == ""
    assert ensure_progress(None).mode == "off"


def test_duration_formatting():
    assert format_duration(1.25) == "1.2s"
    assert format_duration(45) == "45s"
    assert format_duration(125) == "2m 05s"
    assert format_duration(3661) == "1h 01m 01s"


def _legacy_density_adaptation(affmat_nn, seqlet_neighbors, perplexity):
    eps = 0.0000001
    rows, cols, data = [], [], []
    for row in range(len(affmat_nn)):
        for col, datum in zip(seqlet_neighbors[row], affmat_nn[row]):
            rows.append(row)
            cols.append(col)
            data.append(datum)

    affinity = scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(affmat_nn), len(affmat_nn)),
        dtype="float64",
    )
    affinity.data = np.maximum(
        np.log((1.0 / (0.5 * np.maximum(affinity.data, eps))) - 1),
        0,
    )
    affinity.eliminate_zeros()

    counts = scipy.sparse.csr_matrix(
        (np.ones_like(affinity.data), affinity.indices, affinity.indptr),
        shape=affinity.shape,
        dtype="float64",
    )
    affinity += affinity.T
    counts += counts.T
    affinity.data /= counts.data

    betas = [
        util.binary_search_perplexity(perplexity, affinity[i].data)
        for i in range(affinity.shape[0])
    ]
    normfactors = np.array(
        [
            np.exp(-np.array(affinity[i].data) / beta).sum() + 1
            for i, beta in enumerate(betas)
        ]
    )

    for i in range(affinity.shape[0]):
        for j_idx in range(affinity.indptr[i], affinity.indptr[i + 1]):
            j = affinity.indices[j_idx]
            distance = affinity.data[j_idx]
            rbf_i = np.exp(-distance / betas[i]) / normfactors[i]
            rbf_j = np.exp(-distance / betas[j]) / normfactors[j]
            affinity.data[j_idx] = np.sqrt(rbf_i * rbf_j)

    affinity += scipy.sparse.diags(1.0 / normfactors)
    return affinity


def test_instrumented_density_adaptation_preserves_numerical_results():
    neighbors = np.tile(np.arange(4, dtype="int32"), (4, 1))
    affinity = np.array(
        [
            [1.0, 0.8, 0.3, 0.2],
            [0.8, 1.0, 0.4, 0.25],
            [0.3, 0.4, 1.0, 0.7],
            [0.2, 0.25, 0.7, 1.0],
        ],
        dtype="float64",
    )

    expected = _legacy_density_adaptation(
        affinity.copy(), neighbors, perplexity=2.0)
    observed = tfmodisco._density_adaptation(
        affinity.copy(),
        neighbors,
        tsne_perplexity=2.0,
        progress=False,
    )

    np.testing.assert_allclose(
        observed.toarray(), expected.toarray(), rtol=0, atol=0)


def test_cli_defaults_to_auto_progress_in_noninteractive_logs(
    tmp_path, monkeypatch
):
    sequence_path, attribution_path = _write_cli_inputs(tmp_path)
    output_path = tmp_path / "results.h5"
    observed = {}

    def fake_tfmodisco(**kwargs):
        observed["progress_mode"] = kwargs["progress"].mode
        return None, None

    monkeypatch.setattr(
        modiscolite.tfmodisco, "TFMoDISco", fake_tfmodisco)

    result = CliRunner().invoke(
        motifs,
        [
            "-s",
            str(sequence_path),
            "-a",
            str(attribution_path),
            "-n",
            "100",
            "-o",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed["progress_mode"] == "log"
    assert "RNA-MoDISco" in result.output
    assert "Loading input arrays" in result.output
    assert "Saving HDF5 results" in result.output
    assert "RNA-MoDISco completed" in result.output
    assert output_path.exists()


def test_cli_progress_can_be_disabled(tmp_path, monkeypatch):
    sequence_path, attribution_path = _write_cli_inputs(tmp_path)
    output_path = tmp_path / "results.h5"

    monkeypatch.setattr(
        modiscolite.tfmodisco,
        "TFMoDISco",
        lambda **kwargs: (None, None),
    )

    result = CliRunner().invoke(
        motifs,
        [
            "-s",
            str(sequence_path),
            "-a",
            str(attribution_path),
            "-n",
            "100",
            "-o",
            str(output_path),
            "--progress",
            "off",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output == ""


def test_cli_help_documents_auto_as_the_default_progress_mode():
    result = CliRunner().invoke(motifs, ["--help"])

    assert result.exit_code == 0
    assert "--progress [auto|bar|log|off]" in result.output
    assert "default: auto" in result.output
