import h5py
import numpy as np
import pandas as pd
from click.testing import CliRunner

from modiscolite.cli import cli
from modiscolite.fancy_report import (
    build_tsv,
    load_h5_patterns,
    normalize_logo_layout,
    summarize_logo_layout,
)


def _write_minimal_modisco_h5(path):
    sequence = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    contrib = np.array([
        [0.2, 0.0, 0.0, 0.0],
        [0.0, -0.3, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.0],
        [0.0, 0.0, 0.0, -0.5],
    ])

    with h5py.File(path, "w") as h5:
        for group_name, pattern_name, n_seqlets in [
            ("pos_patterns", "pattern_0", 4),
            ("neg_patterns", "pattern_1", 2),
        ]:
            pattern = h5.create_group(group_name).create_group(pattern_name)
            pattern.create_dataset("sequence", data=sequence)
            pattern.create_dataset("contrib_scores", data=contrib)
            seqlets = pattern.create_group("seqlets")
            seqlets.create_dataset("n_seqlets", data=np.array([n_seqlets]))
            seqlets.create_dataset("is_revcomp", data=np.array([False, True]))


def test_load_h5_patterns_and_build_tsv(tmp_path):
    h5_path = tmp_path / "modisco_results.h5"
    _write_minimal_modisco_h5(h5_path)

    patterns = load_h5_patterns(h5_path)

    assert [pattern["tag"] for pattern in patterns] == [
        "pos_patterns.pattern_0",
        "neg_patterns.pattern_1",
    ]
    assert patterns[0]["iupac"] == "ACGT"
    assert patterns[0]["n_seqlets"] == 4
    assert patterns[0]["pct_revcomp"] == 50.0

    tomtom_df = pd.DataFrame([
        {
            "tag": "pos_patterns.pattern_0",
            "match0": "MA0139.1 CTCF",
            "pval0": 1e-6,
        },
        {
            "tag": "neg_patterns.pattern_1",
            "match0": "MA0477.1 FOS",
            "pval0": 2e-5,
        },
    ])
    tsv = build_tsv(patterns, tomtom_df, top_n=1)

    assert list(tsv["motif_id0"]) == ["MA0139.1", "MA0477.1"]
    assert list(tsv["tf_name0"]) == ["CTCF", "FOS"]
    assert list(tsv["family0"]) == ["CTCF/CTCFL", "AP-1/bZIP"]


def test_normalize_logo_layout(tmp_path):
    logos_dir = tmp_path / "logos"
    logos_dir.mkdir()
    (logos_dir / "pos_patterns.pattern_0.cwm.fwd.png").write_bytes(b"fwd")
    (logos_dir / "pos_patterns.pattern_0.cwm.rev.png").write_bytes(b"rev")

    count = normalize_logo_layout(logos_dir, ["pos_patterns.pattern_0"])
    summary = summarize_logo_layout(logos_dir, ["pos_patterns.pattern_0"])

    assert count == 2
    assert (logos_dir / "pos_patterns.pattern_0" / "trimmed_cwm_fwd_logo.png").read_bytes() == b"fwd"
    assert (logos_dir / "pos_patterns.pattern_0" / "trimmed_cwm_rev_logo.png").read_bytes() == b"rev"
    assert summary["missing_required"] == {}


def test_report_fancy_cli_help():
    result = CliRunner().invoke(cli, ["report-fancy", "--help"])

    assert result.exit_code == 0
    assert "--force-logos" in result.output
    assert "--trim-min-length" in result.output
    assert "--meme-db" in result.output
