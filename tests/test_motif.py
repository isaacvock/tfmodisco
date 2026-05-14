import random

import h5py
import numpy as np
from click.testing import CliRunner

from modiscolite.cli import motifs
from conftest import data_ohe_hyps


def test_modisco_motif_smoke(data_ohe_hyps, tmp_path):
    random.seed(42)
    np.random.seed(42)

    ohe, hyps = data_ohe_hyps
    output = tmp_path / "modisco_results.h5"

    runner = CliRunner()
    result = runner.invoke(
        motifs,
        ["-s", ohe, "-a", hyps, "-n", 2000, "-o", output, "-v"],
    )
    assert result.exit_code == 0, result.output

    with h5py.File(output, "r") as f:
        assert f.attrs["tool"] == "RNA-MoDISco"
        assert f.attrs["alphabet"] == "ACGU"
        assert bool(f.attrs["reverse_complement"]) is False
        assert "pos_patterns" in f

        for pattern_group in ("pos_patterns", "neg_patterns"):
            if pattern_group not in f:
                continue

            for pattern in f[pattern_group].values():
                assert pattern["sequence"].shape[1] == 4
                assert pattern["contrib_scores"].shape[1] == 4
                assert not np.any(pattern["seqlets"]["is_revcomp"][:])
