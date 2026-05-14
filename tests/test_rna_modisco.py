import numpy as np
import pytest
from click.testing import CliRunner

from modiscolite import affinitymat
from modiscolite import aggregator
from modiscolite import core
from modiscolite import rna
from modiscolite import tfmodisco
from modiscolite.cli import motifs


BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3}


def _one_hot(seq, channels=4, length=None):
    length = len(seq) if length is None else length
    arr = np.zeros((length, channels), dtype="float32")
    for i, base in enumerate(seq):
        arr[i, BASE_TO_IDX[base]] = 1
    return arr


def _seqlet_set(seq):
    seqlet = core.Seqlet(example_idx=0, start=0, end=len(seq))
    seqlet.sequence = _one_hot(seq)
    seqlet.contrib_scores = seqlet.sequence.copy()
    seqlet.hypothetical_contribs = seqlet.sequence.copy()
    seqlet.sequence_features = seqlet.sequence.copy()
    seqlet.position_mask = np.ones(len(seq), dtype=bool)
    return core.SeqletSet([seqlet])


def _annotated_sequence():
    sequence = np.zeros((1, 24, 6), dtype="float32")
    sequence[0, :20, :4] = _one_hot("ACGU" * 5)
    sequence[0, [6, 9, 12], 4] = 1
    sequence[0, [4, 10, 16], 5] = 1
    return sequence


def test_reverse_complement_seqlets_are_disabled():
    seqlet = core.Seqlet(example_idx=0, start=0, end=4)
    with pytest.raises(ValueError, match="stranded"):
        seqlet.revcomp()

    parent = _seqlet_set("ACGUAC")
    child = _seqlet_set("GUACGU")
    _, is_rc, _ = aggregator._align_patterns(
        parent,
        child,
        affinitymat.pearson_correlation,
        min_overlap=1.0,
        transformer="magnitude",
        include_hypothetical=False,
    )
    assert is_rc is False


def test_padding_mask_infers_only_trailing_zero_columns():
    sequence = np.zeros((1, 12, 4), dtype="float32")
    sequence[0, :8] = _one_hot("ACGUACGU")

    mask = rna.infer_padding_mask(sequence)
    np.testing.assert_array_equal(mask, [[True] * 8 + [False] * 4])

    sequence[0, 4] = 0
    with pytest.raises(ValueError, match="all-zero nucleotide columns"):
        rna.infer_padding_mask(sequence)


def test_candidate_windows_respect_padding_and_flanks():
    position_mask = np.array([[True] * 8 + [False] * 4])
    mask = rna.candidate_window_mask(position_mask, window_size=3, flank=2)

    expected = np.zeros((1, 10), dtype=bool)
    expected[0, 2:4] = True
    np.testing.assert_array_equal(mask, expected)


def test_prepare_inputs_accepts_n_by_4_by_l_sequence_and_attributions():
    sequence = _one_hot("ACGUACGU", length=8).T[None]
    attributions = np.ones((1, 4, 8), dtype="float32")

    one_hot, hyps, features, position_mask, padding_mask, region, metadata = (
        tfmodisco._prepare_inputs(sequence, attributions)
    )

    assert one_hot.shape == (1, 8, 4)
    assert hyps.shape == (1, 8, 4)
    assert features.shape == (1, 8, 4)
    assert region == "all"
    assert metadata == []
    assert np.all(position_mask)
    assert np.all(padding_mask)


def test_prepare_inputs_accepts_n_by_6_by_l_and_preserves_extra_channels():
    sequence = _annotated_sequence().transpose(0, 2, 1)
    attributions = np.ones((1, 4, 24), dtype="float32")

    one_hot, hyps, features, position_mask, padding_mask, region, _ = (
        tfmodisco._prepare_inputs(sequence, attributions, region="cds")
    )

    assert one_hot.shape == (1, 24, 4)
    assert hyps.shape == (1, 24, 4)
    assert features.shape == (1, 24, 6)
    assert region == "cds"
    assert np.sum(padding_mask) == 20
    assert np.flatnonzero(position_mask[0]).tolist() == list(range(6, 15))


def test_region_masks_for_5utr_cds_and_3utr():
    sequence = _annotated_sequence()
    padding_mask = rna.infer_padding_mask(sequence)

    mask_5utr, meta = rna.infer_region_mask(sequence, padding_mask, "5utr")
    mask_cds, _ = rna.infer_region_mask(sequence, padding_mask, "cds")
    mask_3utr, _ = rna.infer_region_mask(sequence, padding_mask, "3utr")

    assert meta[0]["cds_start"] == 6
    assert meta[0]["cds_end"] == 15
    assert meta[0]["splice_sites"] == [4, 10, 16]
    np.testing.assert_array_equal(np.flatnonzero(mask_5utr[0]), np.arange(0, 6))
    np.testing.assert_array_equal(np.flatnonzero(mask_cds[0]), np.arange(6, 15))
    np.testing.assert_array_equal(np.flatnonzero(mask_3utr[0]), np.arange(15, 20))


def test_region_masks_skip_missing_cds_examples_by_default():
    sequence = np.concatenate(
        [_annotated_sequence(), np.zeros((1, 24, 6), dtype="float32")],
        axis=0,
    )
    sequence[1, :18, :4] = _one_hot("ACGU" * 5)[:18]
    padding_mask = rna.infer_padding_mask(sequence)

    with pytest.warns(UserWarning, match="lack CDS codon-start"):
        mask_3utr, metadata = rna.infer_region_mask(
            sequence, padding_mask, "3utr"
        )

    np.testing.assert_array_equal(np.flatnonzero(mask_3utr[0]), np.arange(15, 20))
    assert not np.any(mask_3utr[1])
    assert metadata[0]["cds_start"] == 6
    assert metadata[1] is None


def test_region_masks_can_error_on_missing_cds_examples():
    sequence = np.zeros((1, 24, 6), dtype="float32")
    sequence[0, :18, :4] = _one_hot("ACGU" * 5)[:18]
    padding_mask = rna.infer_padding_mask(sequence)

    with pytest.raises(ValueError, match="no CDS codon-start annotations"):
        rna.infer_region_mask(
            sequence, padding_mask, "3utr", missing_cds="error"
        )


def test_region_request_requires_six_channels():
    sequence = _one_hot("ACGUACGU", length=8).T[None]
    attributions = np.ones((1, 4, 8), dtype="float32")

    with pytest.raises(ValueError, match="requires N x L x 6"):
        tfmodisco._prepare_inputs(sequence, attributions, region="cds")


def test_cli_rejects_legacy_window_option(tmp_path):
    sequence_path = tmp_path / "seq.npy"
    attribution_path = tmp_path / "attr.npy"
    np.save(sequence_path, np.zeros((1, 4, 12), dtype="float32"))
    np.save(attribution_path, np.zeros((1, 4, 12), dtype="float32"))

    result = CliRunner().invoke(
        motifs,
        [
            "-s",
            str(sequence_path),
            "-a",
            str(attribution_path),
            "-n",
            "1",
            "--window",
            "10",
        ],
    )

    assert result.exit_code != 0
    assert "no longer accepts --window" in result.output
