# RNA-MoDISco

RNA-MoDISco is a stranded RNA-focused refactor of TF-MoDISco for discovering sequence motifs from machine-learning-model-derived importance scores. RNA inputs are assumed to already be in the biological orientation that should be analyzed; motif discovery does not reverse-complement seqlets, compare reverse-complement motifs, or merge patterns across reverse-complement orientation.

These importance scores can be generated using various attribution methods, such as DeepLIFT or SHAP, applied to models like BPNet. The algorithm identifies high-importance regions (seqlets), clusters them into motifs, and provides a report comparing discovered motifs to known databases.

> [!IMPORTANT]  
> Starting from version v2, TF-MoDISco utilizes the [tfmodisco-lite](https://github.com/jmschrei/tfmodisco-lite/) implementation and interface. This implementation is significantly more memory efficient, and in many cases faster, than the original implementation. The original implementation (v0) is still available [here](https://github.com/kundajelab/tfmodisco/tree/v0-final).

## Algorithm Description

The TF-MoDISco algorithm starts with a set of importance scores on genomic sequences and performs the following tasks:

1. Identify high-importance windows of the sequences, termed "seqlets"
2. Divide the seqlets into positive and negative sets (metaclusters) based on the overall importance score of each seqlet
3. Cluster recurring similar seqlets
4. Generate motifs by aligning the clustered seqlets

During clustering, a coarse-grained similarity is calculated as the cosine similarity between gapped k-mer representations between all pairs of seqlets. This information is used to calculate the top nearest neighbors, for which a fine-grained similarity is calculated as the maximum Jaccard index as two seqlets are aligned with all possible offsets. This sparse similarity matrix is then density adapted, similarly to t-SNE, and Leiden clustering is used to extract patterns. Finally, some heuristics are used to merge similar patterns and split apart the seqlets comprising dissimilar ones.

![image](assets/overview.svg)

## References

TF-MoDISco is described in:
> Wang, Tseng, Ramalingam, Schreiber, et al. "Decoding predictive motif lexicons and syntax from deep learning models of transcription factor binding profiles." (manuscript in preparation)

Related tools:
- [Fi-NeMo](https://github.com/kundajelab/Fi-NeMo): Motif instance detection using TF-MoDISco patterns
- [BPNet](https://github.com/kundajelab/bpnet-refactor): Deep learning models for TF binding prediction
- [ChromBPNet](https://github.com/kundajelab/chrombpnet): Deep learning models for chromatin accessibility prediction

## Installation

You can install TF-MoDISco using `pip install modisco`

## RNA Tensor Conventions

RNA-MoDISco accepts attribution tensors with four nucleotide channels and sequence tensors with either four or six channels:

- Attributions: `N x 4 x L` or internal `N x L x 4`.
- Sequence, nucleotide only: `N x 4 x L` or internal `N x L x 4`, ordered `A/C/G/U`.
- Sequence, annotated RNA: `N x 6 x L` or internal `N x L x 6`, where channels 0-3 are `A/C/G/U`, channel 4 is a CDS codon-start indicator, and channel 5 contains 5' splice-site indicators. Multiple 5' splice-site annotations per transcript are valid.

Motif discovery uses only the first four nucleotide channels. The two optional annotation channels are preserved as sequence features and are used for region-restricted runs. Trailing 3' padding is inferred conservatively from all-zero nucleotide columns; interior all-zero nucleotide columns are treated as ambiguous unless an explicit mask is provided through the Python API.

With six-channel sequence input, `modisco motifs --region` can restrict discovery to `all`, `5utr`, `cds`, or `3utr`. Region boundaries are inferred from phase-consistent CDS codon-start annotations and combined with the padding mask.

For region-specific runs, transcripts without CDS codon-start annotations are skipped by default because 5' UTR/CDS/3' UTR boundaries cannot be inferred for non-coding or unannotated RNAs. Use `--missing-cds error` if every input is expected to be coding and you want strict validation.

## Running RNA-MoDISco

You can run TF-MoDISco using the command line tool `modisco` which comes with the TF-MoDISco installation. This tool allows you to run TF-MoDISco on a set of sequences and corresponding attributions, and then to generate a report (like the one seen above) for the output generated from the first step.

`modisco motifs -s ohe.npz -a shap.npz -n 2000 -o modisco_results.h5`

This command will run modisco on the one-hot encoded RNA sequences in `ohe.npz`, use the attributions from `shap.npz`, use a maximum of 2000 positive/negative seqlets, and output the results to `modisco_results.h5`. CLI sequence and attribution arrays are assumed to be length-last (`N x C x L`). Note that you can also use `npy` files if you don't want to use compressed data for some reason.

> [!TIP]
> **Max seqlets:** Seqlets will generally follow the order of the input regions, and hence can be biased by the order in which the regions are provided. `-n` takes top seqlets in order that they are identified, where identification occurs per region (in order that they are inputted), then in descending order of each seqlet's attribution score per region. For unbiased sampling, shuffle the input regions beforehand. Keep the shuffled regions to keep track of the absolute instance positions.

RNA-MoDISco no longer applies a user-specified fixed motif-search window. `region=all` searches the full non-padded RNA sequence, and `--region 5utr`, `--region cds`, or `--region 3utr` searches the full inferred non-padded region.

## Seqlet and Pattern Sizing Parameters

The `modisco motifs` command exposes several size parameters that control two related but different objects:

- A **seqlet** is an individual high-attribution sequence segment extracted from one input example.
- A **pattern** is the aggregate motif built by aligning, clustering, trimming, and sometimes merging many seqlets.

The CLI defaults below are the defaults in `modiscolite/cli.py`.

Intuitively, these parameters split motif discovery into three stages. `--size` controls the initial **finding** step: what width of attribution event should count as an interesting seqlet core? The flank parameters control how much extra **room to maneuver** RNA-MoDISco has around that first guess while aligning and polishing motifs. `--trim-size` controls the polished motif width RNA-MoDISco tries to preserve after many seqlets have been aligned into a pattern. Flanks are useful because the strongest attribution window is not always perfectly centered on the full biological motif; it may capture only the middle, one side, or the most predictive bases.

`--max-seqlets`, `-n` has no default and is required. It is the maximum number of positive seqlets and the maximum number of negative seqlets passed into clustering, so a run may use up to twice this many seqlets total. This cap is applied after seqlet extraction and after seqlets are split by sign. It does not change the extraction threshold; it only limits how many extracted seqlets are used for motif discovery. Because seqlets are extracted example-by-example, with high-scoring windows chosen within each example, this parameter can be sensitive to input order. Increase it when you have a large dataset, expect many motif families, or want better sensitivity to weaker or rarer motifs. Decrease it for quick exploratory runs or when memory/runtime are limiting. For unbiased subsampling, shuffle input examples before running.

`--size`, `-z`, default `20`, is the seqlet core size, passed to the algorithm as `sliding_window_size`. RNA-MoDISco first collapses nucleotide-channel contribution scores into one attribution score per position, then scores every contiguous window of this length. These window sums are used to estimate positive and negative seqlet thresholds and to pick high-attribution cores. The stored seqlet is wider than this core if `--seqlet-flank-size` is nonzero, but the core is the part used for initial thresholding and positive/negative assignment. Intuitively, this should be close to the width of the attribution signal you want one seqlet to capture, not necessarily the exact biological motif width. Increase it for broad motifs or motif combinations whose signal is spread over a wider span. Decrease it for short motifs, sharply localized signal, or very short regions.

`--seqlet-flank-size`, `-f`, default `5`, adds this many bases on each side of each extracted seqlet core. With the defaults, each raw seqlet spans `5 + 20 + 5 = 30` bases. These flanks are included in the seqlet sequence, contribution scores, and similarity calculations, but they are not part of the core window sum used to assign the seqlet as positive or negative. Candidate windows must have enough valid, non-padded positions to include both flanks, so larger flanks exclude more windows near transcript ends, padding, or region boundaries. Increase this when the high-attribution core may miss useful alignment context or when the biological motif extends beyond the highest-scoring positions. Decrease it for short regions, boundary-heavy region-specific runs, or when extra low-signal context makes clustering noisier. Setting it to `0` makes extraction more boundary-friendly and usually cheaper, but gives RNA-MoDISco less sequence context for aligning seqlets whose attribution peaks are slightly off-center.

`--trim-size`, `-t`, default `30`, is the pattern core size used during pattern polishing, passed as `trim_to_window_size`. After seqlets in a cluster are aligned into a provisional pattern, RNA-MoDISco trims poorly supported pattern edges, expands by `--initial-flank-to-add`, computes per-position information content, and keeps the length-`--trim-size` window with the highest total information content. This happens during the iterative cluster-to-pattern and pattern-merging stages. Intuitively, this is the motif width the polishing step tries to preserve before adding working flanks back. Increase it if motifs are being clipped or if you expect longer structured motifs. Decrease it if reported patterns contain too much low-information context or nearby secondary signal. If `--trim-size` equals `--size`, RNA-MoDISco is being asked to find and preserve a motif core of the same width, which is a clean setting when motifs fit inside the extraction window. If `--trim-size` is larger than `--size`, the extra width must come from seqlet or pattern flanks; with zero flanks, `--trim-size` should generally be no larger than `--size`.

`--initial-flank-to-add`, `-g`, default `10`, adds working flanks to each side of a pattern during polishing and merging. This is separate from `--seqlet-flank-size`: seqlet flanks are attached during initial extraction, while initial pattern flanks are added after seqlets have been clustered and aligned. These working flanks give the polishing step room to recenter the high-information window and give later merging/splitting steps more context. Increase it when motif centers are uncertain, motifs are wider than expected, or neighboring context is important for aligning related seqlets. Decrease it if many seqlets lie close to valid-region boundaries, if too many seqlets are dropped during expansion, or if extra context is making patterns diffuse. Setting it to `0` preserves boundary-adjacent seqlets better, but also removes a chance for pattern polishing to recover bases just outside the initially aligned seqlets. Larger values increase memory and runtime because patterns become wider; setting flanks to `0` would usually be expected to reduce runtime rather than increase it.

`--final-flank-to-add`, `-j`, default `0`, adds extra bases on each side only near the end of motif discovery, after final pattern filtering and before subpatterns are computed and results are saved. It is mostly an output-context parameter: it changes the final saved pattern width and seqlet coordinates, but it does not help decide which top-level patterns are discovered or pass the final filters. Increase it when you want the HDF5 output and reports to show additional sequence context around each discovered motif, such as nearby bases or spacing around the motif. Keep it at `0` when you want compact motif logos and coordinates. Very large values can drop boundary-adjacent seqlets during final expansion because each expanded interval must remain inside the valid, non-padded region.

The output saved in `modisco_results.h5` will include all of the patterns and has the following structure:

```
pos_patterns/
    pattern_0/
        sequence: [...]
        contrib_scores: [...]
        hypothetical_contribs: [...]
        seqlets/
            n_seqlets: [...]
            start: [...]
            end: [...]
            example_idx: [...]
            is_revcomp: [...]
            sequence_features: [...]
            sequence: [...]
            contrib_scores: [...]
            hypothetical_contribs: [...]
        subpattern_0/
            ...
    pattern_1/
        ...
    ...
neg_patterns/
    pattern_0/
        ...
    pattern_1/
        ...
    ...
```

where `[...]` denotes that data is stored at that attribute. RNA-MoDISco seqlets are always in the provided orientation, and `is_revcomp` is retained only as a compatibility field that is false for newly generated results. In cases where there are not enough seqlets to consider a metacluster, that attribute (`neg_patterns` or `pos_patterns`) may not appear in the file.

## Generating reports

The TF-MoDISco report can be generated with the following command:
```sh
modisco report -i modisco_results.h5 -o report/ -s report/ -m motifs.txt
```

Each pattern produced by TF-MoDISco is compared against the database of motifs using [TOMTOM](https://meme-suite.org/meme/tools/tomtom). A good default choice is [this collection of human motifs](https://raw.githubusercontent.com/kundajelab/MotifCompendium/refs/heads/main/pipeline/data/MotifCompendium-Database-Human.meme.txt) produced by the [MotifCompendium](https://github.com/kundajelab/MotifCompendium) package.

The report details each pattern, including seqlet importance and spatial distributions, example seqlets at different importance levels, and motif visualizations.

For users who need the legacy report format use:
```sh
modisco report-simple -i modisco_results.h5 -o simple_report/ -s simple_report/ -m motifs.txt
```
