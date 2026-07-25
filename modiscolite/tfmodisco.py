# tfmodisco.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import numpy as np

import scipy
import scipy.sparse

from collections import defaultdict

from . import affinitymat
from . import aggregator
from . import extract_seqlets
from . import core
from . import util
from . import cluster
from . import rna
from .progress import ensure_progress


def _standardize_input_shapes(one_hot, hypothetical_contribs):
	one_hot = np.asarray(one_hot)
	hypothetical_contribs = np.asarray(hypothetical_contribs)

	if one_hot.ndim != 3:
		raise ValueError("one_hot must be a 3D tensor with 4 or 6 sequence channels.")
	if hypothetical_contribs.ndim != 3:
		raise ValueError("hypothetical_contribs must be a 3D tensor with 4 channels.")
	if one_hot.shape[0] != hypothetical_contribs.shape[0]:
		raise ValueError(
			"one_hot and hypothetical_contribs must have the same number of examples."
		)

	length_first = (
		one_hot.shape[2] in (4, 6) and
		hypothetical_contribs.shape[2] == 4 and
		one_hot.shape[1] == hypothetical_contribs.shape[1]
	)
	channel_first = (
		one_hot.shape[1] in (4, 6) and
		hypothetical_contribs.shape[1] == 4 and
		one_hot.shape[2] == hypothetical_contribs.shape[2]
	)

	if length_first and channel_first:
		raise ValueError(
			"Ambiguous input shape: both axes look like sequence channels. "
			"Use non-ambiguous N x L x C or N x C x L tensors."
		)
	if length_first:
		return one_hot, hypothetical_contribs
	if channel_first:
		return one_hot.transpose(0, 2, 1), hypothetical_contribs.transpose(0, 2, 1)

	raise ValueError(
		"Incompatible input shapes. Sequence must be N x L x 4/6 or N x 4/6 x L; "
		"attributions must be N x L x 4 or N x 4 x L."
	)


def _prepare_inputs(one_hot, hypothetical_contribs, sequence_mask=None,
	region="all", missing_cds="skip", progress=None):
	progress = ensure_progress(progress)
	sequence_features, hypothetical_contribs = _standardize_input_shapes(
		one_hot, hypothetical_contribs)
	region = rna.normalize_region(region)
	missing_cds = rna.normalize_missing_cds(missing_cds)

	with progress.task("Validating input values"):
		if not np.all(np.isfinite(sequence_features)):
			raise ValueError("one_hot contains non-finite values.")
		if not np.all(np.isfinite(hypothetical_contribs)):
			raise ValueError("hypothetical_contribs contains non-finite values.")

	padding_mask = rna.infer_padding_mask(
		sequence=sequence_features, sequence_mask=sequence_mask,
		progress=progress)
	region_mask, region_metadata = rna.infer_region_mask(
		sequence=sequence_features, padding_mask=padding_mask, region=region,
		missing_cds=missing_cds, progress=progress)
	position_mask = padding_mask & region_mask

	if not np.any(position_mask):
		raise ValueError(
			"No valid positions remain after applying padding and region masks."
		)

	return (
		sequence_features[:, :, :4].astype("float32", copy=False),
		hypothetical_contribs.astype("float32", copy=False),
		sequence_features.astype("float32", copy=False),
		position_mask,
		padding_mask,
		region,
		region_metadata,
	)

def _density_adaptation(affmat_nn, seqlet_neighbors, tsne_perplexity,
	progress=None):
	progress = ensure_progress(progress)
	eps = 0.0000001

	n_rows = len(affmat_nn)
	with progress.task(
		"Density adaptation",
		total=4 * n_rows,
		unit="row",
	) as task:
		rows, cols, data = [], [], []
		for row in range(n_rows):
			for col, datum in zip(seqlet_neighbors[row], affmat_nn[row]):
				rows.append(row)
				cols.append(col)
				data.append(datum)
			task.advance()

		affmat_nn = scipy.sparse.csr_matrix((data, (rows, cols)), 
			shape=(n_rows, n_rows), dtype='float64')
		
		affmat_nn.data = np.maximum(np.log((1.0/(0.5*np.maximum(affmat_nn.data, eps)))-1), 0)
		affmat_nn.eliminate_zeros()

		counts_nn = scipy.sparse.csr_matrix((np.ones_like(affmat_nn.data), 
			affmat_nn.indices, affmat_nn.indptr), shape=affmat_nn.shape, dtype='float64')

		affmat_nn += affmat_nn.T
		counts_nn += counts_nn.T
		affmat_nn.data /= counts_nn.data
		del counts_nn

		betas = []
		for i in range(affmat_nn.shape[0]):
			betas.append(
				util.binary_search_perplexity(
					tsne_perplexity, affmat_nn[i].data))
			task.advance()

		normfactors = []
		for i, beta in enumerate(betas):
			normfactors.append(
				np.exp(-np.array(affmat_nn[i].data)/beta).sum()+1)
			task.advance()
		normfactors = np.array(normfactors)

		for i in range(affmat_nn.shape[0]):
			for j_idx in range(affmat_nn.indptr[i], affmat_nn.indptr[i+1]):
				j = affmat_nn.indices[j_idx]
				distance = affmat_nn.data[j_idx]

				rbf_i = np.exp(-distance / betas[i]) / normfactors[i]
				rbf_j = np.exp(-distance / betas[j]) / normfactors[j]

				affmat_nn.data[j_idx] = np.sqrt(rbf_i * rbf_j)
			task.advance()
		task.set_summary("{:,} affinity edges".format(affmat_nn.nnz))

	affmat_diags = scipy.sparse.diags(1.0 / normfactors)
	affmat_nn += affmat_diags
	return affmat_nn


def _filter_patterns(patterns, min_seqlet_support, window_size, 
	min_ic_in_window, background, ppm_pseudocount):
	passing_patterns = []
	for pattern in patterns:
		if len(pattern.seqlets) < min_seqlet_support:
			continue

		ppm = pattern.sequence
		per_position_ic = util.compute_per_position_ic(ppm=ppm, 
			background=background, pseudocount=ppm_pseudocount)

		if len(per_position_ic) < window_size:       
			if np.sum(per_position_ic) < min_ic_in_window:
				continue
		else:
			#do the sliding window sum rearrangement
			windowed_ic = np.sum(util.rolling_window(
				a=per_position_ic, window=window_size), axis=-1)

			if np.max(windowed_ic) < min_ic_in_window:
				continue

		passing_patterns.append(pattern)

	return passing_patterns


def _patterns_from_clusters(seqlets, track_set, min_overlap,
	min_frac, min_num, flank_to_add, window_size, bg_freq, cluster_indices, 
	track_sign, progress=None):

	progress = ensure_progress(progress)
	seqlet_sort_metric = lambda x: -np.sum(np.abs(x.contrib_scores))
	num_clusters = max(cluster_indices+1)
	cluster_to_seqlets = defaultdict(list) 

	for seqlet, idx in zip(seqlets, cluster_indices):
		cluster_to_seqlets[idx].append(seqlet)

	patterns = []
	with progress.task(
		"Building cluster patterns",
		total=len(seqlets),
		unit="seqlet",
		detail="{} Leiden clusters".format(num_clusters),
	) as task:
		for i in range(num_clusters):
			sorted_seqlets = sorted(
				cluster_to_seqlets[i], key=seqlet_sort_metric) 
			pattern = core.SeqletSet([sorted_seqlets[0]])
			task.advance()

			if len(sorted_seqlets) > 1:
				pattern = aggregator.merge_in_seqlets_filledges(
					parent_pattern=pattern,
					seqlets_to_merge=sorted_seqlets[1:],
					track_set=track_set, metric=affinitymat.jaccard,
					min_overlap=min_overlap, progress_task=task)

			pattern = aggregator.polish_pattern(pattern, min_frac=min_frac, 
				min_num=min_num, track_set=track_set, flank=flank_to_add, 
				window_size=window_size, bg_freq=bg_freq)

			if pattern is not None:
				if np.sign(np.sum(pattern.contrib_scores)) == track_sign:
					patterns.append(pattern)
		task.set_summary(
			"{} clusters → {} strand-consistent patterns".format(
				num_clusters, len(patterns))
		)

	return patterns


def _filter_by_correlation(seqlets, seqlet_neighbors, coarse_affmat_nn, 
	fine_affmat_nn, correlation_threshold, progress=None):

	progress = ensure_progress(progress)
	correlations = []
	with progress.task(
		"Correlation filtering",
		total=2 * len(seqlets),
		unit="row",
	) as task:
		for fine_affmat_row, coarse_affmat_row in zip(
			fine_affmat_nn, coarse_affmat_nn):
			to_compare_mask = np.abs(fine_affmat_row) > 0
			corr = scipy.stats.spearmanr(fine_affmat_row[to_compare_mask],
				coarse_affmat_row[to_compare_mask])
			correlations.append(corr.correlation)
			task.advance()

		correlations = np.array(correlations)
		filtered_rows_mask = np.array(correlations) > correlation_threshold

		filtered_seqlets = [seqlet for seqlet, mask in zip(seqlets, 
			filtered_rows_mask) if mask == True]

		#figure out a mapping from pre-filtering to the
		# post-filtering indices
		new_idx_mapping = np.cumsum(filtered_rows_mask) - 1
		retained_indices = set(np.where(filtered_rows_mask == True)[0])

		filtered_neighbors = []
		filtered_affmat_nn = []
		for old_row_idx, (old_neighbors, affmat_row) in enumerate(
			zip(seqlet_neighbors, fine_affmat_nn)):
			if old_row_idx in retained_indices:
				filtered_old_neighbors = [neighbor for neighbor in old_neighbors if neighbor in retained_indices]
				filtered_affmat_row = [affmatval for affmatval, neighbor in zip(affmat_row,old_neighbors) if neighbor in retained_indices]
				filtered_neighbors_row = [new_idx_mapping[neighbor] for neighbor in filtered_old_neighbors]
				filtered_neighbors.append(filtered_neighbors_row)
				filtered_affmat_nn.append(filtered_affmat_row)
			task.advance()
		task.set_summary(
			"{:,} → {:,} retained seqlets".format(
				len(seqlets), len(filtered_seqlets))
		)

	return filtered_seqlets, filtered_neighbors, filtered_affmat_nn


def seqlets_to_patterns(seqlets, track_set, track_signs=None, 
	min_overlap_while_sliding=0.7, nearest_neighbors_to_compute=500, 
	affmat_correlation_threshold=0.15, tsne_perplexity=10.0, 
	n_leiden_iterations=-1, n_leiden_runs=50, frac_support_to_trim_to=0.2,
	min_num_to_trim_to=30, trim_to_window_size=20, initial_flank_to_add=5,
	final_flank_to_add=0,
	prob_and_pertrack_sim_merge_thresholds=[(0.8,0.8), (0.5, 0.85), (0.2, 0.9)],
	prob_and_pertrack_sim_dealbreaker_thresholds=[(0.4, 0.75), (0.2,0.8), (0.1, 0.85), (0.0,0.9)],
	subcluster_perplexity=50, merging_max_seqlets_subsample=1000,
	final_min_cluster_size=20,min_ic_in_window=0.6, min_ic_windowsize=6,
	ppm_pseudocount=0.001, progress=None):

	progress = ensure_progress(progress)
	bg_freq = np.mean([seqlet.sequence for seqlet in seqlets], axis=(0, 1)) 
	progress.verbose_note(
		"Background frequencies: A={:.4f} C={:.4f} G={:.4f} U={:.4f}"
		.format(*bg_freq)
	)

	seqlets_sorter = (lambda arr: sorted(arr, key=lambda x:
		-np.sum(np.abs(x.contrib_scores))))

	seqlets = seqlets_sorter(seqlets)

	for round_idx in range(2):
		if len(seqlets) == 0:
			return None

		with progress.section(
			"Refinement round {} of 2 — {:,} seqlets".format(
				round_idx + 1, len(seqlets))
		):
			# Step 1: Generate coarse resolution
			coarse_affmat_nn, seqlet_neighbors = (
				affinitymat.cosine_similarity_from_seqlets(
					seqlets=seqlets,
					n_neighbors=nearest_neighbors_to_compute,
					sign=track_signs,
					progress=progress))

			# Step 2: Generate fine representation
			fine_affmat_nn = affinitymat.jaccard_from_seqlets(
				seqlets=seqlets, seqlet_neighbors=seqlet_neighbors,
				min_overlap=min_overlap_while_sliding,
				progress=progress)

			if round_idx == 0:
				filtered_seqlets, seqlet_neighbors, filtered_affmat_nn = (
					_filter_by_correlation(
						seqlets, seqlet_neighbors, coarse_affmat_nn,
						fine_affmat_nn, affmat_correlation_threshold,
						progress=progress))
			else:
				filtered_seqlets = seqlets
				filtered_affmat_nn = fine_affmat_nn

			del coarse_affmat_nn
			del fine_affmat_nn
			del seqlets

			# Step 4: Density adaptation
			csr_density_adapted_affmat = _density_adaptation(
				filtered_affmat_nn, seqlet_neighbors, tsne_perplexity,
				progress=progress)

			del filtered_affmat_nn
			del seqlet_neighbors

			# Step 5: Clustering
			cluster_indices = cluster.LeidenCluster(
				csr_density_adapted_affmat,
				n_seeds=n_leiden_runs,
				n_leiden_iterations=n_leiden_iterations,
				progress=progress)

			del csr_density_adapted_affmat

			patterns = _patterns_from_clusters(
				filtered_seqlets, 
				track_set=track_set, 
				min_overlap=min_overlap_while_sliding, 
				min_frac=frac_support_to_trim_to, 
				min_num=min_num_to_trim_to, 
				flank_to_add=initial_flank_to_add, 
				window_size=trim_to_window_size, 
				bg_freq=bg_freq, 
				cluster_indices=cluster_indices, 
				track_sign=track_signs,
				progress=progress)

			#obtain unique seqlets from adjusted motifs
			seqlets = list(dict([(y.string, y)
							 for x in patterns for y in x.seqlets]).values())
			progress.note(
				"Round {} complete: {:,} patterns • {:,} aligned seqlets"
				.format(round_idx + 1, len(patterns), len(seqlets))
			)

	del seqlets

	with progress.section("Splitting and merging patterns"):
		merged_patterns, pattern_merge_hierarchy = (
			aggregator._detect_spurious_merging(
				patterns=patterns, track_set=track_set,
				perplexity=subcluster_perplexity, 
				min_in_subcluster=max(
					final_min_cluster_size, subcluster_perplexity), 
				min_overlap=min_overlap_while_sliding,
				prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
				prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
				min_frac=frac_support_to_trim_to,
				min_num=min_num_to_trim_to,
				flank_to_add=initial_flank_to_add,
				window_size=trim_to_window_size, bg_freq=bg_freq,
				max_seqlets_subsample=merging_max_seqlets_subsample,
				n_seeds=n_leiden_runs,
				progress=progress))

	#Now start merging patterns 
	merged_patterns = sorted(merged_patterns, key=lambda x: -len(x.seqlets))

	with progress.task("Filtering final patterns") as task:
		patterns = _filter_patterns(
			merged_patterns, 
			min_seqlet_support=final_min_cluster_size, 
			window_size=min_ic_windowsize,
			min_ic_in_window=min_ic_in_window, 
			background=bg_freq, ppm_pseudocount=ppm_pseudocount)
		task.set_summary(
			"{} merged → {} passing patterns".format(
				len(merged_patterns), len(patterns))
		)

	#apply subclustering procedure on the final patterns
	with progress.task(
		"Final pattern subclustering",
		total=len(patterns),
		unit="pattern",
		record_duration=False,
	) as task:
		for patternidx, pattern in enumerate(patterns):
			with progress.section(
				"Final pattern {}/{} — {:,} seqlets".format(
					patternidx + 1, len(patterns), len(pattern.seqlets))
			):
				pattern = aggregator._expand_seqlets_to_fill_pattern(
					pattern, track_set, 
					left_flank_to_add=final_flank_to_add,
					right_flank_to_add=final_flank_to_add)

				pattern.compute_subpatterns(
					subcluster_perplexity, 
					n_seeds=n_leiden_runs,
					n_iterations=n_leiden_iterations,
					progress=progress)
				
				patterns[patternidx] = pattern
			task.advance()
		task.set_summary("{} final patterns".format(len(patterns)))

	return patterns


def TFMoDISco(one_hot, hypothetical_contribs, sliding_window_size=21, 
	flank_size=10, min_metacluster_size=100,
	weak_threshold_for_counting_sign=0.8, max_seqlets_per_metacluster=20000,
	target_seqlet_fdr=0.2, min_passing_windows_frac=0.03,
	max_passing_windows_frac=0.2, n_leiden_runs=50, n_leiden_iterations=-1, 
	min_overlap_while_sliding=0.7, nearest_neighbors_to_compute=500, 
	affmat_correlation_threshold=0.15, tsne_perplexity=10.0, 
	frac_support_to_trim_to=0.2, min_num_to_trim_to=30, trim_to_window_size=30, 
	initial_flank_to_add=10, final_flank_to_add=0,
	prob_and_pertrack_sim_merge_thresholds=[(0.8,0.8), (0.5, 0.85), (0.2, 0.9)],
	prob_and_pertrack_sim_dealbreaker_thresholds=[(0.4, 0.75), (0.2,0.8), (0.1, 0.85), (0.0,0.9)],
	subcluster_perplexity=50, merging_max_seqlets_subsample=1000,
	final_min_cluster_size=20, min_ic_in_window=0.6, min_ic_windowsize=6,
	ppm_pseudocount=0.001, sequence_mask=None, region="all",
	missing_cds="skip", verbose=False, progress=None):

	if progress is None and verbose:
		progress = True
	progress = ensure_progress(progress)
	if verbose and hasattr(progress, "verbose"):
		progress.verbose = True
	with progress.section("Preparing RNA inputs"):
		(one_hot, hypothetical_contribs, sequence_features, position_mask,
			padding_mask, region, region_metadata) = _prepare_inputs(
				one_hot=one_hot,
				hypothetical_contribs=hypothetical_contribs,
				sequence_mask=sequence_mask, region=region,
				missing_cds=missing_cds, progress=progress)
		usable_examples = int(np.sum(np.any(position_mask, axis=1)))
		progress.note(
			"{:,}/{:,} usable RNAs • {:,} valid {} positions".format(
				usable_examples, position_mask.shape[0],
				int(np.sum(position_mask)), region)
		)
		progress.verbose_note(
			"Prepared sequence shape={} • attribution shape={} • "
			"position-mask density={:.2%}".format(
				one_hot.shape,
				hypothetical_contribs.shape,
				float(np.mean(position_mask)),
			)
		)

	with progress.task("Preparing contribution tracks"):
		contrib_scores = np.multiply(one_hot, hypothetical_contribs)

		track_set = core.TrackSet(one_hot=one_hot, 
			contrib_scores=contrib_scores,
			hypothetical_contribs=hypothetical_contribs,
			position_mask=position_mask,
			sequence_features=sequence_features,
			padding_mask=padding_mask,
			region=region,
			region_metadata=region_metadata)

	with progress.section("Seqlet discovery"):
		seqlet_coords, threshold = extract_seqlets.extract_seqlets(
			attribution_scores=contrib_scores.sum(axis=2),
			window_size=sliding_window_size,
			flank=flank_size,
			suppress=(int(0.5*sliding_window_size) + flank_size),
			target_fdr=target_seqlet_fdr,
			min_passing_windows_frac=min_passing_windows_frac,
			max_passing_windows_frac=max_passing_windows_frac,
			weak_threshold_for_counting_sign=weak_threshold_for_counting_sign,
			position_mask=position_mask,
			progress=progress)

		with progress.task("Materializing seqlet tracks") as task:
			seqlets = track_set.create_seqlets(seqlet_coords) 
			task.set_summary("{:,} candidate seqlets".format(len(seqlets)))

		pos_seqlets, neg_seqlets = [], []
		with progress.task(
			"Classifying seqlet signs",
			total=len(seqlets),
			unit="seqlet",
		) as task:
			for seqlet in seqlets:
				flank = int(0.5*(len(seqlet)-sliding_window_size))
				core_end = len(seqlet) - flank if flank > 0 else len(seqlet)
				attr = np.sum(seqlet.contrib_scores[flank:core_end])

				if attr > threshold:
					pos_seqlets.append(seqlet)
				elif attr < -threshold:
					neg_seqlets.append(seqlet)
				task.advance()
			task.set_summary(
				"{:,} positive • {:,} negative".format(
					len(pos_seqlets), len(neg_seqlets))
			)

	del seqlets

	if len(pos_seqlets) > min_metacluster_size:
		n_extracted_pos = len(pos_seqlets)
		pos_seqlets = pos_seqlets[:max_seqlets_per_metacluster]
		with progress.section(
			"Positive motif discovery — {:,} seqlets".format(
				len(pos_seqlets))
		):
			if n_extracted_pos > len(pos_seqlets):
				progress.note(
					"Capped {:,} extracted positive seqlets to {:,}".format(
						n_extracted_pos, len(pos_seqlets))
				)

			with progress.task(
				"Positive motif pipeline", record_duration=False) as task:
				pos_patterns = seqlets_to_patterns(
					seqlets=pos_seqlets,
					track_set=track_set, 
					track_signs=1,
					min_overlap_while_sliding=min_overlap_while_sliding,
					nearest_neighbors_to_compute=nearest_neighbors_to_compute,
					affmat_correlation_threshold=affmat_correlation_threshold,
					tsne_perplexity=tsne_perplexity,
					n_leiden_iterations=n_leiden_iterations,
					n_leiden_runs=n_leiden_runs,
					frac_support_to_trim_to=frac_support_to_trim_to,
					min_num_to_trim_to=min_num_to_trim_to,
					trim_to_window_size=trim_to_window_size,
					initial_flank_to_add=initial_flank_to_add,
					final_flank_to_add=final_flank_to_add,
					prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
					prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
					subcluster_perplexity=subcluster_perplexity,
					merging_max_seqlets_subsample=merging_max_seqlets_subsample,
					final_min_cluster_size=final_min_cluster_size,
					min_ic_in_window=min_ic_in_window,
					min_ic_windowsize=min_ic_windowsize,
					ppm_pseudocount=ppm_pseudocount,
					progress=progress)
				n_patterns = (
					0 if pos_patterns is None else len(pos_patterns))
				task.set_summary("{} final positive patterns".format(n_patterns))
	else:
		pos_patterns = None
		progress.skipped(
			"Positive motif discovery: {:,} seqlets did not exceed the "
			"minimum metacluster size of {:,}".format(
				len(pos_seqlets), min_metacluster_size)
		)

	if len(neg_seqlets) > min_metacluster_size:
		n_extracted_neg = len(neg_seqlets)
		neg_seqlets = neg_seqlets[:max_seqlets_per_metacluster]
		with progress.section(
			"Negative motif discovery — {:,} seqlets".format(
				len(neg_seqlets))
		):
			if n_extracted_neg > len(neg_seqlets):
				progress.note(
					"Capped {:,} extracted negative seqlets to {:,}".format(
						n_extracted_neg, len(neg_seqlets))
				)

			with progress.task(
				"Negative motif pipeline", record_duration=False) as task:
				neg_patterns = seqlets_to_patterns(
					seqlets=neg_seqlets,
					track_set=track_set, 
					track_signs=-1,
					min_overlap_while_sliding=min_overlap_while_sliding,
					nearest_neighbors_to_compute=nearest_neighbors_to_compute,
					affmat_correlation_threshold=affmat_correlation_threshold,
					tsne_perplexity=tsne_perplexity,
					n_leiden_iterations=n_leiden_iterations,
					n_leiden_runs=n_leiden_runs,
					frac_support_to_trim_to=frac_support_to_trim_to,
					min_num_to_trim_to=min_num_to_trim_to,
					trim_to_window_size=trim_to_window_size,
					initial_flank_to_add=initial_flank_to_add,
					final_flank_to_add=final_flank_to_add,
					prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
					prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
					subcluster_perplexity=subcluster_perplexity,
					merging_max_seqlets_subsample=merging_max_seqlets_subsample,
					final_min_cluster_size=final_min_cluster_size,
					min_ic_in_window=min_ic_in_window,
					min_ic_windowsize=min_ic_windowsize,
					ppm_pseudocount=ppm_pseudocount,
					progress=progress)
				n_patterns = (
					0 if neg_patterns is None else len(neg_patterns))
				task.set_summary("{} final negative patterns".format(n_patterns))
	else:
		neg_patterns = None
		progress.skipped(
			"Negative motif discovery: {:,} seqlets did not exceed the "
			"minimum metacluster size of {:,}".format(
				len(neg_seqlets), min_metacluster_size)
		)

	return pos_patterns, neg_patterns
