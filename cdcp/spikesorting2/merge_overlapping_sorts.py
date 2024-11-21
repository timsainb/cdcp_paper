import spikeinterface as si
from spikeinterface.core import NumpySorting
import spikeinterface.comparison as sc
import numpy as np
from tqdm.autonotebook import tqdm
import pandas as pd


def subset_sort(sort, subset_start, subset_end):
    if subset_end is None:
        subset_end = np.inf
    sort_units = sort.get_unit_ids()
    spike_trains = [
        sort.get_unit_spike_train(unit)
        for unit in tqdm(sort_units, desc="sort units", leave=False)
    ]
    spike_trains = [np.unique(i) for i in spike_trains]
    spike_trains = [
        np.array(spike_train[(spike_train > subset_start) & (spike_train < subset_end)])
        - subset_start
        for spike_train in spike_trains
    ]
    labels = np.concatenate(
        [
            np.repeat(sort_unit, len(train))
            for sort_unit, train in zip(sort_units, spike_trains)
        ]
    )
    times = np.concatenate(spike_trains)
    # sort times and labels
    # times_sort = np.argsort(times)
    # times = times[times_sort]
    # labels=labels[times_sort]
    subset_sort = NumpySorting.from_times_labels(
        times, labels, sort.get_sampling_frequency()
    )
    return subset_sort


def get_putative_overlapping_units(
    merge_row, spikesorting_directory, putative_match_thresh=0.2, recompute=False
):
    # load matching_unit_df if it already exists
    matching_units_df_loc = merge_row.sort_loc / "units_to_merge.pickle"
    if matching_units_df_loc.exists() and not recompute:
        print("loading matching units df: {}".format(merge_row.merge_recording_id))
        matching_units_df = pd.read_pickle(matching_units_df_loc)
    else:
        print(
            "matching_unit_df did not exist, generating: {}".format(
                merge_row.merge_recording_id
            )
        )
        # grab sort for pre
        sort_loc_prev = (
            spikesorting_directory
            / "sorts"
            / merge_row.recording_id_prev
            / merge_row.sorter
            / "denoised_{}".format(merge_row.denoised)
            / "grouping_{}".format(merge_row.grouping)
        )

        # grab sort for pre
        sort_loc_next = (
            spikesorting_directory
            / "sorts"
            / merge_row.recording_id_next
            / merge_row.sorter
            / "denoised_{}".format(merge_row.denoised)
            / "grouping_{}".format(merge_row.grouping)
        )

        try:
            # load sorts
            print("\tLoading sorts")
            merge_sort = si.core.NpzSortingExtractor(merge_row.sort_loc / "sort.npz")
            sort_prev = si.core.NpzSortingExtractor(sort_loc_prev / "sort.npz")
            sort_next = si.core.NpzSortingExtractor(sort_loc_next / "sort.npz")

        except FileNotFoundError as e:
            print(e)
            return None, None, None, None, None

        # grab subset of previous sort
        sort_prev_subset = subset_sort(
            sort_prev,
            merge_row.prev_row_sample_start,
            merge_row.recording_prev_n_samples_total,
        )

        # grab subset of previous sort
        sort_next_subset = subset_sort(sort_next, 0, merge_row.next_row_sample_end)

        # grab the relevant part of each sort for comparison
        merge_sort_beginning = subset_sort(
            merge_sort,
            0,
            merge_row.recording_prev_n_samples_total - merge_row.prev_row_sample_start,
        )
        merge_sort_end = subset_sort(
            merge_sort,
            merge_row.recording_prev_n_samples_total - merge_row.prev_row_sample_start,
            None,
        )

        # compare merge_sort_beginning and sort_prev_subset
        sort_comparison_prev = sc.compare_two_sorters(
            sorting1=merge_sort_beginning,
            sorting2=sort_prev_subset,
            sorting1_name="prev",
            sorting2_name="overlap",
            delta_time=0.5,
            match_score=0.5,
            chance_score=0.1,
        )

        # compare merge_sort_beginning and sort_prev_subset
        sort_comparison_next = sc.compare_two_sorters(
            sorting1=merge_sort_end,
            sorting2=sort_next_subset,
            sorting1_name="next",
            sorting2_name="overlap",
            delta_time=0.5,
            match_score=0.5,
            chance_score=0.1,
        )

        # get matching units
        matching_units_df_prev = (
            sort_comparison_prev.agreement_scores.reset_index().melt(id_vars="index")
        )
        matching_units_df_prev.columns = ["overlap_sort", "prev_sort", "agreement_prev"]
        matching_units_df_prev = matching_units_df_prev[
            matching_units_df_prev.agreement_prev > putative_match_thresh
        ]

        matching_units_df_next = (
            sort_comparison_next.agreement_scores.reset_index().melt(id_vars="index")
        )
        matching_units_df_next.columns = ["overlap_sort", "next_sort", "agreement_next"]
        matching_units_df_next = matching_units_df_next[
            matching_units_df_next.agreement_next > putative_match_thresh
        ]

        # merge matching units across prev->overlap->next
        matching_units_df = matching_units_df_prev.merge(
            matching_units_df_next, how="inner", on="overlap_sort"
        )

        unit_n_spikes_prev = {
            i: len(sort_prev_subset.get_unit_spike_train(i))
            for i in sort_prev_subset.unit_ids
        }
        unit_n_spikes_next = {
            i: len(sort_next_subset.get_unit_spike_train(i))
            for i in sort_next_subset.unit_ids
        }
        matching_units_df["prev_n_spikes"] = [
            unit_n_spikes_prev[i] for i in matching_units_df.prev_sort.values
        ]
        matching_units_df["next_n_spikes"] = [
            unit_n_spikes_next[i] for i in matching_units_df.next_sort.values
        ]

        matching_units_df.to_pickle(matching_units_df_loc)

    # TODO : print # spikes, # units, # overlap at some threshold (e.g. 0.5)
    (
        n_merged_prev,
        total_spikes_prev,
        n_merged_next,
        total_spikes_next,
    ) = get_merged_ratios(
        matching_units_df,
        matching_units_df["prev_n_spikes"].values,
        matching_units_df["next_n_spikes"].values,
    )

    return (
        matching_units_df,
        n_merged_prev,
        total_spikes_prev,
        n_merged_next,
        total_spikes_next,
    )


def get_merged_ratios(
    matching_units_df, unit_n_spikes_prev, unit_n_spikes_next, thresh=0.5
):
    over_thresh = (matching_units_df.agreement_prev > thresh) & (
        matching_units_df.agreement_next > thresh
    )
    prev_merged_units = np.unique(matching_units_df[over_thresh].prev_sort.values)
    next_merged_units = np.unique(matching_units_df[over_thresh].next_sort.values)
    n_merged_prev = np.sum(unit_n_spikes_prev[over_thresh])
    n_merged_next = np.sum(unit_n_spikes_next[over_thresh])
    total_spikes_prev = np.sum(list(unit_n_spikes_prev))
    total_spikes_next = np.sum(list(unit_n_spikes_next))
    return n_merged_prev, total_spikes_prev, n_merged_next, total_spikes_next
