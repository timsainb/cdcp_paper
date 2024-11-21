import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d


def get_psth_from_trial_aligned_spikes(
    trial_aligned_spikes_loc, padding_s=0, nbins=200, gaussian_sigma_ms=5
):
    bin_ms = 1000 / nbins
    gaussian_sigma = gaussian_sigma_ms / bin_ms

    trial_aligned_spikes = pd.read_pickle(trial_aligned_spikes_loc)
    # get cue info
    trial_aligned_spikes["cue"] = [
        i.split("_")[0] for i in trial_aligned_spikes.stim.values
    ]
    trial_aligned_spikes["interp"] = [
        i.split("_")[1] for i in trial_aligned_spikes.stim.values
    ]
    trial_aligned_spikes["interp_point"] = [
        int(i.split("_")[2]) for i in trial_aligned_spikes.stim.values
    ]
    # align times
    trial_aligned_spikes.loc[
        trial_aligned_spikes.cue.isin(["CL1", "CL0", "CN", "CR0", "CR1"]),
        "stim_length",
    ] = 2
    trial_aligned_spikes.loc[trial_aligned_spikes.cue.isin(["NC"]), "stim_length"] = 1

    spike_psth = []
    for idx, row in tqdm(
        trial_aligned_spikes.iterrows(), total=len(trial_aligned_spikes), leave=False
    ):
        if row.cue in ["CL1", "CL0", "CN", "CR0", "CR1"]:
            bins = np.linspace(1, 2, nbins)
        else:
            bins = np.linspace(0, 1, nbins)

        # bins = np.linspace(-padding_s, row.stim_length + padding_s, nbins)
        hist, bin_edges = np.histogram(row.spike_times, bins, density=False)
        gauss_convolved_psth = gaussian_filter1d(
            hist.astype("float"), gaussian_sigma, mode="constant"
        )
        spike_psth.append(gauss_convolved_psth)

    trial_aligned_spikes["psth"] = spike_psth
    # psth_array = np.vstack(
    #    trial_aligned_spikes[trial_aligned_spikes.stim_length == 2]
    #    .sort_values(by=["cue", "interp", "interp_point"])
    #    .psth.values
    #
    # )
    return trial_aligned_spikes
