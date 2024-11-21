from cdcp.paths import ensure_dir
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import spikeinterface as si
import spikeinterface.toolkit as st
from spikeinterface import WaveformExtractor, extract_waveforms
import probeinterface as pi


def compute_unit_metrics(we, unit_features_folder, replace=False):
    ensure_dir(unit_features_folder)
    with tqdm(total=11) as pbar:
        # get templates
        sort_templates_loc = unit_features_folder / "templates.pickle"
        if replace or (not sort_templates_loc.exists()):
            sort_templates = pd.DataFrame(
                {
                    unit_id: [we.get_template(unit_id=unit_id, mode="median")]
                    for unit_id in we.sorting.unit_ids
                }
            ).T
            sort_templates.columns = ["template"]
            sort_templates.to_pickle(unit_features_folder / "templates.pickle")
        pbar.set_description("sort_templates")
        pbar.update(1)

        # template amplitude
        template_amplitude_loc = unit_features_folder / "template_amplitudes.pickle"
        if replace or (not template_amplitude_loc.exists()):
            template_amplitudes = pd.DataFrame(
                si.toolkit.postprocessing.get_template_amplitudes(
                    we, peak_sign="neg", mode="extremum"
                )
            ).T
            template_amplitudes.columns = [
                "amp_channel_{}".format(i) for i in template_amplitudes.columns
            ]
            template_amplitudes.to_pickle(template_amplitude_loc)
        pbar.set_description("template_amplitudes")
        pbar.update(1)

        # max channel
        max_channel_loc = unit_features_folder / "max_channel.pickle"
        if replace or (not max_channel_loc.exists()):
            max_channel = pd.DataFrame(
                {
                    key: [value]
                    for key, value in si.toolkit.postprocessing.get_template_extremum_channel(
                        we, peak_sign="neg", outputs="id"
                    ).items()
                }
            ).T
            max_channel.columns = ["max_channel"]
            max_channel.to_pickle(max_channel_loc)
        pbar.set_description("max_channel")
        pbar.update(1)

        # spike amplitude
        spike_amplitude_loc = unit_features_folder / "spike_amplitude.pickle"
        if replace or (not spike_amplitude_loc.exists()):
            spike_amplitude = pd.DataFrame(
                {
                    key: [value]
                    for key, value in si.toolkit.postprocessing.get_template_extremum_amplitude(
                        we, peak_sign="neg"
                    ).items()
                }
            ).T
            spike_amplitude.columns = ["spike_amplitude"]
            spike_amplitude.to_pickle(spike_amplitude_loc)
        pbar.set_description("sort_templates")
        pbar.update(1)

        # best channel
        best_channels_loc = unit_features_folder / "best_channels.pickle"
        if replace or (not best_channels_loc.exists()):
            best_channels = pd.DataFrame(
                si.toolkit.postprocessing.get_template_channel_sparsity(
                    we, num_channels=10, method="best_channels", peak_sign="neg"
                )
            ).T
            best_channels.columns = [
                "best_channel_{}".format(i) for i in best_channels.columns
            ]
            best_channels.to_pickle(best_channels_loc)
        pbar.set_description("best_channels")
        pbar.update(1)

        # center of mass
        center_of_mass_loc = unit_features_folder / "center_of_mass_loc.pickle"
        if replace or (not center_of_mass_loc.exists()):
            center_of_mass = pd.DataFrame(
                si.toolkit.postprocessing.compute_unit_centers_of_mass(
                    we, peak_sign="neg", num_channels=10
                )
            ).T
            center_of_mass.columns = ["center_of_mass_x", "center_of_mass_y"]
            center_of_mass.to_pickle(center_of_mass_loc)
        pbar.set_description("center_of_mass")
        pbar.update(1)

        # # of spikes
        n_spikes_loc = unit_features_folder / "n_spikes.pickle"
        if replace or (not n_spikes_loc.exists()):
            n_spikes = si.toolkit.qualitymetrics.misc_metrics.compute_num_spikes(we)
            n_spikes = pd.DataFrame({key: [value] for key, value in n_spikes.items()}).T
            n_spikes.columns = ["n_spikes"]
            n_spikes.to_pickle(n_spikes_loc)
        pbar.set_description("n_spikes")
        pbar.update(1)

        # ISI violations
        isi_violations_rate_loc = unit_features_folder / "isi_violations_rate.pickle"
        if replace or (not isi_violations_rate_loc.exists()):
            isi_violations = (
                si.toolkit.qualitymetrics.misc_metrics.compute_isi_violations(we)
            )
            isi_violations_count = pd.DataFrame(
                {
                    key: [value]
                    for key, value in isi_violations.isi_violations_count.items()
                }
            ).T
            isi_violations_count.columns = ["isi_violations_count"]
            isi_violations_count.to_pickle(
                unit_features_folder / "isi_violations_count.pickle"
            )
            isi_violations_rate = pd.DataFrame(
                {
                    key: [value]
                    for key, value in isi_violations.isi_violations_rate.items()
                }
            ).T
            isi_violations_rate.columns = ["isi_violations_rate"]
            isi_violations_rate.to_pickle(isi_violations_rate_loc)
        pbar.set_description("isi_violations_count")
        pbar.update(1)

        # SNR
        snrs_loc = unit_features_folder / "snrs.pickle"
        if replace or (not snrs_loc.exists()):
            snrs = si.toolkit.qualitymetrics.misc_metrics.compute_snrs(we)
            snrs = pd.DataFrame({key: [value] for key, value in snrs.items()}).T
            snrs.columns = ["snrs"]
            snrs.to_pickle(snrs_loc)
        pbar.set_description("snrs")
        pbar.update(1)

        # presence ratio
        presence_ratio_loc = unit_features_folder / "presence_ratio.pickle"
        if replace or (not presence_ratio_loc.exists()):
            presence_ratio = (
                si.toolkit.qualitymetrics.misc_metrics.compute_presence_ratio(we)
            )
            presence_ratio = pd.DataFrame(
                {key: [value] for key, value in presence_ratio.items()}
            ).T
            presence_ratio.columns = ["presence_ratio"]
            presence_ratio.to_pickle(presence_ratio_loc)
        pbar.set_description("presence_ratio")
        pbar.update(1)

        # amplitude cutoff
        amplitude_cutoff_loc = unit_features_folder / "amplitude_cutoff.pickle"
        if replace or (not amplitude_cutoff_loc.exists()):
            amplitude_cutoff = (
                si.toolkit.qualitymetrics.misc_metrics.compute_amplitudes_cutoff(we)
            )
            amplitude_cutoff = pd.DataFrame(
                {key: [value] for key, value in amplitude_cutoff.items()}
            ).T
            amplitude_cutoff.columns = ["amplitude_cutoff"]
            amplitude_cutoff.to_pickle(amplitude_cutoff_loc)
        pbar.set_description("amplitude_cutoff")
        pbar.update(1)
