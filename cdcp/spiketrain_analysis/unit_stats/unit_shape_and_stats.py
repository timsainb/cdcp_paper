import pandas as pd
from cdcp.paths import DATA_DIR, ensure_dir
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib2 import Path
from tqdm.autonotebook import tqdm


def get_unit_shape(unit_to_analyze, unit_features):
    spike_half_widths = []
    spike_full_widths = []
    spike_rates = []
    spike_amplitudes = []
    isi_violations_rates = []
    best_channel_templates = []
    max_channel_templates = []
    presence_ratios = []
    for sort_unit, sort in unit_to_analyze.sort_units:
        unit_row = unit_features[
            (unit_features.unit == sort_unit) & (unit_features.recording_id == sort)
        ]
        if len(unit_row) == 0:
            continue
        unit_row = unit_row.iloc[0]
        best_channel_template = unit_row.template[:, unit_row.best_channel_0]
        max_channel_template = unit_row.template[:, np.argmax(np.nanmax(unit_row.template, axis=0))]
        # add features
        spike_rates.append(unit_row.spike_rate)
        best_channel_templates.append(best_channel_template)
        max_channel_templates.append(max_channel_template)
        spike_amplitudes.append(unit_row.spike_amplitude)
        isi_violations_rates.append(unit_row.isi_violations_rate)
        presence_ratios.append(unit_row.presence_ratio)

        peak = np.argmax(np.abs(best_channel_template))
        sign_change = np.where(
            np.sign(best_channel_template[:-1]) != np.sign(best_channel_template[1:])
        )[0]
        sign_changes_after_peak = sign_change[sign_change > peak]
        sign_changes_before_peak = sign_change[sign_change < peak]
        if len(sign_changes_after_peak) > 0:
            spike_half_widths.append(sign_changes_after_peak[0] - peak)
            if len(sign_changes_before_peak) > 0:
                spike_full_widths.append(
                    sign_changes_after_peak[0] - sign_changes_before_peak[-1]
                )
    if len(spike_half_widths) == 0:
        if len(unit_row) == 0:
            print(sort_unit, sort)
        return pd.Series({})
    #breakme
    # make sure templates are all the same length
    max_template = int(np.max([len(i) for i in best_channel_templates if type(i) == np.ndarray]))
    best_channel_templates = [i for i in best_channel_templates if (type(i) == np.ndarray)]
    best_channel_templates = [i for i in best_channel_templates if len(i) == max_template]
    max_template = int(np.max([len(i) for i in max_channel_templates if type(i) == np.ndarray]))
    max_channel_templates = [i for i in max_channel_templates if (type(i) == np.ndarray)]
    max_channel_templates = [i for i in max_channel_templates if len(i) == max_template]
    
    return pd.Series(
        
        {
            "half_width": np.nanmedian(spike_half_widths), 
            "full_width": np.nanmean(spike_full_widths),
            "best_channel_template": np.nanmedian(best_channel_templates, axis=0),
            "highest_amplitude_channel_template": np.nanmedian(best_channel_templates, axis=0),
            "spike_rate": np.nanmedian(spike_rates),
            "spike_amplitude": np.nanmedian(spike_amplitudes),
            "presence_ratio": np.nanmedian(presence_ratios),
            "isi_violation_rate": np.nanmedian(isi_violations_rates)
        }
        )


def compute_unit_shape(
    unit,
    unit_features,
    unit_to_analyze,
):
    return  get_unit_shape(
        unit_to_analyze, unit_features
    )

