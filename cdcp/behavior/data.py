import numpy as np
import pandas as pd


def bin_data(x, num_bins=20):
    """ Bins x into a num_bins bins
    """
    bins = np.linspace(np.min(x), np.max(x), num_bins)
    digitized = np.digitize(np.array([int(i) for i in x]), bins=bins, right=True)
    # rescaled = rescale(digitized, np.min(x), np.max(x))
    return digitized, bins


def sum_data(
    data, groups=["pos_bin", "condition_type", "cues", "right_stims", "left_stims"]
):
    """
    Create a summary dataframe for responses by the response
    """
    # group/summarize data
    stats_df = data.groupby(groups).agg([np.mean, np.std, len])
    stats_df = stats_df.reset_index(level=groups)
    stats_df = stats_df.reset_index()
    return stats_df


def cue_direction(cue):
    """ specifies the direction of a cue
    """
    return {
        "CL0": "L",
        "CL1": "L",
        "CN": "N",
        "CR0": "R",
        "CR1": "R",
        "N": "N",
        "NC": "N",
    }[cue]


def prepare_behavior_data(data, num_bins=16, num_prev=np.inf):
    """ parses information from raw bahav_data dataframes
    """
    # subset normal trials
    data = data[data["type_"] == "normal"]
    data = data[data["response"] != "none"]
    
    # Subset part of the dataset
    if num_prev != np.inf:
        data = data[-num_prev:]

    # Extract the response as a boolean
    data["response_bool"] = 0
    data.loc[data["response"] == "L", "response_bool"] = 1
    
    # flip interpolation 0-127 so 0 corresponds to left, and 127 corresponds to right (to make it easier to read)
    data['interpolation_point'] = 127 - data['interpolation_point']
    data['response_bool'] = 1 - data['response_bool'] 

    # specify interpolation info
    data["interpolation"] = [
        ls + rs for ls, rs in zip(data.left_stim.values, data.right_stim.values)
    ]

    # bin data into 32 equally sized bins
    data["pos_bin"], bins = bin_data(
        data["interpolation_point"].values.astype("float32"), num_bins
    )
    data["cue_direction"] = data.cue_id.apply(cue_direction)
    return data, bins
