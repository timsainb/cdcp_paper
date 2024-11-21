import numpy as np
from probeinterface.utils import combine_probes
from probeinterface import Probe, get_probe, ProbeGroup


def make_probe_group(probes_list, probe_x_offset=10000):
    if len(probes_list) == 1:
        probe_group = ProbeGroup()
        for manufacturer, name in probes_list:
            probe = make_probes(manufacturer, name)
            probe_group.add_probe(probe)
    else:
        probes_numpy_list = []
        total_probe_x_offset = 0
        site_num = 0
        for manufacturer, name in probes_list:
            probe = make_probes(manufacturer, name)
            probe_numpy = probe.to_numpy()
            for i in range(len(probe_numpy)):
                probe_numpy[i][0] += total_probe_x_offset
                probe_numpy[i][5] = site_num
                site_num += 1
            probes_numpy_list.append(probe_numpy)
            total_probe_x_offset += probe_x_offset
        probe_numpy_full = np.hstack(probes_numpy_list)
        probe = Probe.from_numpy(probe_numpy_full)
        # print(probe.to_numpy())
        probe.set_device_channel_indices(np.arange(len(probe_numpy_full)))
        probe_group = ProbeGroup()
        probe_group.add_probe(probe)
    return probe_group


def make_probes(
    probe_manufacturer,
    probe_name,
    contact_shape="circle",
    contact_shape_params={"radius": 5},
):
    """
    Tries to grab probe from probeinterface if it exists, otherwise, imports it from probe_maps

    Parameters
    ----------
    probe_manufacturer : [type]
        [description]
    probe_name : [type]
        [description]
    contact_shape : str, optional
        [description], by default "circle"
    contact_shape_params : dict, optional
        [description], by default {"radius": 5}

    Returns
    -------
    [type]
        [description]
    """
    try:
        probe = get_probe(probe_manufacturer, probe_name)
    except:
        geom = np.array(probe_maps[probe_name]["geom"])
        channel_groups = probe_maps[probe_name]["channel_groups"]
        # TODO set channel group info
        probe_list = []
        channels = []
        for group in channel_groups:
            probe = Probe(ndim=2, si_units="um")
            group_channels = np.array(channel_groups[group]["channels"])
            probe.set_contacts(
                positions=geom[group_channels],
                shapes=contact_shape,
                shape_params=contact_shape_params,
            )
            channels.append(group_channels)
            # probe.set_device_channel_indices(np.arange(len(geom[group_channels])))
            # probe.create_auto_shape(probe_type='tip', margin=25)
            probe_list.append(probe)
        probe = combine_probes(probe_list)
        probe.set_device_channel_indices(np.concatenate(channels))
    return probe


probe_maps = {
    "Buzsaki32": {
        "geom": [
            [i, j]
            for i, j in zip(
                [
                    0,
                    -13,
                    13,
                    -20,
                    20,
                    -27,
                    27,
                    -34,
                    200,
                    187,
                    213,
                    180,
                    220,
                    173,
                    227,
                    166,
                    400,
                    387,
                    413,
                    380,
                    420,
                    373,
                    427,
                    366,
                    600,
                    587,
                    613,
                    580,
                    620,
                    573,
                    627,
                    566,
                ],
                [
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                ],
            )
        ],
        "channel_groups": {
            1: {"channels": np.arange(0, 8)},
            2: {"channels": np.arange(8, 16)},
            3: {"channels": np.arange(16, 24)},
            4: {"channels": np.arange(24, 32)},
        },
    },
    "Buzsaki64": {
        "geom": [
            [i, j]
            for i, j in zip(
                [
                    0,
                    -13,
                    13,
                    -20,
                    20,
                    -27,
                    27,
                    -34,
                    200,
                    187,
                    213,
                    180,
                    220,
                    173,
                    227,
                    166,
                    400,
                    387,
                    413,
                    380,
                    420,
                    373,
                    427,
                    366,
                    600,
                    587,
                    613,
                    580,
                    620,
                    573,
                    627,
                    566,
                    0 + 800,
                    -13 + 800,
                    13 + 800,
                    -20 + 800,
                    20 + 800,
                    -27 + 800,
                    27 + 800,
                    -34 + 800,
                    200 + 800,
                    187 + 800,
                    213 + 800,
                    180 + 800,
                    220 + 800,
                    173 + 800,
                    227 + 800,
                    166 + 800,
                    400 + 800,
                    387 + 800,
                    413 + 800,
                    380 + 800,
                    420 + 800,
                    373 + 800,
                    427 + 800,
                    366 + 800,
                    600 + 800,
                    587 + 800,
                    613 + 800,
                    580 + 800,
                    620 + 800,
                    573 + 800,
                    627 + 800,
                    566 + 800,
                ],
                [
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                ],
            )
        ],
        "channel_groups": {
            1: {"channels": np.arange(0, 8)},
            2: {"channels": np.arange(8, 16)},
            3: {"channels": np.arange(16, 24)},
            4: {"channels": np.arange(24, 32)},
            5: {"channels": np.arange(32, 40)},
            6: {"channels": np.arange(40, 48)},
            7: {"channels": np.arange(48, 56)},
            8: {"channels": np.arange(56, 64)},
        },
    },
    "A4x2-tet-7mm-150-200-121": {
        "geom": [
            [0 - 18, 0],
            [0 - 18, 150],
            [0, 0 + 18],
            [0, 150 + 18],
            [0, 150 - 18],
            [0, 0 - 18],
            [0 + 18, 150],
            [0 + 18, 0],
            [200 - 18, 0],
            [200 - 18, 150],
            [200, 0 + 18],
            [200, 150 + 18],
            [200, 150 - 18],
            [200, 0 - 18],
            [200 + 18, 150],
            [200 + 18, 0],
            [400 - 18, 0],
            [400 - 18, 150],
            [400, 0 + 18],
            [400, 150 + 18],
            [400, 150 - 18],
            [400, 0 - 18],
            [400 + 18, 150],
            [400 + 18, 0],
            [600 - 18, 0],
            [600 - 18, 150],
            [600, 0 + 18],
            [600, 150 + 18],
            [600, 150 - 18],
            [600, 0 - 18],
            [600 + 18, 150],
            [600 + 18, 0],
        ],
        "channel_groups": {
            0: {"channels": [0, 2, 5, 7]},
            1: {"channels": [1, 3, 4, 6]},
            2: {"channels": [13, 8, 10, 15]},
            3: {"channels": [9, 11, 12, 14]},
            4: {"channels": [16, 18, 21, 23]},
            5: {"channels": [17, 19, 20, 22]},
            6: {"channels": [25, 27, 28, 30]},
            7: {"channels": [24, 26, 29, 31]},
        },
    },
    "Buzsaki32-H32_21mm": {
        "geom": [
            [i, j]
            for i, j in zip(
                [
                    0,
                    -13,
                    13,
                    -20,
                    20,
                    -27,
                    27,
                    -34,
                    200,
                    187,
                    213,
                    180,
                    220,
                    173,
                    227,
                    166,
                    400,
                    387,
                    413,
                    380,
                    420,
                    373,
                    427,
                    366,
                    600,
                    587,
                    613,
                    580,
                    620,
                    573,
                    627,
                    566,
                ],
                [
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                ],
            )
        ],
        "channel_groups": {
            1: {"channels": np.arange(0, 8)},
            2: {"channels": np.arange(8, 16)},
            3: {"channels": np.arange(16, 24)},
            4: {"channels": np.arange(24, 32)},
        },
    },
    "Buzsaki64-H64_30mm": {
        "geom": [
            [i, j]
            for i, j in zip(
                [
                    0,
                    -13,
                    13,
                    -20,
                    20,
                    -27,
                    27,
                    -34,
                    200,
                    187,
                    213,
                    180,
                    220,
                    173,
                    227,
                    166,
                    400,
                    387,
                    413,
                    380,
                    420,
                    373,
                    427,
                    366,
                    600,
                    587,
                    613,
                    580,
                    620,
                    573,
                    627,
                    566,
                    0 + 800,
                    -13 + 800,
                    13 + 800,
                    -20 + 800,
                    20 + 800,
                    -27 + 800,
                    27 + 800,
                    -34 + 800,
                    200 + 800,
                    187 + 800,
                    213 + 800,
                    180 + 800,
                    220 + 800,
                    173 + 800,
                    227 + 800,
                    166 + 800,
                    400 + 800,
                    387 + 800,
                    413 + 800,
                    380 + 800,
                    420 + 800,
                    373 + 800,
                    427 + 800,
                    366 + 800,
                    600 + 800,
                    587 + 800,
                    613 + 800,
                    580 + 800,
                    620 + 800,
                    573 + 800,
                    627 + 800,
                    566 + 800,
                ],
                [
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                ],
            )
        ],
        "channel_groups": {
            1: {"channels": np.arange(0, 8)},
            2: {"channels": np.arange(8, 16)},
            3: {"channels": np.arange(16, 24)},
            4: {"channels": np.arange(24, 32)},
            5: {"channels": np.arange(32, 40)},
            6: {"channels": np.arange(40, 48)},
            7: {"channels": np.arange(48, 56)},
            8: {"channels": np.arange(56, 64)},
        },
    },
    "A4x2-tet-7mm-150-200-121-H32_21mm": {
        "geom": [
            [0 - 18, 0],
            [0 - 18, 150],
            [0, 0 + 18],
            [0, 150 + 18],
            [0, 150 - 18],
            [0, 0 - 18],
            [0 + 18, 150],
            [0 + 18, 0],
            [200 - 18, 0],
            [200 - 18, 150],
            [200, 0 + 18],
            [200, 150 + 18],
            [200, 150 - 18],
            [200, 0 - 18],
            [200 + 18, 150],
            [200 + 18, 0],
            [400 - 18, 0],
            [400 - 18, 150],
            [400, 0 + 18],
            [400, 150 + 18],
            [400, 150 - 18],
            [400, 0 - 18],
            [400 + 18, 150],
            [400 + 18, 0],
            [600 - 18, 0],
            [600 - 18, 150],
            [600, 0 + 18],
            [600, 150 + 18],
            [600, 150 - 18],
            [600, 0 - 18],
            [600 + 18, 150],
            [600 + 18, 0],
        ],
        "channel_groups": {
            0: {"channels": [0, 2, 5, 7]},
            1: {"channels": [1, 3, 4, 6]},
            2: {"channels": [13, 8, 10, 15]},
            3: {"channels": [9, 11, 12, 14]},
            4: {"channels": [16, 18, 21, 23]},
            5: {"channels": [17, 19, 20, 22]},
            6: {"channels": [25, 27, 28, 30]},
            7: {"channels": [24, 26, 29, 31]},
        },
    },
    "A4x1-tet-7mm-150-200-121": {
        "geom": [
            [0 - 18, 0],
            [0, 0 + 18],
            [0, 0 - 18],
            [0 + 18, 0],
            [200 - 18, 0],
            [200, 0 + 18],
            [200, 0 - 18],
            [200 + 18, 0],
            [400 - 18, 0],
            [400, 0 + 18],
            [400, 0 - 18],
            [400 + 18, 0],
            [600 - 18, 0],
            [600, 0 + 18],
            [600, 0 - 18],
            [600 + 18, 0],
        ],
        "channel_groups": {
            0: {"channels": [0, 1, 2, 3]},
            1: {"channels": [4, 5, 6, 7]},
            2: {"channels": [8, 9, 10, 11]},
            3: {"channels": [12, 13, 14, 15]},
        },
    },
    "A1x32-Edge-5mm-20-177-H32_21mm": {
        "geom": np.array(
            [[i, j] for i, j in zip(np.repeat(20, 32), [20 * i for i in range(32)])]
        ),
        "channel_groups": {1: {"channels": np.arange(32)}},
    },
    "A1x32-Edge-5mm-20-177": {
            "geom": [
            [i, j]
            for i, j in zip(
                # i instead of 0 because errors are thrown otherwise...
                [i for i in range(32)],
                [20 * i for i in range(32)],
            )
        ],
       "channel_groups": {
            1: {"channels": np.arange(32)},

        },
    },
    "A4x8-5mm-200-400-177": {
        "geom": np.array(
            [
                [i, j]
                for i, j in zip(
                    np.concatenate([np.repeat(i * 400, 8) for i in range(4)]),
                    np.concatenate([np.linspace(0, 1400, 8) for i in range(4)]).astype(
                        "int"
                    ),
                )
            ]
        ),
        "channel_groups": {
            i + i: {"channels": np.arange(i * 8, (i + 1) * 8)} for i in range(4)
        },
    },
}
