import numpy
import psutil
import socket
import spikeinterface.sorters as ss
from pathlib2 import Path
from cdcp.paths import DATA_DIR, ensure_dir
import logging

# location of the kilosort2 folder
hostname = socket.gethostname()
if "ssrde" in hostname:
    ss.Kilosort2Sorter.set_kilosort2_path(
        "/cube/bigbird/tsainbur/Projects/github_repos/Kilosort2"
    )
elif hostname in ["pakhi", "txori"]:
    ss.Kilosort2Sorter.set_kilosort2_path(
        "/mnt/cube/tsainbur/Projects/github_repos/Kilosort2"
    )


def get_spikesorting_directories(
    hostname, bird, timestamp, is_acute=False, user="tsainbur"
):
    # if on slurm server, change path names to match mount point on server
    raw_data_folder2 = None
    if is_acute:
        local_raw_data_folder = "nyoni"
    else:
        local_raw_data_folder = "Samamba/ephys"

    if "ssrde" in hostname:
        spikesorting_directory = DATA_DIR / "spikesorting" / bird / timestamp

        try:
            sphere_n_available_GB = (
                psutil.disk_usage("/sphere/gentnerlab_NB/tmp_spikesorting").free
                / 1000000000
            )
        except PermissionError:
            print("Permission error reaching sphere")
            sphere_n_available_GB = 0

        ssrde_n_available_GB = (
            psutil.disk_usage("/home/AD/{}/tmp_spikesorting/".format(user)).free
            / 1000000000
        )
        if ssrde_n_available_GB > 1000:
            tmp_dir = Path("/home/AD/{}/tmp_spikesorting".format(user))
        elif sphere_n_available_GB > 1000:
            tmp_dir = Path("/sphere/gentnerlab_NB/tmp_spikesorting")
        else:
            raise ValueError("Not enough disk space anywhere")

        raw_data_folder = (
            list(
                Path("/sphere/gentnerlab_NB/RawData/").glob(
                    "Samamba/ephys/{}/".format(bird)
                )
            )
            + list(
                Path("/sphere/gentnerlab_NB/RawData/").glob("nyoni/{}/".format(bird))
            )
        )[0]
        # raw_data_folder = Path("/sphere/gentnerlab_NB/RawData/Samamba/ephys") / bird

    elif ("txori" in hostname) or ("pakhi" in hostname):
        # tmp_dir = DATA_DIR / "tmp"

        scratch_n_available_GB = psutil.disk_usage("/scratch").free / 1000000000
        try:
            sphere_n_available_GB = (
                psutil.disk_usage("/mnt/sphere/tmp_spikesorting").free / 1000000000
            )
        except PermissionError:
            print("Permission error reaching sphere")
            sphere_n_available_GB = 0
        if scratch_n_available_GB > 800:
            tmp_dir = Path("/scratch/{}".format(user))
        elif sphere_n_available_GB > 1000:
            tmp_dir = Path("/mnt/sphere/spikesorting_tmp")
        else:
            raise ValueError("Not enough disk space anywhere")

        spikesorting_directory = DATA_DIR / "spikesorting" / bird / timestamp

        raw_data_folder = list(
            Path("/mnt/sphere/RawData/").glob("Samamba/ephys/{}/".format(bird))
        ) + list(Path("/mnt/sphere/RawData/").glob("nyoni/{}/".format(bird)))
        if len(raw_data_folder) > 0:
            raw_data_folder = raw_data_folder[0]
        else:
            # default location
            raw_data_folder = Path("/mnt/sphere/RawData/Samamba/ephys/{}/".format(bird))
        # [0]
        raw_data_folder2 = Path("/mnt/uss/gentnerlab/tsainbur/{}/".format(bird))

    else:
        raise ValueError

    # make sure the folder exists
    ensure_dir(spikesorting_directory)
    ensure_dir(tmp_dir)

    print("spikesorting_directory: {}".format(spikesorting_directory))
    print("tmp_dir: {}".format(tmp_dir))
    print("raw_data_folder: {}".format(raw_data_folder))

    return (spikesorting_directory, tmp_dir, raw_data_folder, raw_data_folder2)
