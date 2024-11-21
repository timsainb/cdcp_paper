import numpy
import psutil
import socket
import spikeinterface.sorters as ss
from pathlib2 import Path
from cdcp.paths import DATA_DIR, ensure_dir
import logging

hostname = socket.gethostname()
if "ssrde" in hostname:
    # os.environ["NPY_MATLAB_PATH"] = "/usr/local/bin/matlab20a"
    # os.environ["matlab"] = "/usr/local/bin/matlab20a"
    ss.Kilosort2Sorter.set_kilosort2_path(
        "/cube/bigbird/tsainbur/Projects/github_repos/Kilosort2"
    )
    # ss.Kilosort2_5Sorter.set_kilosort2_5_path(
    #    "/cube/bigbird/tsainbur/Projects/github_repos/Kilosort"
    # )
elif hostname in ["pakhi", "txori"]:
    ss.Kilosort2Sorter.set_kilosort2_path(
        "/mnt/cube/tsainbur/Projects/github_repos/Kilosort2"
    )
    # ss.Kilosort2_5Sorter.set_kilosort2_5_path(
    #    "/mnt/cube/tsainbur/Projects/github_repos/Kilosort"
    # )


def get_spikesorting_directories(hostname, bird, timestamp):
    # if on slurm server, change path names to match mount point on server
    if "ssrde" in hostname:
        spikesorting_directory = DATA_DIR / "spikesorting" / bird / timestamp

        sphere_n_available_GB = (
            psutil.disk_usage("/sphere/gentnerlab_NB/tmp_spikesorting").free
            / 1000000000
        )
        ssrde_n_available_GB = (
            psutil.disk_usage("/home/AD/tsainbur/tmp_spikesorting/").free / 1000000000
        )
        if ssrde_n_available_GB > 1000:
            tmp_dir = Path("/home/AD/tsainbur/tmp_spikesorting")
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
        sphere_n_available_GB = (
            psutil.disk_usage("/mnt/sphere/tmp_spikesorting").free / 1000000000
        )
        if scratch_n_available_GB > 800:
            tmp_dir = Path("/scratch/tsainbur")
        elif sphere_n_available_GB > 1000:
            tmp_dir = Path("/mnt/sphere/spikesorting_tmp")
        else:
            raise ValueError("Not enough disk space anywhere")

        spikesorting_directory = DATA_DIR / "spikesorting" / bird / timestamp

        raw_data_folder = (
            list(Path("/mnt/sphere/RawData/").glob("Samamba/ephys/{}/".format(bird)))
            + list(Path("/mnt/sphere/RawData/").glob("nyoni/{}/".format(bird)))
        )[0]

    else:
        raise ValueError

    # make sure the folder exists
    ensure_dir(spikesorting_directory)
    ensure_dir(tmp_dir)

    print("spikesorting_directory: {}".format(spikesorting_directory))
    print("tmp_dir: {}".format(tmp_dir))
    print("raw_data_folder: {}".format(raw_data_folder))

    return spikesorting_directory, tmp_dir, raw_data_folder


def create_logger(file_loc, log_name="spike_logger"):
    # create logger
    logger = logging.getLogger(log_name)
    print(logger)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    # create file handler which logs even debug messages
    fh = logging.FileHandler(file_loc)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger
