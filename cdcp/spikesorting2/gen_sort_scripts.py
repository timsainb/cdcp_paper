import stat
from pathlib2 import Path


def gen_sort_bash_script(
    bird,
    timestamp,
    sorter,
    denoised,
    grouping,
    recording_id,
    freq_min,
    freq_max,
    max_spikes_per_unit,
    overwrite_sort=False,
    save_loc=None,
):
    bash_script = [
        "#!/bin/bash",
        "module load matlab/R2020a",
        "module load cuda/10.1",
        """if [ "$HOSTNAME" = txori ]; then
            eval "$(conda shell.bash hook)"
            conda activate cdcp_paper
            export TMP='/mnt/sphere/spikesorting_tmp'
            export TMPDIR='/mnt/sphere/spikesorting_tmp'
            export TEMPDIR='/mnt/sphere/spikesorting_tmp'""",
        """elif [ "$HOSTNAME" = pakhi ]; then
            eval "$(conda shell.bash hook)"
            conda activate cdcp_paper_37
            export TMP='/mnt/sphere/spikesorting_tmp'
            export TMPDIR='/mnt/sphere/spikesorting_tmp'
            export TEMPDIR='/mnt/sphere/spikesorting_tmp'""",
        """else
            #source activate /cube/bigbird/tsainbur/conda_envs/cdcp_env_shared
            #conda activate /cube/bigbird/tsainbur/conda_envs/cdcp_env_shared
            export TMP='/sphere/gentnerlab_NB/tmp_spikesorting'
            export TMPDIR='/sphere/gentnerlab_NB/tmp_spikesorting'
            export TEMPDIR='/sphere/gentnerlab_NB/tmp_spikesorting'""",
        "fi",
        "echo $CONDA_PREFIX",
        "export CUDA_CACHE_MAXSIZE=1073741824"
        if sorter in ["kilosort3", "kilosort2_5"]
        else "",
        "".join(
            [
                """python -c "from cdcp.spikesorting2.sort_individual import run_sort; """,
                "run_sort(",
                "bird='{}',".format(bird),
                "timestamp='{}',".format(timestamp),
                "sorter='{}',".format(sorter),
                "denoised = {},".format(denoised),
                "grouping='{}',".format(grouping)
                if grouping is not None
                else "grouping=None,",
                "recording_id='{}',".format(recording_id),
                "freq_min={},".format(freq_min),
                "freq_max={},".format(freq_max),
                "max_spikes_per_unit={},".format(max_spikes_per_unit),
                "overwrite_sort={},".format(overwrite_sort),
                ''')"''',
            ]
        ),
        "exit 0",
    ]

    if save_loc is not None:
        # save bash script to file
        bashfile = open(save_loc, "w")
        bashfile.write("\n".join(bash_script))
        bashfile.close()
        # make job excecutable
        save_loc.chmod(save_loc.stat().st_mode | stat.S_IEXEC)

    return bash_script


def submit_to_slurm(
    bash_loc, sh, automatically_start_SSRDE_jobs, short_node=False, GPU_node=False
):
    # prepare job for SSRDE / slurm
    # choose GPU vs CPU

    sortjob_ssrde = Path(*(["/cube", "bigbird"] + list(bash_loc.parts[3:])))
    if GPU_node:
        machine = "general_gpu_k40,general_gpu_k80,general_gpu_p6000"
        slurm_command = "sbatch --time 02-00 -p {} {}".format(machine, sortjob_ssrde)

    else:
        if short_node:
            machine = "general_short"
            slurm_command = "sbatch -p {} {}".format(machine, sortjob_ssrde)
        else:
            machine = "general"
            slurm_command = "sbatch --time 02-00 -p {} {}".format(
                machine, sortjob_ssrde
            )

    # run command over slurm
    if automatically_start_SSRDE_jobs:
        ssh_stdin, ssh_stdout, ssh_stderr = sh.execute(slurm_command)
    else:
        ssh_stdout = [None]
    return slurm_command, ssh_stdout


def gen_overlap_sort_bash_script(
    bird,
    timestamp,
    sorter,
    denoised,
    grouping,
    merge_recording_id,
    recording_id_prev,
    recording_id_next,
    freq_min,
    freq_max,
    max_spikes_per_unit,
    overwrite_sort=False,
    save_loc=None,
    tmp_spikesorting_dir="/mnt/sphere/spikesorting_tmp",
):
    bash_script = [
        "#!/bin/bash",
        "module load matlab/R2020a",
        "module load cuda/10.1",
        """if [ "$HOSTNAME" = txori ]; then
            eval "$(conda shell.bash hook)"
            conda activate  /mnt/sphere/conda_envs/cdcp_sphere
            export TMP='/mnt/sphere/spikesorting_tmp'
            export TMPDIR='/mnt/sphere/spikesorting_tmp'
            export TEMPDIR='/mnt/sphere/spikesorting_tmp'""",
        """elif [ "$HOSTNAME" = pakhi ]; then
            eval "$(conda shell.bash hook)"
            conda activate  /mnt/sphere/conda_envs/cdcp_sphere
            export TMP='/mnt/sphere/spikesorting_tmp'
            export TMPDIR='/mnt/sphere/spikesorting_tmp'
            export TEMPDIR='/mnt/sphere/spikesorting_tmp'""",
        """else
            export TMP='/sphere/gentnerlab_NB/tmp_spikesorting'
            export TMPDIR='/sphere/gentnerlab_NB/tmp_spikesorting'
            export TEMPDIR='/sphere/gentnerlab_NB/tmp_spikesorting'""",
        "fi",
        "echo $CONDA_PREFIX",
        "export CUDA_CACHE_MAXSIZE=1073741824"
        if sorter in ["kilosort3", "kilosort2_5"]
        else "",
        "".join(
            [
                """python -c "from cdcp.spikesorting2.sort_overlapping import run_sort_overlap; """,
                "run_sort_overlap(",
                "bird='{}',".format(bird),
                "timestamp='{}',".format(timestamp),
                "sorter='{}',".format(sorter),
                "denoised = {},".format(denoised),
                "grouping='{}',".format(grouping)
                if grouping is not None
                else "grouping=None,",
                "merge_recording_id='{}',".format(merge_recording_id),
                "recording_id_prev='{}',".format(recording_id_prev),
                "recording_id_next='{}',".format(recording_id_next),
                "freq_min={},".format(freq_min),
                "freq_max={},".format(freq_max),
                "max_spikes_per_unit={},".format(max_spikes_per_unit),
                "overwrite_sort={},".format(overwrite_sort),
                ''')"''',
            ]
        ),
        "exit 0",
    ]

    if save_loc is not None:
        # save bash script to file
        bashfile = open(save_loc, "w")
        bashfile.write("\n".join(bash_script))
        bashfile.close()
        # make job excecutable
        save_loc.chmod(save_loc.stat().st_mode | stat.S_IEXEC)

    return bash_script
