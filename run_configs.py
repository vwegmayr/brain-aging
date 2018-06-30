from subprocess import call, Popen
import yaml
import os
import time
import GPUtil
import copy


MAX_GPUS = 1
MAX_SLEEP_TIME = 5 * 60  # in seconds
MIN_SLEEP_TIME = 1
CONFIGS_DONE = 'configs_done'
CONFIGS_RUNNING = 'configs_running'
CONFIGS_TO_DO = 'configs_to_do'
MAX_GPU_MEMORY = 0.5  # in percent


class State(object):
    def __init__(self):
        self.gpus_used = []
        self.processes = []
        self.pid_to_config_name = {}
        self.pid_to_gpu = {}

    def update(self):
        # Available configs
        file_names = os.listdir(CONFIGS_TO_DO)
        file_names = [f for f in file_names if f.endswith(".yaml")]
        file_names = sorted(file_names)

        self.configs_to_do = file_names
        # Check for finished configs
        tmp = copy.copy(self.processes)
        for p in tmp:
            if p.poll() is not None:
                self.processes.remove(p)
                self.gpus_used.remove(self.pid_to_gpu[p.pid])
                self.move_running_to_done(p.pid)

    def move_running_to_done(self, pid):
        config_name = self.pid_to_config_name[pid]
        config_id = config_name.split("_")[0]

        for suff in ["_reason.txt", "_tags.txt", "_config.yaml"]:
            os.rename(
                os.path.join(CONFIGS_RUNNING, config_id + suff),
                os.path.join(CONFIGS_DONE, config_id + suff)
            )

    def jobs_running(self):
        self.update()
        return len(self.processes)

    def get_used_gpus(self):
        self.update()
        return self.gpus_used

    def get_next_config_name(self):
        self.update()
        if len(self.configs_to_do) < 1:
            return None
        else:
            return self.configs_to_do[0]

    def run_config(self, config_name, gpus):
        config_id = config_name.split("_")[0]
        # Load reason
        reason = ""
        with open(os.path.join(CONFIGS_TO_DO, config_id + "_reason.txt"), 'r') as f:
            for line in f:
                reason = line.strip()

        # Load tags
        tags = []
        with open(os.path.join(CONFIGS_TO_DO, config_id + "_tags.txt"), 'r') as f:
            for line in f:
                tags.append(line.strip())

        tag = ",".join(tags)
        # Load tags
        # Move files
        for suff in ["_reason.txt", "_tags.txt", "_config.yaml"]:
            os.rename(
                os.path.join(CONFIGS_TO_DO, config_id + suff),
                os.path.join(CONFIGS_RUNNING, config_id + suff)
            )

        # Build command
        gpu_id = gpus[0]
        self.gpus_used.append(gpu_id)
        config_path = os.path.join(CONFIGS_RUNNING, config_name)
        cmd = 'smt run --config {} -a fit -S data -t "{}" -r "{}"'.format(
            config_path,
            tag,
            reason
        )
        print(cmd)

        proc = Popen(cmd, shell=True)
        self.processes.append(proc)
        self.pid_to_config_name[proc.pid] = config_name
        self.pid_to_gpu[proc.pid] = gpu_id


def get_available_gpus(state):
    r = GPUtil.getAvailable(
        order='first',
        limit=8,
        maxLoad=MAX_GPU_MEMORY,
        maxMemory=MAX_GPU_MEMORY
    )
    for g in state.get_used_gpus():
        r.remove(g)

    return r


def run_configs():
    state = State()
    cur_sleep = MIN_SLEEP_TIME
    while (1):
        config_name = state.get_next_config_name()
        print(config_name)
        if config_name is None:
            # No more configs to run
            time.sleep(60)
            state.update()
            continue

        # Not too many jobs running
        while state.jobs_running() >= MAX_GPUS:
            time.sleep(cur_sleep)
            cur_sleep = min(MAX_SLEEP_TIME, cur_sleep * 2)

        cur_sleep = MIN_SLEEP_TIME
        # Find available GPU
        gpus = get_available_gpus(state)
        while len(gpus) < 1:
            time.sleep(cur_sleep)
            cur_sleep = min(MAX_SLEEP_TIME, cur_sleep * 2)
            gpus = get_available_gpus(state)

        cur_sleep = MIN_SLEEP_TIME

        state.run_config(config_name, gpus)


def main():
    run_configs()


if __name__ == "__main__":
    main()
