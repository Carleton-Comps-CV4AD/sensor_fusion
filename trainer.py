import subprocess
import time

sleep_interval = 60

def terminate_process(proc):
    print("Terminating process")
    proc.terminate()

start_time = time.time()
proc = subprocess.Popen(
    ["python3", "train_parallel.py", "--cfg", "config/hrnetv2_c3_mid_fusion.yaml", "--gpus", "0"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True
)
total_time = 60 * 60 * 2

while time.time() - start_time < total_time:
    try:
        output = proc.stdout.readline()
        if output:
            print(output.strip())
    except KeyboardInterrupt:
        terminate_process(proc)
    time.sleep(sleep_interval)
terminate_process(proc)

start_time = time.time()
proc = subprocess.Popen(
    ["python3", "train_base.py", "--cfg", "config/hrnetv2_c1_base.yaml", "--gpus", "0"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True
)
total_time = 60 * 60 * 1

while time.time() - start_time < total_time:
    try:
        output = proc.stdout.readline()
        if output:
            print(output.strip())
    except KeyboardInterrupt:
        terminate_process(proc)
    time.sleep(sleep_interval)
terminate_process(proc)

start_time = time.time()
proc = subprocess.Popen(
    ["python3", "train_parallel.py", "--cfg", "config/hrnetv2_c3_mid_fusion_deepsup.yaml", "--gpus", "0"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True
)
total_time = 60 * 60 * 8

while time.time() - start_time < total_time:
    try:
        output = proc.stdout.readline()
        if output:
            print(output.strip())
    except KeyboardInterrupt:
        terminate_process(proc)
    time.sleep(sleep_interval)
terminate_process(proc)