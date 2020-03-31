import subprocess
import os
import pandas as pd
import argparse

pd.options.display.max_rows = 999

parser = argparse.ArgumentParser(description='Profile Codegen')
parser.add_argument('--trials', default='10', type=str, help='Number of Trials to Execute')
parser.add_argument('--warmup-trials', default='10', type=str, help='Warmup Trials to discard')
parser.add_argument('--cuda-bin', default='/usr/local/cuda/bin/', type=str, help='Cuda Path')
parser.add_argument('--print', action='store_true', help='Print profiler lines.')

args = parser.parse_args()

default_list = [args.cuda_bin + 'nvprof', '--device-buffer-size', '128', '--print-gpu-trace', '--csv', '--log-file', 'prof_file.csv', './build/codegen_perf', str(int(args.warmup_trials)+int(args.trials))]

output = subprocess.run(default_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

assert os.path.exists('prof_file.csv'), "ERROR: Run failed. No profiler output!"
df = pd.read_csv('prof_file.csv', header=3)
time_rep = df.iloc[0]['Duration']
df = df[df['Name'].str.contains("CudaCodeGen::kernel", na=False)]
mean_val = df[int(args.warmup_trials):]['Duration'].astype(float).mean()
#total_elems = int(args.elements) * batch * 2
#total_bytes = total_elems * 2
#expected_val = total_bytes / (1024.0 * 796000000.0) * 1000000.0
#efficiency = expected_val / mean_val * 100.0
#print(">>>Size: {:.03f} MB Batches: {:3d} Elements: {:3d} Time: {:03f} us {:.01f} %EFF".format(total_bytes/1000000.0, batch, int(args.elements), mean_val, efficiency))
if args.print :
    print(df[int(args.warmup_trials):])
print(">>> Time: {:03f} {}".format(mean_val, time_rep))
os.remove("prof_file.csv")
