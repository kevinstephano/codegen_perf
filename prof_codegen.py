import subprocess
import os
import pandas as pd
import argparse
import re

pd.options.display.max_rows = 999

parser = argparse.ArgumentParser(description='Profile Codegen')
parser.add_argument('--trials', default='10', type=int, help='Number of Trials to Execute')
parser.add_argument('--warmup-trials', default='10', type=int, help='Warmup Trials to discard')
parser.add_argument('--cuda-bin', default='/usr/local/cuda/bin/', type=str, help='Cuda Path')
parser.add_argument('--print', action='store_true', help='Print profiler lines.')
parser.add_argument('--mem-clock', default=796.0, type=float, help='Mem Clock Frequency MHz')
parser.add_argument('--block-x-start', default=80, type=int, help='Block X')
parser.add_argument('--block-x-inc', default=5, type=int, help='Block X')
parser.add_argument('--block-x-stop', default=80, type=int, help='Block X')
parser.add_argument('--block-y', default=64, type=int, help='Block Y')
parser.add_argument('--thread-x-start', default=65, type=int, help='Thread X')
parser.add_argument('--thread-x-inc', default=64, type=int, help='Thread X')
parser.add_argument('--thread-x-stop', default=512, type=int, help='Thread X')
parser.add_argument('--elem-size', default=4, type=int, help='Bytes per element')

args = parser.parse_args()

default_list = [args.cuda_bin + 'nvprof', '--device-buffer-size', '128', '--print-gpu-trace', '--csv', '--log-file', 'prof_file.csv', './build/codegen_perf', str(args.warmup_trials+args.trials)]

for bidx in range(args.block_x_start, args.block_x_stop+args.block_x_inc, args.block_x_inc) :
    for tidx in range(args.thread_x_start, args.thread_x_stop+args.thread_x_inc, args.thread_x_inc) :
        cmd_list = default_list + [str(bidx), str(args.block_y), str(tidx)]
        output = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        # Crop the csv file of extra nvprof lines
        assert os.path.exists('prof_file.csv'), "ERROR: Run failed. No profiler output!"
        exclude = re.compile('^==\d+==')
        with open("prof_file.csv","r+") as f:
            new_f = f.readlines()
            f.seek(0)
            for line in new_f:
                match = exclude.match(line)
                if not match:
                    f.write(line)
            f.truncate()

        # Process profile time to gather kernel's average time
        df = pd.read_csv('prof_file.csv', header=0)
        time_scale = df.iloc[0]['Duration']
        df = df[df['Name'].str.contains("CudaCodeGen::kernel", na=False)]
        mean_val = df[args.warmup_trials:]['Duration'].astype(float).mean()
        # Adjust time scale so it is consistent
        if time_scale == 's' :
            mean_val *= 1000000.0
        elif time_scale == 'ms' :
            mean_val *= 1000.0

        # Bandwidth efficiency calculation
        tensor_size = bidx * args.block_y * tidx
        tensors = tensor_size * 3
        total_bytes = tensors * args.elem_size
        expected_val = total_bytes / (1024.0 * args.mem_clock * 1000000.0) * 1000000.0
        efficiency = expected_val / mean_val * 100.0

        if args.print :
            print(df[args.warmup_trials:])
        print(">>>Size: Grids: {} Blocks: {} Total Bytes: {:.03f} MB Elements: {:3d} Time: {:.03f} us {:.01f} %EFF".format(bidx*args.block_y, tidx, total_bytes/1000000.0, tensors, mean_val, efficiency))
        os.remove("prof_file.csv")
