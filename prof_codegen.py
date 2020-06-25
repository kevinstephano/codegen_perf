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
parser.add_argument('--dim0-start', default=80, type=int, help='Block X')
parser.add_argument('--dim0-inc', default=5, type=int, help='Block X')
parser.add_argument('--dim0-stop', default=80, type=int, help='Block X')
parser.add_argument('--dim1-start', default=1024, type=int, help='Thread X')
parser.add_argument('--dim1-inc', default=16, type=int, help='Thread X')
parser.add_argument('--dim1-stop', default=1024, type=int, help='Thread X')
parser.add_argument('--red-axis', default=1, type=int, help='Thread X')
parser.add_argument('--elem_size', default=4, type=int, help='Thread X')

args = parser.parse_args()

default_list = [args.cuda_bin + 'nvprof', '--device-buffer-size', '128', '--print-gpu-trace', '--csv', '--log-file', 'prof_file.csv', './build/codegen_perf', str(args.warmup_trials+args.trials)]

for dim0 in range(args.dim0_start, args.dim0_stop+args.dim0_inc, args.dim0_inc) :
    for dim1 in range(args.dim1_start, args.dim1_stop+args.dim1_inc, args.dim1_inc) :
        cmd_list = default_list + [str(args.red_axis), str(dim0), str(dim1)]
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
        #print(df)
        mean_val = df[args.warmup_trials:]['Duration'].astype(float).mean()
        # Adjust time scale so it is consistent
        if time_scale == 's' :
            mean_val *= 1000000.0
        elif time_scale == 'ms' :
            mean_val *= 1000.0

        # Bandwidth efficiency calculation
        tensor_size = dim0 * dim1
        if args.red_axis == 0 :
            tensor_size += dim1
        else :
            tensor_size += dim0
        total_bytes = tensor_size * args.elem_size
        expected_val = total_bytes / (1024.0 * args.mem_clock * 1000000.0) * 1000000.0
        efficiency = expected_val / mean_val * 100.0

        if args.print :
            print(df[args.warmup_trials:])
        print(">>>Size: Grid: {} {} Block: {} {} Axis: {} Dim0: {} Dim1: {} Total Bytes: {:.03f} MB Elements: {:3d} Time: {:.03f} us {:.01f} %EFF".format( \
            df.iloc[0]['Grid X'], df.iloc[0]['Grid Y'], df.iloc[0]['Block X'], df.iloc[0]['Block Y'], \
            args.red_axis, dim0, dim1, total_bytes/1000000.0, tensor_size, mean_val, efficiency))
        os.remove("prof_file.csv")
