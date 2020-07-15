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
parser.add_argument('--ti', action='store_true', help='Run TensorIterator only.')
parser.add_argument('--csv', action='store_true', help='Run TensorIterator only.')
parser.add_argument('--fp16', action='store_true', help='Run FP16 precision.')
parser.add_argument('--print-stdout', action='store_true', help='Run TensorIterator only.')
parser.add_argument('--mem-clock', default=796.0, type=float, help='Mem Clock Frequency MHz')
parser.add_argument('--dim0-start', default=80, type=int, help='Block X')
parser.add_argument('--dim0-inc', default='pow2', help='Block X')
parser.add_argument('--dim0-stop', default=80, type=int, help='Block X')
parser.add_argument('--dim1-start', default=1024, type=int, help='Thread X')
parser.add_argument('--dim1-inc', default='pow2', help='Thread X')
parser.add_argument('--dim1-stop', default=1024, type=int, help='Thread X')
parser.add_argument('--red-axis', default=1, type=int, help='Thread X')
parser.add_argument('--elem_size', default=4, type=int, help='Thread X')

args = parser.parse_args()
if args.csv :
    print("Reduction Axis,Dim0,Dim1,Ti GridX,Ti GridY,Ti BlkX,Ti BlkY,Cg GridX,Cg GridY,Cg BlkX,Cg BlkY,CodeGen Time(us),Pytorch Eager Mode(us),CodeGen Eff Bw(GB/s),Pytorch Eager Mode Eff Bw(GB/s)")

default_list = [args.cuda_bin + 'nvprof', '--device-buffer-size', '128', '--print-gpu-trace', '--csv', '--log-file', 'prof_file.csv', './build/codegen_perf', str(args.warmup_trials+args.trials)]

dim0_list = []
if args.dim0_inc == 'pow2' :
    curr = args.dim0_start
    while curr <= args.dim0_stop :
      dim0_list.append(curr)
      curr <<= 1
else :
    dim0_list = range(args.dim0_start, args.dim0_stop+int(args.dim0_inc), int(args.dim0_inc))
dim1_list = []
if args.dim1_inc == 'pow2' :
    curr = args.dim1_start
    while curr <= args.dim1_stop :
      dim1_list.append(curr)
      curr <<= 1
else :
    dim1_list = range(args.dim1_start, args.dim1_stop+int(args.dim1_inc), int(args.dim1_inc))

for dim0 in dim0_list :
    for dim1 in dim1_list :
        cmd_list = default_list + [str(args.red_axis), str(dim0), str(dim1), str(1) if args.ti else str(0), str(1) if args.fp16 else str(0) ]
        output = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
 
        if args.print_stdout :
          print(output.stdout)
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
        # Bandwidth efficiency calculation
        tensor_size = dim0 * dim1
        if args.red_axis == 0 :
            tensor_size += dim1
        else :
            tensor_size += dim0
        total_bytes = tensor_size * (2 if args.fp16 else 4) 
        expected_val = total_bytes / (1024.0 * args.mem_clock * 1000000.0) * 1000000.0

        if not args.ti :
             df_cg = df[df['Name'].str.contains("CudaCodeGen::kernel", na=False)]
             mean_val = df_cg[args.warmup_trials:]['Duration'].astype(float).mean()
             # Adjust time scale so it is consistent
             if time_scale == 's' :
                 mean_val *= 1000000.0
             elif time_scale == 'ms' :
                 mean_val *= 1000.0
             efficiency = expected_val / mean_val * 100.0

        df_ti = df[df['Name'].str.contains("ReduceOp", na=False)]
        df_ti = df_ti[:-1]

        #mean_val_ti = df_ti[args.warmup_trials:-1]['Duration'].astype(float).mean()
        kernel_diff = 0
        if not args.ti :
            kernels_ti = df_ti[args.warmup_trials:]['Duration'].count()
            kernels_cg = df_cg[args.warmup_trials:]['Duration'].count()
            #print(kernels_ti, kernels_cg)
            if( kernels_ti > kernels_cg) :
                kernel_diff = kernels_ti - kernels_cg
                mean_val_ti = df_ti[args.warmup_trials:args.warmup_trials+kernel_diff]['Duration'].astype(float).mean()
                mean_val_ti *= ((kernel_diff) / kernels_cg)
                #mean_val_ti += df_ti[args.warmup_trials+kernel_diff:]['Duration'].astype(float).mean()
            else :
                mean_val_ti = df_ti[args.warmup_trials:]['Duration'].astype(float).mean()
        else :
            mean_val_ti = df_ti[args.warmup_trials:]['Duration'].astype(float).mean()
        # Adjust time scale so it is consistent
        if time_scale == 's' :
            mean_val_ti *= 1000000.0
        elif time_scale == 'ms' :
            mean_val_ti *= 1000.0
        efficiency_ti = expected_val / mean_val_ti * 100.0

        if args.print :
            print(df[args.warmup_trials:])
        if args.csv :
            print("{},{},{},{},{},{},{},{},{},{},{},{:.03f},{:.03f},{:.03f},{:.03f}".format( \
                args.red_axis, dim0, dim1, \
                int(df_ti.iloc[0]['Grid X']), int(df_ti.iloc[0]['Grid Y']), int(df_ti.iloc[0]['Block X']), int(df_ti.iloc[0]['Block Y']), \
                int(df_cg.iloc[0]['Grid X']), int(df_cg.iloc[0]['Grid Y']), int(df_cg.iloc[0]['Block X']), int(df_cg.iloc[0]['Block Y']), \
                mean_val, mean_val_ti, total_bytes/mean_val/1000.0, total_bytes/mean_val_ti/1000.0))
        else :
            if not args.ti :
                print(">>>CdGen Size: Grid: {:4} {:4} Block: {:4} {:4} Axis: {} Dim0: {} Dim1: {} Total Bytes: {:.03f} MB Elements: {:3d} Time: {:.03f} us {:.01f} %EFF".format( \
                    int(df_cg.iloc[0]['Grid X']), int(df_cg.iloc[0]['Grid Y']), int(df_cg.iloc[0]['Block X']), int(df_cg.iloc[0]['Block Y']), \
                    args.red_axis, dim0, dim1, total_bytes/1000000.0, tensor_size, mean_val, efficiency))
            print(">>>TIter Size: Grid: {:4} {:4} Block: {:4} {:4} Axis: {} Dim0: {} Dim1: {} Total Bytes: {:.03f} MB Elements: {:3d} Time: {:.03f} us {:.01f} %EFF TI Agg: {}".format( \
                int(df_ti.iloc[0]['Grid X']), int(df_ti.iloc[0]['Grid Y']), int(df_ti.iloc[0]['Block X']), int(df_ti.iloc[0]['Block Y']), \
            args.red_axis, dim0, dim1, total_bytes/1000000.0, tensor_size, mean_val_ti, efficiency_ti, kernel_diff > 0))
        os.remove("prof_file.csv")
