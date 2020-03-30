#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/iriostream.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/tensor.h>
#include <torch/csrc/jit/codegen/cuda/tensor_meta.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/code_write.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>

// fuser and IR parser
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include "torch/csrc/jit/ir/irparser.h"

#include <iostream>

using namespace torch::jit::fuser;

int main() {
  Fusion fusion;
  FusionGuard fg(&fusion);
  //dimensionality of the problem
  int nDims = 3;

  //Set up symbolic sizes for the axes should be dimensionality of the problem
  std::vector<IterDomain*> dom;
  for(int i=0; i<nDims; i++)
    dom.push_back(new IterDomain(new Int()));

  //Set up your input tensor views
  TensorView* tv0 = new TensorView(new TensorDomain(dom), DataType::Float);
  TensorView* tv1 = new TensorView(new TensorDomain(dom), DataType::Float);
  TensorView* tv3 = new TensorView(new TensorDomain(dom), DataType::Float);

  //Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv3);

  //Do math with it, it returns a `Val*` but can be static_casted back to TensorView
  TensorView* tv2 = static_cast<TensorView*>(add(tv0, tv1));
  TensorView* tv4 = static_cast<TensorView*>(add(tv3, tv2));

  //Register your outputs
  fusion.addOutput(tv4);

  // Do transformations, remember, transformations are outputs to inputs
  // This doesn't have to be in this order
  tv4->merge(1);
  tv4->merge(0);

  // Split by n_threads
  tv4->split(-1, 64*512);
  tv4->split(-1, 512);

  //For all inputs, computeAt the output inline, temporaries should be squeezed between them
  tv0->computeAt(tv4, -1);
  tv1->computeAt(tv4, -1);
  tv3->computeAt(tv4, -1);

  //Parallelize TV3
  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(-2)->parallelize(ParallelType::BIDy);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  
  std::stringstream cdg;
  CodeWrite cw(cdg);
  cw.traverse(&fusion);

  std::cout << cdg.str() << std::endl;

  torch::jit::fuser::cuda::CudaKernel prog;
  prog.device_ = 0;
  prog.grid(80,64);     //   1 CTA
  prog.block(512); // 256 Threads

  auto options =
  at::TensorOptions()
    .dtype(at::kFloat)
    .device(at::kCUDA, 0);

  at::Tensor input0 = at::randn({80,64,512}, options);
  at::Tensor input1 = at::randn_like(input0);;
  at::Tensor input3 = at::randn_like(input0);;
  at::Tensor output = at::empty_like(input0);
  std::vector<at::Tensor> inputs{{input0, input1, input3}};
  std::vector<at::Tensor> outputs{{output}};

  torch::jit::fuser::cuda::compileKernel(fusion, prog);
  torch::jit::fuser::cuda::runTestKernel(prog, inputs, outputs);

  at::Tensor tv2_ref = input0 + input1;
  at::Tensor output_ref = input3 + tv2_ref;

  //TORCH_CHECK(output_ref.equal(output));

  return 0;
}
