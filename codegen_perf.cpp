#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/type.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>
#include <ATen/core/ivalue.h>

#include <iostream>

using namespace torch::jit::fuser;

static TensorView* makeDummyTensor(
    int nDims,
    DataType dtype = DataType::Float) {
  std::vector<IterDomain*> dom;
  for (int i = 0; i < nDims; i++)
    dom.push_back(new IterDomain(new Int(0), new Int()));

  return new TensorView(new TensorDomain(dom), dtype);
}

void reduction(int trials, int red_dim, int dim0, int dim1, bool ti_only, bool fp16) {
  torch::jit::fuser::cuda::CudaKernel prog;
  Fusion& fusion = *prog.fusion_;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2, DataType::Float);
  fusion.addInput(tv0);

  TensorView* tv1 = reductionOp(BinaryOpType::Add, {red_dim}, new Float(0), tv0);

	fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::rand({dim0, dim1}, options);
  at::Tensor cg_output = at::empty({(red_dim == 0 ? dim1 : dim0) }, options);
  //at::Tensor cg_output2 = at::empty({(red_dim == 0 ? dim1 : dim0) }, options);

  // Apply reduction heuristic
  const at::ArrayRef<c10::IValue> inputs({input});

  TORCH_CHECK(cuda::scheduleReduction(prog.fusion_.get(), inputs), "Reduction is not found!");
  /*if(fp16) {
	tv3->split(-1, std::get<3>(blocking.value()));
	tv3->axis(-1)->parallelize(ParallelType::TIDx);
	tv3->axis(-2)->parallelize(ParallelType::BIDx);
  }*/

  fusion.printMath();
  GPULower gpulw(&fusion);
  gpulw.printKernel(std::cout);

  std::cout << std::flush << std::endl;

  prog.device_ = 0;
  //prog.grid(520, 8);
  //prog.block(128,4);


  torch::jit::fuser::cuda::compileKernel(&prog);

  at::Tensor aten_output;
  for (int i = 0; i < trials; ++i) {
    if( !ti_only ) {
      at::Tensor flush_cache_1 = at::ones({6000*1024}, options);
      //torch::jit::fuser::cuda::runKernel(&prog, {input}, {cg_output}, c10::nullopt);
      //torch::jit::fuser::cuda::runTestKernel(&prog, {input}, {cg_output1});
      torch::jit::fuser::cuda::runKernel(&prog, {input}, {cg_output}, c10::nullopt);
    }
    at::Tensor flush_cache_2 = at::ones({6000*1024}, options);
    aten_output = input.sum({red_dim});
  }

  //std::cout << aten_output << std::endl;
  // std::cout << cg_output2 << std::endl;
  if( !ti_only)
    TORCH_CHECK(aten_output.allclose(cg_output),
                "Error of: ",
                aten_output.sub(cg_output).abs().max());
}

int main(int argc, char* argv[]) {

  if (argc != 7) {
    throw std::runtime_error("You forgot to input the number of trials!");
  }
  int trials  = atoi(argv[1]);
  int red_dim = atoi(argv[2]);
  int dim0    = atoi(argv[3]);
  int dim1    = atoi(argv[4]);
  bool ti_only= static_cast<bool>(atoi(argv[5]));
  bool fp16   = static_cast<bool>(atoi(argv[6]));

  reduction(trials, red_dim, dim0, dim1, ti_only, fp16);

  return 0;
}
