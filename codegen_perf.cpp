#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/type.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>
#include <torch/csrc/jit/codegen/cuda/ir_graphviz.h>
#include <torch/csrc/jit/codegen/cuda/ir_graphviz.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
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
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2, (fp16 ? DataType::Half : DataType::Float));
  fusion.addInput(tv0);

  torch::jit::fuser::Val* tv0_cast = nullptr;
  if (fp16) {
    tv0_cast = castOp(DataType::Float, tv0);
  }

  TensorView* tv1 = reductionOp(BinaryOpType::Add, {red_dim}, new Float(0), (fp16 ? tv0_cast->as<TensorView>() : tv0));

  TensorView* tv1_cast = nullptr;
  if (fp16) {
    tv1_cast = castOp(DataType::Half, tv1);
  }

	fusion.addOutput((fp16 ? tv1_cast : tv1));
  //IrGraphGenerator::print(fusion, "ir.dot");

  auto options = at::TensorOptions().dtype((fp16 ? at::kHalf : at::kFloat)).device(at::kCUDA, 0);
  at::Tensor input = at::rand({dim0, dim1}, options);

  // Apply reduction heuristic
  const at::ArrayRef<c10::IValue> inputs({input});

  fusion.printMath();
  c10::optional<cuda::ReductionParams> rparams = cuda::scheduleReduction(&fusion, inputs, tv1);
  TORCH_CHECK(rparams != c10::nullopt, "Reduction is not found!");
  if(fp16) {
    if (red_dim == 0 ) {
      int tidx = rparams.value().bdimx.value;
      tv1_cast->split(-1, tidx);
      tv1_cast->axis(-1)->parallelize(ParallelType::TIDx);
      tv1_cast->axis(-2)->parallelize(ParallelType::BIDx);
    } else {
      if (rparams.value().mul_reds_per_blk) {
        int tidy = rparams.value().bdimy.value;
        tv1_cast->split(0, tidy);
        tv1_cast->axis(-1)->parallelize(ParallelType::TIDy);
      }
      tv1_cast->axis(0)->parallelize(ParallelType::BIDx);
    }
  }
  //IrGraphGenerator::print(fusion, "ir.dot");

  fusion.printMath();
  GPULower gpulw(&fusion);
  gpulw.printKernel(std::cout);

  std::cout << std::flush << std::endl;
  torch::jit::fuser::cuda::FusionExecutor fe;
  fe.compileFusion(&fusion);

  at::Tensor aten_output;
  std::vector<at::Tensor> cg_output;
  for (int i = 0; i < trials; ++i) {
    if( !ti_only ) {
      at::Tensor flush_cache_1 = at::ones({6000*1024}, options);
      cg_output = fe.runFusion({input});
    }
    at::Tensor flush_cache_2 = at::ones({6000*1024}, options);
    aten_output = input.sum({red_dim});
  }
  //std::cout << aten_output << std::endl;
  //std::cout << cg_output << std::endl;

  if( !ti_only)
    TORCH_CHECK(
      aten_output.allclose(cg_output[0]),
      "Error of: ",
      aten_output.sub(cg_output[0]).abs().max()); 
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
