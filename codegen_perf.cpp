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

void reduction(int trials, int red_dim, int dim0, int dim1, bool ti_only) {
  torch::jit::fuser::cuda::CudaKernel prog;
  Fusion& fusion = *prog.fusion_;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = reductionOp(BinaryOpType::Add, {red_dim}, new Float(0), tv0);
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::rand({dim0, dim1}, options);
  at::Tensor cg_output = at::empty({(red_dim == 0 ? dim1 : dim0) }, options);

  // Apply reduction heuristic
  const at::ArrayRef<c10::IValue> inputs({input});

  c10::optional<std::tuple<int,int,int,int>> blocking =
        Scheduler::reduction(prog.fusion_.get(), inputs);
  TORCH_CHECK(blocking != c10::nullopt, "Reduction is not found!");

  fusion.printMath();
  GPULower gpulw(&fusion);
  gpulw.printKernel(std::cout);

  prog.device_ = 0;
  prog.grid(std::get<0>(blocking.value()), std::get<1>(blocking.value()));
  prog.block(std::get<2>(blocking.value()), std::get<3>(blocking.value()));

  torch::jit::fuser::cuda::compileKernel(&prog);

  at::Tensor aten_output;
  for (int i = 0; i < trials; ++i) {
    if( !ti_only )
      torch::jit::fuser::cuda::runTestKernel(&prog, {input}, {cg_output});
    aten_output = input.sum({red_dim});
  }

  //if( !ti_only)
  //  if (! aten_output.allclose(cg_output))
  //    std::cout << "ATEN and Codegen mismatch!!!" << std::endl;
  if( !ti_only)
    TORCH_CHECK(aten_output.allclose(cg_output),
                "Error of: ",
                aten_output.sub(cg_output).abs().max());
}

int main(int argc, char* argv[]) {

  if (argc != 6) {
    throw std::runtime_error("You forgot to input the number of trials!");
  }
  int trials  = atoi(argv[1]);
  int red_dim = atoi(argv[2]);
  int dim0    = atoi(argv[3]);
  int dim1    = atoi(argv[4]);
  bool ti_only= static_cast<bool>(atoi(argv[5]));

  reduction(trials, red_dim, dim0, dim1, ti_only);

  return 0;
}
