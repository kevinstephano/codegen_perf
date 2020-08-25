#pragma once

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

static auto name_string = R"(
CudaCodeGen::kernel1
)";

static auto kernel_string = R"(
namespace CudaCodeGen {

typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef short int  int16_t;
typedef long long int int64_t;

template<typename T, int N>
struct Tensor {
  T& operator[](int64_t ind) {
    return data[ind];
  };

  T* data;
  int64_t size[N];
  int64_t stride[N];
};

// Specialization for 0-dim case as it does not need size and stride arrays.
// They will be an error as well since zero-length arrays are not allowed.
template<typename T>
struct Tensor<T, 0> {
  T& operator[](int64_t) {
    return *data;
  };

  T* data;
};


#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
struct __align__(2) __half {
  __host__ __device__ __half() { }
protected:
  unsigned short __x;
};

/* Definitions of intrinsics */
__device__ __half __float2half(const float f) {
  __half val;
  asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(f));
  return val;
}
__device__ float __half2float(const __half h) {
  float val;
  asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__HALF_TO_CUS(h)));
  return val;
}


class Philox {
public:
  __device__ inline Philox(unsigned long long seed,
                           unsigned long long subsequence,
                           unsigned long long offset) {
    key.x = (unsigned int)seed;
    key.y = (unsigned int)(seed >> 32);
    counter = make_uint4(0, 0, 0, 0);
    counter.z = (unsigned int)(subsequence);
    counter.w = (unsigned int)(subsequence >> 32);
    STATE = 0;
    incr_n(offset / 4);
  }
  __device__ inline unsigned long operator()() {
    if(STATE == 0) {
      uint4 counter_ = counter;
      uint2 key_ = key;
      for(int i = 0; i < 9; i++) {
        counter_ = single_round(counter_, key_);
        key_.x += (kPhilox10A); key_.y += (kPhilox10B);
      }
      output = single_round(counter_, key_);
      incr();
    }
    unsigned long ret;
    switch(STATE) {
      case 0: ret = output.x; break;
      case 1: ret = output.y; break;
      case 2: ret = output.z; break;
      case 3: ret = output.w; break;
    }
    STATE = (STATE + 1) % 4;
    return ret;
  }
private:
  uint4 counter;
  uint4 output;
  uint2 key;
  unsigned int STATE;
  __device__ inline void incr_n(unsigned long long n) {
    unsigned int nlo = (unsigned int)(n);
    unsigned int nhi = (unsigned int)(n >> 32);
    counter.x += nlo;
    if (counter.x < nlo)
      nhi++;
    counter.y += nhi;
    if (nhi <= counter.y)
      return;
    if (++counter.z)
      return;
    ++counter.w;
  }
  __device__ inline void incr() {
    if (++counter.x)
      return;
    if (++counter.y)
      return;
    if (++counter.z)
      return;
    ++counter.w;
  }
  __device__ unsigned int mulhilo32(unsigned int a, unsigned int b,
                                    unsigned int *result_high) {
    *result_high = __umulhi(a, b);
    return a*b;
  }
  __device__ inline uint4 single_round(uint4 ctr, uint2 key) {
    unsigned int hi0;
    unsigned int hi1;
    unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
    unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);
    uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
    return ret;
  }
  static const unsigned long kPhilox10A = 0x9E3779B9;
  static const unsigned long kPhilox10B = 0xBB67AE85;
  static const unsigned long kPhiloxSA = 0xD2511F53;
  static const unsigned long kPhiloxSB = 0xCD9E8D57;
};
// Inverse of 2^32.
#define M_RAN_INVM32 2.3283064e-10f
__device__  __inline__ float uniform(unsigned int x) {
  return x * M_RAN_INVM32;
}


__device__ constexpr int ceilDiv(const int a, const int b) {
  return (a + b - 1) / b;
}
__device__ float clamp(const float x, const float minv, const float maxv) {
  return x < minv ? minv : (x > maxv ? maxv : x);
}
__device__ float frac(const float x) {
  return x - truncf(x);
}
__device__ float gelu(const float x) {
  return x * normcdf(x);
}
__device__ float reciprocal(const float x) {
  return 1.f / x;
}
__device__ float relu(const float x) {
  return x <= 0.f ? 0.f : x;
}
__device__ float remainder(const float a, const float b) {
  return a - b * floorf(a / b);
}
__device__ float sigmoid(const float x) {
  return 1.f / (1.f + expf(-x));
}
__device__ float threshold(const float x, const float t, const float v) {
  return x <= t ? v : x;
}
__device__ float where(const bool c, const float a, const float b) {
  return c ? a : b;
}
__device__ float randLike(Philox rnd) {
  return uniform(rnd());
};


// [Z,Y,X]_THREADS is the number of participating threads in the z, y, x
// dimension of the block. If set to 0 it means that dimension doesn't
// participate, otherwise it is the number of threads. We could start with warp
// reductions, then reduce the warps, this could save some shared memory, but
// may actually be slower.
template<bool X_REDUCE, bool Y_REDUCE, bool Z_REDUCE, typename T, typename Func>
__inline__ __device__
void blockReduce(T& out, const T inp_val, Func reduction_op, const dim3& thread_idx, const dim3& block_dim, T* shared_mem) {

  unsigned int reduction_size 
    = (X_REDUCE ? block_dim.x : 1)
    * (Y_REDUCE ? block_dim.y : 1)
    * (Z_REDUCE ? block_dim.z : 1);

  // If this thread will output a final result
  bool should_write = true;

  if (X_REDUCE)
    should_write = should_write && thread_idx.x == 0;
  if (Y_REDUCE)
    should_write = should_write && thread_idx.y == 0;
  if (Z_REDUCE)
    should_write = should_write && thread_idx.z == 0;

  unsigned int reduction_stride;
  unsigned int reduction_tid;
  unsigned int linear_tid;

  if(X_REDUCE && !Y_REDUCE && Z_REDUCE){
    // Transpose Z and Y in the shared memory so Z and X dims are contiguous in smem
    reduction_stride = 1;
    linear_tid = threadIdx.y * blockDim.z * blockDim.x + threadIdx.z * blockDim.x + threadIdx.x;
    reduction_tid = threadIdx.z * blockDim.x + threadIdx.x;
  } else {
    // Normal reduction in order
    reduction_stride 
    = (X_REDUCE ? 1 
    : (Y_REDUCE ? block_dim.x
    : (Z_REDUCE ? block_dim.x * block_dim.y : 0)));

    linear_tid = thread_idx.z * block_dim.y * block_dim.x + thread_idx.y * block_dim.x + thread_idx.x;

    reduction_tid
    = ( Z_REDUCE ? thread_idx.z : 0 ) * ( Y_REDUCE ? block_dim.y : 1 ) * ( X_REDUCE ? block_dim.x : 1 )
    + ( Y_REDUCE ? thread_idx.y : 0 )                                 * ( X_REDUCE ? block_dim.x : 1 )
    + ( X_REDUCE ? thread_idx.x : 0 );
  }

  assert( reduction_stride != 0 );

  shared_mem[linear_tid] = inp_val;
  __syncthreads();
  // Reduce down to nearest power of 2:
  int np2 =  1 << (31 - __clz(reduction_size));

  if( reduction_tid < np2 ){
    if( reduction_tid + np2 < reduction_size){
      reduction_op( shared_mem[linear_tid], shared_mem[linear_tid + np2 * reduction_stride] );
    }
  }
  __syncthreads();
  //for (int factor = np2/2; factor > contig_threads / 2; factor>>=1) {
  for (int factor = np2/2; factor > 0; factor>>=1) {
    if (reduction_tid < factor) {
      reduction_op( shared_mem[linear_tid], shared_mem[linear_tid + factor * reduction_stride] );
    }
    __syncthreads();
  }
  if(should_write)
    out = shared_mem[linear_tid];
  
}


namespace reduction {

// Utility functions
__host__ __device__ __forceinline__ size_t size(const dim3& d) {
  return (size_t)d.x * (size_t)d.y * (size_t)d.z;
}

__host__ __device__ __forceinline__ int isize(const dim3& d) {
  return d.x * d.y * d.z;
}

__host__ __device__ __forceinline__ size_t offset(const dim3& pos, const dim3& dim) {
  return (size_t)pos.x + (size_t)pos.y * (size_t)dim.x +
      (size_t)pos.z * (size_t)dim.x * (size_t)dim.y;
}

__host__ __device__ __forceinline__ size_t ioffset(const dim3& pos, const dim3& dim) {
  return pos.x + pos.y * dim.x + pos.z * dim.x * dim.y;
}

// Returns dim3 of each reduction segment.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK>
__host__ __device__ dim3 dimension_of_reduction_segment(const dim3& grid_dim) {
  return dim3{X_BLOCK ? grid_dim.x : 1,
        Y_BLOCK ? grid_dim.y : 1,
        Z_BLOCK ? grid_dim.z : 1};
}

// Returns the number of blocks in each reduction segment.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK>
__host__ __device__ size_t size_of_reduction_segment(const dim3& grid_dim) {
  return size(dimension_of_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(grid_dim));
}

// Returns the total number of reduction segments.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK>
__host__ __device__ size_t number_of_reduction_segments(const dim3& grid_dim) {
  return (X_BLOCK ? 1: grid_dim.x) *
      (Y_BLOCK ? 1 : grid_dim.y) *
      (Z_BLOCK ? 1 : grid_dim.z);
}

// Returns the 1-D index of the segment of thread block of block_idx.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK>
__host__ __device__ size_t index_of_reduction_segment(const dim3& block_idx,
                                                      const dim3& grid_dim) {
  size_t seg_idx = 0;
  if (!Z_BLOCK)
    seg_idx += block_idx.z;
  if (!Y_BLOCK)
    seg_idx = seg_idx * grid_dim.y + block_idx.y;
  if (!X_BLOCK)
    seg_idx = seg_idx * grid_dim.x + block_idx.x;
  return seg_idx;
}

// Returns the offset of thread block in its reduction segment.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK>
__host__ __device__ size_t offset_in_reduction_segment(const dim3& block_idx,
                                                       const dim3& grid_dim) {
  size_t offset = 0;
  if (Z_BLOCK)
    offset = offset * grid_dim.z + block_idx.z;
  if (Y_BLOCK)
    offset = offset * grid_dim.y + block_idx.y;
  if (X_BLOCK)
    offset = offset * grid_dim.x + block_idx.x;
  return offset;
}

// Returns dim3 of each reduction block.
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD>
__host__ __device__ dim3 dimension_of_reduction_block(const dim3& block_dim) {
  return dim3{X_THREAD ? block_dim.x : 1,
        Y_THREAD ? block_dim.y : 1,
        Z_THREAD ? block_dim.z : 1};
}

// Returns the number of threads of each reduction block.
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD>
__host__ __device__ int size_of_reduction_block(const dim3& block_dim) {
  return isize(dimension_of_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(block_dim));
}

// Returns the linear offset of a thread in a reduction block.
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD>
__host__ __device__ int offset_in_reduction_block(const dim3& thread_idx,
                                                  const dim3& block_dim) {
  int offset = 0;
  if (Z_THREAD)
    offset += thread_idx.z;
  if (Y_THREAD)
    offset = offset * block_dim.y + thread_idx.y;
  if (X_THREAD)
    offset = offset * block_dim.x + thread_idx.x;
  return offset;
}

/** Reduces all the reduction blocks in each reduction segment.

  This is only used by one thread block per reduction segment. The input
  reduction blocks of the segment are stored in an intermediate buffer pointed
  by parameter in. Template parameters X/Y/Z_THREAD denote how the reduction
  block is formed.

  The size of a reduction block is by definition smaller or equal to the size of
  a thread block. We use the remaining threads to parallelize reductions across
  reduction blocks. For example, when X/Y/Z_THREAD = {true, false, false}, we
  use blockDim.y*blockDim.z threads for each output value. This is done first by
  loading the input values in parallel and then by reducing across threads of
  dimensions whose XYZ_THREAD are false.

  Note that what is done here after the loading from global memory is similar to
  what the existing blockReduce function does. The main difference is that the
  logical block to reduce is a 2D domain where the leading dimension is the size
  of a reduction block and the second dimension is the remaining factor in each
  thread block. For example, when X/Y/Z_THREAD = {false, true, false}, the
  threads are arranged as (blockDim.y, blockDim.x*blockDim.z). We do not reduce
  along the first dimension but only the second dimension. So, it is possible to
  reuse the existing blockReduce with dim3{blockDim.y, blockDim.x*blockDim.z}
  instead of blockDim and with X_THREAD and Y_THREAD being false and true,
  respectively. Also, it still need to shuffle the final output values to their
  actual corresponding threads. In the case of when X/Y/Z_THREAD = {false, true,
  false}, after the intra-block reduction, the final results will still be held
  by the first blockDim.y threads, which need to be transferred to threads at
  threadIdx.x == 0 and threadIdx.z == 0.
*/
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD,
          typename T, typename Func>
__device__ void gridReduceLastBlock(T& out, const T *in, const size_t in_size,
                                    Func reduction_op, T* shared_buf) {
  const int tid = ioffset(threadIdx, blockDim);
  const int block_size = isize(blockDim);
  const int rblock_size = size_of_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(blockDim);

  T inp = {0.0, 0.0,0.0,0.0};
  if (tid < in_size) {
    inp = in[tid];
  }
  for (size_t i = tid + block_size; i < in_size; i += block_size) {
    reduction_op(inp, in[i]);
  }

  const auto should_write = (X_THREAD || threadIdx.x == 0) &&
      (Y_THREAD || threadIdx.y == 0) &&
      (Z_THREAD || threadIdx.z == 0);

  auto rem_size = block_size / rblock_size;

  if (rem_size > 1) {
    const int rblock_offset = tid % rblock_size;
    const int rblock_idx = tid / rblock_size;
    blockReduce<false, true, false>(
        inp, inp, reduction_op,
        dim3{(unsigned)rblock_offset, (unsigned)rblock_idx, 0},
        dim3{(unsigned)rblock_size, (unsigned)rem_size},
        shared_buf);
    __syncthreads();
    if (tid < rblock_size) {
      shared_buf[tid] = inp;
    }
    __syncthreads();
    if (should_write) {
      inp = shared_buf[offset_in_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(
          threadIdx, blockDim)];
    }
  }

  if (should_write) {
    out = inp;
  }
}

/** Reduces per-thread values across thread blocks.

Function parameters:
- out: Per-thread output location
- inp_val: Per-thread input value
- reduction_op: Scalar reduction function
- work_buf: Temporary buffer for cross-block reductions
- sync_flags: A vector of integers for synchronizations
- shared_buf: Shared memory buffer for intra-block reduction

Return true when the thread block has the valid result.

Template parameters:
- X/Y/Z_BLOCK: When true, reduces across thread blocks along the X/Y/Z
  dimensions
- X/Y/Z_THREAD: When true, all threads along the X/Y/Z dimensions participate in
  the cross-block reduction. Otherwise, only threads at offset 0 do.
- T: Scalar data type of input/output data
- Func: Type of scalara reduction function

Template parameters X/Y/Z_BLOCK define a group of thread blocks that are reduced together. We call
it a reduction segment. Some examples are:

Case 1: X/Y/Z_BLOCK == true/true/true -> There is only one segment, which includes all
  thread blocks. It is effecively the same as the grid.
Case 2: X/Y/Z_BLOCK == false/false/false -> Each thread block comprises an individual
  segment by itself.
Case 3: X/Y/Z_BLOCK == true/false/false -> Each segment contains thread blocks that have
  the same blockDim.x. There will be blockDim.y*blockDim.z such segments.

X/Y/Z_THREAD defines a sub region of a thread block that should be reduced with
the sub regions of other thread blocks. We call it a reduction block. E.g.,

Case 1: X/Y/Z_THREAD == false/false/false -> Only thread 0 participates in the
  cross-block reductions. The reduction block is 1x1x1 with thread 0.
Case 2: X/Y/Z_THREAD == true/true/true-> All threads in a thread block participate in
  the cross-block reductions. The reduction block in this case is equivalent to
  the thread block.

After the function completes, only one thread block per reduction segment gets
valid reduction results. There is no guarantee which particular block gets the
final results.
*/
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK,
          bool X_THREAD, bool Y_THREAD, bool Z_THREAD,
          typename T, typename Func>
__device__ bool gridReduce(T& out, T inp_val, Func reduction_op,
                           volatile T* work_buf,
                           Tensor<int64_t, 1> sync_flags,
                           T* shared_buf) {
  const auto seg_size =
      size_of_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);
  const auto seg_idx =
      index_of_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
  const auto rblock_size =
      size_of_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(blockDim);

  // advance to the offset for this segment
  work_buf += seg_idx * seg_size * rblock_size;

  if ((X_THREAD || threadIdx.x == 0) &&
      (Y_THREAD || threadIdx.y == 0) &&
      (Z_THREAD || threadIdx.z == 0)) {
    auto rblock_offset =
        offset_in_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
    auto thread_offset =
        offset_in_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(threadIdx, blockDim);
    auto work_buf_offset = rblock_size * rblock_offset + thread_offset;
    work_buf[work_buf_offset].x = inp_val.x;
    work_buf[work_buf_offset].y = inp_val.y;
    work_buf[work_buf_offset].z = inp_val.z;
    work_buf[work_buf_offset].w = inp_val.w;
  }
  __syncthreads();

  __shared__ bool last_block;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    __threadfence();
    auto old = atomicAdd(  (unsigned long long*) &sync_flags[seg_idx], 1);
    last_block = old + 1 == seg_size;
  }
  __syncthreads();

  if (last_block) {
    // final reduction
    gridReduceLastBlock<X_THREAD, Y_THREAD, Z_THREAD>(
        out, (T*)work_buf, seg_size * rblock_size,
        reduction_op, shared_buf);
    return true;
  } else {
    return false;
  }
}
} // namespace reduction


namespace broadcast {

template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD>
__host__ __device__ unsigned offset_of_source(const dim3& block_dim, const dim3& thread_idx) {
  unsigned offset = 0;
  if (!Z_THREAD)
    offset = offset * block_dim.z + thread_idx.z;
  if (!Y_THREAD)
    offset = offset * block_dim.y + thread_idx.y;
  if (!X_THREAD)
    offset = offset * block_dim.x + thread_idx.x;
  return offset;
}

/** Broadcasts within partitioned groups of threads.

    X_THREAD: Broadcast from threadIdx.x == 0 if true
    Y_THREAD: Broadcast from threadIdx.y == 0 if true
    Z_THREAD: Broadcast from threadIdx.z == 0 if true
    inp_val: Per-thread source value. Only valid when the thread is a source.
    out: Per-thread output location
 */
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, typename T>
__device__ void blockBroadcast(T& out, T inp_val) {

  // Use worst case for memory.
  __shared__ T shared_mem[1024];

  const bool has_valid_data =
      (!X_THREAD || threadIdx.x == 0) &&
      (!Y_THREAD || threadIdx.y == 0) &&
      (!Z_THREAD || threadIdx.z == 0);

  const auto shared_offset = offset_of_source<X_THREAD, Y_THREAD, Z_THREAD>(blockDim, threadIdx);

  if (has_valid_data)
    shared_mem[shared_offset] = inp_val;

  __syncthreads();

  out = shared_mem[shared_offset];
}

} // namespace broadcast

__device__ void reduction_add_float4(float4& a, const float4 b) {
  a.x = a.x + b.x;
  a.y = a.y + b.y;
  a.z = a.z + b.z;
  a.w = a.w + b.w;
}
__global__ void kernel1(Tensor<__half, 2> T0, Tensor<__half, 1> T3, Tensor<float, 1> T7, Tensor<int64_t, 1> T8){
  __shared__ float4 shared_mem[1024];
  float T2[4];
  for(size_t i131 = 0; i131 < 4; ++i131 ) {
    T2[ i131 ]
       = float(0);
  }
  float2* f2_data = reinterpret_cast<float2*>(T0.data);
  float T4[4] = {float(0.0), float(0.0), float(0.0), float(0.0)};
  for(size_t i128 = 0; i128 < ( ceilDiv(( ceilDiv(( ceilDiv(T0.size[0], 4) ), 256) ), 4) ); ++i128 ) {
    for(size_t i129 = 0; i129 < 4; ++i129 ) {
      __half *T1;
      float2 tmp;
      size_t addr = ( ( ( ( ( ( ( i128 * 4 ) + i129 ) * 256 ) + blockIdx.y ) * 4 ) + threadIdx.y ) * T0.stride[0]/4) + ( ( ( ( ( blockIdx.x ) ) * 32 ) + threadIdx.x ) * T0.stride[1] ) ;
      tmp = f2_data[addr];
      T1 = reinterpret_cast<__half*>(&tmp);
      for(size_t i666 = 0; i666 < 4; ++i666 ) {
        T4[ i666 ]
           = T4[ i666 ]
           + __half2float(T1[ i666 ]);
      }
    }
  }
  float4 block_result;
  float4* T4_f4 = reinterpret_cast<float4*>(T4);
  float4* T2_f4 = reinterpret_cast<float4*>(T2);
  float4* T7_f4 = reinterpret_cast<float4*>(T7.data);
  blockReduce< false, true, false > ( block_result, *T4_f4, reduction_add_float4, threadIdx, blockDim, reinterpret_cast<float4*>(shared_mem));
  bool T2pred = reduction::gridReduce< false, true, false, true, false, true, float4 > ( *T2_f4, block_result, reduction_add_float4, T7_f4, T8, reinterpret_cast<float4*>(shared_mem));
  for(size_t i127 = 0; i127 < 4; ++i127 ) {
        if ( T2pred && ( threadIdx.y == 0 ) )  {
          T3[  (  blockIdx.x * 32 *4 ) + threadIdx.x*4 + i127 ]
             = __float2half(T2[ i127 ]);
        }
  }
}
}

)";

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
