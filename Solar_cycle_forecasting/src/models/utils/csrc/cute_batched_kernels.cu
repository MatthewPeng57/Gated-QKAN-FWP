// Copyright 2026 Kuo-Chung Peng and Samuel Yen-Chi Chen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ---------------------------------------------------------------------------
// NOTICE (Apache-2.0 4(b)) -- this file is a MODIFIED copy.
//
// Original: https://github.com/Jim137/qkan (Apache-2.0),
//           Copyright (c) Jiun-Cheng Jiang.
// Modifications: Copyright 2026 Kuo-Chung Peng and Samuel Yen-Chi Chen,
//           licensed under the same Apache-2.0 terms.
//
// Summary of changes made to the upstream kernel:
//   * per-sample batched `theta` carrying a leading batch axis
//     `(B, out_dim, in_dim, reps+1, 2)`, so each batch lane reads its own
//     theta -- required because the GQKAN-QKANFWP fast programmer's theta is
//     an input produced by the slow programmer, not a shared parameter;
//   * the block-shared trig cache is dropped, since it assumed one shared
//     theta per (out_dim, in_dim) block;
//   * added the matching batched backward sweep.
// ---------------------------------------------------------------------------

// Implementation note
// -------------------
// Batched-theta pz_encoding forward + backward CuTe kernels.
//
// Extends upstream qkan `cute_pz_fwd_kernel` (Jim137/qkan v0.2.2.post3, Apache-2.0)
// to accept PER-SAMPLE theta `(B, out_dim, in_dim, reps+1, 2)` for the FWP
// fast-programmer readout. Precedent: src/models/utils/cutile_batched_ops.py.
//
// Key delta vs upstream: theta gains a leading batch axis; each thread (one batch
// lane `b`) reads its OWN theta, so the block-shared trig cache is dropped (it
// assumed one shared theta per (o,i) block). CuTe views used for x/out/pw/pb
// (upstream pattern); raw row-major indexing for the batched theta.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include <cute/tensor.hpp>

using namespace cute;

static constexpr float INV_SQRT2 = 0.7071067811865476f;

#define ROWMAJOR2(d0, d1) \
    make_layout(make_shape((d0), (d1)), make_stride((d1), 1))
#define ROWMAJOR3(d0, d1, d2) \
    make_layout(make_shape((d0), (d1), (d2)), make_stride((d1)*(d2), (d2), 1))

static inline torch::Tensor prep(torch::Tensor t, torch::ScalarType dtype) {
    if (!t.is_contiguous()) t = t.contiguous();
    if (t.scalar_type() != dtype) t = t.to(dtype);
    return t;
}

static int select_block_b(int /*n_oi*/, int batch) {
    if (batch >= 512) return 256;
    if (batch >= 128) return 128;
    if (batch >= 32)  return 64;
    return 32;
}

// ── Block-wide reduction → single atomicAdd (for grad_pw/grad_pb, shared across
//    the batch axis). Ported from Jim137/qkan v0.2.2.post3,
//    qkan/csrc/cute_kernels.cu. Only used on the
//    preacts_trainable path; grad_theta (per-sample) needs no reduction. ──
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

template <int BLOCK_B>
__device__ __forceinline__ void block_reduce_atomic_add(
    float val, bool valid, float* reduce_smem, float* target)
{
    constexpr int NUM_WARPS = (BLOCK_B + 31) / 32;
    float masked = valid ? val : 0.0f;
    if constexpr (NUM_WARPS == 1) {
        float sum = warp_reduce_sum(masked);
        if ((threadIdx.x & 31) == 0 && sum != 0.0f) atomicAdd(target, sum);
    } else {
        int tid = threadIdx.x, warp_id = tid >> 5, lane_id = tid & 31;
        float warp_sum = warp_reduce_sum(masked);
        if (lane_id == 0) reduce_smem[warp_id] = warp_sum;
        __syncthreads();
        if (tid < 32) {
            float v = (tid < NUM_WARPS) ? reduce_smem[tid] : 0.0f;
            v = warp_reduce_sum(v);
            if (tid == 0 && v != 0.0f) atomicAdd(target, v);
        }
        __syncthreads();
    }
}

// ── Reverse rotation operators (adjoint-state backward). For Rz and Ry the
//    inverse gate (state reconstruction) and the transpose (adjoint transport)
//    are the SAME map, so each helper serves both. Cross-checked against
//    Jim137/qkan v0.2.2.post3, qkan/csrc/cute_kernels.cu (cute_pz_bwd_kernel). ──
// rev_rz: multiply alpha by (c + i·s), beta by (c - i·s).
__device__ __forceinline__ void rev_rz(
    float c, float s, float& r0, float& i0, float& r1, float& i1)
{
    float nr0 = r0*c - i0*s, ni0 = i0*c + r0*s;
    float nr1 = r1*c + i1*s, ni1 = i1*c - r1*s;
    r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
}
// rev_ry: apply [[c, s], [-s, c]] (transpose/inverse of forward Ry [[c,-s],[s,c]]).
__device__ __forceinline__ void rev_ry(
    float c, float s, float& r0, float& i0, float& r1, float& i1)
{
    float nr0 =  c*r0 + s*r1, ni0 =  c*i0 + s*i1;
    float nr1 = -s*r0 + c*r1, ni1 = -s*i0 + c*i1;
    r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
}

#define DISPATCH_FWD(block_b, IOT_TYPE, KERNEL, grid, smem, ...) \
    do { \
        auto stream = at::cuda::getCurrentCUDAStream(); \
        switch (block_b) { \
            case 32:  KERNEL<IOT_TYPE, 32> <<<grid, 32,  smem, stream>>>(__VA_ARGS__); break; \
            case 64:  KERNEL<IOT_TYPE, 64> <<<grid, 64,  smem, stream>>>(__VA_ARGS__); break; \
            case 128: KERNEL<IOT_TYPE, 128><<<grid, 128, smem, stream>>>(__VA_ARGS__); break; \
            case 256: KERNEL<IOT_TYPE, 256><<<grid, 256, smem, stream>>>(__VA_ARGS__); break; \
            default:  KERNEL<IOT_TYPE, 32> <<<grid, 32,  smem, stream>>>(__VA_ARGS__); break; \
        } \
    } while (0)

// ====================================================================
// Batched PZ-encoding forward kernel
// Circuit: H|0> -> [Rz(th0) Ry(th1) Rz(enc)]xreps -> Rz(th0_f) Ry(th1_f) -> <Z>
// theta: (B, out_dim, in_dim, reps+1, 2), each lane b reads its own theta.
// ====================================================================
template <typename IOT, int BLOCK_B>
__global__ void cute_pz_fwd_batched_kernel(
    const IOT* __restrict__ x_ptr,
    const IOT* __restrict__ theta_ptr,
    const IOT* __restrict__ pw_ptr,
    const IOT* __restrict__ pb_ptr,
    IOT* __restrict__ out_ptr,
    int batch_size, int in_dim, int out_dim, int reps,
    bool preacts_trainable, bool fast_measure)
{
    int pid_oi = blockIdx.x;
    int idx_i  = pid_oi % in_dim;
    int idx_o  = pid_oi / in_dim;
    if (idx_o >= out_dim) return;

    int b = blockIdx.y * BLOCK_B + threadIdx.x;
    if (b >= batch_size) return;

    // CuTe views (x/out/pw/pb) — upstream pattern
    auto gX   = make_tensor(make_gmem_ptr(x_ptr),   ROWMAJOR2(batch_size, in_dim));
    auto gOut = make_tensor(make_gmem_ptr(out_ptr), ROWMAJOR3(batch_size, out_dim, in_dim));
    auto gPW  = make_tensor(make_gmem_ptr(pw_ptr),  ROWMAJOR3(out_dim, in_dim, reps > 0 ? reps : 1));
    auto gPB  = make_tensor(make_gmem_ptr(pb_ptr),  ROWMAJOR3(out_dim, in_dim, reps > 0 ? reps : 1));

    // Per-sample theta base (raw row-major index into (B,out,in,reps+1,2))
    const IOT* th = theta_ptr
        + ((((long)b * out_dim + idx_o) * in_dim + idx_i) * (reps + 1) * 2);
    // th[l*2 + 0] = Rz angle of layer l ; th[l*2 + 1] = Ry angle of layer l

    float x_val = float(gX(b, idx_i));
    float r0 = INV_SQRT2, i0 = 0.0f, r1 = INV_SQRT2, i1 = 0.0f;

    for (int l = 0; l < reps; l++) {
        // Rz(theta[l,0])
        float cz, sz;
        __sincosf(float(th[l * 2 + 0]) * 0.5f, &sz, &cz);
        float nr0 = r0*cz + i0*sz, ni0 = i0*cz - r0*sz, nr1 = r1*cz - i1*sz, ni1 = i1*cz + r1*sz;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;

        // Ry(theta[l,1])
        float cy, sy;
        __sincosf(float(th[l * 2 + 1]) * 0.5f, &sy, &cy);
        nr0 = cy*r0 - sy*r1; ni0 = cy*i0 - sy*i1; nr1 = sy*r0 + cy*r1; ni1 = sy*i0 + cy*i1;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;

        // Rz(enc)
        float enc = x_val;
        if (preacts_trainable)
            enc = float(gPW(idx_o, idx_i, l)) * x_val + float(gPB(idx_o, idx_i, l));
        float ce, se;
        __sincosf(enc * 0.5f, &se, &ce);
        nr0 = r0*ce + i0*se; ni0 = i0*ce - r0*se; nr1 = r1*ce - i1*se; ni1 = i1*ce + r1*se;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
    }

    // Final Rz(theta[reps,0])
    {
        float cz, sz;
        __sincosf(float(th[reps * 2 + 0]) * 0.5f, &sz, &cz);
        float nr0 = r0*cz + i0*sz, ni0 = i0*cz - r0*sz, nr1 = r1*cz - i1*sz, ni1 = i1*cz + r1*sz;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
    }
    // Final Ry(theta[reps,1])
    {
        float cy, sy;
        __sincosf(float(th[reps * 2 + 1]) * 0.5f, &sy, &cy);
        float nr0 = cy*r0 - sy*r1, ni0 = cy*i0 - sy*i1, nr1 = sy*r0 + cy*r1, ni1 = sy*i0 + cy*i1;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
    }

    float result = fast_measure
        ? (sqrtf(r0*r0 + i0*i0) - sqrtf(r1*r1 + i1*i1))
        : ((r0*r0 + i0*i0) - (r1*r1 + i1*i1));
    gOut(b, idx_o, idx_i) = IOT(result);
}

// ====================================================================
// Batched PZ-encoding BACKWARD kernel (adjoint-state, reversible recompute)
// Per-sample theta: each lane b owns its theta slab.
//   grad_theta (B,out,in,reps+1,2) — PLAIN STORE (unique writer, no atomic).
//   grad_x     (B,in)              — atomicAdd (accumulates across out).
//   grad_pw/pb (out,in,reps)       — block-reduce → atomicAdd (shared over B).
// Recomputes the forward to the final state (no state buffer), then walks the
// circuit in reverse: each gate is un-applied to the state (reconstructing its
// pre-gate input) and to the adjoint (transporting it), keeping only the running
// 4-real state in registers.
// ====================================================================
template <typename IOT, int BLOCK_B>
__global__ void cute_pz_bwd_batched_kernel(
    const IOT* __restrict__ x_ptr,
    const IOT* __restrict__ theta_ptr,
    const IOT* __restrict__ pw_ptr,
    const IOT* __restrict__ pb_ptr,
    const IOT* __restrict__ grad_out_ptr,
    IOT* __restrict__ grad_x_ptr,       // (B, in_dim)
    IOT* __restrict__ grad_theta_ptr,   // (B, out, in, reps+1, 2)
    IOT* __restrict__ grad_pw_ptr,      // (out, in, reps)
    IOT* __restrict__ grad_pb_ptr,      // (out, in, reps)
    int batch_size, int in_dim, int out_dim, int reps,
    bool preacts_trainable, bool fast_measure)
{
    extern __shared__ float smem[];          // NUM_WARPS floats (preacts reduce)
    float* s_reduce = smem;

    int pid_oi = blockIdx.x;
    int idx_i  = pid_oi % in_dim;
    int idx_o  = pid_oi / in_dim;
    if (idx_o >= out_dim) return;            // uniform across block

    int tid = threadIdx.x;
    int b   = blockIdx.y * BLOCK_B + tid;
    bool valid = b < batch_size;
    int b_safe = valid ? b : 0;              // OOB lanes read lane 0, write nothing

    auto gX      = make_tensor(make_gmem_ptr(x_ptr),        ROWMAJOR2(batch_size, in_dim));
    auto gGradOut= make_tensor(make_gmem_ptr(grad_out_ptr), ROWMAJOR3(batch_size, out_dim, in_dim));
    auto gPW     = make_tensor(make_gmem_ptr(pw_ptr),       ROWMAJOR3(out_dim, in_dim, reps > 0 ? reps : 1));
    auto gPB     = make_tensor(make_gmem_ptr(pb_ptr),       ROWMAJOR3(out_dim, in_dim, reps > 0 ? reps : 1));

    // Per-sample theta / grad_theta bases (raw row-major, same index as forward).
    long th_off = (((long)b_safe * out_dim + idx_o) * in_dim + idx_i) * (reps + 1) * 2;
    const IOT* th  = theta_ptr      + th_off;   // th[l*2+0]=Rz, th[l*2+1]=Ry
    IOT* gth       = grad_theta_ptr + th_off;

    float x_val = float(gX(b_safe, idx_i));

    // ── Phase 1: forward recompute → final state (r0,i0,r1,i1) ──
    float r0 = INV_SQRT2, i0 = 0.0f, r1 = INV_SQRT2, i1 = 0.0f;
    for (int l = 0; l < reps; l++) {
        float cz, sz; __sincosf(float(th[l*2+0]) * 0.5f, &sz, &cz);
        float nr0 = r0*cz + i0*sz, ni0 = i0*cz - r0*sz, nr1 = r1*cz - i1*sz, ni1 = i1*cz + r1*sz;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;

        float cy, sy; __sincosf(float(th[l*2+1]) * 0.5f, &sy, &cy);
        nr0 = cy*r0 - sy*r1; ni0 = cy*i0 - sy*i1; nr1 = sy*r0 + cy*r1; ni1 = sy*i0 + cy*i1;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;

        float enc = preacts_trainable
            ? (float(gPW(idx_o, idx_i, l)) * x_val + float(gPB(idx_o, idx_i, l))) : x_val;
        float ce, se; __sincosf(enc * 0.5f, &se, &ce);
        nr0 = r0*ce + i0*se; ni0 = i0*ce - r0*se; nr1 = r1*ce - i1*se; ni1 = i1*ce + r1*se;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
    }
    {
        float cz, sz; __sincosf(float(th[reps*2+0]) * 0.5f, &sz, &cz);
        float nr0 = r0*cz + i0*sz, ni0 = i0*cz - r0*sz, nr1 = r1*cz - i1*sz, ni1 = i1*cz + r1*sz;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
    }
    {
        float cy, sy; __sincosf(float(th[reps*2+1]) * 0.5f, &sy, &cy);
        float nr0 = cy*r0 - sy*r1, ni0 = cy*i0 - sy*i1, nr1 = sy*r0 + cy*r1, ni1 = sy*i0 + cy*i1;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
    }

    // ── Phase 2: measurement adjoint from final state ──
    float go = valid ? float(gGradOut(b_safe, idx_o, idx_i)) : 0.0f;
    float ar0, ai0, ar1, ai1;
    if (fast_measure) {
        float an = sqrtf(r0*r0 + i0*i0), bn = sqrtf(r1*r1 + i1*i1);
        float ia = (an > 1e-30f) ? 1.0f/an : 0.0f;
        float ib = (bn > 1e-30f) ? 1.0f/bn : 0.0f;
        ar0 =  go*r0*ia; ai0 =  go*i0*ia; ar1 = -go*r1*ib; ai1 = -go*i1*ib;
    } else {
        ar0 =  2.0f*go*r0; ai0 =  2.0f*go*i0; ar1 = -2.0f*go*r1; ai1 = -2.0f*go*i1;
    }

    // ── Phase 3: reverse sweep. State (r*,i*) now un-applies gate-by-gate; after
    //    rev_* it holds the PRE-gate state used in the angle gradient. ──
    float grad_x_local = 0.0f;

    // Final Ry(th[reps,1])
    {
        float cy, sy; __sincosf(float(th[reps*2+1]) * 0.5f, &sy, &cy);
        rev_ry(cy, sy, r0, i0, r1, i1);
        float gv = 0.5f * (ar0*(-sy*r0 - cy*r1) + ai0*(-sy*i0 - cy*i1)
                         + ar1*( cy*r0 - sy*r1) + ai1*( cy*i0 - sy*i1));
        if (valid) gth[reps*2+1] = IOT(gv);
        rev_ry(cy, sy, ar0, ai0, ar1, ai1);
    }
    // Final Rz(th[reps,0])
    {
        float cz, sz; __sincosf(float(th[reps*2+0]) * 0.5f, &sz, &cz);
        rev_rz(cz, sz, r0, i0, r1, i1);
        float gv = 0.5f * (-sz*(ar0*r0 + ai0*i0 + ar1*r1 + ai1*i1)
                         +  cz*(ar0*i0 - ai0*r0 - ar1*i1 + ai1*r1));
        if (valid) gth[reps*2+0] = IOT(gv);
        rev_rz(cz, sz, ar0, ai0, ar1, ai1);
    }

    for (int l = reps - 1; l >= 0; l--) {
        // Rz(enc)
        float enc = preacts_trainable
            ? (float(gPW(idx_o, idx_i, l)) * x_val + float(gPB(idx_o, idx_i, l))) : x_val;
        float ce, se; __sincosf(enc * 0.5f, &se, &ce);
        rev_rz(ce, se, r0, i0, r1, i1);
        float grad_enc = 0.5f * (-se*(ar0*r0 + ai0*i0 + ar1*r1 + ai1*i1)
                               +  ce*(ar0*i0 - ai0*r0 - ar1*i1 + ai1*r1));
        if (preacts_trainable) {
            float w = float(gPW(idx_o, idx_i, l));
            block_reduce_atomic_add<BLOCK_B>(grad_enc * x_val, valid, s_reduce,
                &grad_pw_ptr[(idx_o * in_dim + idx_i) * reps + l]);
            block_reduce_atomic_add<BLOCK_B>(grad_enc, valid, s_reduce,
                &grad_pb_ptr[(idx_o * in_dim + idx_i) * reps + l]);
            grad_x_local += grad_enc * w;
        } else {
            grad_x_local += grad_enc;
        }
        rev_rz(ce, se, ar0, ai0, ar1, ai1);

        // Ry(th[l,1])
        float cy, sy; __sincosf(float(th[l*2+1]) * 0.5f, &sy, &cy);
        rev_ry(cy, sy, r0, i0, r1, i1);
        float gvy = 0.5f * (ar0*(-sy*r0 - cy*r1) + ai0*(-sy*i0 - cy*i1)
                          + ar1*( cy*r0 - sy*r1) + ai1*( cy*i0 - sy*i1));
        if (valid) gth[l*2+1] = IOT(gvy);
        rev_ry(cy, sy, ar0, ai0, ar1, ai1);

        // Rz(th[l,0])
        float cz, sz; __sincosf(float(th[l*2+0]) * 0.5f, &sz, &cz);
        rev_rz(cz, sz, r0, i0, r1, i1);
        float gvz = 0.5f * (-sz*(ar0*r0 + ai0*i0 + ar1*r1 + ai1*i1)
                          +  cz*(ar0*i0 - ai0*r0 - ar1*i1 + ai1*r1));
        if (valid) gth[l*2+0] = IOT(gvz);
        rev_rz(cz, sz, ar0, ai0, ar1, ai1);
    }

    // grad_x: multiple (o,i) programs accumulate to the same (b,i) → atomic.
    if (valid && grad_x_local != 0.0f)
        atomicAdd(&grad_x_ptr[(long)b * in_dim + idx_i], IOT(grad_x_local));
}

// ====================================================================
// Python-facing launcher (float32 I/O; theta (B,out,in,reps+1,2))
// ====================================================================
torch::Tensor cute_pz_forward_batched(
    torch::Tensor x, torch::Tensor theta,
    torch::Tensor pw, torch::Tensor pb,
    bool preacts_trainable, bool fast_measure)
{
    int batch   = x.size(0);
    int in_dim  = x.size(1);
    int out_dim = theta.size(1);
    int reps    = theta.size(3) - 1;

    if (theta.size(0) == 1 && batch > 1)
        theta = theta.expand({batch, out_dim, in_dim, reps + 1, 2});

    auto io = torch::kFloat32;
    x = prep(x, io); theta = prep(theta, io); pw = prep(pw, io); pb = prep(pb, io);

    auto output = torch::empty({batch, out_dim, in_dim},
        torch::TensorOptions().device(x.device()).dtype(io));

    int n_oi = out_dim * in_dim;
    int block_b = select_block_b(n_oi, batch);
    dim3 grid(n_oi, (batch + block_b - 1) / block_b);

    DISPATCH_FWD(block_b, float, cute_pz_fwd_batched_kernel, grid, 0,
        x.data_ptr<float>(), theta.data_ptr<float>(),
        pw.data_ptr<float>(), pb.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_dim, out_dim, reps, preacts_trainable, fast_measure);

    return output;
}

// ====================================================================
// Backward launcher. Returns {grad_x (B,in), grad_theta (B,out,in,reps+1,2),
// grad_pw, grad_pb}. grad_pw/grad_pb are undefined tensors unless
// preacts_trainable (→ None in Python).
// ====================================================================
std::vector<torch::Tensor> cute_pz_backward_batched(
    torch::Tensor x, torch::Tensor theta,
    torch::Tensor pw, torch::Tensor pb,
    torch::Tensor grad_out,
    bool preacts_trainable, bool fast_measure)
{
    int batch   = x.size(0);
    int in_dim  = x.size(1);
    int out_dim = theta.size(1);
    int reps    = theta.size(3) - 1;

    // Broadcast unbatched theta to a materialized (B,...) tensor so each program
    // is the unique writer of its grad_theta slab; sum back afterwards.
    bool theta_bcast = (theta.size(0) == 1 && batch > 1);
    if (theta_bcast)
        theta = theta.expand({batch, out_dim, in_dim, reps + 1, 2});

    auto io = torch::kFloat32;
    x = prep(x, io); theta = prep(theta, io);
    pw = prep(pw, io); pb = prep(pb, io);
    grad_out = prep(grad_out, io);

    auto opts = torch::TensorOptions().device(x.device()).dtype(io);
    auto grad_x     = torch::zeros({batch, in_dim}, opts);
    auto grad_theta = torch::zeros({batch, out_dim, in_dim, reps + 1, 2}, opts);
    auto grad_pw = preacts_trainable ? torch::zeros(pw.sizes(), opts) : torch::zeros({1}, opts);
    auto grad_pb = preacts_trainable ? torch::zeros(pb.sizes(), opts) : torch::zeros({1}, opts);

    int n_oi = out_dim * in_dim;
    int block_b = select_block_b(n_oi, batch);
    int num_warps = (block_b + 31) / 32;
    int smem = num_warps * sizeof(float);
    dim3 grid(n_oi, (batch + block_b - 1) / block_b);

    DISPATCH_FWD(block_b, float, cute_pz_bwd_batched_kernel, grid, smem,
        x.data_ptr<float>(), theta.data_ptr<float>(),
        pw.data_ptr<float>(), pb.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        grad_x.data_ptr<float>(), grad_theta.data_ptr<float>(),
        grad_pw.data_ptr<float>(), grad_pb.data_ptr<float>(),
        batch, in_dim, out_dim, reps, preacts_trainable, fast_measure);

    if (theta_bcast)
        grad_theta = grad_theta.sum(0, /*keepdim=*/true);

    return {grad_x, grad_theta,
            preacts_trainable ? grad_pw : torch::Tensor(),
            preacts_trainable ? grad_pb : torch::Tensor()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pz_forward_batched", &cute_pz_forward_batched,
          "Batched-theta pz_encoding forward (CuTe)");
    m.def("pz_backward_batched", &cute_pz_backward_batched,
          "Batched-theta pz_encoding backward (CuTe)");
}
