/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/la/helper_kernels.cuh>

#include <cooperative_groups.h>

using namespace cooperative_groups;

NS_CULIP_BEGIN
NS_LA_BEGIN

template<typename T>
__global__
void
k_set_scalar(
    T* mem,
    const T scalar)
{
    *mem = scalar;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_set_scalar(
    T* to,
    const T* from)
{
    *to = *from;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_set_neg_scalar(
    T* to,
    const T* from)
{
    *to = -*from;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_set_div_scalar(
    T* to,
    const T* from_numer,
    const T* from_denom)
{
    *to = *from_numer / * from_denom;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_set_div_neg_scalar(
    T* to,
    const T* from_numer,
    const T* from_denom)
{
    *to = -*from_numer / * from_denom;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_set_mult_scalar(
    T* to,
    const T* from_a,
    const T* from_b)
{
    *to = *from_a * *from_b;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_set_mult_neg_scalar(
    T* to,
    const T* from_a,
    const T* from_b)
{
    *to = - *from_a * *from_b;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_multiply_scalar(
    T* scalar,
    const T* factor)
{
    *scalar *= *factor;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_multiply_scalar(
    T* scalar,
    const T factor)
{
    *scalar *= factor;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_multiply2_scalar(
    T* scalar_a,
    T* scalar_b,
    const T* factor)
{
    *scalar_a *= *factor;
    *scalar_b *= *factor;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_divide_scalar(
    T* scalar,
    const T* factor)
{
    *scalar /= *factor;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_divide_scalar(
    T* scalar,
    const T factor)
{
    *scalar /= factor;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_plus(
    const T* alpha,
    const T* v0,
    const T* beta,
    const T* v1,
    T* result,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= n)
        return;

    __shared__ T s_alpha;
    __shared__ T s_beta;

    if(threadIdx.x == 0)
    {
        s_alpha = *alpha;
        s_beta = *beta;
    }

    __syncthreads();

    result[id] = s_alpha * v0[id] + s_beta * v1[id];
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_add_cwprod(
    const T* a,
    const T* b,
    T* x,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= n)
        return;

    x[id] += a[id] * b[id];
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_cwprod(
    const T* a,
    const T* b,
    T* x,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= n)
        return;

    x[id] = a[id] * b[id];
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_cwdiv(
    const T* a,
    const T* b,
    T* x,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= n)
        return;

    x[id] = a[id] / b[id];
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_scale(
    const T* alpha,
    const T* v0,
    T* result,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= n)
        return;

    result[id] = (*alpha) * v0[id];
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_scalar_norm(
    const T* p,
    const T* q,
    T* result)
{
    const T p_val = *p;
    const T q_val = *q;
    *result = sqrt(p_val * p_val + q_val * q_val);
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_scalar_norm3(
    const T* p,
    const T* q,
    const T* r,
    T* result)
{
    const T p_val = *p;
    const T q_val = *q;
    const T r_val = *r;
    *result = sqrt(p_val * p_val + q_val * q_val + r_val * r_val);
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_add_scalar(
    T* scalar,
    const T* to_add)
{
    *scalar += *to_add;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_add_scalar(
    T* mem,
    const mat_int_t * ix,
    const T to_add,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= n)
        return;

    const mat_int_t my_ix = ix[id];
    mem[my_ix] += to_add;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_init_with_scalar(
    T* mem,
    const T scalar,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= n)
        return;

    mem[id] = scalar;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_add_app(
    T* mem,
    const mat_int_t* ix,
    const T* val,
    const T scalar,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= n)
        return;

    mem[ix[id]] += scalar * val[id];
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_select_bounded(
    const T* full,
    T* subset,
    const mat_int_t* ix,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= n)
        return;

    subset[id] = full[ix[id]];
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_max(
    T* mem,
    const T* op_a,
    const T* op_b)
{
    *mem = max(*op_a, *op_b);
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_max(
    T* mem,
    const T* op_a,
    const T op_b)
{
    *mem = max(*op_a, op_b);
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_min(
    T* mem,
    const T* op_a,
    const T* op_b)
{
    *mem = min(*op_a, *op_b);
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_min(
    T* mem,
    const T* op_a,
    const T op_b)
{
    *mem = min(*op_a, op_b);
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_abs(
    T* op)
{
    *op = fabs(*op);
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_abs(
    T * op,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= n)
        return;

    op[id] = fabs(op[id]);
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_max_abs(
    T* mem,
    const T* op_a,
    const T* op_b)
{
    *mem = max(fabs(*op_a), fabs(*op_b));
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_cwmax_abs(
    T* mem,
    const T* op_a,
    const T* op_b,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= n)
        return;

    mem[id] = max(fabs(op_a[id]), fabs(op_b[id]));
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_check_positive(
    const T* in,
    T * out,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= n)
        return;

    if(in[id] <= 0)
        *out = (T) -1;
}

/* ************************************************************************** */

template<typename T1, typename T2>
__global__
void
k_propagate_values(
    const T1 * ix_list,
    const T2 value,
    T2 * out,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= n)
        return;

    out[ix_list[id]] = value;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_add_scalar(
    T* mem,
    const T scalar,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= n)
        return;

    mem[id] += scalar;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_sqrt(
    const T * in,
    T * out,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= n)
        return;

    out[id] = sqrt(in[id]);
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_cwmin(
    const T * in_a,
    const T * in_b,
    T * out,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= n)
        return;

    out[id] = min(in_a[id], in_b[id]);
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_cwmax(
    const T * in_a,
    const T * in_b,
    T * out,
    const mat_int_t n)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= n)
        return;

    out[id] = max(in_a[id], in_b[id]);
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_row_norm(
    const mat_int_t * csr_row,
    const mat_int_t * csr_col,
    const T* csr_val,
    T * row_norms,
    const mat_int_t m)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= m)
        return;

    const mat_int_t row_offset = csr_row[id];
    const mat_int_t row_size = csr_row[id + 1] - row_offset;

    T my_norm = (T) 0;
    for(mat_int_t i = 0; i < row_size; ++i)
    {
        const T val = csr_val[row_offset + i];
        my_norm += val * val;
    }

    row_norms[id] = sqrt(my_norm);
}

/* ************************************************************************** */

template<typename T>
__forceinline__ __device__
T
sign(
    const T in)
{
    return (T) (in < 0 ? -1 : 1);
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_sym_ortho(
    const T * a,
    const T * b,
    T * c,
    T * s,
    T * r)
{
    if(*b == 0)
    {
        *s = 0;
        *r = abs(*a);

        *c = (*a == 0) ? 1 : sign(*a);
    }
    else if(*a == 0)
    {
        *c = 0;
        *s = sign(*b);
        *r = abs(*b);
    }
    else if(abs(*b) >= abs(*a))
    {
        const T tao = *a / *b;
        *s = sign(*b) / sqrt(1 + tao * tao);
        *c = *s * tao;
        *r = *b / *s;
    }
    else if(abs(*a) > abs(*b))
    {
        const T tao = *b / *a;
        *c = sign(*a) / sqrt(1 + tao * tao);
        *s = *c * tao;
        *r = *a / *c;
    }
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_set_identity(
    T * matrix,
    const mat_int_t m)
{
    const mat_int_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= m)
        return;

    matrix[id * m + id] = 1;
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_solve_D(
    const T * tridiagonal,
    const mat_int_t * is_piv,
    T * b,
    const mat_int_t n)
{
    thread_block tb = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(tb);

    const mat_int_t bidx = tb.group_index().x;
    const mat_int_t tidx = tb.thread_rank();
    const mat_int_t gidx = bidx * tb.size() + tidx;

    /* every thread handles one pivot */
    if(gidx >= n)
        return;

    const mat_int_t this_piv = is_piv[gidx];
    const mat_int_t next_piv = (gidx < n - 1) ? is_piv[gidx + 1] : 1;

    /* !next_piv - is 2x2 pivot */
    const T p_11 = tridiagonal[3 * gidx + 1];
    const T p_12 = tridiagonal[3 * gidx + 2];
    const T p_22 = !next_piv ? tridiagonal[3 * (gidx + 1) + 1] : 1.0;

    const T r_1 = b[gidx];
    const T r_2 = (gidx < n - 1) ? b[gidx + 1] : 0;

    /* compute Givens rotation */
    const T giv_r = sqrt(p_11 * p_11 + p_12 * p_12);
    const T giv_c = p_11 / giv_r;
    const T giv_s = p_12 / giv_r;

    /* apply Givens to RHS */
    T g_r_1 = giv_c * r_1 + giv_s * r_2;
    T g_r_2 = -giv_s * r_1 + giv_c * r_2;

    /* apply Givens rotation to pivot */
    const T tilde_p11 = giv_r;
    const T tilde_p12 = giv_c * p_12 + giv_s * p_22;
    const T tilde_p22 = -giv_s * p_12 + giv_c * p_22;

    g_r_2 /= tilde_p22;
    g_r_1 = (g_r_1 - tilde_p12 * g_r_2) / tilde_p11;

    if(this_piv)
        b[gidx] = g_r_1;
    if(!next_piv)
        b[gidx + 1] = g_r_2;
}

/* ************************************************************************** */

/* push: new to old positions */
template<typename T>
__global__
void
k_permute_P(
    const T * right,
    T * left,
    const mat_int_t * P,
    const mat_int_t n)
{
    thread_block tb = this_thread_block();
    const mat_int_t gidx = tb.group_index().x * tb.size() + tb.thread_rank();

    if(gidx >= n)
        return;

    left[P[gidx]] = right[gidx];
}

/* ************************************************************************** */

/* pull: old to new positions */
template<typename T>
__global__
void
k_permute_Pt(
    const T * right,
    T * left,
    const mat_int_t * P,
    const mat_int_t n)
{
    thread_block tb = this_thread_block();
    const mat_int_t gidx = tb.group_index().x * tb.size() + tb.thread_rank();

    if(gidx >= n)
        return;

    left[gidx] = right[P[gidx]];
}

/* ************************************************************************** */

template<typename T>
__host__ __device__
bool
stencil_is_ge_zero<T>::
operator()(
    T s)
{
    return (s >= (T) 0);
}

template<typename T>
__host__ __device__
bool
stencil_is_gt_zero<T>::
operator()(
    T s)
{
    return (s > (T) 0);
}

template<typename T>
__host__ __device__
bool
stencil_is_zero<T>::
operator()(
    T s)
{
    return (s == (T) 0);
}

template<typename T>
__host__ __device__
bool
stencil_is_one<T>::
operator()(
    T s)
{
    return (s == (T) 1);
}

template<typename T>
__host__ __device__
bool
stencil_is_two<T>::
operator()(
    T s)
{
    return (s == (T) 2);
}

/* ************************************************************************** */

template<typename T>
thrust_map_func<T>::
thrust_map_func(
    const T * map)
: m_map(map)
{

}

template<typename T>
T
thrust_map_func<T>::
operator()(const T in) const
{
    return m_map[in];
}

NS_LA_END
NS_CULIP_END
