/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */
#ifndef __CULIP_LIBS_LA_HELPER_KERNELS_CUH_
#define __CULIP_LIBS_LA_HELPER_KERNELS_CUH_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>

NS_CULIP_BEGIN
NS_LA_BEGIN

/* mem = scalar (from host) */
template<typename T>
__global__
void
k_set_scalar(
    T* mem,
    const T scalar);

/* ************************************************************************** */

/* mem = scalar */
template<typename T>
__global__
void
k_set_scalar(
    T* to,
    const T* from);

/* ************************************************************************** */

/* to = -from */
template<typename T>
__global__
void
k_set_neg_scalar(
    T* to,
    const T* from);

/* ************************************************************************** */

/* to = from_numer / from_denom */
template<typename T>
__global__
void
k_set_div_scalar(
    T* to,
    const T* from_numer,
    const T* from_denom);

/* ************************************************************************** */

/* to = -from_numer / from_denom */
template<typename T>
__global__
void
k_set_div_neg_scalar(
    T* to,
    const T* from_numer,
    const T* from_denom);

/* ************************************************************************** */

/* to = from_a * from_b */
template<typename T>
__global__
void
k_set_mult_scalar(
    T* to,
    const T* from_a,
    const T* from_b);

/* ************************************************************************** */

/* to = -from_a * from_b */
template<typename T>
__global__
void
k_set_mult_neg_scalar(
    T* to,
    const T* from_a,
    const T* from_b);

/* ************************************************************************** */

/* scalar *= factor */
template<typename T>
__global__
void
k_multiply_scalar(
    T* scalar,
    const T* factor);

/* ************************************************************************** */

/* scalar *= factor (from host) */
template<typename T>
__global__
void
k_multiply_scalar(
    T* scalar,
    const T factor);

/* ************************************************************************** */

/* scalar_{a,b} *= factor */
template<typename T>
__global__
void
k_multiply2_scalar(
    T* scalar_a,
    T* scalar_b,
    const T* factor);

/* ************************************************************************** */

/* scalar /= factor */
template<typename T>
__global__
void
k_divide_scalar(
    T* scalar,
    const T* factor);

/* ************************************************************************** */

/* scalar /= factor (on host) */
template<typename T>
__global__
void
k_divide_scalar(
    T* scalar,
    const T factor);

/* ************************************************************************** */

/* result[] = alpha * v0[] + beta * v1[] */
template<typename T>
__global__
void
k_plus(
    const T* alpha,
    const T* v0,
    const T* beta,
    const T* v1,
    T* result,
    const mat_int_t n);

/* ************************************************************************** */

/* x[] += a[] .* b[] */
template<typename T>
__global__
void
k_add_cwprod(
    const T* a,
    const T* b,
    T* x,
    const mat_int_t n);

/* ************************************************************************** */

/* x[] = a[] .* b[] */
template<typename T>
__global__
void
k_cwprod(
    const T* a,
    const T* b,
    T* x,
    const mat_int_t n);

/* ************************************************************************** */

/* x[] = a[] ./ b[] */
template<typename T>
__global__
void
k_cwdiv(
    const T* a,
    const T* b,
    T* x,
    const mat_int_t n);

/* ************************************************************************** */

/* result[] = alpha * v0[] */
template<typename T>
__global__
void
k_scale(
    const T* alpha,
    const T* v0,
    T* result,
    const mat_int_t n);

/* ************************************************************************** */

/* result = sqrt(p * p + q * q) */
template<typename T>
__global__
void
k_scalar_norm(
    const T* p,
    const T* q,
    T* result);

/* ************************************************************************** */

/* result = sqrt(p * p + q * q) */
template<typename T>
__global__
void
k_scalar_norm3(
    const T* p,
    const T* q,
    const T* r,
    T* result);

/* ************************************************************************** */

/* scalar += *to_add */
template<typename T>
__global__
void
k_add_scalar(
    T* scalar,
    const T* to_add);

/* ************************************************************************** */

/* mem[ix] += to_add */
template<typename T>
__global__
void
k_add_scalar(
    T* mem,
    const mat_int_t * ix,
    const T to_add,
    const mat_int_t n);

/* ************************************************************************** */

/* mem[] += to_add */
template<typename T>
__global__
void
k_add_scalar(
    const T * mem,
    const T to_add,
    const mat_int_t n);

/* ************************************************************************** */

/* init mem[] with a scalar in each fieled */
template<typename T>
__global__
void
k_init_with_scalar(
    T* mem,
    const T scalar,
    const mat_int_t n);

/* ************************************************************************** */

/* add scalar * val to mem[ix], n = numel(ix) */
template<typename T>
__global__
void
k_add_app(
    T* mem,
    const mat_int_t* ix,
    const T* val,
    const T scalar,
    const mat_int_t n);

/* ************************************************************************** */

/* extract elements with indices ix from full, pack them into subset */
template<typename T>
__global__
void
k_select_bounded(
    const T* full,
    T* subset,
    const mat_int_t* ix,
    const mat_int_t n);

/* ************************************************************************** */

/* mem = max(op_a, op_b) */
template<typename T>
__global__
void
k_max(
    T* mem,
    const T* op_a,
    const T* op_b);

/* ************************************************************************** */

/* mem = max(op_a, op_b) (op_b on host) */
template<typename T>
__global__
void
k_max(
    T* mem,
    const T* op_a,
    const T op_b);

/* ************************************************************************** */

/* mem = min(op_a, op_b) */
template<typename T>
__global__
void
k_min(
    T* mem,
    const T* op_a,
    const T* op_b);

/* ************************************************************************** */

/* mem = min(op_a, op_b) (op_b on host) */
template<typename T>
__global__
void
k_min(
    T* mem,
    const T* op_a,
    const T op_b);

/* ************************************************************************** */

/* op = abs(op) */
template<typename T>
__global__
void
k_abs(
    T* op);

/* ************************************************************************** */

/* op[] = abs(op[]) */
template<typename T>
__global__
void
k_abs(
    T * op,
    const mat_int_t n);

/* ************************************************************************** */

/* mem = max(abs(op_a), abs(op_b)) */
template<typename T>
__global__
void
k_max_abs(
    T* mem,
    const T* op_a,
    const T* op_b);

/* ************************************************************************** */

/* mem[] = max(abs(op_a[]), abs(op_b[])) */
template<typename T>
__global__
void
k_cwmax_abs(
    T* mem,
    const T* op_a,
    const T* op_b,
    const mat_int_t n);

/* ************************************************************************** */

/* res is negative, if any component is non-positive */
template<typename T>
__global__
void
k_check_positive(
    const T* in,
    T * out,
    const mat_int_t n);

/* ************************************************************************** */

/* propagate values to index list */
template<typename T1, typename T2>
__global__
void
k_propagate_values(
    const T1 * ix_list,
    const T2 value,
    T2 * out,
    const mat_int_t n);

/* ************************************************************************** */

/* mem[] += scalar */
template<typename T>
__global__
void
k_add_scalar(
    T* mem,
    const T scalar,
    const mat_int_t n);

/* ************************************************************************** */

/* out[] = sqrt(in[]) */
template<typename T>
__global__
void
k_sqrt(
    const T * in,
    T * out,
    const mat_int_t n);

/* ************************************************************************** */

/* out[] = min(in_a[], in_b[]) */
template<typename T>
__global__
void
k_cwmin(
    const T * in_a,
    const T * in_b,
    T * out,
    const mat_int_t n);

/* ************************************************************************** */

/* out[] = max(in_a[], in_b[]) */
template<typename T>
__global__
void
k_cwmax(
    const T * in_a,
    const T * in_b,
    T * out,
    const mat_int_t n);

/* ************************************************************************** */

template<typename T>
__global__
void
k_row_norm(
    const mat_int_t * csr_row,
    const mat_int_t * csr_col,
    const T* csr_val,
    T* row_norms,
    const mat_int_t m);

/* ************************************************************************** */

template<typename T>
__forceinline__ __device__
T
sign(
    const T in);

/* ************************************************************************** */

template<typename T>
__global__
void
k_sym_ortho(
    const T * a,
    const T * b,
    T * c,
    T * s,
    T * r);

/* ************************************************************************** */

/* sets diagonal of m x m matrix to 1 */
template<typename T>
__global__
void
k_set_identity(
    T * matrix,
    const mat_int_t m);

/* ************************************************************************** */

template<typename T>
__global__
void
k_solve_D(
    const T * tridiagonal,
    const mat_int_t * is_piv,
    T * b,
    const mat_int_t n);

/* ************************************************************************** */

template<typename T>
__global__
void
k_permute_P(
    const T * right,
    T * left,
    const mat_int_t * P,
    const mat_int_t n);

/* ************************************************************************** */

template<typename T>
__global__
void
k_permute_Pt(
    const T * right,
    T * left,
    const mat_int_t * P,
    const mat_int_t n);

/* ************************************************************************** */

/* Stencil Ops for thrust's copy_if method */
template<typename T>
struct stencil_is_ge_zero
{
    __host__ __device__
    bool operator()(T s);
};
template<typename T>
struct stencil_is_gt_zero
{
    __host__ __device__
    bool operator()(T s);
};
template<typename T>
struct stencil_is_zero
{
    __host__ __device__
    bool operator()(T s);
};
template<typename T>
struct stencil_is_one
{
    __host__ __device__
    bool operator()(T s);
};
template<typename T>
struct stencil_is_two
{
    __host__ __device__
    bool operator()(T s);
};
template<typename T>
struct stencil_is_inf
{
    __device__
    bool operator()(T s);
};
template<typename T>
struct stencil_is_not_inf
{
    __device__
    bool operator()(T s);
};

/* thrust operations */
struct thrust_tuple_minus
{
    __host__ __device__
    mat_int_t
    operator()(const thrust::tuple<mat_int_t, mat_int_t>& t0) const;
};

struct thrust_tuple_sort
{
    __host__ __device__
    bool
    operator()(
        const thrust::tuple<mat_int_t, mat_int_t>& t0,
        const thrust::tuple<mat_int_t, mat_int_t>& t1) const;
};

struct thrust_tuple_equal
{
    __host__ __device__
    bool
    operator()(
        const thrust::tuple<mat_int_t, mat_int_t>& t0,
        const thrust::tuple<mat_int_t, mat_int_t>& t1) const;
};

template<typename T>
struct thrust_map_func
{
    thrust_map_func(const T * map);

    __host__ __device__
    T operator()(const T in) const;

    const T * m_map;
};

NS_LA_END
NS_CULIP_END

#endif /* __CULIP_LIBS_LA_HELPER_KERNELS_CUH_ */
