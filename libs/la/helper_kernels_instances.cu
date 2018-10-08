/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/la/helper_kernels.cuh>
#include <libs/la/helper_kernels.impl.cuh>

#include <math_constants.h>

NS_CULIP_BEGIN
NS_LA_BEGIN

template
__global__
void
k_set_scalar(
    float* mem,
    const float scalar);

template
__global__
void
k_set_scalar(
    double* mem,
    const double scalar);

template
__global__
void
k_set_scalar(
    mat_int_t* mem,
    const mat_int_t scalar);

/* ************************************************************************** */

template
__global__
void
k_set_scalar(
    float* to,
    const float* from);

template
__global__
void
k_set_scalar(
    double* to,
    const double* from);

/* ************************************************************************** */

template
__global__
void
k_set_neg_scalar(
    float* to,
    const float* from);

template
__global__
void
k_set_neg_scalar(
    double* to,
    const double* from);

/* ************************************************************************** */

template
__global__
void
k_set_div_scalar(
    float* to,
    const float* from_numer,
    const float* from_denom);

template
__global__
void
k_set_div_scalar(
    double* to,
    const double* from_numer,
    const double* from_denom);

/* ************************************************************************** */

template
__global__
void
k_set_div_neg_scalar(
    float* to,
    const float* from_numer,
    const float* from_denom);

template
__global__
void
k_set_div_neg_scalar(
    double* to,
    const double* from_numer,
    const double* from_denom);

/* ************************************************************************** */

template
__global__
void
k_set_mult_scalar(
    float* to,
    const float* from_a,
    const float* from_b);

template
__global__
void
k_set_mult_scalar(
    double* to,
    const double* from_a,
    const double* from_b);

/* ************************************************************************** */

template
__global__
void
k_set_mult_neg_scalar(
    float* to,
    const float* from_a,
    const float* from_b);

template
__global__
void
k_set_mult_neg_scalar(
    double* to,
    const double* from_a,
    const double* from_b);

/* ************************************************************************** */

template
__global__
void
k_multiply_scalar(
    float* scalar,
    const float* factor);

template
__global__
void
k_multiply_scalar(
    double* scalar,
    const double* factor);

/* ************************************************************************** */

template
__global__
void
k_multiply_scalar(
    float* scalar,
    const float factor);

template
__global__
void
k_multiply_scalar(
    double* scalar,
    const double factor);

/* ************************************************************************** */

template
__global__
void
k_multiply2_scalar(
    float* scalar_a,
    float* scalar_b,
    const float* factor);

template
__global__
void
k_multiply2_scalar(
    double* scalar_a,
    double* scalar_b,
    const double* factor);

/* ************************************************************************** */

template
__global__
void
k_divide_scalar(
    float* scalar,
    const float* factor);

template
__global__
void
k_divide_scalar(
    double* scalar,
    const double* factor);

/* ************************************************************************** */

template
__global__
void
k_divide_scalar(
    float* scalar,
    const float factor);

template
__global__
void
k_divide_scalar(
    double* scalar,
    const double factor);

/* ************************************************************************** */

template
__global__
void
k_plus(
    const float* alpha,
    const float* v0,
    const float* beta,
    const float* v1,
    float* result,
    const mat_int_t n);

template
__global__
void
k_plus(
    const double* alpha,
    const double* v0,
    const double* beta,
    const double* v1,
    double* result,
    const mat_int_t n);

template
__global__
void
k_plus(
    const mat_int_t* alpha,
    const mat_int_t* v0,
    const mat_int_t* beta,
    const mat_int_t* v1,
    mat_int_t* result,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_add_cwprod(
    const float* a,
    const float* b,
    float* x,
    const mat_int_t n);

template
__global__
void
k_add_cwprod(
    const double* a,
    const double* b,
    double* x,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_cwprod(
    const float* a,
    const float* b,
    float* x,
    const mat_int_t n);

template
__global__
void
k_cwprod(
    const double* a,
    const double* b,
    double* x,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_cwdiv(
    const float* a,
    const float* b,
    float* x,
    const mat_int_t n);

template
__global__
void
k_cwdiv(
    const double* a,
    const double* b,
    double* x,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_scale(
    const float* alpha,
    const float* v0,
    float* result,
    const mat_int_t n);

template
__global__
void
k_scale(
    const double* alpha,
    const double* v0,
    double* result,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_scalar_norm(
    const float* p,
    const float* q,
    float* result);

template
__global__
void
k_scalar_norm(
    const double* p,
    const double* q,
    double* result);

/* ************************************************************************** */

template
__global__
void
k_scalar_norm3(
    const float* p,
    const float* q,
    const float* r,
    float* result);

template
__global__
void
k_scalar_norm3(
    const double* p,
    const double* q,
    const double* r,
    double* result);

/* ************************************************************************** */

template
__global__
void
k_add_scalar(
    float* scalar,
    const float* to_add);

template
__global__
void
k_add_scalar(
    double* scalar,
    const double* to_add);

/* ************************************************************************** */

template
__global__
void
k_add_scalar(
    float* mem,
    const mat_int_t * ix,
    const float to_add,
    const mat_int_t n);

template
__global__
void
k_add_scalar(
    double* mem,
    const mat_int_t * ix,
    const double to_add,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_init_with_scalar(
    float* mem,
    const float scalar,
    const mat_int_t n);

template
__global__
void
k_init_with_scalar(
    double* mem,
    const double scalar,
    const mat_int_t n);

template
__global__
void
k_init_with_scalar(
    mat_int_t* mem,
    const mat_int_t scalar,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_add_app(
    float* mem,
    const mat_int_t* ix,
    const float* val,
    const float scalar,
    const mat_int_t n);

template
__global__
void
k_add_app(
    double* mem,
    const mat_int_t* ix,
    const double* val,
    const double scalar,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_select_bounded(
    const float* full,
    float* subset,
    const mat_int_t* ix,
    const mat_int_t n);

template
__global__
void
k_select_bounded(
    const double* full,
    double* subset,
    const mat_int_t* ix,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_max(
    float* mem,
    const float* op_a,
    const float* op_b);

template
__global__
void
k_max(
    double* mem,
    const double* op_a,
    const double* op_b);

/* ************************************************************************** */

template
__global__
void
k_max(
    float* mem,
    const float* op_a,
    const float op_b);

template
__global__
void
k_max(
    double* mem,
    const double* op_a,
    const double op_b);

/* ************************************************************************** */

template
__global__
void
k_min(
    float* mem,
    const float* op_a,
    const float* op_b);

template
__global__
void
k_min(
    double* mem,
    const double* op_a,
    const double* op_b);

/* ************************************************************************** */

template
__global__
void
k_min(
    float* mem,
    const float* op_a,
    const float op_b);

template
__global__
void
k_min(
    double* mem,
    const double* op_a,
    const double op_b);

/* ************************************************************************** */

template
__global__
void
k_abs(
    float* op);

template
__global__
void
k_abs(
    double* op);

/* ************************************************************************** */

template
__global__
void
k_abs(
    float * op,
    const mat_int_t n);

template
__global__
void
k_abs(
    double * op,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_max_abs(
    float* mem,
    const float* op_a,
    const float* op_b);

template
__global__
void
k_max_abs(
    double* mem,
    const double* op_a,
    const double* op_b);

/* ************************************************************************** */

template
__global__
void
k_cwmax_abs(
    float* mem,
    const float* op_a,
    const float* op_b,
    const mat_int_t n);

template
__global__
void
k_cwmax_abs(
    double* mem,
    const double* op_a,
    const double* op_b,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_check_positive(
    const float* in,
    float * out,
    const mat_int_t n);

template
__global__
void
k_check_positive(
    const double* in,
    double * out,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_propagate_values(
    const mat_int_t * ix_list,
    const float value,
    float * out,
    const mat_int_t n);

template
__global__
void
k_propagate_values(
    const mat_int_t * ix_list,
    const double value,
    double * out,
    const mat_int_t n);

template
__global__
void
k_propagate_values(
    const mat_int_t * ix_list,
    const mat_int_t value,
    mat_int_t * out,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_add_scalar(
    float * mem,
    const float scalar,
    const mat_int_t n);

template
__global__
void
k_add_scalar(
    double * mem,
    const double scalar,
    const mat_int_t n);

template
__global__
void
k_add_scalar(
    mat_int_t * mem,
    const mat_int_t scalar,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_sqrt(
    const float * in,
    float * out,
    const mat_int_t n);

template
__global__
void
k_sqrt(
    const double * in,
    double * out,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_cwmin(
    const float * in_a,
    const float * in_b,
    float * out,
    const mat_int_t n);

template
__global__
void
k_cwmin(
    const double * in_a,
    const double * in_b,
    double * out,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_cwmax(
    const float * in_a,
    const float * in_b,
    float * out,
    const mat_int_t n);

template
__global__
void
k_cwmax(
    const double * in_a,
    const double * in_b,
    double * out,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_row_norm(
    const mat_int_t * csr_row,
    const mat_int_t * csr_col,
    const float * csr_val,
    float * row_norms,
    const mat_int_t m);

template
__global__
void
k_row_norm(
    const mat_int_t * csr_row,
    const mat_int_t * csr_col,
    const double * csr_val,
    double * row_norms,
    const mat_int_t m);

/* ************************************************************************** */

template
__device__
float
sign(
    const float in);

template
__device__
double
sign(
    const double in);

/* ************************************************************************** */

template
__global__
void
k_sym_ortho(
    const float * a,
    const float * b,
    float * c,
    float * s,
    float * r);

template
__global__
void
k_sym_ortho(
    const double * a,
    const double * b,
    double * c,
    double * s,
    double * r);

/* ************************************************************************** */

template
__global__
void
k_set_identity(
    float * matrix,
    const mat_int_t m);

template
__global__
void
k_set_identity(
    double * matrix,
    const mat_int_t m);

/* ************************************************************************** */

template
__global__
void
k_solve_D(
    const float * tridiagonal,
    const mat_int_t * is_piv,
    float * b,
    const mat_int_t n);

template
__global__
void
k_solve_D(
    const double * tridiagonal,
    const mat_int_t * is_piv,
    double * b,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_permute_P(
    const float * right,
    float * left,
    const mat_int_t * P,
    const mat_int_t n);

template
__global__
void
k_permute_P(
    const double * right,
    double * left,
    const mat_int_t * P,
    const mat_int_t n);

/* ************************************************************************** */

template
__global__
void
k_permute_Pt(
    const float * right,
    float * left,
    const mat_int_t * P,
    const mat_int_t n);

template
__global__
void
k_permute_Pt(
    const double * right,
    double * left,
    const mat_int_t * P,
    const mat_int_t n);

/* ************************************************************************** */

template struct stencil_is_ge_zero<float>;
template struct stencil_is_ge_zero<double>;
template struct stencil_is_ge_zero<mat_int_t>;

template struct stencil_is_gt_zero<float>;
template struct stencil_is_gt_zero<double>;
template struct stencil_is_gt_zero<mat_int_t>;

template struct stencil_is_zero<float>;
template struct stencil_is_zero<double>;
template struct stencil_is_zero<mat_int_t>;

template struct stencil_is_one<float>;
template struct stencil_is_one<double>;
template struct stencil_is_one<mat_int_t>;

template struct stencil_is_two<float>;
template struct stencil_is_two<double>;
template struct stencil_is_two<mat_int_t>;

template<>
__device__
bool
stencil_is_inf<float>::
operator()(
    float s)
{
    return (s == CUDART_INF_F);
}

template<>
__device__
bool
stencil_is_inf<double>::
operator()(
    double s)
{
    return (s == CUDART_INF);
}

template<>
__device__
bool
stencil_is_not_inf<float>::
operator()(
    float s)
{
    return (s != CUDART_INF_F);
}

template<>
__device__
bool
stencil_is_not_inf<double>::
operator()(
    double s)
{
    return (s != CUDART_INF);
}

__host__ __device__
mat_int_t
thrust_tuple_minus::
operator()(
    const thrust::tuple<mat_int_t, mat_int_t>& t0)
const
{
    return (thrust::get<1>(t0) - thrust::get<0>(t0));
}

__host__ __device__
bool
thrust_tuple_sort::
operator()(
    const thrust::tuple<mat_int_t, mat_int_t>& t0,
    const thrust::tuple<mat_int_t, mat_int_t>& t1)
const
{
    if(thrust::get<0>(t0) == thrust::get<0>(t1))
        return (thrust::get<1>(t0) < thrust::get<1>(t1));

    return (thrust::get<0>(t0) < thrust::get<0>(t1));
}

__host__ __device__
bool
thrust_tuple_equal::
operator()(
    const thrust::tuple<mat_int_t, mat_int_t>& t0,
    const thrust::tuple<mat_int_t, mat_int_t>& t1)
const
{
    return (thrust::get<0>(t0) == thrust::get<0>(t1) &&
        thrust::get<1>(t0) == thrust::get<1>(t1));
}

template struct thrust_map_func<mat_int_t>;

NS_LA_END
NS_CULIP_END
