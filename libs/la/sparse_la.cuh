/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */
#ifndef __CULIP_LIBS_LA_SPARSE_LA_CUH_
#define __CULIP_LIBS_LA_SPARSE_LA_CUH_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>

NS_CULIP_BEGIN
NS_LA_BEGIN

/**
 * Computes y = alpha * op(A) * x + beta * y.
 *
 * A must be supplied in CSR format.
 */
template<typename T>
__host__
void
T_csrmv(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const csr_matrix_t<T> * gpu_A,
    const dense_vector_t<T> * gpu_x,
    dense_vector_t<T> * gpu_y,
    const T * gpu_alpha,
    const T * gpu_beta);

/* ************************************************************************** */

/**
 * Transpose a CSR-matrix on the GPU.
 */
template<typename T>
__host__
void
T_transpose_csr(
    const gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    csr_matrix_ptr<T>& gpu_At);

/* ************************************************************************** */

/**
 * Solves a triangular system op(A) * x = alpha * b.
 */
template<typename T>
__host__
void
T_triangular_solve(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const csr_matrix_t<T> * gpu_A,
    const dense_vector_t<T> * gpu_b,
    dense_vector_t<T> * gpu_x,
    const T * alpha);

/* ************************************************************************** */

template<typename T>
__host__
void
T_triangular_step_analysis(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const csr_matrix_t<T> * gpu_A,
    cusparseSolveAnalysisInfo_t info);

template<typename T>
__host__
void
T_triangular_step_solve(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    cusparseSolveAnalysisInfo_t info,
    const csr_matrix_t<T> * gpu_A,
    const dense_vector_t<T> * gpu_b,
    dense_vector_t<T> * gpu_x,
    const T * alpha);

/* ************************************************************************** */

template<typename T>
struct T_approx_triangular_info_t
{
    dense_vector_ptr<T> inv_diag;
    dense_vector_ptr<T> tmp;
};

template<typename T>
__host__
void
T_approx_triangular_step_analysis(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const csr_matrix_t<T> * gpu_A,
    T_approx_triangular_info_t<T>& info);

template<typename T>
__host__
void
T_approx_triangular_step_solve(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const T_approx_triangular_info_t<T>& info,
    const csr_matrix_t<T> * gpu_A,
    const dense_vector_t<T> * gpu_b,
    dense_vector_t<T> * gpu_x,
    const T * alpha,
    const mat_int_t sweeps);

/* ************************************************************************** */

/**
 * Symmetrize a graph (C = A + A').
 */
template<typename T>
__host__
void
T_symmetrize(
    const gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    csr_matrix_ptr<T>& gpu_C);

template<typename T>
__host__
void
T_symmetrize_compute(
    const gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    const csr_matrix_t<T> * gpu_At,
    const dense_vector_t<mat_int_t> * C_csr_row,
    const mat_int_t C_Nnz,
    csr_matrix_ptr<T>& gpu_C);

/* ************************************************************************** */

enum SPNE_MODE
{
    SPNE_FULL,
    SPNE_TRIL,
    SPNE_TRIU
};

/**
 * Computes the matrix-matrix product C = A * A' and extracts C's
 * lower triangular part / upper triangular as output - or returns the
 * full product.
 */
template<typename T>
__host__
void
T_SPNE(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    const SPNE_MODE mode,
    csr_matrix_ptr<T>& gpu_C);

template<typename T>
__host__
void
T_SPNE_compute(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    mat_int_t * gpu_C_csr_row,
    mat_int_t * gpu_C_csr_col,
    T * gpu_C_csr_val);

template<typename T>
__host__
void
T_SPNE_extract(
    gpu_handle_ptr& gpu_handle,
    const mat_int_t m,
    const mat_int_t nnz,
    const mat_int_t * C_csr_row,
    const mat_int_t * C_csr_col,
    const T * C_csr_val,
    const SPNE_MODE mode,
    csr_matrix_ptr<T>& gpu_tri_C);

/* ************************************************************************** */

template<typename T>
__host__
void
T_scale_AD(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    const dense_vector_t<T> * gpu_d,
    csr_matrix_ptr<T>& gpu_AD);

/* ************************************************************************** */

/**
 * Computes the row 1-norm of a CSR matrix.
 */
template<typename T>
__host__
void
T_matrix_row_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    dense_vector_t<T> * gpu_norm);

/* ************************************************************************** */

/**
 * Computes the col 1-norm of a CSR matrix.
 */
template<typename T>
__host__
void
T_matrix_col_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    dense_vector_t<T> * gpu_norm);

/* ************************************************************************** */

/**
 * Computes the row/col 1-norm of the augmented matrix
 * K = [diag(d) A'; A 0]
 */
template<typename T>
__host__
void
T_matrix_augmented_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    const dense_vector_t<T> * gpu_d,
    dense_vector_t<T> * gpu_norm);

/* ************************************************************************** */

/**
 * Computes the row/col 1-norm of the normal matrix
 * N = AD^-1A'
 */
template<typename T>
__host__
void
T_matrix_normal_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    const dense_vector_t<T> * gpu_d,
    dense_vector_t<T> * gpu_norm);

/* ************************************************************************** */

/**
 * Given
 * - a row permutation P,
 * - a col permutation Q,
 * - a row scale R and
 * - a col scale C,
 *
 * this procedure returns P * R * A * C * Q (scale before permute).
 */
template<typename T>
__host__
void
T_matrix_permute_scale(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    const dense_vector_t<T> * gpu_row_scale,
    const dense_vector_t<T> * gpu_col_scale,
    const dense_vector_t<mat_int_t> * gpu_row_perm,
    const dense_vector_t<mat_int_t> * gpu_col_perm,
    csr_matrix_ptr<T>& gpu_PRACQ,
    const bool scale_before_match = true,
    const bool perm_is_old_to_new = true);

/* ************************************************************************** */

template<typename T>
__host__
void
T_matrix_permute_row(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    const dense_vector_t<mat_int_t> * old_to_new,
    csr_matrix_ptr<T>& gpu_pA);

/* ************************************************************************** */

template<typename T>
__host__
void
T_matrix_ruiz_scale(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    csr_matrix_ptr<T>& gpu_scaled_A,
    dense_vector_ptr<T>& gpu_row_scale,
    dense_vector_ptr<T>& gpu_col_scale,
    bool symmetric = false);

/* ************************************************************************** */

/**
 * Converts a CSR to a COO matrix.
 */
template<typename T>
__host__
void
T_csr2coo(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_csr_A,
    coo_matrix_ptr<T>& gpu_coo_A);

/* ************************************************************************** */

/**
 * Converts a COO to a CSR matrix.
 */
template<typename T>
__host__
void
T_coo2csr(
    gpu_handle_ptr& gpu_handle,
    const coo_matrix_t<T> * gpu_coo_A,
    csr_matrix_ptr<T>& gpu_csr_A,
    const bool sort = true);

NS_LA_END
NS_CULIP_END

#endif /* __CULIP_LIBS_LA_SPARSE_LA_CUH_ */
