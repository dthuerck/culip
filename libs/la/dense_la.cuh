/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIBS_LA_CUBLAS_WRAPPER_H_
#define __CULIP_LIBS_LA_CUBLAS_WRAPPER_H_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>

NS_CULIP_BEGIN
NS_LA_BEGIN

/**
 * *****************************************************************************
 * ****************************** DEVICE FUNCTIONS *****************************
 * *****************************************************************************
 */

/**
 * A set of CUBLAS wrappers that offer float/double templated functions
 * for all needed CUBLAS primitives, accepting custom data structures
 * as input.
 */

/**
 * 2-norm of a dense vector x.
 *
 * Result is retuned as an host/device scalar.
 */
template<typename T>
__host__
void
T_nrm2(
    const gpu_handle_ptr& gpu_handle,
    const dense_vector_t<T> * gpu_x,
    T* result);

/* ************************************************************************** */

/**
 * Dot product of 2 dense vectors.
 *
 * Result is returned as a host/device scalar.
 */
template<typename T>
__host__
void
T_doti(
    const gpu_handle_ptr& gpu_handle,
    const dense_vector_t<T> * gpu_x,
    const dense_vector_t<T> * gpu_y,
    T * result);

/* ************************************************************************** */

/**
 * Scales vector gpu_y in-place by alpha.
 */
template<typename T>
__host__
void
T_scal(
    const gpu_handle_ptr& gpu_handle,
    dense_vector_t<T> * gpu_y,
    const T * alpha);

/* ************************************************************************** */

/**
 * Saxpy operation on 2 dense vectors, i.e.
 * y = y + alpha * x.
 */
template<typename T>
__host__
void
T_axpy(
    const gpu_handle_ptr& gpu_handle,
    dense_vector_t<T> * gpu_y,
    const dense_vector_t<T> * gpu_x,
    const T * alpha);

/* ************************************************************************** */

/**
 * Sbmv operation for multiplying a dense symmetric banded matrix with a dense
 * vector. The bands, starting from the diagonal downwards, are stored in
 * a dense array A. The full operation is:
 * y = alpha * A * x + beta * y
 *
 * The matrix A is of size n x n with k bands (diagonal counts as 1).
 */
template<typename T>
__host__
void
T_sbmv(
    const gpu_handle_ptr& gpu_handle,
    const dense_vector_t<T> * gpu_A,
    const dense_vector_t<T> * gpu_x,
    dense_vector_t<T> * gpu_y,
    const T * alpha,
    const T * beta,
    const mat_int_t& k,
    const mat_int_t& n);

/* ************************************************************************** */

/**
 * Computes res = sum(|x|).
 */
template<typename T>
__host__
void
T_asum(
    const gpu_handle_ptr& gpu_handle,
    const dense_vector_t<T> * gpu_x,
    T * res);

/* ************************************************************************** */

/**
 * Computes y = alpha * op(A) * x + beta * y.
 * Use only the first k columns of A resp. the first k rows in x;
 */
template<typename T>
__host__
void
T_gemv(
    const gpu_handle_ptr& gpu_handle,
    cublasOperation_t transA,
    const col_major_matrix_t<T> * gpu_A,
    const dense_vector_t<T> * gpu_x,
    dense_vector_t<T> * gpu_y,
    const mat_int_t k,
    const T * alpha,
    const T * beta);

/* ************************************************************************** */

/**
 * Compute C = alpha * op(A) * op(B) + beta * C.
 * Use only the first k columns of A resp. rows of B.
 */
template<typename T>
__host__
void
T_gemm(
    const gpu_handle_ptr& gpu_handle,
    cublasOperation_t transA,
    cublasOperation_t transB,
    const col_major_matrix_t<T> * gpu_A,
    const col_major_matrix_t<T> * gpu_B,
    col_major_matrix_t<T> * gpu_C,
    const mat_int_t k,
    const T * alpha,
    const T * beta);

/* ************************************************************************** */

/**
 * Solve a dense SPD system A x = b by a Cholesky factorization - overwrites b
 * by x. The computation procedure will use A's lower triangular part.
 */
template<typename T>
__host__
void
T_potrfs(
    const gpu_handle_ptr& gpu_handle,
    const col_major_matrix_t<T> * gpu_A,
    dense_vector_t<T> * gpu_xb);

/* ************************************************************************** */

/**
 * Solve a the leading k x k part of triangular lower/upper (modeA) system,
 * i.e. solve op(A(1:k, 1:k)) * x = alpha * b.
 */
template<typename T>
__host__
void
T_trsm(
    const gpu_handle_ptr& gpu_handle,
    cublasOperation_t transA,
    cublasFillMode_t modeA,
    const col_major_matrix_t<T> * gpu_A,
    const dense_vector_t<T> * b,
    dense_vector_t<T> * x,
    const mat_int_t k,
    const T * alpha);

/**
 * *****************************************************************************
 * ******************************* HOST FUNCTIONS ******************************
 * *****************************************************************************
 */

template<typename T>
__host__
void
T_H_gemv(
    const bool transA,
    const col_major_matrix_t<T> * A,
    const dense_vector_t<T> * x,
    dense_vector_t<T> * y,
    const mat_int_t k,
    const T alpha,
    const T beta);

/* ************************************************************************** */

template<typename T>
__host__
void
T_H_gemm(
    const bool transA,
    const bool transB,
    const col_major_matrix_t<T> * A,
    const col_major_matrix_t<T> * B,
    col_major_matrix_t<T> * C,
    const mat_int_t k,
    const T alpha,
    const T beta);

/* ************************************************************************** */

template<typename T>
__host__
void
T_H_axpy(
    dense_vector_t<T> * y,
    const dense_vector_t<T> * x,
    const T alpha);

/* ************************************************************************** */

template<typename T>
__host__
void
T_H_potrfs(
    col_major_matrix_t<T> * AL,
    dense_vector_t<T> * xb);

/* ************************************************************************** */

template<typename T>
__host__
void
T_H_potrf(
    col_major_matrix_t<T> * AL);

/* ************************************************************************** */

template<typename T>
__host__
void
T_H_potrs(
    const col_major_matrix_t<T> * L,
    dense_vector_t<T> * xb);

NS_LA_END
NS_CULIP_END

#endif /* __CULIP_LIBS_LA_CUBLAS_WRAPPER_H_ */
