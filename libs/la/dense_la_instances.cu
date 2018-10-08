/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/la/dense_la.cuh>

extern "C" {
    #include <cblas.h>
}
#include <lapacke.h>

NS_CULIP_BEGIN
NS_LA_BEGIN

/**
 * NRM2
 */
template<>
__host__
void
T_nrm2<float>(
    const gpu_handle_ptr& gpu_handle,
    const dense_vector_t<float> * gpu_x,
    float * result)
{
    gpu_handle->cublas_status =
        cublasSnrm2_v2(
            gpu_handle->cublas_handle,
            gpu_x->m,
            gpu_x->dense_val,
            1,
            result);
}

template<>
__host__
void
T_nrm2<double>(
    const gpu_handle_ptr& gpu_handle,
    const dense_vector_t<double> * gpu_x,
    double * result)
{
    gpu_handle->cublas_status =
        cublasDnrm2_v2(
            gpu_handle->cublas_handle,
            gpu_x->m,
            gpu_x->dense_val,
            1,
            result);
}

/* ************************************************************************** */

/**
 * DOTI
 */
template<>
__host__
void
T_doti<float>(
    const gpu_handle_ptr& gpu_handle,
    const dense_vector_t<float> * gpu_x,
    const dense_vector_t<float> * gpu_y,
    float * result)
{
    gpu_handle->cublas_status =
        cublasSdot_v2(
            gpu_handle->cublas_handle,
            gpu_x->m,
            gpu_x->dense_val,
            1,
            gpu_y->dense_val,
            1,
            result);
}

template<>
__host__
void
T_doti<double>(
    const gpu_handle_ptr& gpu_handle,
    const dense_vector_t<double> * gpu_x,
    const dense_vector_t<double> * gpu_y,
    double * result)
{
    gpu_handle->cublas_status =
        cublasDdot_v2(
            gpu_handle->cublas_handle,
            gpu_x->m,
            gpu_x->dense_val,
            1,
            gpu_y->dense_val,
            1,
            result);
}

/* ************************************************************************** */

/**
 * SCAL
 */
template<>
__host__
void
T_scal<float>(
    const gpu_handle_ptr& gpu_handle,
    dense_vector_t<float> * gpu_y,
    const float * alpha)
{
    gpu_handle->cublas_status =
        cublasSscal_v2(
            gpu_handle->cublas_handle,
            gpu_y->m,
            alpha,
            gpu_y->dense_val,
            1);
}

template<>
__host__
void
T_scal<double>(
    const gpu_handle_ptr& gpu_handle,
    dense_vector_t<double> * gpu_y,
    const double * alpha)
{
    gpu_handle->cublas_status =
        cublasDscal_v2(
            gpu_handle->cublas_handle,
            gpu_y->m,
            alpha,
            gpu_y->dense_val,
            1);
}

/* ************************************************************************** */

/**
 * AXPY
 */
template<>
__host__
void
T_axpy<float>(
    const gpu_handle_ptr& gpu_handle,
    dense_vector_t<float> * gpu_y,
    const dense_vector_t<float> * gpu_x,
    const float * alpha)
{
    gpu_handle->cublas_status =
        cublasSaxpy_v2(
            gpu_handle->cublas_handle,
            gpu_y->m,
            alpha,
            gpu_x->dense_val,
            1,
            gpu_y->dense_val,
            1);
}

template<>
__host__
void
T_axpy<double>(
    const gpu_handle_ptr& gpu_handle,
    dense_vector_t<double> * gpu_y,
    const dense_vector_t<double> * gpu_x,
    const double * alpha)
{
    gpu_handle->cublas_status =
        cublasDaxpy_v2(
            gpu_handle->cublas_handle,
            gpu_y->m,
            alpha,
            gpu_x->dense_val,
            1,
            gpu_y->dense_val,
            1);
}

/* ************************************************************************** */

/**
 * SBMV
 */
template<>
__host__
void
T_sbmv<float>(
    const gpu_handle_ptr& gpu_handle,
    const dense_vector_t<float> * gpu_A,
    const dense_vector_t<float> * gpu_x,
    dense_vector_t<float> * gpu_y,
    const float * alpha,
    const float * beta,
    const mat_int_t& k,
    const mat_int_t& n)
{
    gpu_handle->cublas_status =
        cublasSsbmv_v2(
            gpu_handle->cublas_handle,
            CUBLAS_FILL_MODE_LOWER,
            n,
            k - 1,
            alpha,
            gpu_A->dense_val,
            1,
            gpu_x->dense_val,
            1,
            beta,
            gpu_y->dense_val,
            1);
}

template<>
__host__
void
T_sbmv<double>(
    const gpu_handle_ptr& gpu_handle,
    const dense_vector_t<double> * gpu_A,
    const dense_vector_t<double> * gpu_x,
    dense_vector_t<double> * gpu_y,
    const double * alpha,
    const double * beta,
    const mat_int_t& k,
    const mat_int_t& n)
{
    gpu_handle->cublas_status =
        cublasDsbmv_v2(
            gpu_handle->cublas_handle,
            CUBLAS_FILL_MODE_LOWER,
            n,
            k - 1,
            alpha,
            gpu_A->dense_val,
            1,
            gpu_x->dense_val,
            1,
            beta,
            gpu_y->dense_val,
            1);
}

/* ************************************************************************** */

/**
 * ASUM
 */
template<>
__host__
void
T_asum<float>(
    const gpu_handle_ptr& gpu_handle,
    const dense_vector_t<float> * gpu_x,
    float * res)
{
    gpu_handle->cublas_status =
        cublasSasum(
            gpu_handle->cublas_handle,
            gpu_x->m,
            gpu_x->dense_val,
            1,
            res);
}

template<>
__host__
void
T_asum<double>(
    const gpu_handle_ptr& gpu_handle,
    const dense_vector_t<double> * gpu_x,
    double * res)
{
    gpu_handle->cublas_status =
        cublasDasum(
            gpu_handle->cublas_handle,
            gpu_x->m,
            gpu_x->dense_val,
            1,
            res);
}

/* ************************************************************************** */

/**
 * GEMV
 */

template<>
__host__
void
T_gemv(
    const gpu_handle_ptr& gpu_handle,
    cublasOperation_t transA,
    const col_major_matrix_t<float> * gpu_A,
    const dense_vector_t<float> * gpu_x,
    dense_vector_t<float> * gpu_y,
    const mat_int_t k,
    const float * alpha,
    const float * beta)
{
    gpu_handle->cublas_status =
        cublasSgemv(
            gpu_handle->cublas_handle,
            transA,
            gpu_A->m,
            k,
            alpha,
            gpu_A->dense_val,
            gpu_A->m,
            gpu_x->dense_val,
            1,
            beta,
            gpu_y->dense_val,
            1);
}

template<>
__host__
void
T_gemv(
    const gpu_handle_ptr& gpu_handle,
    cublasOperation_t transA,
    const col_major_matrix_t<double> * gpu_A,
    const dense_vector_t<double> * gpu_x,
    dense_vector_t<double> * gpu_y,
    const mat_int_t k,
    const double * alpha,
    const double * beta)
{
    gpu_handle->cublas_status =
        cublasDgemv(
            gpu_handle->cublas_handle,
            transA,
            gpu_A->m,
            k,
            alpha,
            gpu_A->dense_val,
            gpu_A->m,
            gpu_x->dense_val,
            1,
            beta,
            gpu_y->dense_val,
            1);
}

/* ************************************************************************** */

/**
 * GEMM
 */

template<>
__host__
void
T_gemm(
    const gpu_handle_ptr& gpu_handle,
    cublasOperation_t transA,
    cublasOperation_t transB,
    const col_major_matrix_t<float> * gpu_A,
    const col_major_matrix_t<float> * gpu_B,
    col_major_matrix_t<float> * gpu_C,
    const mat_int_t k,
    const float * alpha,
    const float * beta)
{
    gpu_handle->cublas_status =
        cublasSgemm(
            gpu_handle->cublas_handle,
            transA,
            transB,
            gpu_A->m,
            gpu_B->n,
            k,
            alpha,
            gpu_A->dense_val,
            gpu_A->m,
            gpu_B->dense_val,
            gpu_B->m,
            beta,
            gpu_C->dense_val,
            gpu_C->m);
}

template<>
__host__
void
T_gemm(
    const gpu_handle_ptr& gpu_handle,
    cublasOperation_t transA,
    cublasOperation_t transB,
    const col_major_matrix_t<double> * gpu_A,
    const col_major_matrix_t<double> * gpu_B,
    col_major_matrix_t<double> * gpu_C,
    const mat_int_t k,
    const double * alpha,
    const double * beta)
{
    gpu_handle->cublas_status =
        cublasDgemm(
            gpu_handle->cublas_handle,
            transA,
            transB,
            gpu_A->m,
            gpu_B->n,
            k,
            alpha,
            gpu_A->dense_val,
            gpu_A->m,
            gpu_B->dense_val,
            gpu_B->m,
            beta,
            gpu_C->dense_val,
            gpu_C->m);
}

/* ************************************************************************** */

/**
 * POTRF / POTRS
 */

template<>
__host__
void
T_potrfs(
    const gpu_handle_ptr& gpu_handle,
    const col_major_matrix_t<float> * gpu_A,
    dense_vector_t<float> * gpu_xb)
{
    /* 1st: compute size of necessary buffers and allocate buffer */
    int Lwork;
    gpu_handle->cusolver_status =
        cusolverDnSpotrf_bufferSize(
            gpu_handle->cusolver_handle,
            CUBLAS_FILL_MODE_LOWER,
            gpu_A->m,
            gpu_A->dense_val,
            gpu_A->m,
            &Lwork);
    CHECK_CUSOLVER(gpu_handle);

    dense_vector_ptr<float> workspace = make_managed_dense_vector_ptr<float>(
        Lwork, true);

    /* 2nd: factorize A using a dense Cholesky factorization */
    int devinfo;
    gpu_handle->cusolver_status =
        cusolverDnSpotrf(
            gpu_handle->cusolver_handle,
            CUBLAS_FILL_MODE_LOWER,
            gpu_A->m,
            gpu_A->dense_val,
            gpu_A->m,
            workspace->dense_val,
            Lwork,
            &devinfo);
    CHECK_CUSOLVER(gpu_handle);

    if(devinfo < 0)
        printf("cuSOLVER rf: %d-th parameter is wrong.\n", -devinfo);
    if(devinfo > 0)
        printf("cuSOLVER rf: %d-th minor is not positive definite.\n", devinfo);

    /* 3rd: solver A x = b using the previously computed factorization */
    gpu_handle->cusolver_status =
        cusolverDnSpotrs(
            gpu_handle->cusolver_handle,
            CUBLAS_FILL_MODE_LOWER,
            gpu_A->m,
            1,
            gpu_A->dense_val,
            gpu_A->m,
            gpu_xb->dense_val,
            gpu_A->m,
            &devinfo);

    if(devinfo < 0)
        printf("cuSOLVER rs: %d-th parameter is wrong.\n", -devinfo);
}

/* ************************************************************************** */

template<>
__host__
void
T_trsm(
    const gpu_handle_ptr& gpu_handle,
    cublasOperation_t transA,
    cublasFillMode_t modeA,
    const col_major_matrix_t<float> * gpu_A,
    const dense_vector_t<float> * b,
    dense_vector_t<float> * x,
    const mat_int_t k,
    const float * alpha)
{
    cudaMemcpyAsync(x->dense_val, b->dense_val, k * sizeof(float),
        cudaMemcpyDeviceToDevice, gpu_handle->get_stream());
    gpu_handle->cublas_status =
        cublasStrsm(
            gpu_handle->cublas_handle,
            CUBLAS_SIDE_LEFT,
            modeA,
            transA,
            CUBLAS_DIAG_NON_UNIT,
            k,
            1,
            alpha,
            gpu_A->dense_val,
            gpu_A->m,
            x->dense_val,
            x->m);
    CHECK_CUSPARSE(gpu_handle);
}

template<>
__host__
void
T_trsm(
    const gpu_handle_ptr& gpu_handle,
    cublasOperation_t transA,
    cublasFillMode_t modeA,
    const col_major_matrix_t<double> * gpu_A,
    const dense_vector_t<double> * b,
    dense_vector_t<double> * x,
    const mat_int_t k,
    const double * alpha)
{
    cudaMemcpyAsync(x->dense_val, b->dense_val, k * sizeof(double),
        cudaMemcpyDeviceToDevice, gpu_handle->get_stream());
    gpu_handle->cublas_status =
        cublasDtrsm(
            gpu_handle->cublas_handle,
            CUBLAS_SIDE_LEFT,
            modeA,
            transA,
            CUBLAS_DIAG_NON_UNIT,
            k,
            1,
            alpha,
            gpu_A->dense_val,
            gpu_A->m,
            x->dense_val,
            x->m);
    CHECK_CUSPARSE(gpu_handle);
}

/* ************************************************************************** */

template<>
__host__
void
T_H_gemv(
    const bool transA,
    const col_major_matrix_t<float> * A,
    const dense_vector_t<float> * x,
    dense_vector_t<float> * y,
    const mat_int_t k,
    const float alpha,
    const float beta)
{
    cblas_sgemv(
        CblasColMajor,
        transA ? CblasTrans : CblasNoTrans,
        A->m,
        k,
        alpha,
        A->dense_val,
        A->m,
        x->dense_val,
        1,
        beta,
        y->dense_val,
        1);
}

template<>
__host__
void
T_H_gemv(
    const bool transA,
    const col_major_matrix_t<double> * A,
    const dense_vector_t<double> * x,
    dense_vector_t<double> * y,
    const mat_int_t k,
    const double alpha,
    const double beta)
{
    cblas_dgemv(
        CblasColMajor,
        transA ? CblasTrans : CblasNoTrans,
        A->m,
        k,
        alpha,
        A->dense_val,
        A->m,
        x->dense_val,
        1,
        beta,
        y->dense_val,
        1);
}

/* ************************************************************************** */

template<>
__host__
void
T_H_gemm(
    const bool transA,
    const bool transB,
    const col_major_matrix_t<float> * A,
    const col_major_matrix_t<float> * B,
    col_major_matrix_t<float> * C,
    const mat_int_t k,
    const float alpha,
    const float beta)
{
    cblas_sgemm(
        CblasColMajor,
        transA ? CblasTrans : CblasNoTrans,
        transB ? CblasTrans : CblasNoTrans,
        transA ? A->n : A->m,
        transB ? B->m : B->n,
        k,
        alpha,
        A->dense_val,
        A->m,
        B->dense_val,
        B->m,
        beta,
        C->dense_val,
        C->m);
}

template<>
__host__
void
T_H_gemm(
    const bool transA,
    const bool transB,
    const col_major_matrix_t<double> * A,
    const col_major_matrix_t<double> * B,
    col_major_matrix_t<double> * C,
    const mat_int_t k,
    const double alpha,
    const double beta)
{
    cblas_dgemm(
        CblasColMajor,
        transA ? CblasTrans : CblasNoTrans,
        transB ? CblasTrans : CblasNoTrans,
        transA ? A->n : A->m,
        transB ? B->m : B->n,
        k,
        alpha,
        A->dense_val,
        A->m,
        B->dense_val,
        B->m,
        beta,
        C->dense_val,
        C->m);
}

/* ************************************************************************** */

template<>
__host__
void
T_H_axpy(
    dense_vector_t<float> * y,
    const dense_vector_t<float> * x,
    const float alpha)
{
    cblas_saxpy(
        y->m,
        alpha,
        x->dense_val,
        1,
        y->dense_val,
        1);
}

template<>
__host__
void
T_H_axpy(
    dense_vector_t<double> * y,
    const dense_vector_t<double> * x,
    const double alpha)
{
    cblas_daxpy(
        y->m,
        alpha,
        x->dense_val,
        1,
        y->dense_val,
        1);
}

/* ************************************************************************** */

template<>
__host__
void
T_H_potrfs(
    col_major_matrix_t<float> * A,
    dense_vector_t<float> * xb)
{
    const mat_int_t err = LAPACKE_sposv(
        LAPACK_COL_MAJOR,
        'L',
        A->n,
        1,
        A->dense_val,
        A->n,
        xb->dense_val,
        xb->m);

    if(err != 0)
        printf("LAPACK error %d in %s:%d...\n", err, __FILE__, __LINE__);
}

template<>
__host__
void
T_H_potrfs(
    col_major_matrix_t<double> * A,
    dense_vector_t<double> * xb)
{
    const mat_int_t err = LAPACKE_dposv(
        LAPACK_COL_MAJOR,
        'L',
        A->n,
        1,
        A->dense_val,
        A->n,
        xb->dense_val,
        xb->m);

    if(err != 0)
        printf("LAPACK error %d in %s:%d...\n", err, __FILE__, __LINE__);
}

/* ************************************************************************** */

template<>
__host__
void
T_H_potrf(
    col_major_matrix_t<float> * AL)
{
    const mat_int_t err = LAPACKE_spotrf(
        LAPACK_COL_MAJOR,
        'L',
        AL->n,
        AL->dense_val,
        AL->n);

    if(err != 0)
        printf("LAPACK error %d in %s:%d...\n", err, __FILE__, __LINE__);
}

template<>
__host__
void
T_H_potrf(
    col_major_matrix_t<double> * AL)
{
    const mat_int_t err = LAPACKE_dpotrf(
        LAPACK_COL_MAJOR,
        'L',
        AL->n,
        AL->dense_val,
        AL->n);

    if(err != 0)
        printf("LAPACK error %d in %s:%d...\n", err, __FILE__, __LINE__);
}

/* ************************************************************************** */

template<>
__host__
void
T_H_potrs(
    const col_major_matrix_t<float> * L,
    dense_vector_t<float> * xb)
{
    const mat_int_t err = LAPACKE_spotrs(
        LAPACK_COL_MAJOR,
        'L',
        L->n,
        1,
        L->dense_val,
        L->n,
        xb->dense_val,
        xb->m);

    if(err != 0)
        printf("LAPACK error %d in %s:%d...\n", err, __FILE__, __LINE__);
}

template<>
__host__
void
T_H_potrs(
    const col_major_matrix_t<double> * L,
    dense_vector_t<double> * xb)
{
    const mat_int_t err = LAPACKE_dpotrs(
        LAPACK_COL_MAJOR,
        'L',
        L->n,
        1,
        L->dense_val,
        L->n,
        xb->dense_val,
        xb->m);

    if(err != 0)
        printf("LAPACK error %d in %s:%d...\n", err, __FILE__, __LINE__);
}

NS_LA_END
NS_CULIP_END
