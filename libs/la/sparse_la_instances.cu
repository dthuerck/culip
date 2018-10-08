/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/la/sparse_la.cuh>
#include <libs/la/sparse_la.impl.cuh>

NS_CULIP_BEGIN
NS_LA_BEGIN

template
__host__
void
T_symmetrize(
    const gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<float> * gpu_A,
    csr_matrix_ptr<float>& gpu_C);

template
__host__
void
T_symmetrize(
    const gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<double> * gpu_A,
    csr_matrix_ptr<double>& gpu_C);

template<>
__host__
void
T_symmetrize_compute(
    const gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<float> * gpu_A,
    const csr_matrix_t<float> * gpu_At,
    const dense_vector_t<mat_int_t> * C_csr_row,
    const mat_int_t C_Nnz,
    csr_matrix_ptr<float>& gpu_C)
{
    const mat_int_t m = gpu_A->m;

    gpu_handle->push_scalar_mode();
    gpu_handle->set_scalar_mode(false);

    gpu_C = make_csr_matrix_ptr<float>(m, m, C_Nnz, true);
    CHECK_CUDA(cudaMemcpy(gpu_C->csr_row, C_csr_row->dense_val,
        (m + 1) * sizeof(mat_int_t), cudaMemcpyDeviceToDevice));

    float one = 1.0;
    gpu_handle->cusparse_status =
        cusparseScsrgeam(
            gpu_handle->cusparse_handle,
            m,
            m,
            &one,
            gpu_A->get_description(),
            gpu_A->nnz,
            gpu_A->csr_val,
            gpu_A->csr_row,
            gpu_A->csr_col,
            &one,
            gpu_At->get_description(),
            gpu_At->nnz,
            gpu_At->csr_val,
            gpu_At->csr_row,
            gpu_At->csr_col,
            gpu_C->get_description(),
            gpu_C->csr_val,
            gpu_C->csr_row,
            gpu_C->csr_col);
    CHECK_CUSPARSE(gpu_handle);

    gpu_handle->pop_scalar_mode();
}

template<>
__host__
void
T_symmetrize_compute(
    const gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<double> * gpu_A,
    const csr_matrix_t<double> * gpu_At,
    const dense_vector_t<mat_int_t> * C_csr_row,
    const mat_int_t C_Nnz,
    csr_matrix_ptr<double>& gpu_C)
{
    const mat_int_t m = gpu_A->m;

    gpu_handle->push_scalar_mode();
    gpu_handle->set_scalar_mode(false);

    gpu_C = make_csr_matrix_ptr<double>(m, m, C_Nnz, true);
    CHECK_CUDA(cudaMemcpy(gpu_C->csr_row, C_csr_row->dense_val,
        (m + 1) * sizeof(mat_int_t), cudaMemcpyDeviceToDevice));

    double one = 1.0;
    gpu_handle->cusparse_status =
        cusparseDcsrgeam(
            gpu_handle->cusparse_handle,
            m,
            m,
            &one,
            gpu_A->get_description(),
            gpu_A->nnz,
            gpu_A->csr_val,
            gpu_A->csr_row,
            gpu_A->csr_col,
            &one,
            gpu_At->get_description(),
            gpu_At->nnz,
            gpu_At->csr_val,
            gpu_At->csr_row,
            gpu_At->csr_col,
            gpu_C->get_description(),
            gpu_C->csr_val,
            gpu_C->csr_row,
            gpu_C->csr_col);
    CHECK_CUSPARSE(gpu_handle);

    gpu_handle->pop_scalar_mode();
}

/* ************************************************************************** */

/**
 * CSRMV
 */
template<>
__host__
void
T_csrmv<float>(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const csr_matrix_t<float> * gpu_A,
    const dense_vector_t<float> * gpu_x,
    dense_vector_t<float> * gpu_y,
    const float * gpu_alpha,
    const float * gpu_beta)
{
    // if(transA == CUSPARSE_OPERATION_NON_TRANSPOSE)
    // {
    //     gpu_handle->cusparse_status =
    //         cusparseScsrmv_mp(
    //             gpu_handle->cusparse_handle,
    //             transA,
    //             gpu_A->m,
    //             gpu_A->n,
    //             gpu_A->nnz,
    //             gpu_alpha,
    //             gpu_A->get_description(),
    //             gpu_A->csr_val,
    //             gpu_A->csr_row,
    //             gpu_A->csr_col,
    //             gpu_x->dense_val,
    //             gpu_beta,
    //             gpu_y->dense_val);
    // }
    // else
    // {
        gpu_handle->cusparse_status =
            cusparseScsrmv(
                gpu_handle->cusparse_handle,
                transA,
                gpu_A->m,
                gpu_A->n,
                gpu_A->nnz,
                gpu_alpha,
                gpu_A->get_description(),
                gpu_A->csr_val,
                gpu_A->csr_row,
                gpu_A->csr_col,
                gpu_x->dense_val,
                gpu_beta,
                gpu_y->dense_val);
    // }
    CHECK_CUSPARSE(gpu_handle);
}

template<>
__host__
void
T_csrmv<double>(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const csr_matrix_t<double> * gpu_A,
    const dense_vector_t<double> * gpu_x,
    dense_vector_t<double> * gpu_y,
    const double * gpu_alpha,
    const double * gpu_beta)
{
    // if(transA == CUSPARSE_OPERATION_NON_TRANSPOSE)
    // {
    //     gpu_handle->cusparse_status =
    //         cusparseDcsrmv_mp(
    //             gpu_handle->cusparse_handle,
    //             transA,
    //             gpu_A->m,
    //             gpu_A->n,
    //             gpu_A->nnz,
    //             gpu_alpha,
    //             gpu_A->get_description(),
    //             gpu_A->csr_val,
    //             gpu_A->csr_row,
    //             gpu_A->csr_col,
    //             gpu_x->dense_val,
    //             gpu_beta,
    //             gpu_y->dense_val);
    // }
    // else
    // {
        gpu_handle->cusparse_status =
            cusparseDcsrmv(
                gpu_handle->cusparse_handle,
                transA,
                gpu_A->m,
                gpu_A->n,
                gpu_A->nnz,
                gpu_alpha,
                gpu_A->get_description(),
                gpu_A->csr_val,
                gpu_A->csr_row,
                gpu_A->csr_col,
                gpu_x->dense_val,
                gpu_beta,
                gpu_y->dense_val);
    // }
    CHECK_CUSPARSE(gpu_handle);
}

/* ************************************************************************** */

template<>
__host__
void
T_transpose_csr<float>(
    const gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<float> * gpu_A,
    csr_matrix_ptr<float>& gpu_At)
{
    gpu_At = make_csr_matrix_ptr<float>(gpu_A->n, gpu_A->m, gpu_A->nnz,
        true);
    gpu_handle->cusparse_status =
        cusparseScsr2csc(
            gpu_handle->cusparse_handle,
            gpu_A->m,
            gpu_A->n,
            gpu_A->nnz,
            gpu_A->csr_val,
            gpu_A->csr_row,
            gpu_A->csr_col,
            gpu_At->csr_val,
            gpu_At->csr_col,
            gpu_At->csr_row,
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO);
}

template<>
__host__
void
T_transpose_csr<double>(
    const gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<double> * gpu_A,
    csr_matrix_ptr<double>& gpu_At)
{
    gpu_At = make_csr_matrix_ptr<double>(gpu_A->n, gpu_A->m, gpu_A->nnz,
        true);
    gpu_handle->cusparse_status =
        cusparseDcsr2csc(
            gpu_handle->cusparse_handle,
            gpu_A->m,
            gpu_A->n,
            gpu_A->nnz,
            gpu_A->csr_val,
            gpu_A->csr_row,
            gpu_A->csr_col,
            gpu_At->csr_val,
            gpu_At->csr_col,
            gpu_At->csr_row,
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO);
}

/* ************************************************************************** */

template<>
__host__
void
T_triangular_solve(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const csr_matrix_t<float> * gpu_A,
    const dense_vector_t<float> * gpu_b,
    dense_vector_t<float> * gpu_x,
    const float * alpha)
{
    // /* create necessary data */
    // cusparseSolveAnalysisInfo_t info;
    // gpu_handle->cusparse_status =
    //     cusparseCreateSolveAnalysisInfo(&info);
    // CHECK_CUSPARSE(gpu_handle);

    // /* first phase: analyse */
    // gpu_handle->cusparse_status =
    //     cusparseScsrsv_analysis(
    //         gpu_handle->cusparse_handle,
    //         transA,
    //         gpu_A->m,
    //         gpu_A->nnz,
    //         gpu_A->get_description(),
    //         gpu_A->csr_val,
    //         gpu_A->csr_row,
    //         gpu_A->csr_col,
    //         info);
    // CHECK_CUSPARSE(gpu_handle);

    // /* second phase: (numerically) solve */
    // gpu_handle->cusparse_status =
    //     cusparseScsrsv_solve(
    //         gpu_handle->cusparse_handle,
    //         transA,
    //         gpu_A->m,
    //         alpha,
    //         gpu_A->get_description(),
    //         gpu_A->csr_val,
    //         gpu_A->csr_row,
    //         gpu_A->csr_col,
    //         info,
    //         gpu_b->dense_val,
    //         gpu_x->dense_val);
    // CHECK_CUSPARSE(gpu_handle);

    /* allocate buffer */
    csrsv2Info_t info;
    gpu_handle->cusparse_status =
        cusparseCreateCsrsv2Info(&info);

    mat_int_t buffer_size;
    gpu_handle->cusparse_status =
        cusparseScsrsv2_bufferSize(
            gpu_handle->cusparse_handle,
            transA,
            gpu_A->m,
            gpu_A->nnz,
            gpu_A->get_description(),
            gpu_A->csr_val,
            gpu_A->csr_row,
            gpu_A->csr_col,
            info,
            &buffer_size);

    void * buffer;
    cudaMalloc(&buffer, buffer_size);
    gpu_handle->cusparse_status =
        cusparseScsrsv2_analysis(
            gpu_handle->cusparse_handle,
            transA,
            gpu_A->m,
            gpu_A->nnz,
            gpu_A->get_description(),
            gpu_A->csr_val,
            gpu_A->csr_row,
            gpu_A->csr_col,
            info,
            CUSPARSE_SOLVE_POLICY_USE_LEVEL,
            &buffer_size);

    gpu_handle->cusparse_status =
        cusparseScsrsv2_solve(
            gpu_handle->cusparse_handle,
            transA,
            gpu_A->m,
            gpu_A->nnz,
            alpha,
            gpu_A->get_description(),
            gpu_A->csr_val,
            gpu_A->csr_row,
            gpu_A->csr_col,
            info,
            gpu_b->dense_val,
            gpu_x->dense_val,
            CUSPARSE_SOLVE_POLICY_USE_LEVEL,
            &buffer_size);

    cudaFree(buffer);
    gpu_handle->cusparse_status =
        cusparseDestroyCsrsv2Info(info);
}

template<>
__host__
void
T_triangular_solve(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const csr_matrix_t<double> * gpu_A,
    const dense_vector_t<double> * gpu_b,
    dense_vector_t<double> * gpu_x,
    const double * alpha)
{
    // /* create necessary data */
    // cusparseSolveAnalysisInfo_t info;
    // gpu_handle->cusparse_status =
    //     cusparseCreateSolveAnalysisInfo(&info);
    // CHECK_CUSPARSE(gpu_handle);

    // /* first phase: analyse */
    // gpu_handle->cusparse_status =
    //     cusparseDcsrsv_analysis(
    //         gpu_handle->cusparse_handle,
    //         transA,
    //         gpu_A->m,
    //         gpu_A->nnz,
    //         gpu_A->get_description(),
    //         gpu_A->csr_val,
    //         gpu_A->csr_row,
    //         gpu_A->csr_col,
    //         info);
    // CHECK_CUSPARSE(gpu_handle);

    // /* second phase: (numerically) solve */
    // gpu_handle->cusparse_status =
    //     cusparseDcsrsv_solve(
    //         gpu_handle->cusparse_handle,
    //         transA,
    //         gpu_A->m,
    //         alpha,
    //         gpu_A->get_description(),
    //         gpu_A->csr_val,
    //         gpu_A->csr_row,
    //         gpu_A->csr_col,
    //         info,
    //         gpu_b->dense_val,
    //         gpu_x->dense_val);
    // CHECK_CUSPARSE(gpu_handle);

    // cusparseDestroySolveAnalysisInfo(info);
    // CHECK_CUSPARSE(gpu_handle);

    /* allocate buffer */
    csrsv2Info_t info;
    gpu_handle->cusparse_status =
        cusparseCreateCsrsv2Info(&info);

    mat_int_t buffer_size;
    gpu_handle->cusparse_status =
        cusparseDcsrsv2_bufferSize(
            gpu_handle->cusparse_handle,
            transA,
            gpu_A->m,
            gpu_A->nnz,
            gpu_A->get_description(),
            gpu_A->csr_val,
            gpu_A->csr_row,
            gpu_A->csr_col,
            info,
            &buffer_size);

    void * buffer;
    cudaMalloc(&buffer, buffer_size);
    gpu_handle->cusparse_status =
        cusparseDcsrsv2_analysis(
            gpu_handle->cusparse_handle,
            transA,
            gpu_A->m,
            gpu_A->nnz,
            gpu_A->get_description(),
            gpu_A->csr_val,
            gpu_A->csr_row,
            gpu_A->csr_col,
            info,
            CUSPARSE_SOLVE_POLICY_USE_LEVEL,
            &buffer_size);

    gpu_handle->cusparse_status =
        cusparseDcsrsv2_solve(
            gpu_handle->cusparse_handle,
            transA,
            gpu_A->m,
            gpu_A->nnz,
            alpha,
            gpu_A->get_description(),
            gpu_A->csr_val,
            gpu_A->csr_row,
            gpu_A->csr_col,
            info,
            gpu_b->dense_val,
            gpu_x->dense_val,
            CUSPARSE_SOLVE_POLICY_USE_LEVEL,
            &buffer_size);

    cudaFree(buffer);
    gpu_handle->cusparse_status =
        cusparseDestroyCsrsv2Info(info);
}

/* ************************************************************************** */

template<>
__host__
void
T_triangular_step_analysis(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const csr_matrix_t<float> * gpu_A,
    cusparseSolveAnalysisInfo_t info)
{
    gpu_handle->cusparse_status =
        cusparseScsrsv_analysis(
            gpu_handle->cusparse_handle,
            transA,
            gpu_A->m,
            gpu_A->nnz,
            gpu_A->get_description(),
            gpu_A->csr_val,
            gpu_A->csr_row,
            gpu_A->csr_col,
            info);
    CHECK_CUSPARSE(gpu_handle);
}

template<>
__host__
void
T_triangular_step_solve(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    cusparseSolveAnalysisInfo_t info,
    const csr_matrix_t<float> * gpu_A,
    const dense_vector_t<float> * gpu_b,
    dense_vector_t<float> * gpu_x,
    const float * alpha)
{
    gpu_handle->cusparse_status =
        cusparseScsrsv_solve(
            gpu_handle->cusparse_handle,
            transA,
            gpu_A->m,
            alpha,
            gpu_A->get_description(),
            gpu_A->csr_val,
            gpu_A->csr_row,
            gpu_A->csr_col,
            info,
            gpu_b->dense_val,
            gpu_x->dense_val);
    CHECK_CUSPARSE(gpu_handle);
}

template<>
__host__
void
T_triangular_step_analysis(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const csr_matrix_t<double> * gpu_A,
    cusparseSolveAnalysisInfo_t info)
{
    gpu_handle->cusparse_status =
        cusparseDcsrsv_analysis(
            gpu_handle->cusparse_handle,
            transA,
            gpu_A->m,
            gpu_A->nnz,
            gpu_A->get_description(),
            gpu_A->csr_val,
            gpu_A->csr_row,
            gpu_A->csr_col,
            info);
    CHECK_CUSPARSE(gpu_handle);
}

template<>
__host__
void
T_triangular_step_solve(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    cusparseSolveAnalysisInfo_t info,
    const csr_matrix_t<double> * gpu_A,
    const dense_vector_t<double> * gpu_b,
    dense_vector_t<double> * gpu_x,
    const double * alpha)
{
    gpu_handle->cusparse_status =
        cusparseDcsrsv_solve(
            gpu_handle->cusparse_handle,
            transA,
            gpu_A->m,
            alpha,
            gpu_A->get_description(),
            gpu_A->csr_val,
            gpu_A->csr_row,
            gpu_A->csr_col,
            info,
            gpu_b->dense_val,
            gpu_x->dense_val);
    CHECK_CUSPARSE(gpu_handle);
}

/* ************************************************************************** */

template
__host__
void
T_approx_triangular_step_analysis(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const csr_matrix_t<float> * gpu_A,
    T_approx_triangular_info_t<float>& info);

template
__host__
void
T_approx_triangular_step_analysis(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const csr_matrix_t<double> * gpu_A,
    T_approx_triangular_info_t<double>& info);

template
__host__
void
T_approx_triangular_step_solve(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const T_approx_triangular_info_t<float>& info,
    const csr_matrix_t<float> * gpu_A,
    const dense_vector_t<float> * gpu_b,
    dense_vector_t<float> * gpu_x,
    const float * alpha,
    const mat_int_t sweeps);

template
__host__
void
T_approx_triangular_step_solve(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const T_approx_triangular_info_t<double>& info,
    const csr_matrix_t<double> * gpu_A,
    const dense_vector_t<double> * gpu_b,
    dense_vector_t<double> * gpu_x,
    const double * alpha,
    const mat_int_t sweeps);

/* ************************************************************************** */

template
__host__
void
T_SPNE(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<float> * gpu_A,
    const SPNE_MODE mode,
    csr_matrix_ptr<float>& gpu_C);

template
__host__
void
T_SPNE(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<double> * gpu_A,
    const SPNE_MODE mode,
    csr_matrix_ptr<double>& gpu_C);

template<>
__host__
void
T_SPNE_compute(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<float> * gpu_A,
    mat_int_t * gpu_C_csr_row,
    mat_int_t * gpu_C_csr_col,
    float * gpu_C_csr_val)
{
    cusparseMatDescr_t C_descr = gpu_A->get_description();
    cusparseScsrgemm(
        gpu_handle->cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE,
        gpu_A->m,
        gpu_A->m,
        gpu_A->n,
        gpu_A->get_description(),
        gpu_A->nnz,
        gpu_A->csr_val,
        gpu_A->csr_row,
        gpu_A->csr_col,
        gpu_A->get_description(),
        gpu_A->nnz,
        gpu_A->csr_val,
        gpu_A->csr_row,
        gpu_A->csr_col,
        C_descr,
        gpu_C_csr_val,
        gpu_C_csr_row,
        gpu_C_csr_col);
    CHECK_CUSPARSE(gpu_handle);
}

template<>
__host__
void
T_SPNE_compute(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<double> * gpu_A,
    mat_int_t * gpu_C_csr_row,
    mat_int_t * gpu_C_csr_col,
    double * gpu_C_csr_val)
{
    cusparseMatDescr_t C_descr = gpu_A->get_description();
    cusparseDcsrgemm(
        gpu_handle->cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE,
        gpu_A->m,
        gpu_A->m,
        gpu_A->n,
        gpu_A->get_description(),
        gpu_A->nnz,
        gpu_A->csr_val,
        gpu_A->csr_row,
        gpu_A->csr_col,
        gpu_A->get_description(),
        gpu_A->nnz,
        gpu_A->csr_val,
        gpu_A->csr_row,
        gpu_A->csr_col,
        C_descr,
        gpu_C_csr_val,
        gpu_C_csr_row,
        gpu_C_csr_col);
    CHECK_CUSPARSE(gpu_handle);
}

template
__host__
void
T_SPNE_extract(
    gpu_handle_ptr& gpu_handle,
    const mat_int_t m,
    const mat_int_t nnz,
    const mat_int_t * C_csr_row,
    const mat_int_t * C_csr_col,
    const float * C_csr_val,
    const SPNE_MODE mode,
    csr_matrix_ptr<float>& gpu_tri_C);

template
__host__
void
T_SPNE_extract(
    gpu_handle_ptr& gpu_handle,
    const mat_int_t m,
    const mat_int_t nnz,
    const mat_int_t * C_csr_row,
    const mat_int_t * C_csr_col,
    const double * C_csr_val,
    const SPNE_MODE mode,
    csr_matrix_ptr<double>& gpu_tri_C);

/* ************************************************************************** */

template
__host__
void
T_scale_AD(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<float> * gpu_A,
    const dense_vector_t<float> * gpu_d,
    csr_matrix_ptr<float>& gpu_AD);

template
__host__
void
T_scale_AD(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<double> * gpu_A,
    const dense_vector_t<double> * gpu_d,
    csr_matrix_ptr<double>& gpu_AD);

/* ************************************************************************** */

template
__host__
void
T_matrix_row_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<float> * gpu_A,
    dense_vector_t<float> * gpu_norm);

template
__host__
void
T_matrix_row_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<double> * gpu_A,
    dense_vector_t<double> * gpu_norm);

/* ************************************************************************** */

template
__host__
void
T_matrix_col_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<float> * gpu_A,
    dense_vector_t<float> * gpu_norm);

template
__host__
void
T_matrix_col_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<double> * gpu_A,
    dense_vector_t<double> * gpu_norm);

/* ************************************************************************** */

template
__host__
void
T_matrix_augmented_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<float> * gpu_A,
    const dense_vector_t<float> * gpu_d,
    dense_vector_t<float> * gpu_norm);

template
__host__
void
T_matrix_augmented_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<double> * gpu_A,
    const dense_vector_t<double> * gpu_d,
    dense_vector_t<double> * gpu_norm);

/* ************************************************************************** */

template
__host__
void
T_matrix_normal_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<float> * gpu_A,
    const dense_vector_t<float> * gpu_d,
    dense_vector_t<float> * gpu_norm);

template
__host__
void
T_matrix_normal_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<double> * gpu_A,
    const dense_vector_t<double> * gpu_d,
    dense_vector_t<double> * gpu_norm);

/* ************************************************************************** */

template
__host__
void
T_matrix_permute_scale(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<float> * gpu_A,
    const dense_vector_t<float> * gpu_row_norm,
    const dense_vector_t<float> * gpu_col_norm,
    const dense_vector_t<mat_int_t> * gpu_row_perm,
    const dense_vector_t<mat_int_t> * gpu_col_perm,
    csr_matrix_ptr<float>& gpu_PRACQ,
    const bool scale_before_match,
    const bool perm_is_old_to_new);

template
__host__
void
T_matrix_permute_scale(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<double> * gpu_A,
    const dense_vector_t<double> * gpu_row_norm,
    const dense_vector_t<double> * gpu_col_norm,
    const dense_vector_t<mat_int_t> * gpu_row_perm,
    const dense_vector_t<mat_int_t> * gpu_col_perm,
    csr_matrix_ptr<double>& gpu_PRACQ,
    const bool scale_before_match,
    const bool perm_is_old_to_new);

/* ************************************************************************** */

template
__host__
void
T_matrix_permute_row(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<float> * gpu_A,
    const dense_vector_t<mat_int_t> * old_to_new,
    csr_matrix_ptr<float>& gpu_pA);

template
__host__
void
T_matrix_permute_row(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<double> * gpu_A,
    const dense_vector_t<mat_int_t> * old_to_new,
    csr_matrix_ptr<double>& gpu_pA);

/* ************************************************************************** */

template
__host__
void
T_matrix_ruiz_scale(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<float> * gpu_A,
    csr_matrix_ptr<float>& gpu_scaled_A,
    dense_vector_ptr<float>& gpu_row_scale,
    dense_vector_ptr<float>& gpu_col_scale,
    bool symmetric);

template
__host__
void
T_matrix_ruiz_scale(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<double> * gpu_A,
    csr_matrix_ptr<double>& gpu_scaled_A,
    dense_vector_ptr<double>& gpu_row_scale,
    dense_vector_ptr<double>& gpu_col_scale,
    bool symmetric);

/* ************************************************************************** */

template
__host__
void
T_csr2coo(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<float> * gpu_csr_A,
    coo_matrix_ptr<float>& gpu_coo_A);

template
__host__
void
T_csr2coo(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<double> * gpu_csr_A,
    coo_matrix_ptr<double>& gpu_coo_A);

/* ************************************************************************** */

template
__host__
void
T_coo2csr(
    gpu_handle_ptr& gpu_handle,
    const coo_matrix_t<float> * gpu_coo_A,
    csr_matrix_ptr<float>& gpu_csr_A,
    const bool sort);

template
__host__
void
T_coo2csr(
    gpu_handle_ptr& gpu_handle,
    const coo_matrix_t<double> * gpu_coo_A,
    csr_matrix_ptr<double>& gpu_csr_A,
    const bool sort);

NS_LA_END
NS_CULIP_END
