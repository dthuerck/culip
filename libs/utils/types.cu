/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/utils/types.cuh>
#include <libs/utils/types.impl.cuh>

NS_CULIP_BEGIN

/**
 * *****************************************************************************
 * ************************* TEMPLATE INSTANTIATIONS ***************************
 * *****************************************************************************
 */

template class dense_vector_t<char>;
template class dense_vector_t<mat_size_t>;

template class dense_vector_t<mat_int_t>;
template class col_major_matrix_t<mat_int_t>;
template class csr_matrix_t<mat_int_t>;
template class coo_matrix_t<mat_int_t>;

template class dense_vector_t<float>;
template class col_major_matrix_t<float>;
template class csr_matrix_t<float>;
template class coo_matrix_t<float>;

template class dense_vector_t<double>;
template class col_major_matrix_t<double>;
template class csr_matrix_t<double>;
template class coo_matrix_t<double>;

/* ************************************************************************** */

template
col_major_matrix_ptr<mat_int_t>
make_col_major_matrix_ptr(
    const mat_size_t _m,
    const mat_size_t _n,
    const bool _on_device);

template
col_major_matrix_ptr<float>
make_col_major_matrix_ptr(
    const mat_size_t _m,
    const mat_size_t _n,
    const bool _on_device);

template
col_major_matrix_ptr<double>
make_col_major_matrix_ptr(
    const mat_size_t _m,
    const mat_size_t _n,
    const bool _on_device);

template
col_major_matrix_ptr<mat_int_t>
make_col_major_matrix_ptr(
    const bool _on_device);

template
col_major_matrix_ptr<float>
make_col_major_matrix_ptr(
    const bool _on_device);

template
col_major_matrix_ptr<double>
make_col_major_matrix_ptr(
    const bool _on_device);

template
col_major_matrix_ptr<mat_int_t>
make_col_major_matrix_ptr(
    const mat_size_t _m,
    const mat_size_t _n,
    mat_int_t * _dense_val,
    const bool _on_device);

template
col_major_matrix_ptr<float>
make_col_major_matrix_ptr(
    const mat_size_t _m,
    const mat_size_t _n,
    float * _dense_val,
    const bool _on_device);

template
col_major_matrix_ptr<double>
make_col_major_matrix_ptr(
    const mat_size_t _m,
    const mat_size_t _n,
    double * _dense_val,
    const bool _on_device);

/* ************************************************************************** */

template
csr_matrix_ptr<mat_int_t>
make_csr_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const bool on_device);

template
csr_matrix_ptr<float>
make_csr_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const bool on_device);

template
csr_matrix_ptr<double>
make_csr_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const bool on_device);

template
csr_matrix_ptr<mat_int_t>
make_csr_matrix_ptr(
    const bool on_device);

template
csr_matrix_ptr<float>
make_csr_matrix_ptr(
    const bool on_device);

template
csr_matrix_ptr<double>
make_csr_matrix_ptr(
    const bool on_device);

template
csr_matrix_ptr<mat_int_t>
make_csr_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const mat_int_t * csr_row,
    const mat_int_t * csr_col,
    const mat_int_t * csr_val,
    const bool on_device);

template
csr_matrix_ptr<float>
make_csr_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const mat_int_t * csr_row,
    const mat_int_t * csr_col,
    const float * csr_val,
    const bool on_device);

template
csr_matrix_ptr<double>
make_csr_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const mat_int_t * csr_row,
    const mat_int_t * csr_col,
    const double * csr_val,
    const bool on_device);

/* ************************************************************************** */

template
coo_matrix_ptr<float>
make_coo_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const bool on_device);

template
coo_matrix_ptr<double>
make_coo_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const bool on_device);

template
coo_matrix_ptr<float>
make_coo_matrix_ptr(
    const bool on_device);

template
coo_matrix_ptr<double>
make_coo_matrix_ptr(
    const bool on_device);

template
coo_matrix_ptr<float>
make_coo_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const mat_int_t * coo_row,
    const mat_int_t * coo_col,
    const float * coo_val,
    const bool on_device);

template
coo_matrix_ptr<double>
make_coo_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const mat_int_t * coo_row,
    const mat_int_t * coo_col,
    const double * coo_val,
    const bool on_device);

/* ************************************************************************** */

template
dense_vector_ptr<float>
make_raw_dense_vector_ptr();

template
dense_vector_ptr<double>
make_raw_dense_vector_ptr();

template
dense_vector_ptr<char>
make_raw_dense_vector_ptr();

template
dense_vector_ptr<mat_int_t>
make_raw_dense_vector_ptr();

template
dense_vector_ptr<mat_size_t>
make_raw_dense_vector_ptr();

template
dense_vector_ptr<float>
make_raw_dense_vector_ptr(
    const mat_size_t,
    const bool on_device,
    float * dense_val);

template
dense_vector_ptr<double>
make_raw_dense_vector_ptr(
    const mat_size_t,
    const bool on_device,
    double * dense_val);

template
dense_vector_ptr<char>
make_raw_dense_vector_ptr(
    const mat_size_t,
    const bool on_device,
    char * dense_val);

template
dense_vector_ptr<mat_int_t>
make_raw_dense_vector_ptr(
    const mat_size_t,
    const bool on_device,
    mat_int_t * dense_val);

template
dense_vector_ptr<mat_size_t>
make_raw_dense_vector_ptr(
    const mat_size_t,
    const bool on_device,
    mat_size_t * dense_val);

template
dense_vector_ptr<float>
make_managed_dense_vector_ptr(
    const mat_size_t m,
    const bool on_device);

template
dense_vector_ptr<double>
make_managed_dense_vector_ptr(
    const mat_size_t m,
    const bool on_device);

template
dense_vector_ptr<char>
make_managed_dense_vector_ptr(
    const mat_size_t m,
    const bool on_device);

template
dense_vector_ptr<mat_int_t>
make_managed_dense_vector_ptr(
    const mat_size_t m,
    const bool on_device);

template
dense_vector_ptr<mat_size_t>
make_managed_dense_vector_ptr(
    const mat_size_t m,
    const bool on_device);

template
dense_vector_ptr<float>
make_managed_dense_vector_ptr(
    const bool on_device);

template
dense_vector_ptr<double>
make_managed_dense_vector_ptr(
    const bool on_device);

template
dense_vector_ptr<char>
make_managed_dense_vector_ptr(
    const bool on_device);

template
dense_vector_ptr<mat_int_t>
make_managed_dense_vector_ptr(
    const bool on_device);

template
dense_vector_ptr<mat_size_t>
make_managed_dense_vector_ptr(
    const bool on_device);

/**
 * *****************************************************************************
 * ******************************* GPU_HANDLE_T ********************************
 * *****************************************************************************
 */

gpu_handle_t::
gpu_handle_t()
{
    cublasCreate_v2(&cublas_handle);
    cusparseCreate(&cusparse_handle);
    cusolverDnCreate(&cusolver_handle);

    cublas_status = CUBLAS_STATUS_SUCCESS;
    cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusolver_status = CUSOLVER_STATUS_SUCCESS;

    /* initialize to default stream */
    set_stream(cudaStreamDefault);

    /* initialize to host scalar mode */
    set_scalar_mode(false);
}

/* ************************************************************************** */

gpu_handle_t::
~gpu_handle_t()
{
    cublasDestroy_v2(cublas_handle);
    cusparseDestroy(cusparse_handle);
    cusolverDnDestroy(cusolver_handle);
}

/* ************************************************************************** */

void
gpu_handle_t::
set_stream(
    const cudaStream_t& stream)
{
    cublasSetStream_v2(cublas_handle, stream);
    cusparseSetStream(cusparse_handle, stream);
    cusolverDnSetStream(cusolver_handle, stream);

    m_stream = stream;
}

/* ************************************************************************** */

cudaStream_t&
gpu_handle_t::
get_stream()
{
    return m_stream;
}

/* ************************************************************************** */

bool
gpu_handle_t::
get_scalar_mode()
{
    cublasPointerMode_t cublas_mode;
    cublas_status = cublasGetPointerMode_v2(cublas_handle, &cublas_mode);

    cusparsePointerMode_t cusparse_mode;
    cusparse_status = cusparseGetPointerMode(cusparse_handle,
        &cusparse_mode);

    return ((cublas_mode == CUBLAS_POINTER_MODE_DEVICE) &&
        (cusparse_mode == CUSPARSE_POINTER_MODE_DEVICE));
}

/* ************************************************************************** */

void
gpu_handle_t::
set_scalar_mode(
    const bool scalar_device)
{
    cublas_status = cublasSetPointerMode_v2(cublas_handle,
        scalar_device ? CUBLAS_POINTER_MODE_DEVICE :
        CUBLAS_POINTER_MODE_HOST);
    cusparse_status = cusparseSetPointerMode(cusparse_handle,
        scalar_device ? CUSPARSE_POINTER_MODE_DEVICE :
        CUSPARSE_POINTER_MODE_HOST);
}

/* ************************************************************************** */

void
gpu_handle_t::
push_scalar_mode()
{
    m_modes.push(get_scalar_mode());
}

/* ************************************************************************** */

void
gpu_handle_t::
pop_scalar_mode()
{
    if(!m_modes.empty())
    {
        set_scalar_mode(m_modes.top());
        m_modes.pop();
    }
}

/* ************************************************************************** */

void
gpu_handle_t::
__status_check(const char* s,
                const int f,
                const char* fname,
                const size_t line)
{
    if (f) {
        std::cerr << s << " (error " << f << ") at" << fname << ":" <<
            line << ", exiting..." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/* ************************************************************************** */

void
gpu_handle_t::
__cublas_check(
    const char* fname,
    const size_t line)
{
    if(cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "cuBLAS error " << cublas_err_str(cublas_status)
            << " at" << fname << ":" << line << ", exiting..." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/* ************************************************************************** */

void
gpu_handle_t::
__cusparse_check(
    const char* fname,
    const size_t line)
{
    if(cusparse_status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "cuSPARSE error " << cusparse_err_str(cusparse_status)
            << " at" << fname << ":" << line << ", exiting..." << std::endl;
            std::exit(EXIT_FAILURE);
    }
}

/* ************************************************************************** */

void
gpu_handle_t::
__cusolver_check(
    const char* fname,
    const size_t line)
{
    if(cusolver_status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cout << "cuSOLVER error " << cusolver_err_str(cusolver_status)
            << " at" << fname << ":" << line << ", exiting..." << std::endl;
            std::exit(EXIT_FAILURE);
    }
}

/* ************************************************************************** */

/**
    * Utility functions for error retrieval & evaluation.
    */
const char *
gpu_handle_t::
cublas_err_str(
    cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "CUBLAS_UNKNOWN";
    }
}

/* ************************************************************************** */

const char *
gpu_handle_t::
cusparse_err_str(
    cusparseStatus_t status)
{
    switch(status)
    {
        case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        default:
            return "CUSPARSE_UNKNOWN";
    }
}

/* ************************************************************************** */

const char *
gpu_handle_t::
cusolver_err_str(
    cusolverStatus_t status)
{
    switch(status)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        default:
            return "CUSOLVER_UNKNOWN";
    }
}

NS_CULIP_END
