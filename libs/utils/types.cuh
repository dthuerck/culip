/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIBS_UTILS_GPU_DEFINES_H_
#define __CULIP_LIBS_UTILS_GPU_DEFINES_H_

#include <libs/utils/defines.h>
#include <libs/utils/mem_pool.cuh>

#include <memory>
#include <iostream>
#include <cstdio>
#include <stack>
#include <signal.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <device_launch_parameters.h>

#include <thrust/device_ptr.h>

/* turns CUDA errors into segfaults */
#define CHECK_CUDA(err) ({cudaError_t v = err; if(v != cudaSuccess) { printf("CUDA error in %s:%d: %s\n", __FILE__, __LINE__, _cudaGetErrorEnum(v)); raise(SIGSEGV); }})
//#define CHECK_CUDA(err) (checkCudaErrors(err))

#define CHECK_CUBLAS(hndl_ptr) (hndl_ptr->__cublas_check(__FILE__, __LINE__))
#define CHECK_CUSPARSE(hndl_ptr) (hndl_ptr->__cusparse_check(__FILE__, __LINE__))
#define CHECK_CUSOLVER(hndl_ptr) (hndl_ptr->__cusolver_check(__FILE__, __LINE__))
#define CHECK_STATUS(hndl_ptr,s,f) (hndl_ptr->__status_check(s,f,__FILE__, __LINE__))
#define DIV_UP(a, b) (((a) / (b) + ((a) % (b) == 0 ? 0 : 1)))

NS_CULIP_BEGIN

enum original_var_type_t
{
    O_NONNEGATIVE,
    O_NONPOSITIVE,
    O_FREE_POS_PART,
    O_FREE_NEG_PART,
    O_SLACK
};


/**
 * *****************************************************************************
 * ****************************** DENSE_VECTOR_T *******************************
 * *****************************************************************************
 */

/**
 * Abstract interface for a dense vector: just the raw pointer, location info
 * and metadata.
 */
template<typename T>
class dense_vector_t
{
public:
    mat_size_t m;
    bool on_device;

    T * dense_val;

public:
    virtual ~dense_vector_t();

    /* self-information */
    virtual bool is_managed() const = 0;

    /* assignment */
    virtual void operator=(const dense_vector_t<T> * vec) = 0;

    /* access */
    T& operator[](const mat_int_t i);
    const T& operator[] (const mat_int_t i) const;

    /* debugging */
    void print(const char * s) const;

    thrust::device_ptr<T> dense_val_ptr() const;

protected:
    dense_vector_t(const mat_size_t _m, const bool _on_device,
        T * _dense_val);
};

template<typename T>
using dense_vector_ptr = std::unique_ptr<dense_vector_t<T>>;

/* ************************************************************************** */

template<typename T>
class raw_dense_vector_t final : public dense_vector_t<T>
{
public:
    raw_dense_vector_t();
    raw_dense_vector_t(
        const mat_size_t _m,
        const bool _on_device,
        T * _dense_val);
    ~raw_dense_vector_t();

    /* parent overrides */
    virtual bool is_managed() const;
    virtual void operator=(const dense_vector_t<T> * vec);
};

/* ************************************************************************** */

template<typename T>
class managed_dense_vector_t final : public dense_vector_t<T>
{
public:
    managed_dense_vector_t(
        const mat_size_t _m,
        const bool _on_device);
    managed_dense_vector_t(
        const bool _on_device);
    ~managed_dense_vector_t();

    /* parent overrides */
    virtual bool is_managed() const;
    virtual void operator=(const dense_vector_t<T> * vec);

private:
    void _alloc(const mat_size_t _m, bool _on_device);
    void _free();
};

/* ************************************************************************** */

/* constructor for empty raw pointers */
template<typename T>
dense_vector_ptr<T>
make_raw_dense_vector_ptr();

/* constructor for raw pointer with content */
template<typename T>
dense_vector_ptr<T>
make_raw_dense_vector_ptr(
    const mat_size_t,
    const bool on_device,
    T * dense_val);

/* constructor for managed allocation */
template<typename T>
dense_vector_ptr<T>
make_managed_dense_vector_ptr(
    const mat_size_t m,
    const bool on_device);

/* constructor for empty managed allocation */
template<typename T>
dense_vector_ptr<T>
make_managed_dense_vector_ptr(
    const bool on_device);


/**
 * *****************************************************************************
 * **************************** COL_MAJOR_MATRIX_T *****************************
 * *****************************************************************************
 */

/**
 * Dense matrix stored in column - major format (for e.g. solving multiple
 * right - hand sides with a matrix).
 */

template<typename T>
struct col_major_matrix_t
{
    mat_size_t m;
    mat_size_t n;
    bool on_device;

    T * dense_val; /* col i starts at i * n */

    /* constructor for pool-managed allocation */
    col_major_matrix_t(
        const mat_size_t _m,
        const mat_size_t _n,
        const bool _on_device);

    /* constructor for non allocated pool-managed memory */
    col_major_matrix_t(
        const bool _on_device);

    /* constructor for manually allocated memory */
    col_major_matrix_t(
        const mat_size_t _m,
        const mat_size_t _n,
        T * _dense_val,
        const bool _on_device);

    ~col_major_matrix_t();

    void operator=(const col_major_matrix_t<T> * mat);
    void print(const char * s) const;

    /* get access to single columns or elements */
    dense_vector_ptr<T> col(const mat_size_t j);
    T * elem(const mat_size_t i, const mat_size_t j);

    T& operator[](const mat_size_t i);
    const T& operator[] (const mat_size_t i) const;

    thrust::device_ptr<T> dense_val_ptr() const;

private:
    void _alloc(const mat_size_t _m, const mat_size_t _n);
    void _free();
    bool _managed;
};

template<typename T>
using col_major_matrix_ptr = std::unique_ptr<col_major_matrix_t<T>>;

/* constructor for pool-managed allocation */
template<typename T>
col_major_matrix_ptr<T>
make_col_major_matrix_ptr(
    const mat_size_t _m,
    const mat_size_t _n,
    const bool _on_device);

/* constructor for non allocated pool-managed memory */
template<typename T>
col_major_matrix_ptr<T>
make_col_major_matrix_ptr(
    const bool _on_device);

/* constructor for manually allocated memory */
template<typename T>
col_major_matrix_ptr<T>
make_col_major_matrix_ptr(
    const mat_size_t _m,
    const mat_size_t _n,
    T * _dense_val,
    const bool _on_device);

/**
 * *****************************************************************************
 * ******************************* CSR_MATRIX_T ********************************
 * *****************************************************************************
 */

/**
 * Sparse matrix in compressed row storage format (CSR) with
 * cuSPARSE-compatible types.
 *
 * Supports only real inputs; uses an 0-based index.
 */
template<typename T>
struct csr_matrix_t
{
    mat_size_t m;
    mat_size_t n;
    mat_size_t nnz;
    bool on_device;

    mat_int_t * csr_row; /* row length table */
    mat_int_t * csr_col; /* col index table */
    T * csr_val; /* value table */

    /* constructor for pool-managed allocation */
    csr_matrix_t(
        const mat_size_t _m,
        const mat_size_t _n,
        const mat_size_t _nnz,
        const bool _on_device);

    /* constructor for non allocated pool-managed memory */
    csr_matrix_t(
        const bool _on_device);

    /* constructor for manually allocated memory */
    csr_matrix_t(
        const mat_size_t _m,
        const mat_size_t _n,
        const mat_size_t _nnz,
        const mat_int_t * _csr_row,
        const mat_int_t * _csr_col,
        const T * _csr_val,
        const bool _on_device);

    ~csr_matrix_t();

    void operator=(const csr_matrix_t<T> * mat);
    void print(const char *s, const bool print_dense = false) const;

    thrust::device_ptr<mat_int_t> csr_row_ptr() const;
    thrust::device_ptr<mat_int_t> csr_col_ptr() const;
    thrust::device_ptr<T> csr_val_ptr() const;

    cusparseMatDescr_t get_description() const;

private:
    void _alloc(const mat_size_t _m, const mat_size_t _nnz);
    void _free();
    bool _managed;
};

template<typename T>
using csr_matrix_ptr = std::unique_ptr<csr_matrix_t<T>>;

/* constructor for pool-managed allocation */
template<typename T>
csr_matrix_ptr<T>
make_csr_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const bool on_device);

/* constructor for non allocated pool-managed memory */
template<typename T>
csr_matrix_ptr<T>
make_csr_matrix_ptr(
    const bool on_device);

/* constructor for manually allocated memory */
template<typename T>
csr_matrix_ptr<T>
make_csr_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const mat_int_t * csr_row,
    const mat_int_t * csr_col,
    const T * csr_val,
    const bool on_device);

/**
 * *****************************************************************************
 * ******************************* COO_MATRIX_T ********************************
 * *****************************************************************************
 */

/**
 * Sparse matrix in coordinate storage format (COO) with
 * SPRAL-compatible types (in case of double!).
 *
 * Supports only real inputs; uses an 0-based index.
 */
template<typename T>
struct coo_matrix_t
{
    mat_size_t m;
    mat_size_t n;
    mat_size_t nnz;
    bool on_device;

    mat_int_t * coo_row; /* row index table */
    mat_int_t * coo_col; /* col index table */
    T * coo_val; /* value table */

    /* constructor for pool-managed allocation */
    coo_matrix_t(
        const mat_size_t _m,
        const mat_size_t _n,
        const mat_size_t _nnz,
        const bool _on_device);

    /* constructor for non allocated pool-managed memory */
    coo_matrix_t(
        const bool _on_device);

    /* constructor for manually allocated memory */
    coo_matrix_t(
        const mat_size_t _m,
        const mat_size_t _n,
        const mat_size_t _nnz,
        const mat_int_t * coo_row,
        const mat_int_t * coo_col,
        const T * coo_val,
        const bool on_device);

    ~coo_matrix_t();

    thrust::device_ptr<mat_int_t> coo_row_ptr() const;
    thrust::device_ptr<mat_int_t> coo_col_ptr() const;
    thrust::device_ptr<T> coo_val_ptr() const;

    void operator=(const coo_matrix_t<T> * mat);
    void print(const char *s, const bool print_dense = false) const;

private:
    void _alloc(const mat_size_t _nnz);
    void _free();
    bool _managed;
};

template<typename T>
using coo_matrix_ptr = std::unique_ptr<coo_matrix_t<T>>;

/* constructor for pool-managed allocation */
template<typename T>
coo_matrix_ptr<T>
make_coo_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const bool on_device);

/* constructor for non allocated pool-managed memory */
template<typename T>
coo_matrix_ptr<T>
make_coo_matrix_ptr(
    const bool on_device);

/* constructor for manually allocated memory */
template<typename T>
coo_matrix_ptr<T>
make_coo_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const mat_int_t * coo_row,
    const mat_int_t * coo_col,
    const T * coo_val,
    const bool on_device);

/**
 * *****************************************************************************
 * ******************************* GPU_HANDLE_T ********************************
 * *****************************************************************************
 */

/**
 * A wrapper around cublas and cusparse's handle types.
 */
struct gpu_handle_t
{
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    cusolverDnHandle_t cusolver_handle;

    cublasStatus_t cublas_status;
    cusparseStatus_t cusparse_status;
    cusolverStatus_t cusolver_status;

    /* saving and switching modes */
    std::stack<bool> m_modes;

    gpu_handle_t();
    ~gpu_handle_t();

    void set_stream(const cudaStream_t& stream);
    cudaStream_t& get_stream();

    bool get_scalar_mode();
    void set_scalar_mode(const bool scalar_device);
    void push_scalar_mode();
    void pop_scalar_mode();

    void __status_check(const char* s, const int f,
        const char* fname, const size_t line);
    void __cublas_check(const char* fname, const size_t line);
    void __cusparse_check(const char* fname, const size_t line);
    void __cusolver_check(const char* fname, const size_t line);

    /**
     * Utility functions for error retrieval & evaluation.
     */
    const char * cublas_err_str(cublasStatus_t status);
    const char * cusparse_err_str(cusparseStatus_t status);
    const char * cusolver_err_str(cusolverStatus_t status);

protected:
    cudaStream_t m_stream;
};

using gpu_handle_ptr = std::shared_ptr<gpu_handle_t>;

NS_CULIP_END

#endif /* __CULIP_LIBS_UTILS_GPU_DEFINES_H_ */
