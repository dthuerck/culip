/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIBS_LA_SPMV_CUH_
#define __CULIP_LIBS_LA_SPMV_CUH_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>

NS_CULIP_BEGIN
NS_LA_BEGIN

template<typename T>
class SPMV
{
public:
    virtual ~SPMV();

    virtual mat_int_t m() const = 0;
    virtual mat_int_t n() const = 0;

    virtual void multiply(const dense_vector_t<T> * x,
        dense_vector_t<T> * b, const bool transpose = false) const = 0;
    virtual void multiply(const dense_vector_t<T> * x,
        dense_vector_t<T> * b, const T * alpha, const T * beta,
        const bool transpose = false) const = 0 ;
};

template<typename T>
using spmv_ptr = std::unique_ptr<SPMV<T>>;

/* ************************************************************************** */

/**
 * Multiply with a fixed csr-matrix A, i.e.
 * b = alpha * op(A) * x + beta * b
 */
template<typename T>
class CSRMatrixSPMV : public SPMV<T>
{
public:
    CSRMatrixSPMV(gpu_handle_ptr& gpu_handle,
        const csr_matrix_t<T> * A, const bool explicit_At = false);
    ~CSRMatrixSPMV();

    mat_int_t m() const;
    mat_int_t n() const;

    virtual void multiply(const dense_vector_t<T> * x,
        dense_vector_t<T> * b, const bool transpose = false) const;
    virtual void multiply(const dense_vector_t<T> * x,
        dense_vector_t<T> * b, const T * alpha, const T * beta,
        const bool transpose = false) const;

protected:
    gpu_handle_ptr m_handle;
    const csr_matrix_t<T> * m_A;

    const bool m_explicit_At;
    csr_matrix_ptr<T> m_At;
};

/* ************************************************************************** */

/**
 * Multiplies with A D^-1 A^T + (delta * I)(ix), where D = diag(d) and
 * delta is a scalar with ix a vector determining where to apply the shift.
 */
template<typename T>
class NormalMatrixSPMV : public SPMV<T>
{
public:
    NormalMatrixSPMV(gpu_handle_ptr& gpu_handle, const csr_matrix_t<T> * A,
        const dense_vector_t<T> * d, const bool explicit_At = false);
    NormalMatrixSPMV(gpu_handle_ptr& gpu_handle, const csr_matrix_t<T> * A,
        const dense_vector_t<T> * d, const T delta,
        const dense_vector_t<mat_int_t> * ix,
        const bool explicit_At = false);
    ~NormalMatrixSPMV();

    mat_int_t m() const;
    mat_int_t n() const;

    virtual void multiply(const dense_vector_t<T> * x,
        dense_vector_t<T> * b, const bool transpose = false) const;
    virtual void multiply(const dense_vector_t<T> * x,
        dense_vector_t<T> * b, const T * alpha, const T * beta,
        const bool transpose = false) const;

protected:
    gpu_handle_ptr m_handle;
    const csr_matrix_t<T> * m_A;
    csr_matrix_ptr<T> m_At;
    const dense_vector_t<T> * m_d;
    const bool m_explicit_At;

    const T m_delta;
    const dense_vector_t<mat_int_t> * m_ix;

    mutable dense_vector_ptr<T> m_Atx;
    mutable dense_vector_ptr<T> m_v;
};

/* ************************************************************************** */

/**
 * Multiplies with K = [D A'; A (delta*I)(ix)] where D = diag(d), delta is a
 * scalar constant and ix is an array which decides where the delta*I matrix
 * is applied.
 */
template<typename T>
class AugmentedMatrixSPMV : public SPMV<T>
{
public:
    AugmentedMatrixSPMV(gpu_handle_ptr& gpu_handle, const csr_matrix_t<T> * A,
        const dense_vector_t<T> * d, const bool explicit_At = false);
    AugmentedMatrixSPMV(gpu_handle_ptr& gpu_handle, const csr_matrix_t<T> * A,
        const dense_vector_t<T> * d, const T delta,
        const dense_vector_t<mat_int_t> * ix, const bool explicit_At = false);
    ~AugmentedMatrixSPMV();

    mat_int_t m() const;
    mat_int_t n() const;

    virtual void multiply(const dense_vector_t<T> * x,
        dense_vector_t<T> * b, const bool transpose = false) const;
    virtual void multiply(const dense_vector_t<T> * x,
        dense_vector_t<T> * b, const T * alpha, const T * beta,
        const bool transpose = false) const;

protected:
    gpu_handle_ptr m_handle;
    const csr_matrix_t<T> * m_A;
    csr_matrix_ptr<T> m_At;
    const dense_vector_t<T> * m_d;
    const bool m_explicit_At;

    const T m_delta;
    const dense_vector_t<mat_int_t> * m_ix;

    /* copied data for multiplication with abs */
    csr_matrix_ptr<T> * m_abs_A;
    mutable dense_vector_ptr<T> m_v;
};

NS_LA_END
NS_CULIP_END

#endif /* __CULIP_LA_SPMV_CUH_ */
