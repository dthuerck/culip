/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/la/spmv.cuh>
#include <libs/la/sparse_la.cuh>
#include <libs/la/dense_la.cuh>
#include <libs/la/helper_kernels.cuh>

NS_CULIP_BEGIN
NS_LA_BEGIN

template<typename T>
SPMV<T>::
~SPMV()
{

}

/**
 * *****************************************************************************
 * ****************************** CSRMatrixSPMV ********************************
 * *****************************************************************************
 */

template<typename T>
CSRMatrixSPMV<T>::
CSRMatrixSPMV(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * A,
    const bool explicit_At)
: m_handle(gpu_handle),
  m_A(A),
  m_explicit_At(explicit_At)
{
    if(explicit_At)
        T_transpose_csr(m_handle, m_A, m_At);
}

/* ************************************************************************** */

template<typename T>
CSRMatrixSPMV<T>::
~CSRMatrixSPMV()
{

}

/* ************************************************************************** */

template<typename T>
mat_int_t
CSRMatrixSPMV<T>::
m()
const
{
    return m_A->m;
}

/* ************************************************************************** */

template<typename T>
mat_int_t
CSRMatrixSPMV<T>::
n()
const
{
    return m_A->n;
}

/* ************************************************************************** */

template<typename T>
void
CSRMatrixSPMV<T>::
multiply(
    const dense_vector_t<T> * x,
    dense_vector_t<T> * b,
    const bool transpose)
const
{
    m_handle->push_scalar_mode();
    m_handle->set_scalar_mode(false);
    const T one = 1.0;
    const T zero = 0.0;
    multiply(x, b, &one, &zero, transpose);
    m_handle->pop_scalar_mode();
}

/* ************************************************************************** */

template<typename T>
void
CSRMatrixSPMV<T>::
multiply(
    const dense_vector_t<T> * x,
    dense_vector_t<T> * b,
    const T * alpha,
    const T * beta,
    const bool transpose)
const
{
    T_csrmv(
        m_handle,
        (transpose && !m_explicit_At) ?
            CUSPARSE_OPERATION_TRANSPOSE :
            CUSPARSE_OPERATION_NON_TRANSPOSE,
        (transpose && m_explicit_At) ? m_At.get() : m_A,
        x, b, alpha, beta);
}

/**
 * *****************************************************************************
 * ***************************** NormalMatrixSPMV ******************************
 * *****************************************************************************
 */

template<typename T>
NormalMatrixSPMV<T>::
NormalMatrixSPMV(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * A,
    const dense_vector_t<T> * d,
    const bool explicit_At)
: NormalMatrixSPMV<T>(gpu_handle, A, d, 0, nullptr, explicit_At)
{

}

/* ************************************************************************** */

template<typename T>
NormalMatrixSPMV<T>::
NormalMatrixSPMV(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * A,
    const dense_vector_t<T> * d,
    const T delta,
    const dense_vector_t<mat_int_t> * ix,
    const bool explicit_At)
: m_handle(gpu_handle),
  m_A(A),
  m_d(d),
  m_delta(delta),
  m_ix(ix),
  m_v(make_managed_dense_vector_ptr<T>(A->m, true)),
  m_Atx(make_managed_dense_vector_ptr<T>(A->n, true)),
  m_explicit_At(explicit_At)
{
    if(explicit_At)
        T_transpose_csr(gpu_handle, m_A, m_At);
}

/* ************************************************************************** */

template<typename T>
NormalMatrixSPMV<T>::
~NormalMatrixSPMV()
{

}

/* ************************************************************************** */

template<typename T>
mat_int_t
NormalMatrixSPMV<T>::
m()
const
{
    return m_A->m;
}

/* ************************************************************************** */

template<typename T>
mat_int_t
NormalMatrixSPMV<T>::
n()
const
{
    return m_A->m;
}

/* ************************************************************************** */

template<typename T>
void
NormalMatrixSPMV<T>::
multiply(
    const dense_vector_t<T> * x,
    dense_vector_t<T> * b,
    const bool transpose)
const
{
    m_handle->push_scalar_mode();
    m_handle->set_scalar_mode(false);

    const T one = 1.0;
    const T zero = 0.0;
    multiply(x, b, &one, &zero, transpose);

    m_handle->pop_scalar_mode();
}

/* ************************************************************************** */

template<typename T>
void
NormalMatrixSPMV<T>::
multiply(
    const dense_vector_t<T> * x,
    dense_vector_t<T> * b,
    const T * alpha,
    const T * beta,
    const bool transpose)
const
{
    *m_v = b;

    m_handle->push_scalar_mode();
    m_handle->set_scalar_mode(false);

    /* define host constants */
    const T one = 1.0;
    const T zero = 0.0;

    /* note: ignore transpose, since NE matrix symmetric */
    const mat_int_t block_size = 256;
    const mat_int_t grid_size_n = DIV_UP(m_A->n, block_size);

    /* multiply with At */
    T_csrmv(
        m_handle,
        m_explicit_At ? CUSPARSE_OPERATION_NON_TRANSPOSE :
            CUSPARSE_OPERATION_TRANSPOSE,
        m_explicit_At ? m_At.get() : m_A,
        x, m_Atx.get(), &one, &zero);

    /* scale by d^-1 */
    k_cwdiv<<<grid_size_n, block_size, 0, m_handle->get_stream()>>>(
        m_Atx->dense_val, m_d->dense_val, m_Atx->dense_val, m_A->n);

    /* multiply by A */
    T_csrmv(
        m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        m_A, m_Atx.get(), b, &one, &zero);

    /* add delta I * x where applicable */
    if(m_ix != nullptr)
    {
        const mat_int_t grid_size_ix = DIV_UP(m_ix->m, block_size);
        k_add_app<T><<<grid_size_ix, block_size, 0, m_handle->get_stream()>>>
            (b->dense_val, m_ix->dense_val, x->dense_val, m_delta, m_ix->m);
    }
    m_handle->pop_scalar_mode();

    /* multiply b by alpha and add v */
    T_scal(m_handle, b, alpha);
    T_axpy(m_handle, b, m_v.get(), beta);
}

/**
 * *****************************************************************************
 * ************************** AugmentedMatrixSPMV ******************************
 * *****************************************************************************
 */

template<typename T>
AugmentedMatrixSPMV<T>::
AugmentedMatrixSPMV(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * A,
    const dense_vector_t<T> * d,
    const bool explicit_At)
: AugmentedMatrixSPMV<T>(gpu_handle, A, d, 0, nullptr, explicit_At)
{

}

/* ************************************************************************** */

template<typename T>
AugmentedMatrixSPMV<T>::
AugmentedMatrixSPMV(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * A,
    const dense_vector_t<T> * d,
    const T delta,
    const dense_vector_t<mat_int_t> * ix,
    const bool explicit_At)
: m_handle(gpu_handle),
  m_A(A),
  m_d(d),
  m_delta(delta),
  m_ix(ix),
  m_v(make_managed_dense_vector_ptr<T>(A->m + A->n, true)),
  m_explicit_At(explicit_At)
{
    if(explicit_At)
        T_transpose_csr(gpu_handle, m_A, m_At);
}

/* ************************************************************************** */

template<typename T>
AugmentedMatrixSPMV<T>::
~AugmentedMatrixSPMV()
{

}

/* ************************************************************************** */

template<typename T>
mat_int_t
AugmentedMatrixSPMV<T>::
m()
const
{
    return m_A->m;
}

/* ************************************************************************** */

template<typename T>
mat_int_t
AugmentedMatrixSPMV<T>::
n()
const
{
    return m_A->m;
}

/* ************************************************************************** */

template<typename T>
void
AugmentedMatrixSPMV<T>::
multiply(
    const dense_vector_t<T> * x,
    dense_vector_t<T> * b,
    const bool transpose)
const
{
    m_handle->push_scalar_mode();
    m_handle->set_scalar_mode(false);

    const T one = 1.0;
    const T zero = 0.0;
    multiply(x, b, &one, &zero, transpose);

    m_handle->pop_scalar_mode();
}

/* ************************************************************************** */

template<typename T>
void
AugmentedMatrixSPMV<T>::
multiply(
    const dense_vector_t<T> * x,
    dense_vector_t<T> * b,
    const T * alpha,
    const T * beta,
    const bool transpose)
const
{
    *m_v = b;

    /* note: ignore transpose, since matrix symmetric */
    const mat_int_t A_m = m_A->m;
    const mat_int_t A_n = m_A->n;

    const mat_int_t block_size = 256;
    const mat_int_t grid_size_n = DIV_UP(A_n, block_size);

    /* split x, b */
    dense_vector_ptr<T> x_1 = make_raw_dense_vector_ptr<T>(A_n, true,
        x->dense_val);
    dense_vector_ptr<T> x_2 = make_raw_dense_vector_ptr<T>(A_m, true,
        x->dense_val + A_n);

    dense_vector_ptr<T> b_1 = make_raw_dense_vector_ptr<T>(A_n, true,
        b->dense_val);
    dense_vector_ptr<T> b_2 = make_raw_dense_vector_ptr<T>(A_m, true,
        b->dense_val + A_n);

    /* multiply D with x1 [size n] */
    k_cwprod<T><<<grid_size_n, block_size, 0, m_handle->get_stream()>>>(
        x_1->dense_val, m_d->dense_val, b_1->dense_val, A_n);

    /* multiply (delta * I)(ix) with x2 [size m] */
    cudaMemsetAsync(b_2->dense_val, 0, A_m * sizeof(T), m_handle->get_stream());
    if(m_ix != nullptr)
    {
        const mat_int_t grid_size_ix = DIV_UP(m_ix->m, block_size);
        k_add_app<T><<<grid_size_ix, block_size, 0, m_handle->get_stream()>>>(
            b_2->dense_val, m_ix->dense_val, x_2->dense_val,
            m_delta, m_ix->m);
    }

    m_handle->push_scalar_mode();
    m_handle->set_scalar_mode(false);
    T host_one = 1.0;

    /* add A x1 to (delta * I)(ix) to b2 */
    T_csrmv(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        m_A, x_1.get(),
        b_2.get(), &host_one, &host_one);

    /* add A^t x2 to D * x1 */
    T_csrmv(m_handle, m_explicit_At ?
        CUSPARSE_OPERATION_NON_TRANSPOSE :
        CUSPARSE_OPERATION_TRANSPOSE,
        m_explicit_At ? m_At.get() : m_A,
        x_1.get(), b_1.get(), &host_one, &host_one);
    m_handle->pop_scalar_mode();

    /* multiply b by alpha and add v */
    T_scal(m_handle, b, alpha);
    T_axpy(m_handle, b, m_v.get(), beta);
}

NS_LA_END
NS_CULIP_END
