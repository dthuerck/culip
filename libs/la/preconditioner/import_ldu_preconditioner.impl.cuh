/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/la/preconditioner/import_ldu_preconditioner.cuh>

#include <libs/la/sparse_la.cuh>
#include <libs/la/helper_kernels.cuh>

#include <cooperative_groups.h>

using namespace cooperative_groups;

NS_CULIP_BEGIN
NS_LA_BEGIN

template<typename T>
ImportLDUPreconditioner<T>::
ImportLDUPreconditioner(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * L,
    const csr_matrix_t<T> * U,
    const bool exact_solve)
: Preconditioner<T>(gpu_handle),
  m_exact_solve(exact_solve),
  m_symmetric(false),
  m_has_D(false),
  m_L(L),
  m_D(nullptr),
  m_U(U),
  m_P_r(nullptr),
  m_P_c(nullptr),
  m_S_r(nullptr),
  m_S_c(nullptr),
  m_has_P(false),
  m_has_S(false),
  m_tmp(make_managed_dense_vector_ptr<T>(L->n, true)),
  m_tmp_ps(make_managed_dense_vector_ptr<T>(L->n, true))
{
    analyze_L();
    analyze_U();
}

/* ************************************************************************** */

template<typename T>
ImportLDUPreconditioner<T>::
ImportLDUPreconditioner(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * L,
    const bool exact_solve)
: Preconditioner<T>(gpu_handle),
  m_exact_solve(exact_solve),
  m_symmetric(true),
  m_has_D(false),
  m_L(L),
  m_D(nullptr),
  m_D_is_piv(nullptr),
  m_U(nullptr),
  m_P_r(nullptr),
  m_P_c(nullptr),
  m_S_r(nullptr),
  m_S_c(nullptr),
  m_has_P(false),
  m_has_S(false),
  m_tmp(make_managed_dense_vector_ptr<T>(L->n, true)),
  m_tmp_ps(make_managed_dense_vector_ptr<T>(L->n, true))
{
    analyze_L();
}

/* ************************************************************************** */

template<typename T>
ImportLDUPreconditioner<T>::
ImportLDUPreconditioner(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * L,
    const dense_vector_t<T> * D,
    const dense_vector_t<mat_int_t> * D_is_piv,
    const bool exact_solve)
: Preconditioner<T>(gpu_handle),
  m_exact_solve(exact_solve),
  m_symmetric(true),
  m_has_D(true),
  m_L(L),
  m_D(D),
  m_D_is_piv(D_is_piv),
  m_U(nullptr),
  m_P_r(nullptr),
  m_P_c(nullptr),
  m_S_r(nullptr),
  m_S_c(nullptr),
  m_has_P(false),
  m_has_S(false),
  m_tmp(make_managed_dense_vector_ptr<T>(L->n, true)),
  m_tmp_ps(make_managed_dense_vector_ptr<T>(L->n, true))
{
    analyze_L();
}

/* ************************************************************************** */

template<typename T>
ImportLDUPreconditioner<T>::
~ImportLDUPreconditioner()
{
    if(m_info_L != nullptr)
        this->m_handle->cusparse_status =
            cusparseCreateSolveAnalysisInfo(&m_info_L);
    if(m_info_L_t != nullptr)
        this->m_handle->cusparse_status =
            cusparseCreateSolveAnalysisInfo(&m_info_L_t);

    if(m_info_U != nullptr)
        this->m_handle->cusparse_status =
            cusparseCreateSolveAnalysisInfo(&m_info_U);
    if(m_info_U_t != nullptr)
        this->m_handle->cusparse_status =
            cusparseCreateSolveAnalysisInfo(&m_info_U_t);
}

/* ************************************************************************** */

template<typename T>
void
ImportLDUPreconditioner<T>::
set_permutation(
    const dense_vector_t<mat_int_t> * P_r,
    const dense_vector_t<mat_int_t> * P_c)
{
    m_has_P = true;

    m_P_r = P_r;
    m_P_c = P_c;
}

/* ************************************************************************** */

template<typename T>
void
ImportLDUPreconditioner<T>::
set_scaling(
    const dense_vector_t<T> * S_r,
    const dense_vector_t<T> * S_c)
{
    m_has_S = true;

    m_S_r = S_r;
    m_S_c = S_c;
}

/* ************************************************************************** */

template<typename T>
mat_int_t
ImportLDUPreconditioner<T>::
n()
const
{
    return m_L->m;
}

/* ************************************************************************** */

template<typename T>
bool
ImportLDUPreconditioner<T>::
is_left()
const
{
    return true;
}

/* ************************************************************************** */

template<typename T>
bool
ImportLDUPreconditioner<T>::
is_middle()
const
{
    return m_has_D;
}

/* ************************************************************************** */

template<typename T>
bool
ImportLDUPreconditioner<T>::
is_right()
const
{
    return true;
}

/* ************************************************************************** */

template<typename T>
void
ImportLDUPreconditioner<T>::
solve_left(
    const dense_vector_t<T> * b,
    dense_vector_t<T> * x,
    const bool transpose)
const
{
    T one = 1.0;
    this->m_handle->push_scalar_mode();
    this->m_handle->set_scalar_mode(false);

    cudaStream_t stream = this->m_handle->get_stream();

    if(transpose)
    {
        /* P_l = L * P_r' * inv(S_r) -> inv(P_l) = S_r * P_r * inv(L) */
        if(m_exact_solve)
        {
            T_triangular_step_solve(
                this->m_handle, CUSPARSE_OPERATION_TRANSPOSE,
                m_info_L_t, m_L, b, m_tmp.get(), &one);
        }
        else
        {
            T_approx_triangular_step_solve(
                this->m_handle, CUSPARSE_OPERATION_TRANSPOSE,
                m_approx_info_L_t, m_L, b, m_tmp.get(), &one, 10);
        }

        solve_PS(true, true, m_tmp.get(), x);
    }
    else
    {
        /* P_l = inv(S_r) * P_r * L -> inv(P_l) = inv(L) * P_r' * S_r */
        solve_PS(true, false, b, m_tmp.get());

        if(m_exact_solve)
        {
            T_triangular_step_solve(this->m_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE, m_info_L, m_L, m_tmp.get(), x,
                    &one);
        }
        else
        {
            T_approx_triangular_step_solve(this->m_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE, m_approx_info_L, m_L,
                m_tmp.get(), x, &one, 10);
        }
    }

    this->m_handle->pop_scalar_mode();
}

/* ************************************************************************** */

template<typename T>
void
ImportLDUPreconditioner<T>::
solve_middle(
    const dense_vector_t<T> * b,
    dense_vector_t<T> * x,
    const bool transpose)
const
{
    CHECK_CUDA(cudaMemcpyAsync(x->dense_val, b->dense_val, b->m * sizeof(T),
        cudaMemcpyDeviceToDevice, this->m_handle->get_stream()));

    if(m_has_D)
    {
        /* now solve with D on x */
        const mat_int_t block_size = 256;
        const mat_int_t grid_size = DIV_UP(m_L->n, block_size);

        k_solve_D<<<grid_size, block_size, 0, this->m_handle->get_stream()>>>(
            m_D->dense_val,
            m_D_is_piv->dense_val,
            x->dense_val,
            m_L->n);
    }
};

/* ************************************************************************** */

template<typename T>
void
ImportLDUPreconditioner<T>::
solve_right(
    const dense_vector_t<T> * b,
    dense_vector_t<T> * x,
    const bool transpose)
const
{
    const csr_matrix_t<T> * R = m_symmetric ? m_L : m_U;

    T one = 1.0;
    this->m_handle->push_scalar_mode();
    this->m_handle->set_scalar_mode(false);

    cudaStream_t stream = this->m_handle->get_stream();

    if(transpose)
    {
        /* P_r = inv(S_c) * P_c' * R' -> inv(P_r) = inv(R') * P_c * S_c */
        solve_PS(false, false, b, m_tmp.get());

        if(m_exact_solve)
        {
            T_triangular_step_solve(this->m_handle,
                m_symmetric ? CUSPARSE_OPERATION_NON_TRANSPOSE :
                CUSPARSE_OPERATION_TRANSPOSE,
                m_symmetric ? m_info_L : m_info_U_t,
                R, m_tmp.get(), x, &one);
        }
        else
        {
            T_approx_triangular_step_solve(this->m_handle,
                m_symmetric ? CUSPARSE_OPERATION_NON_TRANSPOSE :
                CUSPARSE_OPERATION_TRANSPOSE,
                m_symmetric ? m_approx_info_L : m_approx_info_U_t,
                R, m_tmp.get(), x, &one, 10);
        }
    }
    else
    {
        /* P_r = R * P_c' * inv(S_c) -> inv(P_r) = S_c * P_c * inv(R) */
        if(m_exact_solve)
        {
            T_triangular_step_solve(this->m_handle,
                m_symmetric ? CUSPARSE_OPERATION_TRANSPOSE :
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                m_symmetric ? m_info_L_t : m_info_U,
                R, b, m_tmp.get(), &one);
        }
        else
        {
            T_approx_triangular_step_solve(this->m_handle,
                m_symmetric ? CUSPARSE_OPERATION_TRANSPOSE :
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                m_symmetric ? m_approx_info_L_t : m_approx_info_U,
                R, b, m_tmp.get(), &one, 10);
        }

        solve_PS(false, true, m_tmp.get(), x);
    }

    this->m_handle->pop_scalar_mode();
}

/* ************************************************************************** */

template<typename T>
void
ImportLDUPreconditioner<T>::
analyze_L()
{
    if(m_exact_solve)
    {
        this->m_handle->cusparse_status =
            cusparseCreateSolveAnalysisInfo(&m_info_L);
        CHECK_CUSPARSE(this->m_handle);
        this->m_handle->cusparse_status =
            cusparseCreateSolveAnalysisInfo(&m_info_L_t);
        CHECK_CUSPARSE(this->m_handle);

        T_triangular_step_analysis(this->m_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, m_L, m_info_L);
        T_triangular_step_analysis(this->m_handle,
            CUSPARSE_OPERATION_TRANSPOSE, m_L, m_info_L_t);
    }
    else
    {
        T_approx_triangular_step_analysis(this->m_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, m_L, m_approx_info_L);
        T_approx_triangular_step_analysis(this->m_handle,
            CUSPARSE_OPERATION_TRANSPOSE, m_L, m_approx_info_L_t);
    }
}

/* ************************************************************************** */

template<typename T>
void
ImportLDUPreconditioner<T>::
analyze_U()
{
    if(m_exact_solve)
    {
        this->m_handle->cusparse_status =
            cusparseCreateSolveAnalysisInfo(&m_info_U);
        CHECK_CUSPARSE(this->m_handle);
        this->m_handle->cusparse_status =
            cusparseCreateSolveAnalysisInfo(&m_info_U_t);
        CHECK_CUSPARSE(this->m_handle);

        T_triangular_step_analysis(this->m_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, m_U, m_info_U);
        T_triangular_step_analysis(this->m_handle,
            CUSPARSE_OPERATION_TRANSPOSE, m_U, m_info_U_t);
    }
    else
    {
        T_approx_triangular_step_analysis(this->m_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, m_U, m_approx_info_U);
        T_approx_triangular_step_analysis(this->m_handle,
            CUSPARSE_OPERATION_TRANSPOSE, m_U, m_approx_info_U_t);
    }
}

/* ************************************************************************** */

template<typename T>
void
ImportLDUPreconditioner<T>::
solve_PS(
    const bool left_data,
    const bool transpose,
    const dense_vector_t<T> * b,
    dense_vector_t<T> * x)
const
{
    const mat_int_t grid_size_n = DIV_UP(m_L->n, m_block_size);
    cudaStream_t& stream = this->m_handle->get_stream();

    const dense_vector_t<mat_int_t> * P = left_data ? m_P_r : m_P_c;
    const dense_vector_t<T> * S = left_data ? m_S_r : m_S_c;

    if(!transpose)
    {
        /* P' * S * b */
        if(m_has_P && m_has_S)
        {
            k_cwprod<T><<<grid_size_n, m_block_size, 0, stream>>>(
                b->dense_val, S->dense_val, m_tmp_ps->dense_val, m_L->n);

            k_permute_Pt<T><<<grid_size_n, m_block_size, 0, stream>>>(
                m_tmp_ps->dense_val, x->dense_val, P->dense_val, m_L->n);
        }
        else if(m_has_P)
        {
            k_permute_Pt<T><<<grid_size_n, m_block_size, 0, stream>>>(
                b->dense_val, x->dense_val, P->dense_val, m_L->n);
        }
        else if(m_has_S)
        {
            k_cwprod<T><<<grid_size_n, m_block_size, 0, stream>>>(
                b->dense_val, S->dense_val, x->dense_val, m_L->n);
        }
        else
        {
            cudaMemcpyAsync(x->dense_val, b->dense_val, b->m * sizeof(T),
                cudaMemcpyDeviceToDevice, this->m_handle->get_stream());
        }
    }
    else
    {
        /* S * P * b */
        if(m_has_P && m_has_S)
        {
            k_permute_P<T><<<grid_size_n, m_block_size, 0, stream>>>(
                b->dense_val, m_tmp_ps->dense_val, P->dense_val, m_L->n);

            k_cwprod<T><<<grid_size_n, m_block_size, 0, stream>>>(
                m_tmp_ps->dense_val, S->dense_val, x->dense_val, m_L->n);
        }
        else if(m_has_P)
        {
            k_permute_P<T><<<grid_size_n, m_block_size, 0, stream>>>(
                b->dense_val, x->dense_val, P->dense_val, m_L->n);
        }
        else if(m_has_S)
        {
            k_cwprod<T><<<grid_size_n, m_block_size, 0, stream>>>(
                b->dense_val, S->dense_val, x->dense_val, m_L->n);
        }
        else
        {
            cudaMemcpyAsync(x->dense_val, b->dense_val, b->m * sizeof(T),
                cudaMemcpyDeviceToDevice, this->m_handle->get_stream());
        }
    }
}

NS_LA_END
NS_CULIP_END