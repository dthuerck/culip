/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIBS_LA_SQMR_CUH_
#define __CULIP_LIBS_LA_SQMR_CUH_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>
#include <libs/la/spmv.cuh>
#include <libs/la/preconditioner.cuh>

NS_CULIP_BEGIN
NS_LA_BEGIN

/**
 * Simplified (symmetric) QMR variant that assumes a preconditioner has
 * the form M = L * D * L' with D symmetric. In this case, the A'
 * multiplications can be substituted with (faster) D solves.
 *
 * This way, this variant allows using an indefinite preconditioner; contratry
 * to SQMR-from-BiCG, the quasi residual is tracked each iteration and used
 * for convergence testing instead of a simple upper bound.
 */
template<typename T>
class SQMR
{
public:
    SQMR(gpu_handle_ptr& gpu_handle, const SPMV<T> * spmv,
        const T tol, const mat_int_t max_iterations,
        const Preconditioner<T> * preconditioner);
    SQMR(gpu_handle_ptr&, const SPMV<T> * spmv,
        const T tol, const mat_int_t max_iterations);
    ~SQMR();

    void solve(const dense_vector_t<T> * b, dense_vector_t<T> * x,
        T& residual, mat_int_t& iterations);

protected:
    void prec_spmv(const dense_vector_t<T> * b, dense_vector_t<T> * x);

    void apply_preconditioner_left(const dense_vector_t<T> * b,
        dense_vector_t<T> * x);
    void apply_preconditioner_middle(const dense_vector_t<T> * b,
        dense_vector_t<T> * x);
    void apply_preconditioner_right(const dense_vector_t<T> * b,
        dense_vector_t<T> * x);

protected:
    gpu_handle_ptr& m_handle;

    const T m_tol;
    const mat_int_t m_max_it;

    /* SPMV & Preconditioner */
    const SPMV<T> * m_spmv;
    const Preconditioner<T> * m_prec;

    /* QMR working data */
    cudaStream_t m_stream_a;
    cudaStream_t m_stream_b;
    cudaEvent_t m_compute_event;
    cudaEvent_t m_copy_event;

    dense_vector_ptr<T> m_scalars;
    dense_vector_ptr<T> m_v[3];
    dense_vector_ptr<T> m_w[3];
    dense_vector_ptr<T> m_p[3];
    dense_vector_ptr<T> m_Av;
    dense_vector_ptr<T> m_ev, m_ev2;
    dense_vector_ptr<T> m_tmp;

    /* data for convergence checks */
    mat_int_t * m_host_test;
    dense_vector_ptr<mat_int_t> m_test;
};

NS_LA_END
NS_CULIP_END

#endif /*__CULIP_LIBS_LA_SQMR_CUH_ */