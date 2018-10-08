/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/la/iterative_solvers/sqmr.cuh>

#include <libs/la/sparse_la.cuh>
#include <libs/la/dense_la.cuh>
#include <libs/la/helper_kernels.cuh>

NS_CULIP_BEGIN
NS_LA_BEGIN

#define ACC(r) (((r) % 3) + (((r) < 0) ? 3 : 0))

template<typename T>
SQMR<T>::
SQMR(
    gpu_handle_ptr& gpu_handle,
    const SPMV<T> * spmv,
    const T tol,
    const mat_int_t max_iterations,
    const Preconditioner<T> * preconditioner)
: m_handle(gpu_handle),
  m_spmv(spmv),
  m_tol(tol),
  m_max_it(max_iterations),
  m_prec(preconditioner),
  m_scalars(make_managed_dense_vector_ptr<T>(27, true)),
  m_Av(make_managed_dense_vector_ptr<T>(spmv->m(), true)),
  m_ev(make_managed_dense_vector_ptr<T>(spmv->m(), true)),
  m_ev2(make_managed_dense_vector_ptr<T>(spmv->m(), true)),
  m_tmp(make_managed_dense_vector_ptr<T>(spmv->m(), true)),
  m_test(make_managed_dense_vector_ptr<mat_int_t>(2, true))
{
    /* page-locked memory for async transfers */
    CHECK_CUDA(cudaMallocHost(&m_host_test, 2 * sizeof(mat_int_t)));

    CHECK_CUDA(cudaStreamCreateWithFlags(&m_stream_a, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&m_stream_b, cudaStreamNonBlocking));

    CHECK_CUDA(cudaEventCreate(&m_compute_event));
    CHECK_CUDA(cudaEventCreate(&m_copy_event));

    /* create working arrays */
    for(mat_int_t i = 0; i < 3; ++i)
    {
        m_v[i] = make_managed_dense_vector_ptr<T>(spmv->m(), true);
        m_w[i] = make_managed_dense_vector_ptr<T>(spmv->m(), true);
        m_p[i] = make_managed_dense_vector_ptr<T>(spmv->m(), true);
    }
}

/* ************************************************************************** */

template<typename T>
SQMR<T>::
SQMR(
    gpu_handle_ptr& gpu_handle,
    const SPMV<T> * spmv,
    const T tol,
    const mat_int_t max_iterations)
: SQMR<T>(gpu_handle, spmv, tol, max_iterations, nullptr)
{

}

/* ************************************************************************** */

template<typename T>
SQMR<T>::
~SQMR()
{
    CHECK_CUDA(cudaFreeHost(m_host_test));

    /* delete streams and set back to default stream */
    CHECK_CUDA(cudaStreamDestroy(m_stream_b));
    CHECK_CUDA(cudaStreamDestroy(m_stream_a));

    CHECK_CUDA(cudaEventDestroy(m_copy_event));
    CHECK_CUDA(cudaEventDestroy(m_compute_event));
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_sqmr_test(
    const T * delta,
    const T * res,
    const T * norm_A,
    const T * norm_x,
    const T * norm_pb,
    const T tol,
    mat_int_t * test,
    const mat_int_t iteration)
{
    test[0] |= (abs(*delta) <= tol);
    test[1] |= (*res / (*norm_A * (*norm_x) + *norm_pb) <= tol);

    if(iteration % 4 == 0)
        printf("%d: %g\n", iteration, *res / *norm_pb);
}

template<typename T>
void
SQMR<T>::
solve(
    const dense_vector_t<T> * b,
    dense_vector_t<T> * x,
    T& residual,
    mat_int_t& iterations)
{
    const mat_int_t m = m_spmv->m();

    m_handle->set_stream(m_stream_a);

    m_handle->push_scalar_mode();
    m_handle->set_scalar_mode(true);

    /* initialize vectors */
    cudaMemsetAsync(m_scalars->dense_val, 0, m_scalars->m * sizeof(T),
        m_stream_a);
    for(mat_int_t i = 0; i < 3; ++i)
    {
        cudaMemsetAsync(m_v[i]->dense_val, 0, m * sizeof(T), m_stream_a);
        cudaMemsetAsync(m_w[i]->dense_val, 0, m * sizeof(T), m_stream_a);
        cudaMemsetAsync(m_p[i]->dense_val, 0, m * sizeof(T), m_stream_a);
    }
    cudaMemsetAsync(m_test->dense_val, 0, 2 * sizeof(mat_int_t), m_stream_a);
    cudaMemsetAsync(x->dense_val, 0, x->m * sizeof(T), m_stream_a);

    /* initialize scalars */
    T * alpha = m_scalars->dense_val;
    T * beta = m_scalars->dense_val + 1;
    T * delta = m_scalars->dense_val + 4;
    T * norm_A = m_scalars->dense_val + 7;
    T * epsilon = m_scalars->dense_val + 8;
    T * beta_rot = m_scalars->dense_val + 9;
    T * beta_rot_rot = m_scalars->dense_val + 10;
    T * alpha_rot = m_scalars->dense_val + 11;
    T * c = m_scalars->dense_val + 12;
    T * s = m_scalars->dense_val + 15;
    T * tau = m_scalars->dense_val + 18;
    T * rho = m_scalars->dense_val + 19;
    T * gamma = m_scalars->dense_val + 20;
    T * norm_pb = m_scalars->dense_val + 21;
    T * norm_x = m_scalars->dense_val + 22;
    T * one = m_scalars->dense_val + 23;
    T * zero = m_scalars->dense_val + 24;
    T * tmp = m_scalars->dense_val + 25;
    T * eta = m_scalars->dense_val + 26;

    k_set_scalar<T><<<1, 1, 0, m_stream_a>>>(&c[ACC(0)], -1.0);
    k_set_scalar<T><<<1, 1, 0, m_stream_a>>>(one, 1.0);
    k_set_scalar<T><<<1, 1, 0, m_stream_a>>>(zero, 0.0);

    /* compute initial RHS, v_1 and w_1 */
    apply_preconditioner_left(b, m_v[0].get());

    /* v_0 = (L \ b) / norm(L \ b) */
    T_nrm2(m_handle, m_v[0].get(), norm_pb);
    k_set_div_scalar<T><<<1, 1, 0, m_stream_a>>>(tmp, one, norm_pb);
    T_scal(m_handle, m_v[0].get(), tmp);

    /* eta = 1 / norm(D \ v[0]); w[0] = eta * (D \ v[0]) */
    apply_preconditioner_middle(m_v[0].get(), m_w[0].get());

    T_nrm2(m_handle, m_w[0].get(), eta);
    k_set_div_scalar<T><<<1, 1, 0, m_stream_a>>>(eta, one, eta);
    T_scal(m_handle, m_w[0].get(), eta);

    /* gamma = norm(L \ b) */
    k_set_scalar<T><<<1, 1, 0, m_stream_a>>>(gamma, norm_pb);

    /* lookahead: Av = Pl \ (A * (Pr \ (D \ v[ACC(k)]))) */
    apply_preconditioner_middle(m_v[0].get(), m_tmp.get());
    prec_spmv(m_tmp.get(), m_Av.get());

    iterations = 0;
    while(iterations < m_max_it)
    {
        /**
         * Simplified Bi-Lanczos step
         */

        /* alpha = v[ACC(k)]' * (D \ Av) / v[ACC(k)]' * (D \ v[ACC(k)]) */
        apply_preconditioner_middle(m_Av.get(), m_tmp.get());
        T_doti(m_handle, m_v[ACC(iterations)].get(), m_tmp.get(), alpha);
        apply_preconditioner_middle(m_v[ACC(iterations)].get(), m_tmp.get());
        T_doti(m_handle, m_v[ACC(iterations)].get(), m_tmp.get(), tmp);
        k_set_div_scalar<T><<<1, 1, 0, m_stream_a>>>(alpha, alpha, tmp);

        /**
         * v[ACC(k + 1)] = Av - alpha * v[ACC(k)]
         *                    - beta[ACC(k)] * v[ACC(k - 1)]
         */
        cudaMemcpyAsync(m_v[ACC(iterations + 1)]->dense_val, m_Av->dense_val,
            m * sizeof(T), cudaMemcpyDeviceToDevice, m_stream_a);
        k_set_neg_scalar<T><<<1, 1, 0, m_stream_a>>>(tmp, alpha);
        T_axpy(m_handle, m_v[ACC(iterations + 1)].get(),
            m_v[ACC(iterations)].get(), tmp);
        k_set_neg_scalar<T><<<1, 1, 0, m_stream_a>>>(tmp,
            &beta[ACC(iterations)]);
        T_axpy(m_handle, m_v[ACC(iterations + 1)].get(),
            m_v[ACC(iterations - 1)].get(), tmp);

        /* delta[ACC(k + 1)] = norm(v[ACC(k + 1)]) */
        T_nrm2(m_handle, m_v[ACC(iterations + 1)].get(),
            &delta[ACC(iterations + 1)]);

        /* scale v[ACC(k + 1)] by delta[ACC(k + 1)] */
        k_set_div_scalar<T><<<1, 1, 0, m_stream_a>>>(tmp, one,
            &delta[ACC(iterations + 1)]);
        T_scal(m_handle, m_v[ACC(iterations + 1)].get(), tmp);

        /* w[ACC(k + 1)] = delta[ACC(k + 1)] * eta * D \ v[ACC(k + 1)] */
        apply_preconditioner_middle(m_v[ACC(iterations + 1)].get(),
            m_w[ACC(iterations + 1)].get());
        k_set_mult_scalar<T><<<1, 1, 0, m_stream_a>>>(tmp, eta,
            &delta[ACC(iterations + 1)]);
        T_scal(m_handle, m_w[ACC(iterations + 1)].get(), tmp);

        /* lookahead: Av = Pl \ (A * (Pr \ (D \ v[ACC(k + 1)]))) */
        apply_preconditioner_middle(m_v[ACC(iterations + 1)].get(),
            m_tmp.get());
        prec_spmv(m_tmp.get(), m_Av.get());

        /* beta[ACC(k + 1)] = w[ACC(k)]' * Av / w[ACC(k)]' * v[ACC(k)]' */
        T_doti(m_handle, m_w[ACC(iterations)].get(), m_Av.get(),
            &beta[ACC(iterations + 1)]);
        T_doti(m_handle, m_w[ACC(iterations)].get(), m_v[ACC(iterations)].get(),
            tmp);
        k_set_div_scalar<T><<<1, 1, 0, m_stream_a>>>(&beta[ACC(iterations + 1)],
            &beta[ACC(iterations + 1)], tmp);

        /* normalize w[ACC(k + 1)] with beta[ACC(k + 1)] */
        k_set_div_scalar<T><<<1, 1, 0, m_stream_a>>>(tmp, one,
            &beta[ACC(iterations + 1)]);
        T_scal(m_handle, m_w[ACC(iterations + 1)].get(), tmp);

        /* upate eta */
        k_set_div_scalar<T><<<1, 1, 0, m_stream_a>>>(tmp,
            &delta[ACC(iterations + 1)], &beta[ACC(iterations + 1)]);
        k_set_mult_scalar<T><<<1, 1, 0, m_stream_a>>>(eta,
            tmp, eta);

        /**
         * Update matrix norm
         */
        k_scalar_norm3<T><<<1, 1, 0, m_stream_a>>>(&beta[ACC(iterations)],
            alpha, &delta[ACC(iterations + 1)], tmp);
        k_max<T><<<1, 1, 0, m_stream_a>>>(norm_A, norm_A, tmp);

        /**
         * Apply the last two Givens rotation to tridiagonal matrix
         */
        k_set_mult_scalar<T><<<1, 1, 0, m_stream_a>>>(epsilon,
            &s[ACC(iterations - 1)], &beta[ACC(iterations)]);
        k_set_neg_scalar<T><<<1, 1, 0, m_stream_a>>>(tmp,
            &c[ACC(iterations - 1)]);
        k_set_mult_scalar<T><<<1, 1, 0, m_stream_a>>>(beta_rot, tmp,
            &beta[ACC(iterations)]);

        k_plus<T><<<1, 1, 0, m_stream_a>>>(&c[ACC(iterations)], beta_rot,
            &s[ACC(iterations)], alpha, beta_rot_rot, 1);
        k_set_neg_scalar<T><<<1, 1, 0, m_stream_a>>>(tmp, &c[ACC(iterations)]);
        k_plus<T><<<1, 1, 0, m_stream_a>>>(&s[ACC(iterations)], beta_rot,
            tmp, alpha, alpha_rot, 1);

        /* compute new Givens rotation */
        k_sym_ortho<T><<<1, 1, 0, m_stream_a>>>(alpha_rot,
            &delta[ACC(iterations + 1)], &c[ACC(iterations + 1)],
            &s[ACC(iterations + 1)], rho);

        /* apply Givens rotation to RHS */
        k_set_mult_scalar<T><<<1, 1, 0, m_stream_a>>>(tau,
            &c[ACC(iterations + 1)], gamma);

        /* solve least-squares system */
        cudaMemcpyAsync(m_p[ACC(iterations)]->dense_val,
            m_v[ACC(iterations)]->dense_val, m * sizeof(T),
            cudaMemcpyDeviceToDevice, m_stream_a);
        k_set_neg_scalar<T><<<1, 1, 0, m_stream_a>>>(tmp, beta_rot_rot);
        T_axpy(m_handle, m_p[ACC(iterations)].get(),
            m_p[ACC(iterations - 1)].get(), tmp);
        k_set_neg_scalar<T><<<1, 1, 0, m_stream_a>>>(tmp, epsilon);
        T_axpy(m_handle, m_p[ACC(iterations)].get(),
            m_p[ACC(iterations - 2)].get(), tmp);
        k_set_div_scalar<T><<<1, 1, 0, m_stream_a>>>(tmp, one, rho);
        T_scal(m_handle, m_p[ACC(iterations)].get(), tmp);

        T_axpy(m_handle, x, m_p[ACC(iterations)].get(), tau);
        T_nrm2(m_handle, x, norm_x);

        /* update quasi-residual */
        k_set_mult_scalar<T><<<1, 1, 0, m_stream_a>>>(gamma,
            &s[ACC(iterations + 1)], gamma);

        /* check for convergence */
        k_sqmr_test<T><<<1, 1, 0, m_stream_a>>>(&delta[ACC(iterations + 1)],
            gamma, norm_A, norm_x, norm_pb, m_tol, m_test->dense_val,
            iterations);

        /**
         * Throttling: Check convergence from last round.
         */
        const int tcheck = 4;

        if (iterations > 0 && (iterations % tcheck) == 0) {

            /* stop computation */
            cudaEventSynchronize(m_compute_event);

            /* make sure copying has finished */
            cudaEventSynchronize(m_copy_event);

            if(m_host_test[0] || m_host_test[1])
                break;
        }
        else if(((iterations + 1) % tcheck) == 0) {
            cudaStreamSynchronize(m_stream_a);
            cudaMemcpyAsync(m_host_test, m_test->dense_val,
                2 * sizeof(mat_int_t), cudaMemcpyDeviceToHost, m_stream_b);
            cudaEventRecord(m_copy_event, m_stream_b);
            cudaEventRecord(m_compute_event, m_stream_a);
        }

        /* step to next iteration */
        ++iterations;
    };

    CHECK_CUDA(cudaDeviceSynchronize());

    /* reset libs to default stream */
    m_handle->set_stream(cudaStreamDefault);

    /* copy last residual for return */
    k_set_div_scalar<<<1, 1>>>(gamma, gamma, norm_pb);
    CHECK_CUDA(cudaMemcpy(&residual, gamma, sizeof(T), cudaMemcpyDeviceToHost));

    /* solve for x with preconditioner (middle + left) */
    CHECK_CUDA(cudaMemcpy(m_Av->dense_val, x->dense_val, m * sizeof(T),
        cudaMemcpyDeviceToDevice));
    apply_preconditioner_middle(m_Av.get(), m_tmp.get());
    apply_preconditioner_right(m_tmp.get(), x);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* reset pointer mode */
    m_handle->pop_scalar_mode();
}

/* ************************************************************************** */

template<typename T>
void
SQMR<T>::
prec_spmv(
    const dense_vector_t<T> * x,
    dense_vector_t<T> * b)
{
    apply_preconditioner_right(x, m_ev.get());
    m_spmv->multiply(m_ev.get(), m_ev2.get());
    apply_preconditioner_left(m_ev2.get(), b);
}

/* ************************************************************************** */

template<typename T>
void
SQMR<T>::
apply_preconditioner_left(
    const dense_vector_t<T> * b,
    dense_vector_t<T> * x)
{
    if(m_prec != nullptr && m_prec->is_left())
        m_prec->solve_left(b, x, false);
    else
        cudaMemcpyAsync(x->dense_val, b->dense_val, b->m * sizeof(T),
            cudaMemcpyDeviceToDevice, m_stream_a);
}

/* ************************************************************************** */

template<typename T>
void
SQMR<T>::
apply_preconditioner_middle(
    const dense_vector_t<T> * b,
    dense_vector_t<T> * x)
{
    if(m_prec != nullptr && m_prec->is_middle())
        m_prec->solve_middle(b, x, false);
    else
        cudaMemcpyAsync(x->dense_val, b->dense_val, b->m * sizeof(T),
            cudaMemcpyDeviceToDevice, m_stream_a);
}

/* ************************************************************************** */

template<typename T>
void
SQMR<T>::
apply_preconditioner_right(
    const dense_vector_t<T> * b,
    dense_vector_t<T> * x)
{
    if(m_prec != nullptr && m_prec->is_right())
        m_prec->solve_right(b, x, false);
    else
        cudaMemcpyAsync(x->dense_val, b->dense_val, b->m * sizeof(T),
            cudaMemcpyDeviceToDevice, m_stream_a);
}

NS_LA_END
NS_CULIP_END