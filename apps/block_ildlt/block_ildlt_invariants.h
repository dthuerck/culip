/**
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <cstdio>
#include <set>
#include <vector>
#include <numeric>

#include <libs/utils/types.cuh>
#include <libs/test/test.h>
#include <libs/la/sparse_la.cuh>
#include <libs/la/dense_la.cuh>

#include <libs/la/preconditioner/block_ildlt.cuh>

#include <libs/la/spmv.cuh>
#include <libs/la/iterative_solvers/sqmr.cuh>

using namespace NS_CULIP;
using namespace NS_CULIP::NS_LA;
using namespace NS_CULIP::NS_TEST;

/* ************************************************************************** */

template<typename T>
csr_matrix_ptr<T>
host_permute(
    gpu_handle_ptr& cu_handle,
    const csr_matrix_t<T> * A,
    const dense_vector_t<mat_int_t> * permutation,
    const bool scale_before_match,
    const bool perm_is_old_to_new)
{
    csr_matrix_ptr<T> d_A = make_csr_matrix_ptr<T>(true);
    *d_A = A;

    dense_vector_ptr<T> h_scale =
        make_managed_dense_vector_ptr<T>(A->m, false);
    std::fill(h_scale->dense_val, h_scale->dense_val + A->m, 1.0);

    dense_vector_ptr<mat_int_t> d_permutation =
        make_managed_dense_vector_ptr<mat_int_t>(true);
    dense_vector_ptr<T> d_scaling =
        make_managed_dense_vector_ptr<T>(true);

    *d_permutation = permutation;
    *d_scaling = h_scale.get();

    csr_matrix_ptr<T> d_B;
    T_matrix_permute_scale(
        cu_handle,
        d_A.get(),
        d_scaling.get(),
        d_scaling.get(),
        d_permutation.get(),
        d_permutation.get(),
        d_B,
        scale_before_match,
        perm_is_old_to_new);

    csr_matrix_ptr<T> B = make_csr_matrix_ptr<T>(false);
    *B = d_B.get();

    return B;
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
bildlt_fun(
    const std::string& A_path,
    const std::string& block_starts_path,
    const std::string& permutation_path,
    const std::string& is_piv_path,
    const mat_int_t fill_level,
    const T fill_factor,
    const T threshold)
{
    gpu_handle_ptr cu_handle(new gpu_handle_t);

    /* load fine, mc64-processed csr matrix */
    printf("Loading %s...\n", A_path.c_str());
    csr_matrix_ptr<T> h_A = Test<T>::read_matrix_csr(A_path.c_str(),
        false);

    /* load block starts, permutation and piv starts */
    printf("Loading %s...\n", block_starts_path.c_str());
    dense_vector_ptr<mat_int_t> h_block_starts =
        Test<mat_int_t>::read_dense_vector(block_starts_path.c_str());

    printf("Loading %s...\n", permutation_path.c_str());
    dense_vector_ptr<mat_int_t> h_permutation =
        Test<mat_int_t>::read_dense_vector(permutation_path.c_str());

    printf("Loading %s...\n", is_piv_path.c_str());
    dense_vector_ptr<mat_int_t> h_is_piv =
        Test<mat_int_t>::read_dense_vector(is_piv_path.c_str());

    /* compute pivot starts */
    std::vector<mat_int_t> piv_starts;
    for(mat_int_t i = 0; i < h_is_piv->m; ++i)
    {
        if((*h_is_piv)[i] == 1)
            piv_starts.push_back(i);
    }
    piv_starts.push_back(h_A->m);

    printf("Input matrix: \n");
    printf("\tm: %ld\n", h_A->m);
    printf("\tn: %ld\n", h_A->n);
    printf("\tnnz: %ld\n", h_A->nnz);

    /* permute matrix */
    csr_matrix_ptr<T> h_p_A = host_permute<T>(
        cu_handle,
        h_A.get(),
        h_permutation.get(),
        false,
        false);

    /* block & compute preconditioner */
    printf("Start blocky...\n");
    fflush(stdout);

    START_TIMER("Block-iLDLt");
    BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting> blocky(
        cu_handle,
        h_p_A.get(),
        piv_starts.size() - 1,
        piv_starts.data(),
        h_block_starts->m - 1,
        h_block_starts->dense_val);
    blocky.compute(fill_level, fill_factor, threshold);
    STOP_TIMER("Block-iLDLt");
    PRINT_TIMER("Block-iLDLt");

    /* analysis for solve */
    // blocky.set_solve_algorithm(SOLVE_STRIPE, 0);
    blocky.set_solve_algorithm(SOLVE_BLOCK, 0);
    // blocky.set_solve_algorithm(SOLVE_SCALAR, 0);
    // blocky.set_solve_algorithm(SOLVE_JACOBI, 1);

    START_TIMER("Solve analysis");
    blocky.solve_analysis();
    STOP_TIMER("Solve analysis");
    PRINT_TIMER("Solve analysis");

    /* compute test right hand side */
    dense_vector_ptr<T> h_x =
        make_managed_dense_vector_ptr<T>(h_A->m, false);
    std::fill(h_x->dense_val, h_x->dense_val + h_A->m, 1.0);

    csr_matrix_ptr<T> d_p_A = make_csr_matrix_ptr<T>(true);
    dense_vector_ptr<T> d_x =
        make_managed_dense_vector_ptr<T>(true);

    *d_p_A = h_p_A.get();
    *d_x = h_x.get();

    cu_handle->push_scalar_mode();
    cu_handle->set_scalar_mode(false);
    dense_vector_ptr<T> d_b =
        make_managed_dense_vector_ptr<T>(h_A->m, true);

    T one = 1.0;
    T zero = 0.0;
    T_csrmv<T>(
        cu_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_p_A.get(),
        d_x.get(),
        d_b.get(),
        &one,
        &zero);
    cu_handle->pop_scalar_mode();
    CHECK_CUDA(cudaDeviceSynchronize());

    /* solve with SQMR */
    START_TIMER("Solve");
    CSRMatrixSPMV<T> spmv(cu_handle, d_p_A.get());
    SQMR<T> sqmr(cu_handle, (SPMV<T> *) &spmv, 1e-8, 1000,
        (Preconditioner<T> *) &blocky);

    dense_vector_ptr<T> d_solve =
        make_managed_dense_vector_ptr<T>(h_A->m, true);

    T residual;
    mat_int_t iterations;
    sqmr.solve(d_b.get(), d_solve.get(), residual, iterations);
    STOP_TIMER("Solve");
    PRINT_TIMER("Solve");

    /* compute error and residual */
    dense_vector_ptr<T> diff_x =
        make_managed_dense_vector_ptr<T>(true);
    *diff_x = d_solve.get();

    dense_vector_ptr<T> d_solve_b =
        make_managed_dense_vector_ptr<T>(true);
    *d_solve_b = d_b.get();

    cu_handle->push_scalar_mode();
    cu_handle->set_scalar_mode(false);

    T norm_x, norm_diff;
    T minus_one = -1.0;
    T_nrm2<T>(
        cu_handle,
        d_x.get(),
        &norm_x);
    T_axpy<T>(
        cu_handle,
        diff_x.get(),
        d_x.get(),
        &minus_one);
    T_nrm2<T>(
        cu_handle,
        diff_x.get(),
        &norm_diff);
    CHECK_CUDA(cudaDeviceSynchronize());

    T norm_b, norm_r;
    T_csrmv<T>(
        cu_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_p_A.get(),
        d_solve.get(),
        d_solve_b.get(),
        &one,
        &minus_one);
    T_nrm2<T>(
        cu_handle,
        d_solve_b.get(),
        &norm_r);
    T_nrm2<T>(
        cu_handle,
        d_b.get(),
        &norm_b);
    CHECK_CUDA(cudaDeviceSynchronize());

    cu_handle->pop_scalar_mode();

    printf("SQMR: residual %g after %d iterations\n",
        residual, iterations);
    printf("True relative error: %g\n", norm_diff / norm_x);
    printf("True relative residual: %g\n", norm_r / norm_b);
}