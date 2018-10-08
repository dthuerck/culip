/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <iostream>

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>

#include <libs/la/iterative_solvers/sqmr.cuh>
#include <libs/la/preconditioner/import_ldu_preconditioner.cuh>
#include <libs/la/spmv.cuh>
#include <libs/la/dense_la.cuh>
#include <libs/test/test.h>

NS_CULIP_BEGIN

using namespace NS_LA;

template<typename T>
class SQMRTest : public ::testing::Test
{
public:
    using real_t = typename T::real_t;

public:
    SQMRTest()
    {

    }

    ~SQMRTest()
    {

    }

    void
    SetUp()
    {
        /* initialize cuSPARSE for each test */
        m_handles = gpu_handle_ptr(new gpu_handle_t);
        CHECK_CUBLAS(m_handles);
        CHECK_CUSPARSE(m_handles);
    }

    void
    TearDown()
    {

    }

    void
    generate_problem(
        const bool precondition,
        const char * A_path,
        const char * L_path,
        const char * D_path,
        const char * P_path,
        const char * S_path,
        csr_matrix_ptr<real_t>& A,
        csr_matrix_ptr<real_t>& L,
        dense_vector_ptr<real_t>& D,
        dense_vector_ptr<mat_int_t>& D_is_piv,
        dense_vector_ptr<mat_int_t>& P,
        dense_vector_ptr<real_t>& S,
        dense_vector_ptr<real_t>& x,
        dense_vector_ptr<real_t>& b,
        real_t& norm_b)
    {
        /* read matrix on host */
        csr_matrix_ptr<real_t> h_A =
            NS_TEST::Test<real_t>::read_matrix_csr(A_path);
        csr_matrix_ptr<real_t> h_L = precondition ?
            NS_TEST::Test<real_t>::read_matrix_csr(L_path) :
            make_csr_matrix_ptr<real_t>(false);
        csr_matrix_ptr<real_t> h_csr_D = precondition ?
            NS_TEST::Test<real_t>::read_matrix_csr(D_path) :
            make_csr_matrix_ptr<real_t>(false);

        /* read vectors on host */
        dense_vector_ptr<mat_int_t> h_P = precondition ?
            NS_TEST::Test<mat_int_t>::read_dense_vector(P_path) :
            make_managed_dense_vector_ptr<mat_int_t>(false);
        dense_vector_ptr<real_t> h_S = precondition ?
            NS_TEST::Test<real_t>::read_dense_vector(S_path) :
            make_managed_dense_vector_ptr<real_t>(false);

        /* generate RHS on host & compute b */
        dense_vector_ptr<real_t> h_x = make_managed_dense_vector_ptr<real_t>(
            h_A->n, false);
        std::fill(h_x->dense_val, h_x->dense_val + h_x->m, 1.0);

        dense_vector_ptr<real_t> h_b =
            NS_TEST::TestLA<real_t>::mat_vec_multiply(h_A.get(), h_x.get(),
            m_handles);

        /* convert D to tridiagonal form w/1x1 and 2x2 pivots */
        const mat_int_t n = h_A->n;
        dense_vector_ptr<real_t> h_D =
            make_managed_dense_vector_ptr<real_t>(3 * n, false);
        dense_vector_ptr<mat_int_t> h_D_is_piv =
            make_managed_dense_vector_ptr<mat_int_t>(n, false);
        if(precondition)
        {
            std::fill(h_D->dense_val, h_D->dense_val + 3 * n, 0);
            std::fill(h_D_is_piv->dense_val, h_D_is_piv->dense_val + n, 1);

            mat_int_t i = 0;
            while(i < n)
            {
                const mat_int_t i_len = h_csr_D->csr_row[i + 1] -
                    h_csr_D->csr_row[i];

                const real_t * i_val = h_csr_D->csr_val + h_csr_D->csr_row[i];

                if(i_len == 1)
                {
                    /* 1x1 pivot */
                    if(i > 0)
                        (*h_D)[3 * (i - 1) + 2] = 0;
                    (*h_D)[3 * i + 1] = i_val[0];
                    if(i < n - 1)
                        (*h_D)[3 * (i + 1)] = 0;

                    ++i;
                }
                else
                {
                    /* 2x2 pivot */
                    const real_t * ip_val = h_csr_D->csr_val +
                        h_csr_D->csr_row[i + 1];

                    (*h_D)[3 * i] = 0;
                    (*h_D)[3 * i + 1] = i_val[0];
                    (*h_D)[3 * i + 2] = i_val[1];

                    (*h_D)[3 * (i + 1) + 0] = ip_val[0];
                    (*h_D)[3 * (i + 1) + 1] = ip_val[1];
                    (*h_D)[3 * (i + 1) + 2] = 0;

                    (*h_D_is_piv)[i + 1] = 0;

                    i += 2;
                }
            }
        }

        /* transfer data to GPU */
        A = make_csr_matrix_ptr<real_t>(true);
        L = make_csr_matrix_ptr<real_t>(true);
        D = make_managed_dense_vector_ptr<real_t>(true);
        D_is_piv = make_managed_dense_vector_ptr<mat_int_t>(true);
        P = make_managed_dense_vector_ptr<mat_int_t>(true);
        S = make_managed_dense_vector_ptr<real_t>(true);
        x = make_managed_dense_vector_ptr<real_t>(true);
        b = make_managed_dense_vector_ptr<real_t>(true);

        *A = h_A.get();
        if(precondition)
            *L = h_L.get();
        if(precondition)
        {
            *D = h_D.get();
            *D_is_piv = h_D_is_piv.get();
        }
        if(precondition)
            *P = h_P.get();
        if(precondition)
            *S = h_S.get();

        *x = h_x.get();
        *b = h_b.get();

        /* compute b's norm */
        this->m_handles->push_scalar_mode();
        this->m_handles->set_scalar_mode(false);
        T_nrm2(this->m_handles, b.get(), &norm_b);
        this->m_handles->pop_scalar_mode();
    }

protected:
    gpu_handle_ptr m_handles;
};
TYPED_TEST_CASE_P(SQMRTest);

TYPED_TEST_P(SQMRTest, TestSolve)
{
    using real_t = typename TypeParam::real_t;
    static const real_t term_tol = TypeParam::term_tol;
    static const real_t test_tol = TypeParam::test_tol;
    static const char * prefix_path = TypeParam::prefix_path;
    static const bool precondition = TypeParam::precondition;

    std::string A_path(prefix_path); A_path += "A.mtx";
    std::string L_path(prefix_path); L_path += "prec_L.mtx";
    std::string D_path(prefix_path); D_path += "prec_D.mtx";
    std::string p_path(prefix_path); p_path += "prec_p.mtx";
    std::string s_path(prefix_path); s_path += "prec_s.mtx";

    /* create data */
    csr_matrix_ptr<real_t> A, L;
    dense_vector_ptr<real_t> x, b, D, S;
    dense_vector_ptr<mat_int_t> D_is_piv, P;
    real_t nrm_b;
    this->generate_problem(
            precondition,
            A_path.c_str(),
            L_path.c_str(),
            D_path.c_str(),
            p_path.c_str(),
            s_path.c_str(),
            A, L, D, D_is_piv, P, S, x, b, nrm_b);

    /* construct spmv module */
    CSRMatrixSPMV<real_t> spmv(this->m_handles, A.get(), true);

    /* load an incomplete Cholesky preconditioner if desired */
    Preconditioner<real_t> * prec = nullptr;
    if(precondition)
    {
        ImportLDUPreconditioner<real_t> * ldu =
            new ImportLDUPreconditioner<real_t>(this->m_handles, L.get(),
            D.get(), D_is_piv.get(), true);

        /* add permutation / scale if prec was computed on p/s matrix */
        if(precondition)
        {
            ldu->set_permutation(P.get(), P.get());
            ldu->set_scaling(S.get(), S.get());
        }

        prec = ldu;
    }

    /* solve system by SQMR */
    dense_vector_ptr<real_t> mr_x =
        make_managed_dense_vector_ptr<real_t>(A->n, true);
    SQMR<real_t> sqmr(this->m_handles, &spmv, term_tol, A->n,
        prec);

    real_t residual;
    mat_int_t iterations;
    sqmr.solve(b.get(), mr_x.get(), residual, iterations);

    /* compute (real) residual */
    dense_vector_ptr<real_t> Ax = make_managed_dense_vector_ptr<real_t>(true);
    *Ax = b.get();

    this->m_handles->push_scalar_mode();
    this->m_handles->set_scalar_mode(false);
    const real_t alpha = -1.0;
    const real_t beta = 1.0;
    spmv.multiply(mr_x.get(), Ax.get(), &alpha, &beta, false);

    real_t real_res;
    T_nrm2(this->m_handles, Ax.get(), &real_res);
    this->m_handles->pop_scalar_mode();

    /* check residual and iteration count */
    ASSERT_LE(real_res, test_tol * nrm_b);
    ASSERT_LE(iterations, A->n);

    /* clean up */
    if(prec != nullptr)
        delete prec;
}

#define TEST_CASE(name, _real_t, _term_tol, _test_tol, _prefix_path, _prec) \
    struct name { \
        typedef _real_t real_t; \
        static constexpr _real_t term_tol = _term_tol; \
        static constexpr _real_t test_tol = _test_tol; \
        static constexpr char * prefix_path = (char*) _prefix_path; \
        static const bool precondition = _prec; \
    };

TEST_CASE(small_S, float, 1e-6, 1e-4, "data/sqmr_small_", false);
TEST_CASE(small_D, double, 1e-10, 1e-5, "data/sqmr_small_", false);

TEST_CASE(small_S_P, float, 1e-6, 1e-4, "data/sqmr_small_", true);
TEST_CASE(small_D_P, double, 1e-10, 1e-5, "data/sqmr_small_", true);

TEST_CASE(big_S_P, float, 1e-6, 1e-4, "data/sqmr_big_", true);
TEST_CASE(big_D_P, double, 1e-10, 1e-5, "data/sqmr_big_", true);

TEST_CASE(huge_S_P, float, 1e-6, 1e-4, "data/sqmr_huge_", true);
TEST_CASE(huge_D_P, double, 1e-10, 1e-5, "data/sqmr_huge_", true);

REGISTER_TYPED_TEST_CASE_P(SQMRTest,
    TestSolve);
typedef ::testing::Types<
    small_S,
    small_D,
    small_S_P,
    small_D_P,
    big_S_P,
    big_D_P,
    huge_S_P,
    huge_D_P
    > TestTupleInstances;
INSTANTIATE_TYPED_TEST_CASE_P(SolveTest,
    SQMRTest, TestTupleInstances);

NS_CULIP_END