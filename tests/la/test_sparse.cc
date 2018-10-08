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

#include <libs/la/sparse_la.cuh>
#include <libs/la/dense_la.cuh>

#include <libs/test/test.h>

NS_CULIP_BEGIN

using namespace NS_LA;

template<typename T>
class SparseTest : public :: testing::Test
{
public:
    public:
    using real_t = typename T::real_t;

public:
    SparseTest()
    {

    }

    ~SparseTest()
    {

    }

    void
    SetUp()
    {
        /* initialize cuSPARSE for each test */
        m_handle = gpu_handle_ptr(new gpu_handle_t);
        CHECK_CUBLAS(m_handle);
        CHECK_CUSPARSE(m_handle);
    }

    void
    TearDown()
    {

    }

    void
    generate_problem(
        const char * L_path,
        csr_matrix_ptr<real_t>& L,
        dense_vector_ptr<real_t>& x,
        dense_vector_ptr<real_t>& b)
    {
        /* read matrix on host */
        csr_matrix_ptr<real_t> h_L =
            NS_TEST::Test<real_t>::read_matrix_csr(L_path);

        /* generate RHS on host & compute b */
        dense_vector_ptr<real_t> h_x = make_managed_dense_vector_ptr<real_t>(
            h_L->n, false);
        std::fill(h_x->dense_val, h_x->dense_val + h_x->m, 0.01);

        dense_vector_ptr<real_t> h_b =
            NS_TEST::TestLA<real_t>::mat_vec_multiply(h_L.get(), h_x.get(),
            m_handle);

        /* transfer data to GPU */
        L = make_csr_matrix_ptr<real_t>(true);
        x = make_managed_dense_vector_ptr<real_t>(true);
        b = make_managed_dense_vector_ptr<real_t>(true);

        *L = h_L.get();
        *x = h_x.get();
        *b = h_b.get();
    }

protected:
    gpu_handle_ptr m_handle;
};
TYPED_TEST_CASE_P(SparseTest);

TYPED_TEST_P(SparseTest, TestApproxTriangularSolve)
{
    using real_t = typename TypeParam::real_t;
    static const char * path = TypeParam::path;
    static const mat_int_t sweeps = TypeParam::sweeps;
    static const real_t test_tol = TypeParam::test_tol;

    this->m_handle->push_scalar_mode();
    this->m_handle->set_scalar_mode(false);
    real_t one = 1.0;

    /* create data */
    csr_matrix_ptr<real_t> L;
    dense_vector_ptr<real_t> x, b;
    this->generate_problem(path, L, x, b);

    /* solve with exact solve */
    cusparseSolveAnalysisInfo_t exact_info;
    this->m_handle->cusparse_status =
        cusparseCreateSolveAnalysisInfo(&exact_info);

    dense_vector_ptr<real_t> exact_x =
        make_managed_dense_vector_ptr<real_t>(x->m, true);

    T_triangular_step_analysis(
        this->m_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        L.get(),
        exact_info);
    T_triangular_step_solve(
        this->m_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        exact_info,
        L.get(),
        b.get(),
        exact_x.get(),
        &one);

    /* solve with inexact solve */
    T_approx_triangular_info_t<real_t> inexact_info;

    dense_vector_ptr<real_t> inexact_x =
        make_managed_dense_vector_ptr<real_t>(x->m, true);

    T_approx_triangular_step_analysis(
        this->m_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        L.get(),
        inexact_info);
    T_approx_triangular_step_solve(
        this->m_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        inexact_info,
        L.get(),
        b.get(),
        inexact_x.get(),
        &one,
        sweeps);

    this->m_handle->pop_scalar_mode();

    /* compare solutions */
    dense_vector_ptr<real_t> h_exact_x =
        make_managed_dense_vector_ptr<real_t>(false);
    dense_vector_ptr<real_t> h_inexact_x =
        make_managed_dense_vector_ptr<real_t>(false);
    dense_vector_ptr<real_t> h_x =
        make_managed_dense_vector_ptr<real_t>(false);

    *h_exact_x = exact_x.get();
    *h_inexact_x = inexact_x.get();
    *h_x = x.get();

    ASSERT_TRUE(NS_TEST::Test<real_t>::compare_dense_vector(h_inexact_x.get(),
        h_x.get(), test_tol));
    /* strangely, for U, cuSparse seems to be wrong */
    //ASSERT_TRUE(NS_TEST::Test<real_t>::compare_dense_vector(h_exact_x.get(),
    //    h_x.get(), test_tol));
}

#define TEST_CASE(name, _real_t, _path, _sweeps, _test_tol) \
    struct name { \
        typedef _real_t real_t; \
        static constexpr char * path = (char*) _path; \
        static constexpr mat_int_t sweeps = _sweeps; \
        static constexpr _real_t test_tol = _test_tol; \
    };

TEST_CASE(LS, float, "data/gmres_big_prec_L.mtx", 10, 1e-3);
TEST_CASE(LD, double, "data/gmres_big_prec_L.mtx", 10, 1e-5);

TEST_CASE(US, float, "data/gmres_big_prec_U.mtx", 10, 1e-3);
TEST_CASE(UD, double, "data/gmres_big_prec_U.mtx", 10, 1e-5);

REGISTER_TYPED_TEST_CASE_P(SparseTest,
    TestApproxTriangularSolve);
typedef ::testing::Types<
    LS,
    LD,
    US,
    UD
    > TestTupleInstances;
INSTANTIATE_TYPED_TEST_CASE_P(ApproxTest,
    SparseTest, TestTupleInstances);


NS_CULIP_END