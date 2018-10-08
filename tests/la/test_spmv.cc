/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <iostream>

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <libs/la/spmv.cuh>
#include <libs/la/sparse_la.cuh>
#include <libs/test/test.h>

NS_CULIP_BEGIN

using namespace NS_LA;

template<typename T>
class SPMVTest : public ::testing::Test
{
public:
    using real_t = typename T::real_t;
public:
    SPMVTest()
    {

    }

    ~SPMVTest()
    {

    }

    void
    SetUp()
    {
        /* initialize cuSPARSE / cuBLAS */
        m_handles = gpu_handle_ptr(new gpu_handle_t);
        CHECK_CUBLAS(m_handles);
        CHECK_CUSPARSE(m_handles);
    }

    void
    TearDown()
    {

    }

    void
    load_problem(
        csr_matrix_ptr<real_t>& A,
        dense_vector_ptr<real_t>& d,
        dense_vector_ptr<real_t>& x_m,
        dense_vector_ptr<real_t>& x_n,
        dense_vector_ptr<real_t>& x_mn)
    {
        /* load A, d from disk and put them onto the GPU */
        csr_matrix_ptr<real_t> h_A =
            NS_TEST::Test<real_t>::read_matrix_csr("data/spmv_A.mtx", false);
        dense_vector_ptr<real_t> h_d =
            NS_TEST::Test<real_t>::read_dense_vector("data/spmv_d.mtx", 0);

        A = make_csr_matrix_ptr<real_t>(true);
        d = make_managed_dense_vector_ptr<real_t>(true);

        *A = h_A.get();
        *d = h_d.get();

        /* create a vector of 0.1 for multiplication */
        dense_vector_ptr<real_t> h_x_m =
            make_managed_dense_vector_ptr<real_t>(A->m, false);
        dense_vector_ptr<real_t> h_x_n =
            make_managed_dense_vector_ptr<real_t>(A->n, false);
        dense_vector_ptr<real_t> h_x_mn =
            make_managed_dense_vector_ptr<real_t>(A->m + A->n, false);

        std::fill(h_x_m->dense_val, h_x_m->dense_val + h_x_m->m, 0.1);
        std::fill(h_x_n->dense_val, h_x_n->dense_val + h_x_n->m, 0.1);
        std::fill(h_x_mn->dense_val, h_x_mn->dense_val + h_x_mn->m, 0.1);

        x_m = make_managed_dense_vector_ptr<real_t>(true);
        x_n = make_managed_dense_vector_ptr<real_t>(true);
        x_mn = make_managed_dense_vector_ptr<real_t>(true);

        *x_m = h_x_m.get();
        *x_n = h_x_n.get();
        *x_mn = h_x_mn.get();
    }

    void
    load_matrix(
        csr_matrix_ptr<real_t>& A,
        const char * path)
    {
        csr_matrix_ptr<real_t> h_A =
            NS_TEST::Test<real_t>::read_matrix_csr(path, false);
        A = make_csr_matrix_ptr<real_t>(true);
        *A = h_A.get();
    }

    bool
    compare_spmv(
        const SPMV<real_t> * A_spmv,
        const csr_matrix_t<real_t> * A,
        const dense_vector_t<real_t> * x,
        const bool transpose)
    {
        const mat_int_t m = A_spmv->m();
        const mat_int_t n = A_spmv->n();

        /* compute A * x (possibly) matrix-free */
        dense_vector_ptr<real_t> b_spmv = make_managed_dense_vector_ptr<real_t>(
            transpose ? n : m, true);
        A_spmv->multiply(x, b_spmv.get(), transpose);

        /* compute A * x using conventional SPMV */
        dense_vector_ptr<real_t> b = make_managed_dense_vector_ptr<real_t>(
            transpose ? n : m, true);

        m_handles->push_scalar_mode();
        m_handles->set_scalar_mode(false);
        real_t one = 1.0;
        real_t zero = 0.0;
        cusparseOperation_t transA = transpose ? CUSPARSE_OPERATION_TRANSPOSE :
            CUSPARSE_OPERATION_NON_TRANSPOSE;
        T_csrmv(m_handles, transA, A, x, b.get(), &one, &zero);
        m_handles->pop_scalar_mode();

        /* download results */
        dense_vector_ptr<real_t> h_b_spmv =
            make_managed_dense_vector_ptr<real_t>(false);
        dense_vector_ptr<real_t> h_b =
            make_managed_dense_vector_ptr<real_t>(false);

        *h_b_spmv = b_spmv.get();
        *h_b = b.get();

        return NS_TEST::Test<real_t>::compare_dense_vector(h_b_spmv.get(),
            h_b.get(), T::test_tol);
    }

protected:
    gpu_handle_ptr m_handles;
};
TYPED_TEST_CASE_P(SPMVTest);

TYPED_TEST_P(SPMVTest, CSRSPMVTest)
{
    using real_t = typename TypeParam::real_t;
    static const real_t test_tol = TypeParam::test_tol;

    /* load data */
    csr_matrix_ptr<real_t> A;
    dense_vector_ptr<real_t> d, x_m, x_n, x_mn;
    this->load_problem(A, d, x_m, x_n, x_mn);

    /* generate SPMV module */
    CSRMatrixSPMV<real_t> spmv(this->m_handles, A.get(), true);

    /* check for A * x */
    ASSERT_TRUE(this->compare_spmv(&spmv, A.get(), x_n.get(), false));

    /* check for A' * x */
    ASSERT_TRUE(this->compare_spmv(&spmv, A.get(), x_m.get(), true));
}

TYPED_TEST_P(SPMVTest, NormalMatrixSPMVTest)
{
    using real_t = typename TypeParam::real_t;
    static const real_t test_tol = TypeParam::test_tol;

    /* load data */
    csr_matrix_ptr<real_t> A, N;
    dense_vector_ptr<real_t> d, x_m, x_n, x_mn;
    this->load_problem(A, d, x_m, x_n, x_mn);
    this->load_matrix(N, "data/spmv_N.mtx");

    /* generate SPMV module */
    NormalMatrixSPMV<real_t> n_no_t(this->m_handles, A.get(), d.get(), true);

    /* check for A * x */
    ASSERT_TRUE(this->compare_spmv(&n_no_t, N.get(), x_m.get(), false));

    /* check for A' * x */
    ASSERT_TRUE(this->compare_spmv(&n_no_t, N.get(), x_n.get(), true));
}

TYPED_TEST_P(SPMVTest, AugmentedMatrixSPMVTest)
{
    using real_t = typename TypeParam::real_t;
    static const real_t test_tol = TypeParam::test_tol;

    /* load data */
    csr_matrix_ptr<real_t> A, K;
    dense_vector_ptr<real_t> d, x_m, x_n, x_mn;
    this->load_problem(A, d, x_m, x_n, x_mn);
    this->load_matrix(K, "data/spmv_K.mtx");

    /* generate SPMV module (w/o explicit transpose) */
    AugmentedMatrixSPMV<real_t> n_no_t(this->m_handles, A.get(), d.get(), true);

    /* check for A * x */
    ASSERT_TRUE(this->compare_spmv(&n_no_t, K.get(), x_mn.get(), false));

    /* check for A' * x */
    ASSERT_TRUE(this->compare_spmv(&n_no_t, K.get(), x_mn.get(), true));
}

#define TEST_CASE(name, _real_t, _test_tol) \
    struct name { \
        typedef _real_t real_t; \
        static constexpr _real_t test_tol = _test_tol; \
    };

TEST_CASE(tS, float, 1e-3);
TEST_CASE(tD, double, 1e-10);

REGISTER_TYPED_TEST_CASE_P(SPMVTest,
    CSRSPMVTest,
    NormalMatrixSPMVTest,
    AugmentedMatrixSPMVTest);
typedef ::testing::Types<
    tS,
    tD> TestTupleInstances;
INSTANTIATE_TYPED_TEST_CASE_P(SPMVMatrixTest,
    SPMVTest, TestTupleInstances);

NS_CULIP_END