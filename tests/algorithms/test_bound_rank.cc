/**
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>
#include <libs/test/test.h>

#include <libs/algorithms/sprank.cuh>
#include <libs/algorithms/permute_btf.h>

NS_CULIP_BEGIN
NS_TEST_BEGIN

using namespace NS_ALGORITHMS;

template<typename T>
class BoundRankTest : public ::testing::Test
{
public:
    BoundRankTest()
    {

    }

    ~BoundRankTest()
    {

    }

    void
    SetUp()
    {
        printf("SETUP!!!!\n");
    }

    void
    TearDown()
    {

    }
};
TYPED_TEST_CASE_P(BoundRankTest);

TYPED_TEST_P(BoundRankTest, TestFindRankBounds)
{
    static const char * path = TypeParam::path;
    static const mat_int_t lb = TypeParam::lb;
    static const mat_int_t ub = TypeParam::ub;

    /* load matrix on host */
    csr_matrix_ptr<double> A, At;
    printf("Reading data....\n");
    A = NS_TEST::Test<double>::read_matrix_csr(path, false);
    At = NS_TEST::Test<double>::read_matrix_csr(path, true);
    printf("Read data....\n");

    /* compute lower bounds (permute_btf) */
    std::vector<mat_int_t> row_order(A->m);
    std::vector<mat_int_t> col_order(A->n);
    BTF btf;
    mat_int_t t_lb = btf.permute(A->m, A->n, At->csr_row, At->csr_col,
        A->csr_row, A->csr_col, row_order.data(), col_order.data());
    ASSERT_EQ(t_lb, lb);

    /* compute upper bounds (sprank) */
    mat_int_t t_ub = sprank(A->m, A->n, A->csr_row, A->csr_col);
    ASSERT_EQ(t_ub, ub);
}

/* instantiate for types and matrix file paths */
#define TEST_CASE(name, _path, _lb, _ub) \
    struct name { \
        static constexpr char * path = (char*) _path; \
        static constexpr mat_int_t lb = _lb; \
        static constexpr mat_int_t ub = _ub; \
    };

TEST_CASE(rank00, "data/bound_rank_51_800.mtx", 51, 800);
TEST_CASE(rank01, "data/bound_rank_920_930.mtx", 920, 930);
TEST_CASE(rank02, "data/bound_rank_4380_4380.mtx", 4380, 4380);

typedef ::testing::Types<
    rank00,
    rank01,
    rank02
    > TestTupleInstances;
REGISTER_TYPED_TEST_CASE_P(BoundRankTest,
    TestFindRankBounds);
INSTANTIATE_TYPED_TEST_CASE_P(TestBoundRank,
    BoundRankTest, TestTupleInstances);

NS_TEST_END
NS_CULIP_END