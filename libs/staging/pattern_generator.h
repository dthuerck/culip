/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_STAGING_PATTERN_GENERATOR_H_
#define __CULIP_STAGING_PATTERN_GENERATOR_H_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>

#include <libs/staging/triangular.h>
#include <libs/staging/block_diagonal.h>
#include <libs/staging/flexible_triangular.h>
#include <libs/staging/elimination_tree.h>

#include <atomic>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

template<typename T>
class PatternGenerator
{
public:
    PatternGenerator();
    virtual ~PatternGenerator();

    /* one-shot pattern generation, no static pivoting */
    virtual Triangular_ptr<T> compute_pattern(
        const csr_matrix_t<T> * A, const mat_int_t num_piv_starts,
        const mat_int_t * piv_starts) = 0;

    /* row-wise pattern generation with pivoting (1x1, 2x2) */
    virtual void init_pivot_pattern(const csr_matrix_t<T> * A) = 0;
    virtual void pivot_1x1(const mat_int_t cur_row, const mat_int_t piv_a) = 0;
    virtual void pivot_2x2(const mat_int_t cur_row, const mat_int_t piv_a,
        const mat_int_t piv_b) = 0;
    virtual mat_int_t row_pattern(const mat_int_t row, mat_int_t * buf_ix) = 0;

protected:
    const csr_matrix_t<T> * m_A;
    std::vector<mat_int_t> m_piv_starts;
};

template<typename T>
using PatternGenerator_ptr = std::unique_ptr<PatternGenerator<T>>;

/* ************************************************************************** */

template<typename T>
class ZeroFillInPattern : public PatternGenerator<T>
{
public:
    ZeroFillInPattern();
    ~ZeroFillInPattern();

    /* one-shot pattern generation, no static pivoting */
    Triangular_ptr<T> compute_pattern(const csr_matrix_t<T> * A,
        const mat_int_t num_piv_starts, const mat_int_t * piv_starts);

    /* row-wise pattern generation with pivoting (1x1, 2x2) */
    virtual void init_pivot_pattern(const csr_matrix_t<T> * A);
    virtual void pivot_1x1(const mat_int_t cur_row, const mat_int_t piv_a);
    virtual void pivot_2x2(const mat_int_t cur_row, const mat_int_t piv_a,
        const mat_int_t piv_b);
    virtual mat_int_t row_pattern(const mat_int_t row, mat_int_t * buf_ix);

protected:
    FlexibleTriangular_ptr<T> m_pL;
    std::vector<mat_int_t> m_is_piv;
};

/* ************************************************************************** */

template<typename T>
class ExactPattern : public PatternGenerator<T>
{
public:
    ExactPattern();
    ~ExactPattern();

    /* row-wise pattern generation with pivoting (1x1, 2x2) */
    Triangular_ptr<T> compute_pattern(const csr_matrix_t<T> * A,
        const mat_int_t num_piv_starts, const mat_int_t * piv_starts);

    /* row-wise pattern generation with pivoting (1x1, 2x2) */
    virtual void init_pivot_pattern(const csr_matrix_t<T> * A);
    virtual void pivot_1x1(const mat_int_t cur_row, const mat_int_t piv_a);
    virtual void pivot_2x2(const mat_int_t cur_row, const mat_int_t piv_a,
        const mat_int_t piv_b);
    virtual mat_int_t row_pattern(const mat_int_t row, mat_int_t * buf_ix);

protected:
    EliminationTree_ptr<T> m_etree;
};

/* ************************************************************************** */

/**
 * Note: assume that A is a full, symmetric matrix
 */
template<typename T>
class LevelPattern : public PatternGenerator<T>
{
public:
    LevelPattern(const mat_int_t level);
    ~LevelPattern();

    /* row-wise pattern generation with pivoting (1x1, 2x2) */
    Triangular_ptr<T> compute_pattern(const csr_matrix_t<T> * A,
        const mat_int_t num_piv_starts, const mat_int_t * piv_starts);

    /* row-wise pattern generation with pivoting (1x1, 2x2) */
    virtual void init_pivot_pattern(const csr_matrix_t<T> * A);
    virtual void pivot_1x1(const mat_int_t cur_row, const mat_int_t piv_a);
    virtual void pivot_2x2(const mat_int_t cur_row, const mat_int_t piv_a,
        const mat_int_t piv_b);
    virtual mat_int_t row_pattern(const mat_int_t row, mat_int_t * buf_ix);

protected:
    mat_int_t m_level;

    std::vector<mat_int_t> m_row_starts;
    std::vector<mat_int_t> m_row_ix;

    /* permuted input matrix */
    FlexibleTriangular_ptr<T> m_pL;

    /* save pivot order (for 2x2 stuff) */
    std::vector<mat_int_t> m_is_piv;
};

/* ************************************************************************** */

template<typename T>
class BlockRestrictedPattern : public PatternGenerator<T>
{
public:
    BlockRestrictedPattern(const mat_int_t m, const csr_matrix_t<T> * coarse,
        const mat_int_t num_blocks, const mat_int_t * block_starts);
    ~BlockRestrictedPattern();

    /* row-wise pattern generation with pivoting (1x1, 2x2) */
    Triangular_ptr<T> compute_pattern(const csr_matrix_t<T> * A,
        const mat_int_t num_piv_starts, const mat_int_t * piv_starts);

    /* row-wise pattern generation with pivoting (1x1, 2x2) */
    virtual void init_pivot_pattern(const csr_matrix_t<T> * A);
    virtual void pivot_1x1(const mat_int_t cur_row, const mat_int_t piv_a);
    virtual void pivot_2x2(const mat_int_t cur_row, const mat_int_t piv_a,
        const mat_int_t piv_b);
    virtual mat_int_t row_pattern(const mat_int_t row, mat_int_t * buf_ix);

protected:
    void create_block_map();

protected:
    /* for output */
    std::vector<mat_int_t> m_row_starts;
    std::vector<mat_int_t> m_row_ix;

    /* for coarse / blocked pattern restriction */
    const mat_int_t m_m;
    const csr_matrix_t<T> * m_coarse;
    const mat_int_t m_num_blocks;
    const mat_int_t * m_block_starts;

    std::vector<mat_int_t> m_block_map;

    /* permuted & filtered fine input matrix */
    FlexibleTriangular_ptr<T> m_pL;
    BlockDiagonal_ptr<T> m_pD;

    /* save pivot order (for 2x2 stuff) */
    std::vector<mat_int_t> m_is_piv;
};

/* ************************************************************************** */

template<typename T>
class BlockRestrictedExactPattern : public PatternGenerator<T>
{
public:
    BlockRestrictedExactPattern(const mat_int_t m,
        const csr_matrix_t<T> * coarse, const mat_int_t num_blocks,
        const mat_int_t * block_starts);
    ~BlockRestrictedExactPattern();

    /* row-wise pattern generation with pivoting (1x1, 2x2) */
    Triangular_ptr<T> compute_pattern(const csr_matrix_t<T> * A,
        const mat_int_t num_piv_starts, const mat_int_t * piv_starts);

    /* row-wise pattern generation with pivoting (1x1, 2x2) */
    virtual void init_pivot_pattern(const csr_matrix_t<T> * A);
    virtual void pivot_1x1(const mat_int_t cur_row, const mat_int_t piv_a);
    virtual void pivot_2x2(const mat_int_t cur_row, const mat_int_t piv_a,
        const mat_int_t piv_b);
    virtual mat_int_t row_pattern(const mat_int_t row, mat_int_t * buf_ix);

protected:
    void create_dense_block_row(const mat_int_t block_row, char * dense_row);
    mat_int_t filter_row(const mat_int_t row, const mat_int_t row_len,
        const mat_int_t * in_row_ix, mat_int_t * out_row_ix,
        const char * dense_map = nullptr);

protected:
    EliminationTree_ptr<T> m_etree;

    /* for coarse / blocked pattern restriction */
    const mat_int_t m_m;
    const csr_matrix_t<T> * m_coarse;
    const mat_int_t m_num_blocks;
    const mat_int_t * m_block_starts;
    std::vector<mat_int_t> m_rowcol_in_block;
};

NS_STAGING_END
NS_CULIP_END

#endif /* __CULIP_STAGING_PATTERN_GENERATOR_H_ */