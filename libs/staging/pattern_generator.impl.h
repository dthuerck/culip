/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */


#include <libs/staging/pattern_generator.h>
#include <libs/staging/elimination_tree.h>

#include <set>
#include <cstdio>
#include <iostream>
#include <queue>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

/**
 * *****************************************************************************
 * **************************** PatternGenerator *******************************
 * *****************************************************************************
 */

template<typename T>
PatternGenerator<T>::
PatternGenerator()
{
}

/* ************************************************************************** */

template<typename T>
PatternGenerator<T>::
~PatternGenerator()
{
}

/**
 * *****************************************************************************
 * *************************** ZeroFillInPattern *******************************
 * *****************************************************************************
 */

template<typename T>
ZeroFillInPattern<T>::
ZeroFillInPattern()
: PatternGenerator<T>()
{
}

/* ************************************************************************** */

template<typename T>
ZeroFillInPattern<T>::
~ZeroFillInPattern()
{
}

/* ************************************************************************** */

template<typename T>
Triangular_ptr<T>
ZeroFillInPattern<T>::
compute_pattern(
    const csr_matrix_t<T> * A,
    const mat_int_t num_piv_starts,
    const mat_int_t * piv_starts)
{
    /* save data */
    this->m_A = A;
    this->m_piv_starts.assign(piv_starts, piv_starts + num_piv_starts);

    /* remove remove subdiagonal entries in block pivots */
    std::vector<mat_int_t> sub_m(this->m_A->m);
    for(mat_int_t i = 0; i < this->m_piv_starts.size(); ++i)
    {
        const mat_int_t piv_start = this->m_piv_starts[i];
        const mat_int_t piv_end = (i < this->m_piv_starts.size() - 1 ?
            this->m_piv_starts[i + 1] : this->m_A->m);

        for(mat_int_t j = piv_start; j < piv_end; ++j)
            sub_m[j] = piv_start;
    }

    /* determine row sizes of lower triangular */
    std::vector<mat_int_t> row_sizes(this->m_A->m, 0);

    mat_int_t nnz = 0;
    for(mat_int_t i = 0; i < this->m_A->m; ++i)
    {
        const mat_int_t i_len = this->m_A->csr_row[i + 1] -
            this->m_A->csr_row[i];
        const mat_int_t * i_col = this->m_A->csr_col + this->m_A->csr_row[i];

        for(mat_int_t j = 0; j < i_len && i_col[j] < sub_m[i]; ++j)
            ++row_sizes[i];

        /* plus one for the diagonal */
        ++row_sizes[i];

        nnz += row_sizes[i];
    }

    /* allocate output layout and copy data */
    Triangular_ptr<T> L = Triangular_ptr<T>(
            new Triangular<T>(this->m_A->m, nnz));

    mat_int_t * L_csr_row = L->raw_row_ptr();
    mat_int_t * L_csr_col = L->raw_col_ptr();

    mat_int_t offset = 0;
    L_csr_row[0] = 0;
    for(mat_int_t i = 0; i < this->m_A->m; ++i)
    {
        const mat_int_t * i_col = this->m_A->csr_col + this->m_A->csr_row[i];

        for(mat_int_t j = 0; j < row_sizes[i] - 1; ++j)
            L_csr_col[offset++] = i_col[j];

        /* diagonal entry */
        L_csr_col[offset++] = i;

        L_csr_row[i + 1] = offset;
    }

    return L;
}

/* ************************************************************************** */

template<typename T>
void
ZeroFillInPattern<T>::
init_pivot_pattern(
    const csr_matrix_t<T> * A)
{
    m_pL = FlexibleTriangular_ptr<T>(new FlexibleTriangular<T>(A));
    m_is_piv = std::vector<mat_int_t>(A->m, 1);
}

/* ************************************************************************** */

template<typename T>
void
ZeroFillInPattern<T>::
pivot_1x1(
    const mat_int_t cur_row,
    const mat_int_t piv_a)
{
    m_pL->pivot(cur_row, piv_a);

    m_is_piv[cur_row] = 1;
}

/* ************************************************************************** */

template<typename T>
void
ZeroFillInPattern<T>::
pivot_2x2(
    const mat_int_t cur_row,
    const mat_int_t piv_a,
    const mat_int_t piv_b)
{
    m_pL->pivot(cur_row, piv_a);
    m_pL->pivot(cur_row + 1, piv_b);

    m_is_piv[cur_row] = 1;
    m_is_piv[cur_row + 1] = 0;
}

/* ************************************************************************** */

template<typename T>
mat_int_t
ZeroFillInPattern<T>::
row_pattern(
    const mat_int_t row,
    mat_int_t * buf_ix)
{
    mat_int_t len = 0;

    /* pattern: row of pL (- subdiagonal element for 2x2 pivot) */
    mat_int_t * pL_ix;
    T * pL_val;
    const mat_int_t pL_len = m_pL->row(row, pL_ix, pL_val);

    for(mat_int_t i = 0; i < pL_len; ++i)
    {
        if(m_is_piv[row] || pL_ix[i] != row - 1)
        {
            buf_ix[len] = pL_ix[i];
            ++len;
        }
    }

    return len;
}

/**
 * *****************************************************************************
 * ****************************** ExactPattern *********************************
 * *****************************************************************************
 */

template<typename T>
ExactPattern<T>::
ExactPattern()
: PatternGenerator<T>()
{
}

/* ************************************************************************** */

template<typename T>
ExactPattern<T>::
~ExactPattern()
{
}

/* ************************************************************************** */

template<typename T>
Triangular_ptr<T>
ExactPattern<T>::
compute_pattern(
    const csr_matrix_t<T> * A,
    const mat_int_t num_piv_starts,
    const mat_int_t * piv_starts)
{
    /* save data */
    this->m_A = A;
    this->m_piv_starts.assign(piv_starts, piv_starts + num_piv_starts);

    m_etree = EliminationTree_ptr<T>(new EliminationTree<T>(this->m_A->m,
        this->m_piv_starts.size(), this->m_piv_starts.data()));
    Triangular_ptr<T> L = m_etree->extract_pattern(this->m_A);

    return L;
}

/* ************************************************************************** */

template<typename T>
void
ExactPattern<T>::
init_pivot_pattern(
    const csr_matrix_t<T> * A)
{
    this->m_A = A;

    m_etree = EliminationTree_ptr<T>(new EliminationTree<T>(this->m_A->m));
    m_etree->init_pivot_pattern(A);
}

/* ************************************************************************** */

template<typename T>
void
ExactPattern<T>::
pivot_1x1(
    const mat_int_t cur_row,
    const mat_int_t piv_a)
{
    m_etree->pivot_1x1(cur_row, piv_a);
}

/* ************************************************************************** */

template<typename T>
void
ExactPattern<T>::
pivot_2x2(
    const mat_int_t cur_row,
    const mat_int_t piv_a,
    const mat_int_t piv_b)
{
    m_etree->pivot_2x2(cur_row, piv_a, piv_b);
}

/* ************************************************************************** */

template<typename T>
mat_int_t
ExactPattern<T>::
row_pattern(
    const mat_int_t row,
    mat_int_t * buf_ix)
{
    return m_etree->row_pattern(row, buf_ix);
}

/**
 * *****************************************************************************
 * ******************************* LevelPattern ********************************
 * *****************************************************************************
 */

template<typename T>
LevelPattern<T>::
LevelPattern(
    const mat_int_t level)
: m_level(std::max(level, 0)),
  PatternGenerator<T>()
{

}

/* ************************************************************************** */

template<typename T>
LevelPattern<T>::
~LevelPattern<T>()
{

}

/* ************************************************************************** */

template<typename T>
Triangular_ptr<T>
LevelPattern<T>::
compute_pattern(
    const csr_matrix_t<T> * A,
    const mat_int_t num_piv_starts,
    const mat_int_t * piv_starts)
{
    /* save data */
    this->m_A = A;
    m_pL = FlexibleTriangular_ptr<T>(new FlexibleTriangular<T>(A));

    /* save pivots */
    m_is_piv.resize(A->m);
    std::fill(m_is_piv.begin(), m_is_piv.end(), 0);
    for(mat_int_t i = 0; i < num_piv_starts; ++i)
        m_is_piv[piv_starts[i]] = 1;

    /* use BFS to find fill-paths (levels after sum rule) */
    std::vector<mat_int_t> L_csr_col;

    std::vector<mat_int_t> lvl_lens(A->m + 1);

    #pragma omp parallel for
    for(mat_int_t i = 0; i < A->m; ++i)
    {
        std::vector<mat_int_t> buf(i + 1);
        lvl_lens[i] = row_pattern(i, buf.data());
    }
    lvl_lens[A->m] = 0;

    /* compute offsets */
    mat_int_t hold = lvl_lens[0];
    lvl_lens[0] = 0;
    for(mat_int_t i = 1; i < A->m + 1; ++i)
    {
        const mat_int_t res = lvl_lens[i - 1] + hold;
        hold = lvl_lens[i];
        lvl_lens[i] = res;
    }

    const mat_int_t nnz = lvl_lens[A->m];
    L_csr_col.resize(nnz);

    #pragma omp parallel for
    for(mat_int_t i = 0; i < A->m; ++i)
    {
        row_pattern(i, L_csr_col.data() + lvl_lens[i]);
    }

    /* import pattern into triangular matrix */
    Triangular_ptr<T> L = Triangular_ptr<T>(new Triangular<T>(A->m, nnz));
    std::copy(lvl_lens.begin(), lvl_lens.end(), L->raw_row_ptr());
    std::copy(L_csr_col.begin(), L_csr_col.end(), L->raw_col_ptr());
    std::fill(L->raw_val_ptr(), L->raw_val_ptr() + nnz, 1.0);

    return L;
}

/* ************************************************************************** */

template<typename T>
void
LevelPattern<T>::
init_pivot_pattern(
    const csr_matrix_t<T> * A)
{
    this->m_A = A;
    m_pL = FlexibleTriangular_ptr<T>(new FlexibleTriangular<T>(A));

    /* initialize with 1x1 pivots */
    m_is_piv.resize(A->m);
    std::fill(m_is_piv.begin(), m_is_piv.end(), 1);
}

/* ************************************************************************** */

template<typename T>
void
LevelPattern<T>::
pivot_1x1(
    const mat_int_t cur_row,
    const mat_int_t piv_a)
{
    /* pivot source matrix */
    m_pL->pivot(cur_row, piv_a);

    /* mark cur_row as 1x1 pivot */
    m_is_piv[cur_row] = 1;
}

/* ************************************************************************** */

template<typename T>
void
LevelPattern<T>::
pivot_2x2(
    const mat_int_t cur_row,
    const mat_int_t piv_a,
    const mat_int_t piv_b)
{
    /* pivot source matrix */
    m_pL->pivot(cur_row, piv_a);
    m_pL->pivot(cur_row + 1, piv_b);

    /* mark cur_row as 2x2 pivot */
    m_is_piv[cur_row] = 1;
    m_is_piv[cur_row + 1] = 0;
}

/* ************************************************************************** */

template<typename T>
mat_int_t
LevelPattern<T>::
row_pattern(
    const mat_int_t row,
    mat_int_t * buf_ix)
{
    /**
     * Note: in the factorization, we solve L_11 (D_11 x)), hence the level
     * fill computes L's fill-in first and then cares for D, i.e. 2x2 pivots
     */
    const mat_int_t sub_m = (m_is_piv[row] == 0 ? (row - 1) : row);

    mat_int_t nz_len = 0;

    std::vector<mat_int_t> level(m_pL->m(), m_pL->m());
    std::vector<mat_int_t> lpred(m_pL->m(), m_pL->m());
    std::vector<bool> added(m_pL->m(), false);

    std::queue<mat_int_t> bfs;

    mat_int_t * cur_ix;
    T * cur_val;
    mat_int_t cur_len;
    lpred[row] = -1;
    level[row] = -1;
    buf_ix[nz_len++] = row;
    added[row] = true;

    cur_len = m_pL->row(row, cur_ix, cur_val);
    for(mat_int_t j = 0; j < cur_len; ++j)
    {
        if(cur_ix[j] < sub_m)
        {
            bfs.push(cur_ix[j]);
            level[cur_ix[j]] = 0;
            lpred[cur_ix[j]] = cur_ix[j];
        }
    }

    auto level_step = [&](const mat_int_t cur, const mat_int_t next_pred,
        const mat_int_t next_level)
    {
        cur_len = m_pL->row(cur, cur_ix, cur_val);
        for(mat_int_t j = 0; j < cur_len; ++j)
        {
            if(cur_ix[j] < cur && next_level <= m_level &&
                next_pred < lpred[cur_ix[j]])
            {
                bfs.push(cur_ix[j]);
                level[cur_ix[j]] = next_level;
                lpred[cur_ix[j]] = next_pred;
            }
        }

        cur_len = m_pL->col(cur, cur_ix);
        for(mat_int_t j = 0; j < cur_len; ++j)
        {
            if(cur_ix[j] < sub_m && next_level <= m_level &&
                next_pred < lpred[cur_ix[j]])
            {
                bfs.push(cur_ix[j]);
                level[cur_ix[j]] = next_level;
                lpred[cur_ix[j]] = next_pred;
            }
        }
    };

    while(!bfs.empty())
    {
        const mat_int_t cur = bfs.front();
        bfs.pop();

        mat_int_t partner2x2 = -1;
        if(cur < m_pL->m() - 1 && m_is_piv[cur] && !m_is_piv[cur + 1])
            partner2x2 = cur + 1;
        if(cur > 0 && !m_is_piv[cur] && m_is_piv[cur - 1])
            partner2x2 = cur - 1;

        if(cur >= lpred[cur])
        {
            if(!added[cur])
            {
                buf_ix[nz_len++] = cur;
                added[cur] = true;
            }
            if(partner2x2 != -1 && !added[partner2x2])
            {
                buf_ix[nz_len++] = partner2x2;
                added[partner2x2] = true;
            }
        }

        const mat_int_t next_pred = std::max(cur, lpred[cur]);
        const mat_int_t next_level = level[cur] + 1;

        level_step(cur, next_pred, next_level);
        if(partner2x2 != -1)
            level_step(partner2x2, next_pred, next_level);
    }

    std::sort(buf_ix, buf_ix + nz_len);

    return nz_len;
}

/**
 * *****************************************************************************
 * ************************** BlockRestrictedPattern ***************************
 * *****************************************************************************
 */

template<typename T>
BlockRestrictedPattern<T>::
BlockRestrictedPattern(
    const mat_int_t m,
    const csr_matrix_t<T> * coarse,
    const mat_int_t num_blocks,
    const mat_int_t * block_starts)
: m_m(m),
  m_coarse(coarse),
  m_num_blocks(num_blocks),
  m_block_starts(block_starts),
  PatternGenerator<T>()
{
    create_block_map();
}

/* ************************************************************************** */

template<typename T>
BlockRestrictedPattern<T>::
~BlockRestrictedPattern()
{

}

/* ************************************************************************** */

template<typename T>
Triangular_ptr<T>
BlockRestrictedPattern<T>::
compute_pattern(
    const csr_matrix_t<T> * A,
    const mat_int_t num_piv_starts,
    const mat_int_t * piv_starts)
{
    /* save data */
    this->m_A = A;
    m_pL = FlexibleTriangular_ptr<T>(new FlexibleTriangular<T>(A));

    /* save pivots */
    m_is_piv.resize(A->m);
    std::fill(m_is_piv.begin(), m_is_piv.end(), 0);
    for(mat_int_t i = 0; i < num_piv_starts; ++i)
        m_is_piv[piv_starts[i]] = 1;

    /* create block diagonal matrix */
    m_pD = BlockDiagonal_ptr<T>(new BlockDiagonal<T>(A->m,
        num_piv_starts, piv_starts));

    /* use BFS to find fill-paths (levels after sum rule) */
    std::vector<mat_int_t> buf(A->m);

    m_row_starts.resize(A->m + 1);
    m_row_ix.clear();
    m_row_ix.reserve(A->nnz);

    m_row_starts[0] = 0;
    for(mat_int_t i = 0; i < A->m; ++i)
    {
        const mat_int_t i_len = row_pattern(i, buf.data());

        m_row_starts[i + 1] = m_row_starts[i] + i_len;
        std::copy(buf.begin(), buf.begin() + i_len,
            std::back_inserter(m_row_ix));
    }
    const mat_int_t nnz = m_row_starts.back();

    /* import pattern into triangular matrix */
    Triangular_ptr<T> L = Triangular_ptr<T>(new Triangular<T>(A->m,
        nnz));
    std::copy(m_row_starts.begin(), m_row_starts.end(), L->raw_row_ptr());
    std::copy(m_row_ix.begin(), m_row_ix.end(), L->raw_col_ptr());
    std::fill(L->raw_val_ptr(), L->raw_val_ptr() + L->nnz(), (T) 1.0);

    return L;
}

/* ************************************************************************** */

template<typename T>
void
BlockRestrictedPattern<T>::
init_pivot_pattern(
    const csr_matrix_t<T> * A)
{
    this->m_A = A;
    m_pL = FlexibleTriangular_ptr<T>(new FlexibleTriangular<T>(A));
    m_pD = BlockDiagonal_ptr<T>(new BlockDiagonal<T>(A->m));

    /* initialize with 1x1 pivots */
    m_is_piv.resize(A->m);
    std::fill(m_is_piv.begin(), m_is_piv.end(), 1);

    for(mat_int_t i = 0; i < A->m; ++i)
    {
        Block_ptr<T> b = BlockFactory<T>::create_block(1, i);
        m_pD->add_block(b.get());
    }
}

/* ************************************************************************** */

template<typename T>
void
BlockRestrictedPattern<T>::
pivot_1x1(
    const mat_int_t cur_row,
    const mat_int_t piv_a)
{
    /* pivot source matrix */
    m_pL->pivot(cur_row, piv_a);

    /* mark cur_row as 1x1 pivot */
    m_is_piv[cur_row] = 1;

    /* recreate block diagonal matrix */
    std::vector<Block_ptr<T>> prev_blocks;
    for(mat_int_t i = 0; i < m_pD->num_blocks(); ++i)
    {
        const Block_ptr<T>& i_block = m_pD->raw_blocks()[i];
        prev_blocks.emplace_back(i_block->copy());
    }

    /* add current_pivot */
    prev_blocks.emplace_back(BlockFactory<T>::create_block(1, cur_row));

    /* add remaining blocks */
    for(mat_int_t i = cur_row + 1; i < m_m; ++i)
        prev_blocks.emplace_back(BlockFactory<T>::create_block(1, i));

    m_pD = BlockDiagonal_ptr<T>(new BlockDiagonal<T>(m_m, prev_blocks.size(),
        prev_blocks.data()));
}

/* ************************************************************************** */

template<typename T>
void
BlockRestrictedPattern<T>::
pivot_2x2(
    const mat_int_t cur_row,
    const mat_int_t piv_a,
    const mat_int_t piv_b)
{
    /* pivot source matrix */
    m_pL->pivot(cur_row, piv_a);
    m_pL->pivot(cur_row + 1, piv_b);

    /* mark cur_row as 2x2 pivot */
    m_is_piv[cur_row] = 1;
    m_is_piv[cur_row + 1] = 0;

    /* recreate block diagonal matrix */
    std::vector<Block_ptr<T>> prev_blocks;
    for(mat_int_t i = 0; i < m_pD->num_blocks(); ++i)
    {
        const Block_ptr<T>& i_block = m_pD->raw_blocks()[i];
        prev_blocks.emplace_back(i_block->copy());
    }

    /* add current_pivot */
    prev_blocks.emplace_back(BlockFactory<T>::create_block(2, cur_row));

    /* add remaining blocks */
    for(mat_int_t i = cur_row + 2; i < m_m; ++i)
        prev_blocks.emplace_back(BlockFactory<T>::create_block(1, i));

    m_pD = BlockDiagonal_ptr<T>(new BlockDiagonal<T>(m_m, prev_blocks.size(),
        prev_blocks.data()));
}

/* ************************************************************************** */

template<typename T>
mat_int_t
BlockRestrictedPattern<T>::
row_pattern(
    const mat_int_t row,
    mat_int_t * buf_ix)
{
    const mat_int_t sub_m = (m_is_piv[row] == 0 ? (row - 1) : row);

    /**
     * block restriction is similar to threshold dropping, hence there
     * is no easy fill-in strategy -> every row must be generated from
     * triangular solves
     */

    /* extract A's pattern */
    mat_int_t * A_ix;
    T * A_val;
    mat_int_t A_len = m_pL->row(row, A_ix, A_val);

    /* solve with L first */
    std::vector<mat_int_t> L_ix;
    mat_int_t L_len = m_pL->sub_sanalysis(sub_m, A_len, A_ix);
    L_ix.resize(L_len);
    m_pL->sub_sanalysis_export(L_ix.data());

    /* solve with D then */
    std::vector<mat_int_t> LD_ix;
    mat_int_t LD_len = m_pD->sub_sanalysis(sub_m, L_len, L_ix.data());
    LD_ix.resize(LD_len);
    m_pD->sub_sanalysis_export(LD_ix.data());

    /* add diagonal element */
    LD_ix.push_back(row);

    /* remove all elements outside of blocks */
    const mat_int_t coarse_row = m_block_map[row];
    std::vector<bool> coarse_dense_row(m_coarse->m, false);
    for(mat_int_t i = m_coarse->csr_row[coarse_row];
        i < m_coarse->csr_row[coarse_row + 1]; ++i)
        coarse_dense_row[m_coarse->csr_col[i]] = true;

    LD_ix.erase(
        std::remove_if(
            LD_ix.begin(),
            LD_ix.end(),
            [&](const mat_int_t col)
            {
                const mat_int_t col_block = m_block_map[col];

                return !coarse_dense_row[col_block];
            }),
        LD_ix.end());
    std::sort(LD_ix.begin(), LD_ix.end());

    /* copy to output */
    std::copy(LD_ix.begin(), LD_ix.end(), buf_ix);

    /* update L */
    std::vector<T> vals(LD_ix.size(), 1.0);
    m_pL->set_row(row, LD_ix.size(), LD_ix.data(), vals.data());

    return LD_ix.size();
}

/* ************************************************************************** */

template<typename T>
void
BlockRestrictedPattern<T>::
create_block_map()
{
    m_block_map.resize(m_m);

    for(mat_int_t i = 0; i < m_num_blocks; ++i)
    {
        const mat_int_t i_from = m_block_starts[i];
        const mat_int_t i_to = (i < m_num_blocks - 1) ? m_block_starts[i + 1] :
            m_m;

        for(mat_int_t j = i_from; j < i_to; ++j)
            m_block_map[j] = i;
    }
}

/**
 * *****************************************************************************
 * ************************ BlockRestrictedExactPattern ************************
 * *****************************************************************************
 */

template<typename T>
BlockRestrictedExactPattern<T>::
BlockRestrictedExactPattern(
    const mat_int_t m,
    const csr_matrix_t<T> * coarse,
    const mat_int_t num_blocks,
    const mat_int_t * block_starts)
: m_m(m),
  m_coarse(coarse),
  m_num_blocks(num_blocks),
  m_block_starts(block_starts),
  PatternGenerator<T>()
{
    /* create a row-in-block map */
    m_rowcol_in_block.resize(m);
    for(mat_int_t i = 0; i < num_blocks; ++i)
    {
        const mat_int_t i_from = block_starts[i];
        const mat_int_t i_to = (i < num_blocks - 1) ? block_starts[i + 1] :
            m;

        for(mat_int_t j = i_from; j < i_to; ++j)
            m_rowcol_in_block[j] = i;
    }
}

/* ************************************************************************** */

template<typename T>
BlockRestrictedExactPattern<T>::
~BlockRestrictedExactPattern()
{
}

/* ************************************************************************** */

template<typename T>
Triangular_ptr<T>
BlockRestrictedExactPattern<T>::
compute_pattern(
    const csr_matrix_t<T> * A,
    const mat_int_t num_piv_starts,
    const mat_int_t * piv_starts)
{
    /* save data */
    this->m_A = A;
    this->m_piv_starts.assign(piv_starts, piv_starts + num_piv_starts);

    m_etree = EliminationTree_ptr<T>(new EliminationTree<T>(this->m_A->m,
        this->m_piv_starts.size(), this->m_piv_starts.data()));
    Triangular_ptr<T> L = m_etree->extract_pattern(this->m_A);

    /* filter entries - remove if they are not in any block */
    std::vector<mat_int_t> bigbuf(L->nnz());
    std::vector<mat_int_t> filtered_sizes(m_m + 1);
    for(mat_int_t i = 0; i < m_num_blocks; ++i)
    {
        const mat_int_t i_from = m_block_starts[i];
        const mat_int_t i_to = (i < m_num_blocks - 1) ? m_block_starts[i + 1] :
            m_m;

        /* create a dense map and reuse in fine rows */
        std::vector<char> map(i + 1);
        create_dense_block_row(i, map.data());

        #pragma omp parallel for
        for(mat_int_t j = i_from; j < i_to; ++j)
        {
            filtered_sizes[j] = filter_row(j, L->row_length(j),
                L->row_col(j), bigbuf.data() + L->raw_row_ptr()[j],
                map.data());
        }
    }
    filtered_sizes[m_m] = 0;

    /* compute offsets */
    mat_int_t hold = filtered_sizes[0];
    filtered_sizes[0] = 0;
    for(mat_int_t i = 1; i < m_m + 1; ++i)
    {
        const mat_int_t res = filtered_sizes[i - 1] + hold;
        hold = filtered_sizes[i];
        filtered_sizes[i] = res;
    }

    /* create filtered L */
    const mat_int_t filtered_nnz = filtered_sizes[m_m];
    Triangular_ptr<T> filt_L = Triangular_ptr<T>(new Triangular<T>(m_m,
        filtered_nnz));
    std::copy(filtered_sizes.begin(), filtered_sizes.end(),
        filt_L->raw_row_ptr());
    std::fill(filt_L->raw_val_ptr(), filt_L->raw_val_ptr() + filtered_nnz,
        1.0);

    for(mat_int_t i = 0; i < m_num_blocks; ++i)
    {
        const mat_int_t i_from = m_block_starts[i];
        const mat_int_t i_to = (i < m_num_blocks - 1) ? m_block_starts[i + 1] :
            m_m;

        /* create a dense map and reuse in fine rows */
        std::vector<char> map(i + 1);
        create_dense_block_row(i, map.data());

        #pragma omp parallel for
        for(mat_int_t j = i_from; j < i_to; ++j)
        {
            filtered_sizes[j] = filter_row(j, L->row_length(j),
                L->row_col(j), filt_L->raw_col_ptr() + filt_L->raw_row_ptr()[j],
                map.data());
        }
    }

    return filt_L;
}

/* ************************************************************************** */

template<typename T>
void
BlockRestrictedExactPattern<T>::
init_pivot_pattern(
    const csr_matrix_t<T> * A)
{
    this->m_A = A;

    m_etree = EliminationTree_ptr<T>(new EliminationTree<T>(this->m_A->m));
    m_etree->init_pivot_pattern(A);
}

/* ************************************************************************** */

template<typename T>
void
BlockRestrictedExactPattern<T>::
pivot_1x1(
    const mat_int_t cur_row,
    const mat_int_t piv_a)
{
    m_etree->pivot_1x1(cur_row, piv_a);
}

/* ************************************************************************** */

template<typename T>
void
BlockRestrictedExactPattern<T>::
pivot_2x2(
    const mat_int_t cur_row,
    const mat_int_t piv_a,
    const mat_int_t piv_b)
{
    m_etree->pivot_2x2(cur_row, piv_a, piv_b);
}

/* ************************************************************************** */

template<typename T>
mat_int_t
BlockRestrictedExactPattern<T>::
row_pattern(
    const mat_int_t row,
    mat_int_t * buf_ix)
{
    std::vector<mat_int_t> buf(row + 1);
    const mat_int_t orig_len = m_etree->row_pattern(row, buf.data());

    return filter_row(row, orig_len, buf.data(), buf_ix);
}

/* ************************************************************************** */

template<typename T>
void
BlockRestrictedExactPattern<T>::
create_dense_block_row(
    const mat_int_t block_row,
    char * dense_row)
{
    std::fill(dense_row, dense_row + block_row + 1, 0);
    for(mat_int_t j = m_coarse->csr_row[block_row];
        j < m_coarse->csr_row[block_row + 1]; ++j)
    {
        dense_row[m_coarse->csr_col[j]] = 1;
    }
}

/* ************************************************************************** */

template<typename T>
mat_int_t
BlockRestrictedExactPattern<T>::
filter_row(
    const mat_int_t row,
    const mat_int_t row_len,
    const mat_int_t * in_row_ix,
    mat_int_t * out_row_ix,
    const char * dense_map)
{
    const mat_int_t block_row = m_rowcol_in_block[row];

    /* create block row map if not given */
    const char * use_dense_map = dense_map;
    std::vector<char> map;
    if(use_dense_map == nullptr)
    {
        map.resize(block_row + 1);
        create_dense_block_row(block_row, map.data());
        use_dense_map = map.data();
    }

    mat_int_t ptr = 0;
    for(mat_int_t i = 0; i < row_len; ++i)
    {
        const mat_int_t block_col = m_rowcol_in_block[in_row_ix[i]];
        if(use_dense_map[block_col])
            out_row_ix[ptr++] = in_row_ix[i];
    }

    return ptr;
}

NS_STAGING_END
NS_CULIP_END