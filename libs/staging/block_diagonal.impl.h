/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/staging/block_diagonal.h>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

/*
 * *****************************************************************************
 * ********************************* BLOCK *************************************
 * *****************************************************************************
 */

template<typename T>
Block<T>::
Block(
    const mat_int_t first_row)
: m_f_row(first_row)
{

}

/* ************************************************************************** */

template<typename T>
Block<T>::
~Block()
{

}

/* ************************************************************************** */

template<typename T>
mat_int_t
Block<T>::
first_row()
const
{
    return m_f_row;
}

/* ************************************************************************** */

template<typename T>
bool
Block<T>::
covers_row(
    const mat_int_t row)
const
{
    return (first_row() <= row && row < first_row() + order());
}

/* ************************************************************************** */

template<typename T>
bool
Block<T>::
is_posdef()
const
{
    return (determinant() > 0);
}

/*
 * *****************************************************************************
 * ******************************** BLOCK1x1 ***********************************
 * *****************************************************************************
 */

template<typename T>
Block1x1<T>::
Block1x1(
    const mat_int_t first_row)
: Block<T>(first_row)
{
}

/* ************************************************************************** */

template<typename T>
Block1x1<T>::
~Block1x1<T>()
{
}

/* ************************************************************************** */

template<typename T>
mat_int_t
Block1x1<T>::
order()
const
{
    return 1;
}

/* ************************************************************************** */

template<typename T>
T
Block1x1<T>::
determinant()
const
{
    return m_alpha;
}

/* ************************************************************************** */

template<typename T>
void
Block1x1<T>::
solve(
    const T * in,
    T * out)
const
{
    *out = *in / m_alpha;
}

/* ************************************************************************** */

template<typename T>
void
Block1x1<T>::
multiply(
    const T * in,
    T * out)
const
{
    *out = *in * m_alpha;
}

/* ************************************************************************** */

template<typename T>
void
Block1x1<T>::
read_elems(
    const T * elems)
{
    m_alpha = *elems;
}

/* ************************************************************************** */

template<typename T>
void
Block1x1<T>::
write_elems(
    T * elems)
const
{
    *elems = m_alpha;
}

/* ************************************************************************** */

template<typename T>
Block_ptr<T>
Block1x1<T>::
copy()
const
{
    Block1x1<T> * ret = new Block1x1<T>(this->m_f_row);
    ret->read_elems(&m_alpha);

    return Block_ptr<T>(ret);
}

/* ************************************************************************** */

template<typename T>
void
Block1x1<T>::
make_posdef()
{
    m_alpha = std::abs(m_alpha);
}

/*
 * *****************************************************************************
 * ******************************** BLOCK2x2 ***********************************
 * *****************************************************************************
 */

template<typename T>
Block2x2<T>::
Block2x2(
    const mat_int_t first_row)
: Block<T>(first_row)
{
}

/* ************************************************************************** */

template<typename T>
Block2x2<T>::
~Block2x2<T>()
{
}

/* ************************************************************************** */

template<typename T>
mat_int_t
Block2x2<T>::
order()
const
{
    return 2;
}

/* ************************************************************************** */

template<typename T>
T
Block2x2<T>::
determinant()
const
{
    return (m_alpha * m_gamma - m_beta * m_beta);
}

/* ************************************************************************** */

template<typename T>
void
Block2x2<T>::
solve(
    const T * in,
    T * out)
const
{
    /* Cramer's rule */
    // const T det = m_alpha * m_gamma - m_beta * m_beta;
    // const T in0 = in[0];
    // const T in1 = in[1];

    // out[0] = m_gamma * in0 - m_beta * in1;
    // out[1] = - m_beta * in0 + m_alpha * in1;

    // out[0] /= det;
    // out[1] /= det;

    /* solve using Givens rotations */
    const T giv_r = sqrt(m_alpha * m_alpha + m_beta * m_beta);
    const T giv_c = m_alpha / giv_r;
    const T giv_s = m_beta / giv_r;

    const T tmp0 = giv_c * in[0] + giv_s * in[1];
    const T tmp1 = -giv_s * in[0] + giv_c * in[1];

    const T tilde_d11 = giv_c * m_alpha + giv_s * m_beta;
    const T tilde_d12 = giv_c * m_beta + giv_s * m_gamma;
    const T tilde_d22 = -giv_s * m_beta + giv_c * m_gamma;

    out[1] = tmp1 / tilde_d22;
    out[0] = (tmp0 - tilde_d12 * out[1]) / tilde_d11;
}

/* ************************************************************************** */

template<typename T>
void
Block2x2<T>::
multiply(
    const T * in,
    T * out)
const
{
    const T in0 = in[0];
    const T in1 = in[1];

    out[0] = m_alpha * in0 + m_beta * in1;
    out[1] = m_beta * in0 + m_gamma * in1;
}

/* ************************************************************************** */

template<typename T>
void
Block2x2<T>::
read_elems(
    const T * elems)
{
    m_alpha = elems[0];
    m_beta = elems[2];
    m_gamma = elems[3];
}

/* ************************************************************************** */

template<typename T>
void
Block2x2<T>::
write_elems(
    T * elems)
const
{
    elems[0] = m_alpha;
    elems[1] = m_beta;
    elems[2] = m_beta;
    elems[3] = m_gamma;
}

/* ************************************************************************** */

template<typename T>
Block_ptr<T>
Block2x2<T>::
copy()
const
{
    Block2x2<T> * ret = new Block2x2<T>(this->m_f_row);

    T tmp[4];
    write_elems(tmp);
    ret->read_elems(tmp);

    return Block_ptr<T>(ret);
}

/* ************************************************************************** */

template<typename T>
void
Block2x2<T>::
make_posdef()
{
    /* manual Eigendecomposition, then set Eigenvalues to positive */
    const T trace = m_alpha + m_gamma;
    const T det = determinant();

    /* Eigenvalues */
    T lambda_1 = trace / 2 + std::sqrt(trace * trace  / 2 - det);
    T lambda_2 = trace / 2 - std::sqrt(trace * trace  / 2 - det);

    /* Eigendecomposition */
    T v_1[] = {m_beta, lambda_1 - m_alpha};
    T v_2[] = {m_beta, lambda_2 - m_alpha};
    const T nrm_v_1 = std::sqrt(v_1[0] * v_1[0] + v_1[1] * v_1[1]);
    const T nrm_v_2 = std::sqrt(v_2[0] * v_2[0] + v_2[1] * v_2[1]);
    v_1[0] /= nrm_v_1;
    v_1[1] /= nrm_v_1;
    v_2[0] /= nrm_v_2;
    v_2[1] /= nrm_v_2;

    /* force matrix to be posdef */
    lambda_1 = std::abs(lambda_1);
    lambda_2 = std::abs(lambda_2);

    /* compute new elements */
    m_alpha = lambda_1 * v_1[0] * v_1[0] + lambda_2 * v_2[0] * v_2[0];
    m_gamma = lambda_1 * v_1[1] * v_1[1] + lambda_2 * v_2[1] * v_2[1];
    m_beta = lambda_1 * v_1[0] * v_1[1] + lambda_2 * v_1[0] * v_1[1];
}

/*
 * *****************************************************************************
 * ****************************** BLOCKFACTORY *********************************
 * *****************************************************************************
 */

template<typename T>
BlockFactory<T>::
BlockFactory()
{

}

/* ************************************************************************** */

template<typename T>
BlockFactory<T>::
~BlockFactory()
{

}

/* ************************************************************************** */

template<typename T>
Block_ptr<T>
BlockFactory<T>::
create_block(
    const mat_int_t order,
    const mat_int_t first_row)
{
    if(order == 1)
    {
        return Block_ptr<T>(new Block1x1<T>(first_row));
    }
    else if(order == 2)
    {
        return Block_ptr<T>(new Block2x2<T>(first_row));
    }
    else
    {
        std::cerr << "No pivots > 2x2 supported..." << std::endl;

        return nullptr;
    }
}

/*
 * *****************************************************************************
 * ****************************** BLOCKDIAGONAL ********************************
 * *****************************************************************************
 */

template<typename T>
BlockDiagonal<T>::
BlockDiagonal(
    const mat_int_t m,
    const mat_int_t num_blocks,
    const mat_int_t * block_starts)
: m_m(m),
  m_num_blocks(num_blocks)
{
    init_blocks(block_starts);
}

/* ************************************************************************** */

template<typename T>
BlockDiagonal<T>::
BlockDiagonal(
    const mat_int_t m,
    const mat_int_t num_blocks,
    const Block_ptr<T> * blocks)
: m_m(m),
  m_num_blocks(num_blocks)
{
    m_blocks.resize(num_blocks);

    for(mat_int_t i = 0; i < num_blocks; ++i)
        m_blocks[i] = blocks[i]->copy();
}

/* ************************************************************************** */

template<typename T>
BlockDiagonal<T>::
BlockDiagonal(
    const mat_int_t m)
: m_m(m),
  m_num_blocks(0)
{

}

/* ************************************************************************** */

template<typename T>
BlockDiagonal<T>::
~BlockDiagonal()
{
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
add_block(
    const Block<T> * blk)
{
    m_blocks.emplace_back(blk->copy());
    ++m_num_blocks;
}

/* ************************************************************************** */

template<typename T>
mat_int_t
BlockDiagonal<T>::
num_blocks()
const
{
    return m_num_blocks;
}

/* ************************************************************************** */

template<typename T>
const Block_ptr<T> *
BlockDiagonal<T>::
raw_blocks()
const
{
    return (const Block_ptr<T> *) m_blocks.data();
}

/* ************************************************************************** */

template<typename T>
Block_ptr<T> *
BlockDiagonal<T>::
raw_blocks_rw()
{
    return m_blocks.data();
}

/* ************************************************************************** */

template<typename T>
mat_int_t
BlockDiagonal<T>::
block_with_row(
    const mat_int_t row)
const
{
    /* find blocks that contains this row */
    for(mat_int_t i = 0; i < m_num_blocks; ++i)
        if(m_blocks[i]->covers_row(row))
            return i;

    return (m_num_blocks + 1);
}

/* ************************************************************************** */

template<typename T>
csr_matrix_ptr<T>
BlockDiagonal<T>::
to_csr()
const
{
    std::vector<mat_int_t> csr_row(m_m  + 1);
    std::fill(csr_row.begin(), csr_row.end(), 0);

    /* compute number of nonzeros */
    mat_int_t nnz = 0;
    for(const Block_ptr<T>& b : m_blocks)
        nnz += b->order() * b->order();

    /* create matrix structure */
    csr_matrix_ptr<T> csr = make_csr_matrix_ptr<T>(m_m, m_m, nnz, false);
    std::fill(csr->csr_row, csr->csr_row + m_m + 1, 0);

    /* compute row offsets */
    for(const Block_ptr<T>& b : m_blocks)
    {
        /* count row sizes */
        for(mat_int_t i = 0; i < b->order(); ++i)
            csr->csr_row[b->first_row() + i] += b->order();
    }

    mat_int_t hold = csr->csr_row[0];
    csr->csr_row[0] = 0;
    for(mat_int_t i = 1; i < m_m + 1; ++i)
    {
        const mat_int_t res = csr->csr_row[i - 1] + hold;
        hold = csr->csr_row[i];
        csr->csr_row[i] = res;
    }

    /* compute CSR form */
    mat_int_t ptr = 0;
    for(const Block_ptr<T>& b : m_blocks)
    {
        /* write out values */
        const mat_int_t row_offset = csr->csr_row[b->first_row()];
        b->write_elems(csr->csr_val + row_offset);

        /* write out indices */
        for(mat_int_t i = 0; i < b->order(); ++i)
            for(mat_int_t j = 0; j < b->order(); ++j)
                csr->csr_col[ptr++] = b->first_row() + j;
    }

    /* assemble CSR matrix and return */
    return csr;
}

/* ************************************************************************** */

template<typename T>
mat_int_t
BlockDiagonal<T>::
sanalysis(
    const mat_int_t b_len,
    const mat_int_t * b_ix)
{
    return sub_sanalysis(m_m, b_len, b_ix);
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
sanalysis_export(
    mat_int_t * x_ix)
const
{
    sub_sanalysis_export(x_ix);
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
ssolve(
    const T * b_val,
    T * x_val)
const
{
    sub_ssolve(m_m, b_val, x_val);
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
dsolve(
    const T * b_val,
    T * x_val)
const
{
    sub_dsolve(m_m, b_val, x_val);
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
smultiply(
    const T * x_val,
    T * b_val)
const
{
    sub_smultiply(m_m, x_val, b_val);
}

/* ************************************************************************** */

template<typename T>
mat_int_t
BlockDiagonal<T>::
sub_sanalysis(
    const mat_int_t sub_m,
    const mat_int_t b_len,
    const mat_int_t * b_ix)
{
    m_x_ix.clear();
    m_cov_blocks.clear();

    /* copy b's data */
    m_b_len = b_len;
    m_b_ix.resize(b_len);
    std::copy(b_ix, b_ix + b_len, m_b_ix.begin());

    /* for 1x1 blocks only, would be b's pattern, otherwise add nz's */
    mat_int_t d_ptr = 0;
    mat_int_t last_ptr = (mat_int_t) -1;
    for(mat_int_t i = 0; i < b_len; ++i)
    {
        const mat_int_t ix = b_ix[i];

        /* exit if not in submatrix anymore */
        if(ix >= sub_m)
            return m_x_ix.size();

        while(d_ptr < m_num_blocks && !m_blocks[d_ptr]->covers_row(ix))
            ++d_ptr;

        if(m_blocks[d_ptr]->covers_row(ix))
        {
            /* skip if already covered */
            if(d_ptr != last_ptr)
            {
                m_cov_blocks.push_back(d_ptr);

                for(mat_int_t j = 0; j < m_blocks[d_ptr]->order(); ++j)
                    m_x_ix.push_back(m_blocks[d_ptr]->first_row() + j);
            }

            last_ptr = d_ptr;
        }
    }

    return m_x_ix.size();
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
sub_sanalysis(
    const mat_int_t sub_m,
    const mat_int_t b_len,
    const mat_int_t * b_ix,
    std::vector<mat_int_t>& x_ix,
    std::vector<mat_int_t>& cov_blocks)
const
{
    x_ix.clear();
    cov_blocks.clear();

    /* for 1x1 blocks only, would be b's pattern, otherwise add nz's */
    mat_int_t d_ptr = 0;
    mat_int_t last_ptr = (mat_int_t) -1;
    for(mat_int_t i = 0; i < b_len; ++i)
    {
        const mat_int_t ix = b_ix[i];

        /* exit if not in submatrix anymore */
        if(ix >= sub_m)
            return;

        while(d_ptr < m_num_blocks && !m_blocks[d_ptr]->covers_row(ix))
            ++d_ptr;

        if(m_blocks[d_ptr]->covers_row(ix))
        {
            /* skip if already covered */
            if(d_ptr != last_ptr)
            {
                cov_blocks.push_back(d_ptr);

                for(mat_int_t j = 0; j < m_blocks[d_ptr]->order(); ++j)
                    x_ix.push_back(m_blocks[d_ptr]->first_row() + j);
            }

            last_ptr = d_ptr;
        }
    }
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
sub_sanalysis_export(
    mat_int_t * x_ix)
const
{
    std::copy(m_x_ix.begin(), m_x_ix.end(), x_ix);
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
sub_ssolve(
    const mat_int_t sub_m,
    const T * b_val,
    T * x_val)
const
{
    std::vector<T> full_x(sub_m, 0);

    /* scatter b */
    for(mat_int_t i = 0; i < m_b_len; ++i)
        if(m_b_ix[i] < sub_m)
            full_x[m_b_ix[i]] = b_val[i];

    /* solve with blocks */
    for(const mat_int_t p : m_cov_blocks)
    {
        const Block_ptr<T>& blk = m_blocks[p];
        blk->solve(full_x.data() + blk->first_row(),
            full_x.data() + blk->first_row());
    }

    /* save sparse solution */
    for(mat_int_t i = 0; i < m_x_ix.size(); ++i)
        x_val[i] = full_x[m_x_ix[i]];
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
sub_dsolve(
    const mat_int_t sub_m,
    const T * b_val,
    T * x_val)
const
{
    /* copy b */
    std::copy(b_val, b_val + sub_m, x_val);

    /* solve with blocks */
    for(mat_int_t p = 0; p < m_num_blocks; ++p)
    {
        const Block_ptr<T>& blk = m_blocks[p];
        blk->solve(x_val + blk->first_row(),
            x_val + blk->first_row());
    }
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
sub_smultiply(
    const mat_int_t sub_m,
    const T * x_val,
    T * b_val)
const
{
    std::vector<T> full_x(sub_m, 0);

    /* scatter x */
    for(mat_int_t i = 0; i < m_b_len; ++i)
        if(m_b_ix[i] < sub_m)
            full_x[m_b_ix[i]] = x_val[i];

    /* solve with blocks */
    for(const mat_int_t p : m_cov_blocks)
    {
        const Block_ptr<T>& blk = m_blocks[p];
        blk->multiply(full_x.data() + blk->first_row(),
            full_x.data() + blk->first_row());
    }

    /* save sparse solution */
    for(mat_int_t i = 0; i < m_x_ix.size(); ++i)
        b_val[i] = full_x[m_x_ix[i]];
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
sub_smultiply(
    const mat_int_t sub_m,
    const mat_int_t num_cov_blocks,
    const mat_int_t * cov_blocks,
    const mat_int_t x_len,
    const mat_int_t * x_ix,
    const T * x_val,
    const mat_int_t b_len,
    const mat_int_t * b_ix,
    T * b_val)
const
{
    std::vector<T> full_x(sub_m, 0);

    /* scatter x */
    for(mat_int_t i = 0; i < x_len; ++i)
        if(x_ix[i] < sub_m)
            full_x[x_ix[i]] = x_val[i];

    /* solve with blocks */
    for(mat_int_t p = 0; p < num_cov_blocks; ++p)
    {
        const Block_ptr<T>& blk = m_blocks[cov_blocks[p]];
        blk->multiply(full_x.data() + blk->first_row(),
            full_x.data() + blk->first_row());
    }

    /* save sparse solution */
    for(mat_int_t i = 0; i < b_len; ++i)
        b_val[i] = full_x[b_ix[i]];
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
ssolve(
    const mat_int_t b_len,
    const mat_int_t * b_ix,
    const T * b_val,
    const mat_int_t x_len,
    const mat_int_t * x_ix,
    T * x_val)
const
{
    sub_ssolve(m_m, b_len, b_ix, b_val, x_len, x_ix, x_val);
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
sub_ssolve(
    const mat_int_t sub_m,
    const mat_int_t b_len,
    const mat_int_t * b_ix,
    const T * b_val,
    const mat_int_t x_len,
    const mat_int_t * x_ix,
    T * x_val)
const
{
    /* in-situ analysis: which blocks are covered? */
    std::vector<mat_int_t> cov_blocks;

    /* for 1x1 blocks only, would be b's pattern, otherwise add nz's */
    mat_int_t d_ptr = 0;
    mat_int_t last_ptr = (mat_int_t) -1;
    for(mat_int_t i = 0; i < b_len; ++i)
    {
        const mat_int_t ix = b_ix[i];

        /* exit if not in submatrix anymore */
        if(ix >= sub_m)
            break;

        while(d_ptr < m_num_blocks && !m_blocks[d_ptr]->covers_row(ix))
            ++d_ptr;

        if(m_blocks[d_ptr]->covers_row(ix))
        {
            /* skip if already covered */
            if(d_ptr != last_ptr)
                cov_blocks.push_back(d_ptr);

            last_ptr = d_ptr;
        }
    }

    /* start solving! */
    std::vector<T> full_x(sub_m, 0);

    /* scatter b */
    for(mat_int_t i = 0; i < b_len; ++i)
        if(b_ix[i] < sub_m)
            full_x[b_ix[i]] = b_val[i];

    /* solve with blocks */
    for(const mat_int_t p : cov_blocks)
    {
        const Block_ptr<T>& blk = m_blocks[p];
        blk->solve(full_x.data() + blk->first_row(),
            full_x.data() + blk->first_row());
    }

    /* save sparse solution */
    for(mat_int_t i = 0; i < x_len; ++i)
        x_val[i] = full_x[x_ix[i]];
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
dprint()
const
{
    sub_dprint(m_m);
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
sub_dprint(
    const mat_int_t sub_m)
const
{
    std::vector<T> D(sub_m * sub_m, 0);

    for(const Block_ptr<T>& b : m_blocks)
    {
        if(b->first_row() >= sub_m)
            continue;

        /* write out block */
        std::vector<T> blk(b->order() * b->order(), 0);
        b->write_elems(blk.data());

        /* copy block to D */
        for(mat_int_t i = 0; i < b->order(); ++i)
            for(mat_int_t j = 0; j < b->order(); ++j)
                if(b->first_row() + i < sub_m)
                    D[(b->first_row() + i) * sub_m + b->first_row() + j] =
                        blk[i * b->order() + j];
    }

    /* write out dense matrix */
    std::cout << "D(" << sub_m << ") = " << std::endl;
    for(mat_int_t i = 0; i < sub_m; ++i)
    {
        for(mat_int_t j = 0; j < sub_m; ++j)
            std::cout << D[i * sub_m + j] << " ";
        std::cout << std::endl;
    }

}

/* ************************************************************************** */

template<typename T>
BlockDiagonal_ptr<T>
BlockDiagonal<T>::
copy()
const
{
    return BlockDiagonal_ptr<T>(new BlockDiagonal<T>(m_m, m_num_blocks,
        (const Block_ptr<T> *) m_blocks.data()));
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
init_blocks(
    const mat_int_t * block_starts)
{
    m_blocks.clear();
    m_blocks.reserve(m_num_blocks);

    for(mat_int_t i = 0; i < m_num_blocks; ++i)
    {
        const mat_int_t block_start = block_starts[i];
        const mat_int_t block_end =
            ((i == m_num_blocks - 1) ? m_m : block_starts[i + 1]);

        m_blocks.emplace_back(BlockFactory<T>::create_block(
            block_end - block_start, block_start));
    }
}

/* ************************************************************************** */

template<typename T>
bool
BlockDiagonal<T>::
is_posdef()
const
{
    bool posdef = true;

    for(const Block_ptr<T>& b : m_blocks)
        posdef &= b->is_posdef();

    return posdef;
}

/* ************************************************************************** */

template<typename T>
void
BlockDiagonal<T>::
make_posdef()
{
    for(Block_ptr<T>& b : m_blocks)
    {
        if(!b->is_posdef())
            b->make_posdef();
    }
}

NS_STAGING_END
NS_CULIP_END