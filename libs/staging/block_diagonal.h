/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_STAGING_BLOCK_DIAGONAL_H_
#define __CULIP_STAGING_BLOCK_DIAGONAL_H_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

/* forward declaration for pointer type */
template<typename T>
class Block;

template<typename T>
using Block_ptr = std::unique_ptr<Block<T>>;

template<typename T>
class Block
{
public:
    Block(const mat_int_t first_row);
    virtual ~Block();

    virtual mat_int_t first_row() const;
    virtual mat_int_t order() const = 0;
    virtual T determinant() const = 0;

    virtual bool covers_row(const mat_int_t row) const;

    virtual void solve(const T * in, T * out) const = 0;
    virtual void multiply(const T * in, T * out) const = 0;
    virtual void read_elems(const T * elems) = 0;
    virtual void write_elems(T * elems) const = 0;

    virtual Block_ptr<T> copy() const = 0;

    /* make positive definite */
    virtual bool is_posdef() const;
    virtual void make_posdef() = 0;

protected:
    mat_int_t m_f_row;
};

/* ************************************************************************** */

template<typename T>
class Block1x1 : public Block<T>
{
public:
    Block1x1(const mat_int_t first_row);
    ~Block1x1();

    mat_int_t order() const;
    T determinant() const;

    void solve(const T * in, T * out) const;
    void multiply(const T * in, T * out) const;
    void read_elems(const T * elems);
    void write_elems(T * elems) const;

    Block_ptr<T> copy() const;

    /* make positive definite */
    void make_posdef();

protected:
    T m_alpha;
};

/* ************************************************************************** */

template<typename T>
class Block2x2 : public Block<T>
{
public:
    Block2x2(const mat_int_t first_row);
    ~Block2x2();

    mat_int_t order() const;
    T determinant() const;

    void solve(const T * in, T * out) const;
    void multiply(const T * in, T * out) const;
    void read_elems(const T * elems);
    void write_elems(T * elems) const;

    Block_ptr<T> copy() const;

    /* make positive definite */
    void make_posdef();

protected:
    /* [alpha beta; beta gamma] */
    T m_alpha;
    T m_beta;
    T m_gamma;
};

/* ************************************************************************** */

template<typename T>
class BlockFactory
{
public:
    BlockFactory();
    ~BlockFactory();

    static Block_ptr<T> create_block(const mat_int_t order,
        const mat_int_t first_row);
};

/* ************************************************************************** */

/* forward declaration for pointer */
template<typename T>
class BlockDiagonal;

template<typename T>
using BlockDiagonal_ptr = std::unique_ptr<BlockDiagonal<T>>;

template<typename T>
class BlockDiagonal
{
public:
    BlockDiagonal(const mat_int_t m, const mat_int_t num_blocks,
        const mat_int_t * block_starts);
    BlockDiagonal(const mat_int_t m, const mat_int_t num_blocks,
        const Block_ptr<T> * blocks);
    BlockDiagonal(const mat_int_t m);
    ~BlockDiagonal();

    /* build up piece by piece */
    void add_block(const Block<T> * blk);

    /* lower level (data access) */
    mat_int_t num_blocks() const;
    const Block_ptr<T> * raw_blocks() const;
    Block_ptr<T> * raw_blocks_rw();
    mat_int_t block_with_row(const mat_int_t row) const;

    csr_matrix_ptr<T> to_csr() const;

    /* higher level (linear algebra) functions */
    mat_int_t sanalysis(const mat_int_t b_len, const mat_int_t * b_ix);
    void sanalysis_export(mat_int_t * x_ix) const;

    void ssolve(const T * b_val, T * x_val) const;
    void dsolve(const T * b_val, T * x_val) const;
    void smultiply(const T * x_val, T * b_val) const;

    mat_int_t sub_sanalysis(const mat_int_t sub_m, const mat_int_t b_len,
        const mat_int_t * b_ix);
    void sub_sanalysis(const mat_int_t sub_m, const mat_int_t b_len,
        const mat_int_t * b_ix, std::vector<mat_int_t>& x_ix,
        std::vector<mat_int_t>& cov_blocks) const;
    void sub_sanalysis_export(mat_int_t * x_ix) const;

    void sub_ssolve(const mat_int_t sub_m, const T * b_val, T * x_val) const;
    void sub_dsolve(const mat_int_t sub_m, const T * b_val, T * x_val) const;
    void sub_smultiply(const mat_int_t sub_m, const T * x_val, T * b_val) const;
    void sub_smultiply(const mat_int_t sub_m, const mat_int_t num_cov_blocks,
        const mat_int_t * cov_blocks, const mat_int_t x_len,
        const mat_int_t * x_ix, const T * x_val, const mat_int_t b_len,
        const mat_int_t * b_ix, T * b_val) const;

    /* const access procedures for parallel computing */
    void ssolve(const mat_int_t b_len, const mat_int_t * b_ix, const T * b_val,
        const mat_int_t x_len, const mat_int_t * x_ix, T * x_val) const;
    void sub_ssolve(const mat_int_t sub_m, const mat_int_t b_len,
        const mat_int_t * b_ix, const T * b_val, const mat_int_t x_len,
        const mat_int_t * x_ix, T * x_val) const;

    /* debugging functions */
    void dprint() const;
    void sub_dprint(const mat_int_t sub_m) const;

    /* output */
    BlockDiagonal_ptr<T> copy() const;

    /* for preconditioning */
    bool is_posdef() const;
    void make_posdef();

protected:
    void init_blocks(const mat_int_t * block_starts);

protected:
    mat_int_t m_m;
    mat_int_t m_num_blocks;

    std::vector<Block_ptr<T>> m_blocks;

    std::vector<mat_int_t> m_x_ix;
    std::vector<mat_int_t> m_cov_blocks;

    mat_int_t m_b_len;
    std::vector<mat_int_t> m_b_ix;
};

NS_STAGING_END
NS_CULIP_END

#endif /* __CULIP_STAGING_BLOCK_DIAGONAL_H_ */