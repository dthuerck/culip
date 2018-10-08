/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIBS_LA_BLOCK_LDLT_CUH_
#define __CULIP_LIBS_LA_BLOCK_LDLT_CUH_

#include <libs/la/preconditioner.cuh>

#include <limits>

NS_CULIP_BEGIN
NS_LA_BEGIN

enum BLOCK_FORMAT
{
    BLOCK_SPARSE = 0,
    BLOCK_DENSE = 1
};

enum SOLVE_ALGORITHM
{
    SOLVE_STRIPE = 0,
    SOLVE_BLOCK = 1,
    SOLVE_SCALAR = 2,
    SOLVE_JACOBI = 3
};

struct matrix_block
{
    mat_int_t id;
    BLOCK_FORMAT format;
    mat_int_t nnz;
    mat_int_t max_nnz;

    mat_int_t got_extra;

    mat_int_t block_row;
    mat_int_t block_col;

    mat_int_t col_ptr;
    mat_int_t val_ptr;

    /* default constructor */
    matrix_block()
    {

    }

    /* copy constructor & assignment operator */
    matrix_block(
        const matrix_block& other)
    {
        set(other);
    }

    matrix_block&
    operator=(
        const matrix_block& other)
    {
        set(other);

        return *this;
    }

    void set(
        const matrix_block& other)
    {
        this->id = other.id;
        this->format = other.format;
        this->nnz = other.nnz;
        this->max_nnz = other.max_nnz;
        this->block_row = other.block_row;
        this->block_col = other.block_col;
        this->col_ptr = other.col_ptr;
        this->val_ptr = other.val_ptr;
    }

    ~matrix_block()
    {

    }
};

struct compressed_block_list
{
    mat_int_t m = -1;
    mat_int_t nnz = 1;

    dense_vector_ptr<mat_int_t> offsets = nullptr;
    dense_vector_ptr<mat_int_t> indices = nullptr;

    /* constructors */
    compressed_block_list(
        const bool on_device,
        const mat_int_t m,
        const mat_int_t nnz)
    {
        set(on_device, m, nnz);
    }

    compressed_block_list(
        const bool on_device)
    {
        offsets = make_managed_dense_vector_ptr<mat_int_t>(on_device);
        indices = make_managed_dense_vector_ptr<mat_int_t>(on_device);
    }

    /* copy constructor & assignment operator */
    compressed_block_list(
        const compressed_block_list& other)
    {
        set(other);
    }

    compressed_block_list&
    operator=(
        const compressed_block_list& other)
    {
        set(other);

        return *this;
    }

    void set(
        const bool on_device,
        const mat_int_t m,
        const mat_int_t nnz)
    {
        this->m = m;
        this->nnz = nnz;

        offsets = make_managed_dense_vector_ptr<mat_int_t>(m + 1, on_device);
        indices = make_managed_dense_vector_ptr<mat_int_t>(nnz, on_device);
    }


    void set(
        const compressed_block_list& other)
    {
        this->m = other.m;
        this->nnz = other.nnz;

        *(this->offsets) = other.offsets.get();
        *(this->indices) = other.indices.get();
    }

    void print(
        const char * s)
    {
        printf("%s\n", s);
        printf("m: %d\n", m);
        printf("nnz: %d\n", nnz);
        for(mat_int_t i = 0; i < m; ++i)
        {
            printf("\t%d: ", i);
            for(mat_int_t j = (*offsets)[i]; j < (*offsets)[i + 1]; ++j)
                printf("%d ", (*indices)[j]);
            printf("\n");
        }
    }

    ~compressed_block_list()
    {

    }

    csr_matrix_ptr<float> to_csr()
    {
        csr_matrix_ptr<float> mat =
            make_csr_matrix_ptr<float>(m, m, nnz, false);
        std::copy(offsets->dense_val, offsets->dense_val + m + 1,
            mat->csr_row);
        std::copy(indices->dense_val, indices->dense_val + nnz,
            mat->csr_col);
        std::fill(mat->csr_val, mat->csr_val + nnz, 1.0);

        return mat;
    }
};

/* ************************************************************************** */

/* options */
template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
class BlockiLDLt : public Preconditioner<T>
{
public:
    /* mode for 1x1 / 2x2 pivots */
    BlockiLDLt(
        gpu_handle_ptr& gpu_handle,
        const csr_matrix_t<T> * h_A,
        const mat_int_t num_pivs,
        const mat_int_t * piv_starts,
        const mat_int_t num_blocks,
        const mat_int_t * block_starts);
    ~BlockiLDLt();

    virtual mat_int_t n() const;

    /* returns Manteuffel shift necessary for factor */
    T compute(
        const mat_int_t fill_level,
        const T fill_factor,
        const T threshold);

    virtual bool is_left() const;
    virtual bool is_middle() const;
    virtual bool is_right() const;

    virtual void solve_left(
        const dense_vector_t<T> * b,
        dense_vector_t<T> * x,
        const bool transpose = false) const;
    virtual void solve_middle(
        const dense_vector_t<T> * b,
        dense_vector_t<T> * x,
        const bool transpose = false) const;
    virtual void solve_right(
        const dense_vector_t<T> * b,
        dense_vector_t<T> * x,
        const bool transpose = false) const;

    void set_solve_algorithm(
        SOLVE_ALGORITHM algorithm,
        const mat_int_t num_jacobi_sweeps);
    void export_factorized(
        const char * folder);

protected:

    /* matrix assembly */
    void offsets_to_indices(
        const mat_int_t m,
        const mat_int_t nnz,
        const mat_int_t * offsets,
        mat_int_t * indices);
    void indices_to_offsets(
        const mat_int_t m,
        const mat_int_t nnz,
        const mat_int_t * indices,
        mat_int_t * offsets);

    void build_initial_coarse(
        dense_vector_ptr<mat_int_t>& block_nnz);
    mat_int_t determine_coarse_level_fill_and_diag(
        const mat_int_t max_level,
        const mat_int_t * in_block_nnz,
        dense_vector_ptr<mat_int_t>& out_block_nnz);
    void build_block_csc();
    void build_blocks(
        const T fill_factor,
        const mat_int_t * block_A_nnz);

    /* preprocessing, upload & re - download */
    void find_level_sets();
    void find_level_block_sets();
    void upload_data();
    void download_data();
    void eval_results();

    /* factorization */
    void factorize(const T threshold);

    /* export resuts */
    void export_coarse();

public:
    /* solve with factorized matrix */
    void solve_analysis();
    void solve_analysis_stripe();
    void solve_analysis_block();
    void solve_analysis_scalar();

    void solve_U(
        dense_vector_t<T> * b) const;
    void solve_U_stripe(
        dense_vector_t<T> * b) const;
    void solve_U_block(
        dense_vector_t<T> * b) const;
    void solve_U_scalar(
        dense_vector_t<T> * b) const;
    void solve_U_jacobi(
        dense_vector_t<T> * b,
        const mat_int_t sweeps) const;

    void solve_Ut(
        dense_vector_t<T> * b) const;
    void solve_Ut_stripe(
        dense_vector_t<T> * b) const;
    void solve_Ut_block(
        dense_vector_t<T> * b) const;
    void solve_Ut_scalar(
        dense_vector_t<T> * b) const;
    void solve_Ut_jacobi(
        dense_vector_t<T> * b,
        const mat_int_t sweeps)
        const;

    void solve_D(
        dense_vector_t<T> * b) const;

    void solve_P(
        const dense_vector_t<T> * b,
        dense_vector_t<T> * x) const;
    void solve_Pt(
        const dense_vector_t<T> * b,
        dense_vector_t<T> * x) const;

protected:
    /* input data */
    const csr_matrix_t<T> * m_in_h_A;
    const mat_int_t m_num_blocks;
    const mat_int_t m_num_pivs;
    const mat_int_t * m_piv_starts;

    SOLVE_ALGORITHM m_solve_algorithm = SOLVE_BLOCK;
    mat_int_t m_jacobi_sweeps = 8;

    dense_vector_ptr<mat_int_t> m_rowcol_in_block;

    /* matrix data on CPU */
    compressed_block_list m_h_coarse_csr;
    compressed_block_list m_h_coarse_csc;

    dense_vector_ptr<mat_int_t> m_h_block_col;
    dense_vector_ptr<T> m_h_block_val;

    dense_vector_ptr<mat_int_t> m_h_block_starts;

    dense_vector_ptr<matrix_block> m_h_blocks;
    dense_vector_ptr<mat_int_t> m_h_block_ix_store;
    dense_vector_ptr<T> m_h_block_val_store;

    dense_vector_ptr<T> m_h_tridiagonal;
    dense_vector_ptr<mat_int_t> m_h_ispiv;

    /* level data on CPU */
    dense_vector_ptr<mat_int_t> m_block_in_levels_ix;

    mat_int_t m_max_level_size;
    mat_int_t m_max_level_blocks;

    compressed_block_list m_h_levels;
    compressed_block_list m_h_levels_blocks;

    dense_vector_ptr<mat_int_t> m_h_row_first_block_offset;

    /* other data on the CPU */
    dense_vector_ptr<mat_int_t> m_h_permutation;

    /* data for solving with U / U' */
    compressed_block_list m_h_levels_blocks_solve;

    /* matrix data on GPU */
    compressed_block_list m_d_coarse_csr;
    compressed_block_list m_d_coarse_csc;

    dense_vector_ptr<matrix_block> m_d_blocks;
    dense_vector_ptr<mat_int_t> m_d_block_ix_store;
    dense_vector_ptr<T> m_d_block_val_store;

    dense_vector_ptr<mat_int_t> m_d_block_starts;

    dense_vector_ptr<T> m_d_tridiagonal;
    dense_vector_ptr<mat_int_t> m_d_ispiv;

    /* level data on GPU */
    compressed_block_list m_d_levels;
    compressed_block_list m_d_levels_blocks;

    dense_vector_ptr<mat_int_t> m_d_row_first_block_offset;

    /* temporary (in-flight) data on the GPU */
    dense_vector_ptr<T> m_d_blocks_inflight;
    dense_vector_ptr<mat_int_t> m_d_piv_block_location;

    dense_vector_ptr<T> m_d_row_norms;
    dense_vector_ptr<mat_int_t> m_d_permutation;

    dense_vector_ptr<mat_int_t> m_d_storage_mgmt;

    /* data for solving with U / U' */
    compressed_block_list m_d_levels_blocks_solve;

    /* converted U */
    csr_matrix_ptr<T> m_d_csr_U;

    /* storage for jacobi solving */
    mutable dense_vector_ptr<T> m_gpu_jacobi_a;
    mutable dense_vector_ptr<T> m_gpu_jacobi_b;

    /* storage for solving */
    mutable dense_vector_ptr<T> m_tmp_solve;

    /* invalid marker */
    mat_int_t m_invalid = -1;
};

NS_LA_END
NS_CULIP_END

#endif /* __CULIP_LIBS_LA_BLOCK_LDLT_CUH_ */