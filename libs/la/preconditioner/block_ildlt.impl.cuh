/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */
#include <libs/la/preconditioner/block_ildlt.cuh>
#include <libs/la/preconditioner/block_ildlt_kernels.impl.cuh>

#include <libs/la/sparse_la.cuh>
#include <libs/la/dense_la.cuh>
#include <libs/la/helper_kernels.cuh>

#include <libs/algorithms/matching.cuh>
#include <libs/test/test.h>

#include <omp.h>

#include <set>
#include <queue>
#include <numeric>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

NS_CULIP_BEGIN

using namespace NS_ALGORITHMS;

NS_LA_BEGIN

/*
 * *****************************************************************************
 * ********************************* PUBLIC ************************************
 * *****************************************************************************
 */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
BlockiLDLt(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * h_A,
    const mat_int_t num_pivs,
    const mat_int_t * pivs,
    const mat_int_t num_blocks,
    const mat_int_t * block_starts)
: Preconditioner<T>(gpu_handle),
  m_in_h_A(h_A),
  m_num_pivs(num_pivs),
  m_piv_starts(pivs),
  m_num_blocks(num_blocks),
  m_h_coarse_csr(false),
  m_h_coarse_csc(false),
  m_h_levels(false),
  m_h_levels_blocks(false),
  m_d_coarse_csr(true),
  m_d_coarse_csc(true),
  m_d_levels(true),
  m_d_levels_blocks(true),
  m_h_levels_blocks_solve(false),
  m_d_levels_blocks_solve(true),
  m_gpu_jacobi_a(make_managed_dense_vector_ptr<T>(h_A->m, true)),
  m_gpu_jacobi_b(make_managed_dense_vector_ptr<T>(h_A->m, true)),
  m_tmp_solve(make_managed_dense_vector_ptr<T>(h_A->m, true))
{
    m_h_block_starts = make_managed_dense_vector_ptr<mat_int_t>(
        num_blocks + 1, false);
    std::copy(block_starts, block_starts + num_blocks,
        m_h_block_starts->dense_val);
    (*m_h_block_starts)[num_blocks] = h_A->m;

    //m_h_block_starts->print("Block starts");
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
~BlockiLDLt()
{

}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
mat_int_t
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
n()
const
{
    return m_in_h_A->m;
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
T
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
compute(
    const mat_int_t fill_level,
    const T fill_factor,
    const T threshold)
{
    dense_vector_ptr<mat_int_t> A_block_nnz, U_block_nnz;

    /* create the initial block-CSR ("coarse") upper-triangular matrix */
    build_initial_coarse(A_block_nnz);

    /* add level-of-fill blocks */
    const mat_int_t fill_in_nz = determine_coarse_level_fill_and_diag(
        fill_level, A_block_nnz->dense_val, U_block_nnz);

    /* add a coarse CSC layer for column traversal later */
    build_block_csc();

    /* find row-based level sets */
    find_level_sets();

    /* fill the created blocks with values from the input matrix */
    build_blocks(fill_factor, U_block_nnz->dense_val);

    /* upload data to GPU */
    upload_data();

    /* compute the factorization */
    factorize(threshold);

    /* download results */
    download_data();

    /* print some statistics */
    eval_results();

    /* export the factorized matrix U */
    // export_coarse();
    // export_factorized();

    return 0;
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
bool
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
is_left()
const
{
    return true;
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
bool
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
is_middle()
const
{
    return true;
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
bool
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
is_right()
const
{
    return true;
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_left(
    const dense_vector_t<T> * b,
    dense_vector_t<T> * x,
    const bool transpose)
const
{
    if(!transpose)
    {
        /* solve with P'U' */
        solve_Pt(b, x);
        solve_Ut(x);
    }
    else
    {
        solve_right(b, x, false);
    }
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_middle(
    const dense_vector_t<T> * b,
    dense_vector_t<T> * x,
    const bool transpose)
const
{
    /* solve with D */
    cudaMemcpyAsync(x->dense_val, b->dense_val, b->m * sizeof(T),
        cudaMemcpyDeviceToDevice, this->m_handle->get_stream());

    solve_D(x);
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_right(
    const dense_vector_t<T> * b,
    dense_vector_t<T> * x,
    const bool transpose)
const
{
    cudaMemcpyAsync(m_tmp_solve->dense_val, b->dense_val, b->m * sizeof(T),
        cudaMemcpyDeviceToDevice, this->m_handle->get_stream());

    if(!transpose)
    {
        /* solve with UP */
        solve_U(m_tmp_solve.get());
        solve_P(m_tmp_solve.get(), x);
    }
    else
    {
        solve_left(b, x, false);
    }
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
set_solve_algorithm(
    SOLVE_ALGORITHM algorithm,
    const mat_int_t num_jacobi_sweeps)
{
    m_solve_algorithm = algorithm;
    m_jacobi_sweeps = num_jacobi_sweeps;
}

/*
 * *****************************************************************************
 * ************************ PROTECTED / FACTORIZATION **************************
 * *****************************************************************************
 */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
offsets_to_indices(
    const mat_int_t m,
    const mat_int_t nnz,
    const mat_int_t * offsets,
    mat_int_t * indices)
{
    struct is_empty_row
    {
        __host__
        bool operator()(const thrust::tuple<mat_int_t, mat_int_t>& t)
        {
            return (thrust::get<0>(t) != thrust::get<1>(t));
        }
    };
    thrust::fill(thrust::host, indices, indices + nnz, 0);
    thrust::scatter_if(
        thrust::host,
        thrust::counting_iterator<mat_int_t>(0),
        thrust::counting_iterator<mat_int_t>(m),
        offsets,
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(thrust::make_tuple(
                offsets,
                offsets + 1)),
                is_empty_row()),
        indices);
    thrust::inclusive_scan(thrust::host, indices, indices + nnz, indices,
        thrust::maximum<mat_int_t>());
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
indices_to_offsets(
    const mat_int_t m,
    const mat_int_t nnz,
    const mat_int_t * indices,
    mat_int_t * offsets)
{
    dense_vector_ptr<mat_int_t> keys =
        make_managed_dense_vector_ptr<mat_int_t>(nnz, false);
    dense_vector_ptr<mat_int_t> cardinalities =
        make_managed_dense_vector_ptr<mat_int_t>(nnz, false);

    /* count rows with nonzero entries */
    const auto found_rows = thrust::reduce_by_key(
        thrust::host,
        indices,
        indices + nnz,
        thrust::make_constant_iterator<mat_int_t>(1),
        keys->dense_val,
        cardinalities->dense_val);
    const mat_int_t nz_rows = thrust::get<0>(found_rows) - keys->dense_val;

    /* reset offsets */
    thrust::fill(
        thrust::host,
        offsets,
        offsets + m,
        0);
    offsets[m] = 0;

    /* scatter row counts to dense offset array */
    thrust::scatter(
        thrust::host,
        cardinalities->dense_val,
        cardinalities->dense_val + nz_rows,
        keys->dense_val,
        offsets);

    /* compute offsets from the row counts */
    thrust::exclusive_scan(
        thrust::host,
        offsets,
        offsets + m + 1,
        offsets);
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
build_initial_coarse(
    dense_vector_ptr<mat_int_t>& block_nnz)
{
    const mat_int_t coarse_m = m_num_blocks;

    /**
     * build a coarse representation of the upper triangular U part
     */

    /* create a row -> block map */
    m_rowcol_in_block = make_managed_dense_vector_ptr<mat_int_t>(
        m_in_h_A->m, false);
    for(mat_int_t i = 0; i < m_num_blocks; ++i)
    {
        const mat_int_t i_from = (*m_h_block_starts)[i];
        const mat_int_t i_to = (*m_h_block_starts)[i + 1];

        for(mat_int_t j = i_from; j < i_to; ++j)
            (*m_rowcol_in_block)[j] = i;
    }

    /**
     * compute initial blocks per row
     */

    /* extract elements of fine U */
    dense_vector_ptr<mat_int_t> coo_row =
        make_managed_dense_vector_ptr<mat_int_t>(m_in_h_A->nnz, false);
    dense_vector_ptr<mat_int_t> coo_col =
        make_managed_dense_vector_ptr<mat_int_t>(m_in_h_A->nnz, false);

    /* csr row offsets to indices */
    offsets_to_indices(m_in_h_A->m, m_in_h_A->nnz, m_in_h_A->csr_row,
        coo_row->dense_val);

    /* copy csr indices */
    thrust::copy(
        thrust::host,
        m_in_h_A->csr_col,
        m_in_h_A->csr_col + m_in_h_A->nnz,
        coo_col->dense_val);

    /* filter the U entries */
    struct is_tril
    {
        __host__
        bool operator()(const thrust::tuple<mat_int_t, mat_int_t>& t)
        {
            /* row > col -> means it's in L and not in U */
            return (thrust::get<0>(t) > thrust::get<1>(t));
        }
    };
    const auto U_nnz_end =
        thrust::remove_if(
            thrust::host,
            thrust::make_zip_iterator(thrust::make_tuple(
                coo_row->dense_val,
                coo_col->dense_val)),
            thrust::make_zip_iterator(thrust::make_tuple(
                coo_row->dense_val + m_in_h_A->nnz,
                coo_col->dense_val + m_in_h_A->nnz)),
            is_tril());

    /* compute number of nonzeros in U, since there may be 0s on diagonal */
    const mat_int_t U_nnz = U_nnz_end -
        thrust::make_zip_iterator(thrust::make_tuple(
            coo_row->dense_val,
            coo_col->dense_val));
    printf("Fine nnz: %d\n", U_nnz);

    /* map U's entries to block row and cols and sort */
    thrust::transform(
        thrust::host,
        coo_row->dense_val,
        coo_row->dense_val + U_nnz,
        coo_row->dense_val,
        thrust_map_func<mat_int_t>(m_rowcol_in_block->dense_val));
    thrust::transform(
        thrust::host,
        coo_col->dense_val,
        coo_col->dense_val + U_nnz,
        coo_col->dense_val,
        thrust_map_func<mat_int_t>(m_rowcol_in_block->dense_val));

    thrust::sort(
        thrust::host,
        thrust::make_zip_iterator(thrust::make_tuple(
            coo_row->dense_val,
            coo_col->dense_val)),
        thrust::make_zip_iterator(thrust::make_tuple(
            coo_row->dense_val + U_nnz,
            coo_col->dense_val + U_nnz)),
        thrust_tuple_sort());

    /* now determine the number of NNZ per block and the block entries */
    dense_vector_ptr<mat_int_t> block_coo_row =
        make_managed_dense_vector_ptr<mat_int_t>(U_nnz, false);
    dense_vector_ptr<mat_int_t> block_coo_col =
        make_managed_dense_vector_ptr<mat_int_t>(U_nnz, false);
    block_nnz = make_managed_dense_vector_ptr<mat_int_t>(U_nnz, false);
    auto start_in_it = thrust::make_zip_iterator(thrust::make_tuple(
        coo_row->dense_val,
        coo_col->dense_val));
    auto start_out_it = thrust::make_zip_iterator(thrust::make_tuple(
        block_coo_row->dense_val,
        block_coo_col->dense_val));
    const auto start_out_end_tpl = thrust::reduce_by_key(
        thrust::host,
        start_in_it,
        start_in_it + U_nnz,
        thrust::make_constant_iterator<mat_int_t>(1),
        start_out_it,
        block_nnz->dense_val,
        thrust_tuple_equal());
    const mat_int_t coarse_nnz = thrust::get<0>(start_out_end_tpl) -
        start_out_it;
    printf("Coarse nnz: %d\n", coarse_nnz);

    /* compute block CSR offsets */
    m_h_coarse_csr = compressed_block_list(false, coarse_m, coarse_nnz);
    indices_to_offsets(coarse_m, coarse_nnz, block_coo_row->dense_val,
        m_h_coarse_csr.offsets->dense_val);

    /* copy block CSR indices */
    thrust::copy(
        thrust::host,
        block_coo_col->dense_val,
        block_coo_col->dense_val + coarse_nnz,
        m_h_coarse_csr.indices->dense_val);
}

/* ************************************************************************** */

/**
 * note: also adds diagonal block if there is one missing; works on fine
 * graph to avoid transposing the coarse graph
 */
template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
mat_int_t
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
determine_coarse_level_fill_and_diag(
    const mat_int_t max_level,
    const mat_int_t * in_block_nnz,
    dense_vector_ptr<mat_int_t>& out_block_nnz)
{
    const mat_int_t coarse_m = m_num_blocks;

    dense_vector_ptr<mat_int_t> block_row_fi_num =
        make_managed_dense_vector_ptr<mat_int_t>(coarse_m + 1, false);
    thrust::fill(thrust::host, block_row_fi_num->dense_val,
        block_row_fi_num->dense_val + coarse_m + 1, 0);

    /**
     * algorithm see "Level-based Incomplete LU factorization" by Hysom et al.,
     */
    #pragma omp parallel for
    for(mat_int_t i = 0; i < coarse_m; ++i)
    {
        /* compute structure for row U */
        std::vector<bool> visited(coarse_m, false);
        std::vector<mat_int_t> length(coarse_m, 0);

        std::queue<mat_int_t> front;
        front.push(i);
        length[i] = 0;
        visited[i] = true;

        while(!front.empty())
        {
            const mat_int_t cur_coarse = front.front();
            front.pop();

            /* iterate over fine rows in this block */
            const mat_int_t cur_from = (*m_h_block_starts)[cur_coarse];
            const mat_int_t cur_to = (*m_h_block_starts)[cur_coarse + 1];
            for(mat_int_t j = cur_from; j < cur_to; ++j)
            {
                /* use input matrix as CSR + CSC */
                const mat_int_t j_len = m_in_h_A->csr_row[j + 1] -
                    m_in_h_A->csr_row[j];
                const mat_int_t * j_ix = m_in_h_A->csr_col +
                    m_in_h_A->csr_row[j];

                for(mat_int_t k = 0; k < j_len; ++k)
                {
                    const mat_int_t k_coarse = (*m_rowcol_in_block)[j_ix[k]];

                    if(!visited[k_coarse])
                    {
                        visited[k_coarse] = true;

                        if(k_coarse < i && length[cur_coarse] < max_level)
                        {
                            front.push(k_coarse);
                            length[k_coarse] = length[cur_coarse] + 1;
                        }
                        if(k_coarse > i && length[cur_coarse] > 0)
                        {
                            ++(*block_row_fi_num)[i];
                        }
                    }
                }
            }
        }

        /* check for diagonal entry */
        bool has_diag_block = false;
        for(mat_int_t j = (*m_h_coarse_csr.offsets)[i];
            j < (*m_h_coarse_csr.offsets)[i] + 1; ++j)
            has_diag_block |= ((*m_h_coarse_csr.indices)[j] == i);

        if(!has_diag_block)
            ++(*block_row_fi_num)[i];
    }
    (*block_row_fi_num)[coarse_m] = 0;

    /* compute offsets from the additional NNZ numbers */
    thrust::exclusive_scan(
        thrust::host,
        block_row_fi_num->dense_val,
        block_row_fi_num->dense_val + coarse_m + 1,
        block_row_fi_num->dense_val);
    const mat_int_t new_nnz = (*block_row_fi_num)[coarse_m];
    printf("Fill-in coarse nnz: %d\n", new_nnz);

    if(new_nnz == 0)
    {
        out_block_nnz = make_managed_dense_vector_ptr<mat_int_t>(
            m_h_coarse_csr.nnz, false);
        thrust::copy(
            in_block_nnz,
            in_block_nnz + m_h_coarse_csr.nnz,
            out_block_nnz->dense_val);
        return 0;
    }

    /* in a second search, add directly found nz to structure */
    compressed_block_list new_coarse_csr(false, coarse_m, m_h_coarse_csr.nnz +
        new_nnz);
    thrust::fill(thrust::host, new_coarse_csr.offsets->dense_val,
        new_coarse_csr.offsets->dense_val + coarse_m + 1, 0);
    thrust::copy(thrust::host, m_h_coarse_csr.indices->dense_val,
        m_h_coarse_csr.indices->dense_val + m_h_coarse_csr.nnz,
        new_coarse_csr.indices->dense_val);

    dense_vector_ptr<mat_int_t> full_coo_row =
        make_managed_dense_vector_ptr<mat_int_t>(m_h_coarse_csr.nnz + new_nnz,
        false);
    offsets_to_indices(coarse_m, m_h_coarse_csr.nnz,
        m_h_coarse_csr.offsets->dense_val, full_coo_row->dense_val);

    #pragma omp parallel for
    for(mat_int_t i = 0; i < coarse_m; ++i)
    {
        /* compute structure for row U */
        std::vector<bool> visited(coarse_m, false);
        std::vector<mat_int_t> length(coarse_m, 0);

        std::queue<mat_int_t> front;
        front.push(i);
        length[i] = 0;
        visited[i] = true;

        while(!front.empty())
        {
            const mat_int_t cur_coarse = front.front();
            front.pop();

            /* iterate over fine rows in this block */
            const mat_int_t cur_from = (*m_h_block_starts)[cur_coarse];
            const mat_int_t cur_to = (*m_h_block_starts)[cur_coarse + 1];
            for(mat_int_t j = cur_from; j < cur_to; ++j)
            {
                const mat_int_t j_len = m_in_h_A->csr_row[j + 1] -
                    m_in_h_A->csr_row[j];
                const mat_int_t * j_ix = m_in_h_A->csr_col +
                    m_in_h_A->csr_row[j];

                for(mat_int_t k = 0; k < j_len; ++k)
                {
                    const mat_int_t k_coarse = (*m_rowcol_in_block)[j_ix[k]];

                    if(!visited[k_coarse])
                    {
                        visited[k_coarse] = true;

                        if(k_coarse < i && length[cur_coarse] < max_level)
                        {
                            front.push(k_coarse);
                            length[k_coarse] = length[cur_coarse] + 1;
                        }
                        if(k_coarse > i && length[cur_coarse] > 0)
                        {
                            (*new_coarse_csr.indices)[
                                m_h_coarse_csr.nnz +
                                (*block_row_fi_num)[i] +
                                (*new_coarse_csr.offsets)[i]] = k_coarse;
                            (*full_coo_row)[
                                m_h_coarse_csr.nnz +
                                (*block_row_fi_num)[i] +
                                (*new_coarse_csr.offsets)[i]] = i;

                            ++(*new_coarse_csr.offsets)[i];
                        }
                    }
                }
            }
        }

        /* check for diagonal entry */
        const mat_int_t predicted_nnz = (*block_row_fi_num)[i + 1] -
            (*block_row_fi_num)[i];
        if(predicted_nnz > (*new_coarse_csr.offsets)[i])
        {
            (*new_coarse_csr.indices)[
                m_h_coarse_csr.nnz +
                (*block_row_fi_num)[i] +
                (*new_coarse_csr.offsets)[i]] = i;
            (*full_coo_row)[
                m_h_coarse_csr.nnz +
                (*block_row_fi_num)[i] +
                (*new_coarse_csr.offsets)[i]] = i;

            ++(*new_coarse_csr.offsets)[i];
        }
    }

    /* update CSR offsets (additional fill in + orig count, the prefix sum) */
    thrust::transform(
        thrust::host,
        m_h_coarse_csr.offsets->dense_val + 1,
        m_h_coarse_csr.offsets->dense_val + coarse_m + 1,
        m_h_coarse_csr.offsets->dense_val,
        block_row_fi_num->dense_val,
        thrust::minus<mat_int_t>());
    thrust::transform(
        thrust::host,
        new_coarse_csr.offsets->dense_val,
        new_coarse_csr.offsets->dense_val + coarse_m + 1,
        block_row_fi_num->dense_val,
        new_coarse_csr.offsets->dense_val,
        thrust::plus<mat_int_t>());
    thrust::exclusive_scan(
        thrust::host,
        new_coarse_csr.offsets->dense_val,
        new_coarse_csr.offsets->dense_val + coarse_m + 1,
        new_coarse_csr.offsets->dense_val);

    /* save how many NNZ from the original matrix go into the blocks */
    out_block_nnz =
        make_managed_dense_vector_ptr<mat_int_t>(new_coarse_csr.nnz, false);
    thrust::copy(thrust::host, in_block_nnz,
        in_block_nnz + m_h_coarse_csr.nnz,
        out_block_nnz->dense_val);
    thrust::fill(thrust::host, out_block_nnz->dense_val + m_h_coarse_csr.nnz,
        out_block_nnz->dense_val + new_coarse_csr.nnz, 0);

    /* reorder entries to get the final, valid coarse CSR */
    thrust::sort_by_key(
        thrust::host,
        thrust::make_zip_iterator(thrust::make_tuple(
            full_coo_row->dense_val,
            new_coarse_csr.indices->dense_val)),
        thrust::make_zip_iterator(thrust::make_tuple(
            full_coo_row->dense_val + new_coarse_csr.nnz,
            new_coarse_csr.indices->dense_val + new_coarse_csr.nnz)),
        out_block_nnz->dense_val,
        thrust_tuple_sort());

    /* override old CSR structure */
    m_h_coarse_csr = new_coarse_csr;

    return new_nnz;
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
build_block_csc()
{
    m_h_coarse_csc = compressed_block_list(false, m_h_coarse_csr.m,
        m_h_coarse_csr.nnz);

    /* expand CSR offsets and copy CSR indices */
    offsets_to_indices(m_h_coarse_csr.m, m_h_coarse_csr.nnz,
        m_h_coarse_csr.offsets->dense_val, m_h_coarse_csc.indices->dense_val);

    dense_vector_ptr<mat_int_t> coo_col =
        make_managed_dense_vector_ptr<mat_int_t>(false);
    *coo_col = m_h_coarse_csr.indices.get();

    /* sort these by column index */
    thrust::sort(
        thrust::host,
        thrust::make_zip_iterator(thrust::make_tuple(
            coo_col->dense_val,
            m_h_coarse_csc.indices->dense_val)),
        thrust::make_zip_iterator(thrust::make_tuple(
            coo_col->dense_val + m_h_coarse_csr.nnz,
            m_h_coarse_csc.indices->dense_val + m_h_coarse_csr.nnz)),
        thrust_tuple_sort());

    /* compress CSC offsets */
    indices_to_offsets(m_h_coarse_csc.m, m_h_coarse_csc.nnz,
        coo_col->dense_val, m_h_coarse_csc.offsets->dense_val);
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
find_level_sets()
{
    const mat_int_t coarse_m = m_h_coarse_csr.m;
    const mat_int_t coarse_nnz = m_h_coarse_csr.nnz;

    /* allocate storage for levels */
    m_h_levels = compressed_block_list(false, coarse_m, coarse_m);

    /* count col dep sizes (assume full diagonal) */
    dense_vector_ptr<mat_int_t> dep_left =
        make_managed_dense_vector_ptr<mat_int_t>(coarse_m, false);
    thrust::transform(
        thrust::host,
        m_h_coarse_csc.offsets->dense_val + 1,
        m_h_coarse_csc.offsets->dense_val + coarse_m + 1,
        m_h_coarse_csc.offsets->dense_val,
        dep_left->dense_val,
        thrust::minus<mat_int_t>());

    dense_vector_ptr<mat_int_t> row_levels =
        make_managed_dense_vector_ptr<mat_int_t>(coarse_m, false);
    dense_vector_ptr<mat_int_t> levels_blocks =
        make_managed_dense_vector_ptr<mat_int_t>(coarse_m, false);

    /* first queue: select all where deps == 1 (singletons) */
    const mat_int_t * in_end = thrust::copy_if(
        thrust::host,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(coarse_m),
        dep_left->dense_val,
        m_h_levels.indices->dense_val,
        stencil_is_one<mat_int_t>());
    (*m_h_levels.offsets)[0] = 0;
    (*m_h_levels.offsets)[1] = in_end - m_h_levels.indices->dense_val;

    m_max_level_size = (*m_h_levels.offsets)[1];
    m_max_level_blocks = 0;
    for(mat_int_t i = 0; i < (*m_h_levels.offsets)[1]; ++i)
    {
        const mat_int_t cur = (*m_h_levels.indices)[i];
        m_max_level_blocks += ((*m_h_coarse_csr.offsets)[cur + 1] -
            (*m_h_coarse_csr.offsets)[cur]);
        (*row_levels)[cur] = 0;
    }
    (*levels_blocks)[0] = m_max_level_blocks;

    mat_int_t level = 0;
    mat_int_t offset = (*m_h_levels.offsets)[1];
    mat_int_t prev_offset = (*m_h_levels.offsets)[0];
    while(prev_offset < offset)
    {
        ++level;

        mat_int_t level_blocks = 0;

        /* next frontal BFS step */
        const mat_int_t cur_offset = offset;

        #pragma omp parallel for
        for(mat_int_t i = prev_offset; i < cur_offset; ++i)
        {
            const mat_int_t cur = (*m_h_levels.indices)[i];

            /* walk successors (in row) and update their level */
            for(mat_int_t j = (*m_h_coarse_csr.offsets)[cur];
                j < (*m_h_coarse_csr.offsets)[cur + 1]; ++j)
            {
                const mat_int_t j_col = (*m_h_coarse_csr.indices)[j];

                /* singleton left, can add node to next level */
                mat_int_t j_dep_left;

                #pragma omp atomic capture
                {
                    --dep_left->dense_val[j_col];
                    j_dep_left = dep_left->dense_val[j_col];
                }

                if(j_dep_left == 1)
                {
                    mat_int_t pos;

                    #pragma omp atomic capture
                    { pos = offset; offset++; }

                    (*m_h_levels.indices)[pos] = j_col;
                    (*row_levels)[j_col] = level;

                    /* consider rows in level size */
                    const mat_int_t j_col_row_size =
                        ((*m_h_coarse_csr.offsets)[j_col + 1] -
                        (*m_h_coarse_csr.offsets)[j_col]);

                    #pragma omp atomic
                    level_blocks += j_col_row_size;
                }
            }
        }

        /* record level size in blocks */
        (*levels_blocks)[level] = level_blocks;

        /* compare with previous levels */
        m_max_level_size = std::max(m_max_level_size, offset - cur_offset);
        m_max_level_blocks = std::max(m_max_level_blocks, level_blocks);

        /* go to next level */
        (*m_h_levels.offsets)[level + 1] = offset;
        prev_offset = cur_offset;
    }

    printf("Levels: %d (max. size: %d, max blocks: %d)\n", level,
        m_max_level_size, m_max_level_blocks);

    /* update level counter */
    m_h_levels.m = level;

    /**
     * create a list of all blocks and their levels for scheduling
     */
    m_h_levels_blocks = compressed_block_list(false, level, coarse_nnz);

    /**
     * Blocks are initialized later, but we need their metadata
     * (row, col) now
     */
    m_h_blocks = make_managed_dense_vector_ptr<matrix_block>(coarse_nnz, false);

    #pragma omp parallel for
    for(mat_int_t i = 0; i < coarse_m; ++i)
    {
        const mat_int_t i_len = (*m_h_coarse_csr.offsets)[i + 1]  -
            (*m_h_coarse_csr.offsets)[i];
        mat_int_t * i_ix = m_h_coarse_csr.indices->dense_val +
            (*m_h_coarse_csr.offsets)[i];

        for(mat_int_t j = 0; j < i_len; ++j)
        {
            const mat_int_t block_id = (*m_h_coarse_csr.offsets)[i] + j;
            matrix_block * j_block = m_h_blocks->dense_val + block_id;

            j_block->id = block_id;
            j_block->block_row = i;
            j_block->block_col = i_ix[j];
        }
    }

    /* sort blocks per level, starting with the pivot blocks */
    struct thrust_pivot_tuple_sort_row
    {
        const mat_int_t * row_level_map;
        const matrix_block * blocks;

        thrust_pivot_tuple_sort_row(
            const mat_int_t * in_level_map,
            const matrix_block * in_blocks)
        {
            row_level_map = in_level_map;
            blocks = in_blocks;
        }

        bool
        operator()(const mat_int_t blk0, const mat_int_t blk1)
        {
            const matrix_block * b_blk0 = blocks + blk0;
            const matrix_block * b_blk1 = blocks + blk1;

            if(row_level_map[b_blk0->block_row] !=
                row_level_map[b_blk1->block_row])
                return (row_level_map[b_blk0->block_row] <
                    row_level_map[b_blk1->block_row]);

            const bool blk0_piv = (blocks[blk0].block_row ==
                blocks[blk0].block_col);
            const bool blk1_piv = (blocks[blk1].block_row ==
                blocks[blk1].block_col);

            return (blk0_piv && !blk1_piv);
        }
    };
    thrust::copy(
        thrust::host,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(coarse_nnz),
        m_h_levels_blocks.indices->dense_val);
    thrust::sort(
        thrust::host,
        m_h_levels_blocks.indices->dense_val,
        m_h_levels_blocks.indices->dense_val + coarse_nnz,
        thrust_pivot_tuple_sort_row(
            row_levels->dense_val,
            m_h_blocks->dense_val));

    /* create offsets for level/block list */
    thrust::exclusive_scan(
        thrust::host,
        levels_blocks->dense_val,
        levels_blocks->dense_val + level + 1,
        m_h_levels_blocks.offsets->dense_val);

    /* reverse m_h_blocks.indices to get a map block -> index in level list */
    m_block_in_levels_ix =
        make_managed_dense_vector_ptr<mat_int_t>(coarse_nnz, false);

    thrust::scatter(
        thrust::host,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(coarse_nnz),
        m_h_levels_blocks.indices->dense_val,
        m_block_in_levels_ix->dense_val);

    /**
     * find offsets of row's first offdiagonal block in each level; this
     * allows efficient row traversal in the dense in-flight storage later
     */
    m_h_row_first_block_offset =
        make_managed_dense_vector_ptr<mat_int_t>(coarse_m, false);

    #pragma omp parallel for
    for(mat_int_t l = 0; l < level; ++l)
    {
        const mat_int_t l_piv_size = (*m_h_levels.offsets)[l + 1] -
            (*m_h_levels.offsets)[l];
        const mat_int_t l_nopiv_size = (*m_h_levels_blocks.offsets)[l + 1] -
            (*m_h_levels_blocks.offsets)[l] - l_piv_size;

        mat_int_t prev_row = -1;
        for(mat_int_t j = (*m_h_levels_blocks.offsets)[l] + l_piv_size;
            j < (*m_h_levels_blocks.offsets)[l + 1]; ++j)
        {
            const matrix_block * j_blk = (m_h_blocks->dense_val +
                (*m_h_levels_blocks.indices)[j]);
            const mat_int_t j_row = j_blk->block_row;

            if(prev_row != j_row)
            {
                (*m_h_row_first_block_offset)[j_row] = j -
                    (*m_h_levels_blocks.offsets)[l] - l_piv_size;
                prev_row = j_row;
            }
        }
    }
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
build_blocks(
    const T fill_factor,
    const mat_int_t * block_A_nnz)
{
    const T speed_loss_factor = 0.5;

    const mat_int_t coarse_m = m_h_coarse_csr.m;
    const mat_int_t coarse_nnz = m_h_coarse_csr.nnz;

    const auto should_be_sparse =
        [&speed_loss_factor](
            const mat_int_t nnz,
            const mat_int_t rows,
            const mat_int_t cols)
        {
            /* comapre storage needs in bytes + compromise factor */
            return ((sizeof(mat_int_t) + sizeof(T)) * nnz <
                 speed_loss_factor * sizeof(T) * rows * cols);
        };

    /* determine blocks' sizes */
    dense_vector_ptr<mat_size_t> block_ix_space =
        make_managed_dense_vector_ptr<mat_size_t>(coarse_nnz + 1, false);
    dense_vector_ptr<mat_size_t> block_val_space =
        make_managed_dense_vector_ptr<mat_size_t>(coarse_nnz + 1, false);

    (*block_ix_space)[coarse_nnz] = 0;
    (*block_val_space)[coarse_nnz] = 0;

    dense_vector_ptr<mat_int_t> csr_coo_row =
        make_managed_dense_vector_ptr<mat_int_t>(coarse_nnz, false);
    dense_vector_ptr<mat_int_t> csr_coo_col =
        make_managed_dense_vector_ptr<mat_int_t>(coarse_nnz, false);

    mat_int_t blks_sparse = 0;
    mat_int_t blks_dense = 0;

    mat_int_t avg_size = 0;
    T avg_fill = 0;
    T avg_fill_sq = 0;

    #pragma omp parallel for
    for(mat_int_t i = 0; i < coarse_m; ++i)
    {
        const mat_int_t i_len = (*m_h_coarse_csr.offsets)[i + 1]  -
            (*m_h_coarse_csr.offsets)[i];
        mat_int_t * i_ix = m_h_coarse_csr.indices->dense_val +
            (*m_h_coarse_csr.offsets)[i];

        /* first pass: compute average number of elements/block in row */
        mat_int_t sum_nnz = 0;
        mat_int_t sum_blocks = 0;
        for(mat_int_t j = 0; j < i_len; ++j)
        {
            const mat_int_t block_id = (*m_h_coarse_csr.offsets)[i] + j;
            sum_nnz += block_A_nnz[block_id];
            sum_blocks += (block_A_nnz[block_id] != 0);
        }
        const mat_int_t avg_nnz = (mat_int_t) std::ceil(sum_nnz / (T)
            sum_blocks);

        for(mat_int_t j = 0; j < i_len; ++j)
        {
            const mat_int_t block_id = (*m_h_coarse_csr.offsets)[i] + j;
            matrix_block * j_block = m_h_blocks->dense_val + block_id;

            j_block->id = block_id;
            j_block->block_row = i;
            j_block->block_col = i_ix[j];
            j_block->nnz = block_A_nnz[block_id];

            j_block->got_extra = 0;

            const mat_int_t j_from_row =
                (*m_h_block_starts)[j_block->block_row];
            const mat_int_t j_to_row =
                (*m_h_block_starts)[j_block->block_row + 1];

            const mat_int_t j_from_col =
                (*m_h_block_starts)[j_block->block_col];
            const mat_int_t j_to_col =
                (*m_h_block_starts)[j_block->block_col + 1];

            const mat_int_t j_rows = j_to_row - j_from_row;
            const mat_int_t j_cols = j_to_col - j_from_col;

            const mat_int_t src_nnz = (j_block->nnz == 0) ?
                avg_nnz : j_block->nnz;
            j_block->max_nnz = std::min(j_rows * j_cols,
                (mat_int_t) std::ceil(fill_factor * src_nnz));

            const T fill = j_block->nnz / (T) (j_rows * j_cols);
            const T fill_sq = fill * fill;
            const mat_int_t blk_size = j_rows * j_cols;

            #pragma omp atomic
            avg_size += blk_size;

            #pragma omp atomic
            avg_fill += fill;

            #pragma omp atomic
            avg_fill_sq += fill_sq;

            /* organize space / pointers according to level ordering */
            const mat_int_t lvl_list_ix = (*m_block_in_levels_ix)[block_id];

            /* decide on the layout and the storage needs */
            if(j_block->block_row != j_block->block_col &&
                j_block->block_row < 0.9 * coarse_m &&
                should_be_sparse(j_block->max_nnz, j_rows, j_cols))
            {
                j_block->format = BLOCK_SPARSE;
                (*block_ix_space)[lvl_list_ix] = j_block->max_nnz;
                (*block_val_space)[lvl_list_ix] = j_block->max_nnz;

                #pragma omp atomic
                ++blks_sparse;
            }
            else
            {
                /* pivot blocks are always dense to avoid dropping */
                j_block->format = BLOCK_DENSE;
                (*block_ix_space)[lvl_list_ix] = 0;
                (*block_val_space)[lvl_list_ix] = j_rows * j_cols;

                #pragma omp atomic
                ++blks_dense;
            }

            /* overwrite block column with block's ID */
            (*csr_coo_row)[block_id] = i;
            (*csr_coo_col)[block_id] = i_ix[j];
            i_ix[j] = block_id;
        }
    }

    const T blk_avg_size = ((T) avg_size) / coarse_nnz;
    const T blk_avg_fill = avg_fill / coarse_nnz;
    const T blk_stddev_fill = std::sqrt(avg_fill_sq / coarse_nnz -
        blk_avg_fill * blk_avg_fill);

    printf("Block formats: sparse %d, dense %d\n", blks_sparse, blks_dense);
    printf("Avg size: %g, avg fill: %g, stddev fill: %g\n",
        blk_avg_size, blk_avg_fill, blk_stddev_fill);

    /* compute storage needs */
    thrust::exclusive_scan(
        thrust::host,
        block_ix_space->dense_val,
        block_ix_space->dense_val + coarse_nnz + 1,
        block_ix_space->dense_val);
    thrust::exclusive_scan(
        thrust::host,
        block_val_space->dense_val,
        block_val_space->dense_val + coarse_nnz + 1,
        block_val_space->dense_val);

    const mat_size_t ix_space = (*block_ix_space)[coarse_nnz];
    const mat_size_t val_space = (*block_val_space)[coarse_nnz];

    const mat_size_t needed_bytes =
        (ix_space * sizeof(mat_int_t) + val_space * sizeof(T)) / (1024 * 1024);
    printf("Allocated space for %ld ix, %ld vals (%ld MB)\n", ix_space,
        val_space, needed_bytes);

    /* copy pointers */
    #pragma omp parallel for
    for(mat_int_t i = 0; i < coarse_nnz; ++i)
    {
        matrix_block * j_block = m_h_blocks->dense_val + i;

        /* organize space / pointers according to level ordering */
        const mat_int_t lvl_list_ix = (*m_block_in_levels_ix)[i];

        j_block->col_ptr = (*block_ix_space)[lvl_list_ix];
        j_block->val_ptr = (*block_val_space)[lvl_list_ix];
    }

    /* allocate space */
    m_h_block_ix_store =
        make_managed_dense_vector_ptr<mat_int_t>(ix_space, false);
    m_h_block_val_store =
        make_managed_dense_vector_ptr<T>(val_space, false);

    thrust::fill(
        thrust::host,
        m_h_block_ix_store->dense_val,
        m_h_block_ix_store->dense_val + ix_space,
        -1);
    thrust::fill(
        thrust::host,
        m_h_block_val_store->dense_val,
        m_h_block_val_store->dense_val + val_space,
        0);

    /* gather A's entries in the blocks */
    #pragma omp parallel for
    for(mat_int_t i = 0; i < coarse_nnz; ++i)
    {
        mat_int_t rec_nnz = 0;

        matrix_block * j_block = m_h_blocks->dense_val + i;
        mat_int_t * j_out_ix = m_h_block_ix_store->dense_val +
            j_block->col_ptr;
        T * j_out_val = m_h_block_val_store->dense_val +
            j_block->val_ptr;

        const mat_int_t from_row = (*m_h_block_starts)[j_block->block_row];
        const mat_int_t to_row = (*m_h_block_starts)[j_block->block_row + 1];

        const mat_int_t from_col = (*m_h_block_starts)[j_block->block_col];
        const mat_int_t to_col = (*m_h_block_starts)[j_block->block_col + 1];
        const mat_int_t col_size = to_col - from_col;

        for(mat_int_t fine_i = from_row; fine_i < to_row; ++fine_i)
        {
            for(mat_int_t j = m_in_h_A->csr_row[fine_i];
                j < m_in_h_A->csr_row[fine_i + 1]; ++j)
            {
                const mat_int_t fine_col = m_in_h_A->csr_col[j];

                const mat_int_t local_row = fine_i - from_row;
                const mat_int_t local_col = fine_col - from_col;

                if(fine_col >= fine_i &&
                    from_col <= fine_col && fine_col < to_col)
                {
                    /* nz is in block, hence save it */
                    if(j_block->format == BLOCK_SPARSE)
                    {
                        /* unordered, fine storage */
                        j_out_ix[rec_nnz] = local_row * col_size + local_col;
                        j_out_val[rec_nnz] = m_in_h_A->csr_val[j];

                        ++rec_nnz;
                    }
                    else
                    {
                        /* dense storage */
                        j_out_val[local_row * col_size + local_col] =
                            m_in_h_A->csr_val[j];

                        ++rec_nnz;
                    }
                }
            }
        }
    }

    /* update coarse CSC with block IDs (by sorting ({row, col}, ix)) */
    *m_h_coarse_csc.indices = m_h_coarse_csr.indices.get();
    thrust::sort_by_key(
        thrust::host,
        thrust::make_zip_iterator(thrust::make_tuple(
            csr_coo_col->dense_val,
            csr_coo_row->dense_val)),
        thrust::make_zip_iterator(thrust::make_tuple(
            csr_coo_col->dense_val + coarse_nnz,
            csr_coo_row->dense_val + coarse_nnz)),
        m_h_coarse_csc.indices->dense_val,
        thrust_tuple_sort());
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
upload_data()
{
    const mat_int_t coarse_m = m_h_coarse_csr.m;
    const mat_int_t fine_m = m_in_h_A->m;

    /* splatter piv_starts into a is_piv array */
    m_h_ispiv = make_managed_dense_vector_ptr<mat_int_t>(fine_m, false);
    thrust::fill(
        m_h_ispiv->dense_val,
        m_h_ispiv->dense_val + fine_m,
        0);
    thrust::scatter(
        thrust::make_constant_iterator(1),
        thrust::make_constant_iterator(1) + m_num_pivs,
        m_piv_starts,
        m_h_ispiv->dense_val);

    m_d_ispiv = make_managed_dense_vector_ptr<mat_int_t>(true);
    *m_d_ispiv = m_h_ispiv.get();

    /* upload blocks starts */
    m_d_block_starts = make_managed_dense_vector_ptr<mat_int_t>(true);
    *m_d_block_starts = m_h_block_starts.get();

    /* upload CSR and CSC structures */
    m_d_coarse_csr = compressed_block_list(true);
    m_d_coarse_csr = m_h_coarse_csr;

    m_d_coarse_csc = compressed_block_list(true);
    m_d_coarse_csc = m_h_coarse_csc;

    /* upload blocks and data */
    m_d_blocks = make_managed_dense_vector_ptr<matrix_block>(true);
    *m_d_blocks = m_h_blocks.get();

    m_d_block_ix_store = make_managed_dense_vector_ptr<mat_int_t>(true);
    *m_d_block_ix_store = m_h_block_ix_store.get();

    m_d_block_val_store = make_managed_dense_vector_ptr<T>(true);
    *m_d_block_val_store = m_h_block_val_store.get();

    /* (dense) storage for D */
    m_h_tridiagonal = make_managed_dense_vector_ptr<T>(3 * fine_m, false);
    std::fill(m_h_tridiagonal->dense_val, m_h_tridiagonal->dense_val + fine_m,
        0.0);

    m_d_tridiagonal = make_managed_dense_vector_ptr<T>(true);
    *m_d_tridiagonal = m_h_tridiagonal.get();

    /* upload level (row / block) structure */
    m_d_levels = compressed_block_list(true);
    m_d_levels = m_h_levels;

    m_d_levels_blocks = compressed_block_list(true);
    m_d_levels_blocks = m_h_levels_blocks;

    m_d_row_first_block_offset =
        make_managed_dense_vector_ptr<mat_int_t>(true);
    *m_d_row_first_block_offset = m_h_row_first_block_offset.get();

    /* create storage for level blocks */
    m_d_blocks_inflight =
        make_managed_dense_vector_ptr<T>(
        (mat_size_t) (m_max_level_blocks * 32 * 32), true);
    m_d_piv_block_location =
        make_managed_dense_vector_ptr<mat_int_t>(coarse_m, true);
    m_d_row_norms =
        make_managed_dense_vector_ptr<T>(fine_m, true);

    /* create an on-device permutation array */
    m_d_permutation =
        make_managed_dense_vector_ptr<mat_int_t>(fine_m, true);
    thrust::copy(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(fine_m),
        m_d_permutation->dense_val_ptr());

    /* create counters for 'dynamic' space management */
    m_d_storage_mgmt = make_managed_dense_vector_ptr<mat_int_t>(3, true);

    CHECK_CUDA(cudaDeviceSynchronize());
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
download_data()
{
    /* download blocks and value storage */
    *m_h_blocks = m_d_blocks.get();

    *m_h_block_ix_store = m_d_block_ix_store.get();
    *m_h_block_val_store = m_d_block_val_store.get();

    /* download permutation (from pivoting) */
    m_h_permutation = make_managed_dense_vector_ptr<mat_int_t>(false);
    *m_h_permutation = m_d_permutation.get();

    /* download tridiagonal */
    *m_h_tridiagonal = m_d_tridiagonal.get();
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
eval_results()
{
    const mat_int_t coarse_nnz = m_h_levels_blocks.nnz;
    const mat_int_t fine_m = m_in_h_A->m;

    /* compute avg fill */
    struct thrust_eval_compute
    {
        const mat_int_t * block_starts;
        const matrix_block * blocks;
        const bool get_nnz;
        const bool use_sq;

        thrust_eval_compute(
            const mat_int_t * in_block_starts,
            const matrix_block * in_blocks,
            const bool in_get_nnz,
            const bool in_use_sq)
        : block_starts(in_block_starts),
          blocks(in_blocks),
          get_nnz(in_get_nnz),
          use_sq(in_use_sq)
        {

        }

        T operator()(
            const mat_int_t blk_id) const
        {
            const matrix_block * blk = blocks + blk_id;

            const mat_int_t from_row = block_starts[blk->block_row];
            const mat_int_t to_row = block_starts[blk->block_row + 1];

            const mat_int_t from_col = block_starts[blk->block_col];
            const mat_int_t to_col = block_starts[blk->block_col + 1];

            const T block_size = (T) (to_row - from_row) *
                (to_col - from_col);

            const T nnz = (blk->format == BLOCK_SPARSE) ? blk->nnz :
                block_size;

            const T fill = nnz / block_size;

            if(get_nnz)
            {
                return nnz;
            }
            if(use_sq)
            {
                return (fill * fill);
            }
            return fill;
        }
    };

    dense_vector_ptr<T> tmp =
        make_managed_dense_vector_ptr<T>(coarse_nnz, false);
    thrust::transform(
        thrust::host,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(coarse_nnz),
        tmp->dense_val,
        thrust_eval_compute(m_h_block_starts->dense_val,
            m_h_blocks->dense_val, true, false));
    const mat_int_t U_nnz = (mat_int_t)
        thrust::reduce(
            thrust::host,
            tmp->dense_val,
            tmp->dense_val + coarse_nnz);

    thrust::transform(
        thrust::host,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(coarse_nnz),
        tmp->dense_val,
        thrust_eval_compute(m_h_block_starts->dense_val,
            m_h_blocks->dense_val, false, false));
    const T U_fill =
        thrust::reduce(
            thrust::host,
            tmp->dense_val,
            tmp->dense_val + coarse_nnz);

    thrust::transform(
        thrust::host,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(coarse_nnz),
        tmp->dense_val,
        thrust_eval_compute(m_h_block_starts->dense_val,
            m_h_blocks->dense_val, false, true));
    const T U_fill_sq =
        thrust::reduce(
            thrust::host,
            tmp->dense_val,
            tmp->dense_val + coarse_nnz);

    const T avg_U_fill = U_fill / coarse_nnz;
    const T stddev_U_fill = std::sqrt(U_fill_sq / coarse_nnz -
        avg_U_fill * avg_U_fill);

    /* get D's nnz */
    const mat_int_t D_nnz =
        (3 * fine_m - 2) -
        thrust::count(
            m_h_tridiagonal->dense_val,
            m_h_tridiagonal->dense_val + 3 * fine_m,
            (T) 0);

    const T fill_ratio = (2 * U_nnz + D_nnz) / ((T) m_in_h_A->nnz);

    printf("Preconditioner stats:\n");
    printf("\tU nnz: %d\n", U_nnz);
    printf("\tD nnz: %d\n", D_nnz);
    printf("\tBlock avg. fill: %g\n", avg_U_fill);
    printf("\tBlock stddev fill: %g\n", stddev_U_fill);
    printf("\tAchieved fill ratio: %g\n", fill_ratio);
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
factorize(
    const T threshold)
{
    START_TIMER("Factorization");

    const mat_int_t fine_m = m_in_h_A->m;

    const mat_int_t num_levels = m_h_levels.m;

    /* reset row norms */
    thrust::fill(
        thrust::device,
        m_d_row_norms->dense_val_ptr(),
        m_d_row_norms->dense_val_ptr() + fine_m,
        0.0);

    /* start with 0 scratch space and at offsets 0 */
    mat_int_t * d_leftover_space = m_d_storage_mgmt->dense_val;
    mat_int_t * d_out_ix_offset = d_leftover_space + 1;
    mat_int_t * d_out_val_offset = d_out_ix_offset + 1;
    k_set_scalar<mat_int_t><<<1, 1>>>(
        d_leftover_space,
        0);
    k_set_scalar<mat_int_t><<<1, 1>>>(
        d_out_ix_offset,
        0);
    k_set_scalar<mat_int_t><<<1, 1>>>(
        d_out_val_offset,
        0);

    for(mat_int_t lvl = 0; lvl < num_levels; ++lvl)
    {
        const mat_int_t level_offset = (*m_h_levels_blocks.offsets)[lvl];
        const mat_int_t level_size = (*m_h_levels_blocks.offsets)[lvl + 1] -
            level_offset;

        /* in the list of levels' blocks, the first blocks are pivot */
        const mat_int_t level_piv_size = (*m_h_levels.offsets)[lvl + 1] -
            (*m_h_levels.offsets)[lvl];
        const mat_int_t level_nonpiv_size = level_size - level_piv_size;

        const mat_int_t * level_start =
            m_d_levels_blocks.indices->dense_val + level_offset;
        const mat_int_t * level_piv_start =
            m_d_levels_blocks.indices->dense_val + level_offset;
        const mat_int_t * level_nonpiv_start =
            level_piv_start + level_piv_size;

        /* pointers for inflight blocks */
        T * level_piv_inflight = m_d_blocks_inflight->dense_val;
        T * level_nopiv_inflight = level_piv_inflight +
            32 * 32 * level_piv_size;

        /**
         * Step 1: apply deferred updates to this level's pivot blocks and
         * factor them (U'DU)
         */
        k_factor_pivot_blocks<T, opt_use_bk_pivoting, opt_use_rook_pivoting>
            <<<level_piv_size, 256>>>(
            m_d_block_starts->dense_val,
            m_d_ispiv->dense_val,
            level_piv_start,
            m_d_coarse_csc.offsets->dense_val,
            m_d_coarse_csc.indices->dense_val,
            m_d_blocks->dense_val,
            m_d_block_ix_store->dense_val,
            m_d_block_val_store->dense_val,
            m_d_piv_block_location->dense_val,
            level_piv_inflight,
            m_d_tridiagonal->dense_val,
            m_d_row_norms->dense_val,
            m_d_permutation->dense_val);

        /**
         * Step 2: apply deferred updates to off-diagonal blocks, apply
         * their respective pivot blocks and store in temporary storage
         *
         * Dependencies are discovered by querying the coarse CSC.
         *
         * After applying previous updates, the rows are permuted according
         * to this row's permutation
         */

        if(level_nonpiv_size > 0)
        {
            k_update_scale_offdiagonal_blocks<T,
                opt_use_bk_pivoting || opt_use_rook_pivoting>
                <<<level_nonpiv_size, 256>>>(
                m_d_block_starts->dense_val,
                m_d_ispiv->dense_val,
                level_nonpiv_start,
                m_d_coarse_csc.offsets->dense_val,
                m_d_coarse_csc.indices->dense_val,
                m_d_blocks->dense_val,
                m_d_block_ix_store->dense_val,
                m_d_block_val_store->dense_val,
                m_d_piv_block_location->dense_val,
                level_piv_inflight,
                level_nopiv_inflight,
                m_d_tridiagonal->dense_val,
                m_d_row_norms->dense_val,
                m_d_permutation->dense_val);
        }

        /* propagate pivoting permutation to columns above piv blocks */
        if(opt_use_bk_pivoting || opt_use_rook_pivoting)
        {
            const mat_int_t level_row_size =
                (*m_h_levels.offsets)[lvl + 1] -
                (*m_h_levels.offsets)[lvl];
            const mat_int_t * level_row_start =
                m_d_levels.indices->dense_val +
                (*m_h_levels.offsets)[lvl];

            k_pivot_col_blocks<T><<<level_row_size, 256>>>(
                m_d_block_starts->dense_val,
                level_row_start,
                m_d_coarse_csc.offsets->dense_val,
                m_d_coarse_csc.indices->dense_val,
                m_d_blocks->dense_val,
                m_d_block_ix_store->dense_val,
                m_d_block_val_store->dense_val,
                m_d_permutation->dense_val);
        }

        /**
         * Step 3: apply dropping heuristics to blocks and finalize blocks
         * by writing them out to final storage
         */

        /* Note: block size must be 256 here (for sorting!) */
        k_apply_block_dual_dropping<T>
            <<<level_size, 256>>>(
            m_d_block_starts->dense_val,
            level_start,
            m_d_blocks->dense_val,
            m_d_block_ix_store->dense_val,
            m_d_block_val_store->dense_val,
            d_leftover_space,
            d_out_ix_offset,
            d_out_val_offset,
            m_d_blocks_inflight->dense_val,
            m_d_row_norms->dense_val,
            threshold);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    STOP_TIMER("Factorization");
    PRINT_TIMER("Factorization");
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
export_coarse()
{
    /* download GPU data */
    m_h_coarse_csr = m_d_coarse_csr;

    csr_matrix_ptr<T> coarse =
        make_csr_matrix_ptr<T>(m_h_coarse_csr.m, m_h_coarse_csr.m,
        m_h_coarse_csr.nnz, false);

    std::copy(m_h_coarse_csr.offsets->dense_val,
        m_h_coarse_csr.offsets->dense_val + m_h_coarse_csr.m + 1,
        coarse->csr_row);
    for(mat_int_t i = 0; i < m_h_coarse_csr.nnz; ++i)
        coarse->csr_col[i] =
            (m_h_blocks->dense_val + (*m_h_coarse_csr.indices)[i])->block_col;
    std::fill(coarse->csr_val, coarse->csr_val + coarse->nnz, 1.0);

    NS_TEST::Test<T>::write_csr_matrix(coarse.get(), "out_coarse.mtx");
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
export_factorized(
    const char * folder)
{
    const mat_int_t fine_m = m_in_h_A->m;
    const mat_int_t coarse_nnz = m_h_coarse_csr.nnz;

    /**
     * Transfer data back from from the GPU
     */
    m_h_coarse_csr = m_d_coarse_csr;

    *m_h_blocks = m_d_blocks.get();
    *m_h_block_ix_store = m_d_block_ix_store.get();
    *m_h_block_val_store = m_d_block_val_store.get();

    *m_h_tridiagonal = m_d_tridiagonal.get();

    /**
     * Export U
     */
    dense_vector_ptr<mat_int_t> row_nz =
        make_managed_dense_vector_ptr<mat_int_t>(fine_m + 1, false);
    std::fill(row_nz->dense_val, row_nz->dense_val + fine_m + 1, 0);

    for(mat_int_t b = 0; b < coarse_nnz; ++b)
    {
        const matrix_block * b_block = m_h_blocks->dense_val + b;

        const mat_int_t from_i = (*m_h_block_starts)[b_block->block_row];
        const mat_int_t to_i = (*m_h_block_starts)[b_block->block_row + 1];

        const mat_int_t from_j = (*m_h_block_starts)[b_block->block_col];
        const mat_int_t to_j = (*m_h_block_starts)[b_block->block_col + 1];

        if(b_block->format == BLOCK_DENSE)
        {
            const mat_int_t lda = to_j - from_j;
            const T * val_data = m_h_block_val_store->dense_val +
                b_block->val_ptr;

            /* exlude 0 entries by value comparison */
            for(mat_int_t i = 0; i < (to_i - from_i); ++i)
            {
                for(mat_int_t j = 0; j < (to_j - from_j); ++j)
                    (*row_nz)[from_i + i] += (val_data[i * lda + j] != 0.0);
            }
        }
        else
        {
            const mat_int_t lda = to_j - from_j;
            const mat_int_t * off_data = m_h_block_ix_store->dense_val +
                b_block->col_ptr;
            const T * val_data = m_h_block_val_store->dense_val +
                b_block->val_ptr;

            /* exclude 0 entries by value comparison */
            for(mat_int_t i = 0; i < b_block->nnz; ++i)
                (*row_nz)[from_i + (off_data[i] / lda)] +=
                    (val_data[i] != 0.0);
        }
    }

    /* compute row offsets */
    thrust::exclusive_scan(
        thrust::host,
        row_nz->dense_val,
        row_nz->dense_val + fine_m + 1,
        row_nz->dense_val);
    const mat_int_t fine_nnz = (*row_nz)[fine_m];

    csr_matrix_ptr<T> out_U = make_csr_matrix_ptr<T>(fine_m, fine_m,
        fine_nnz, false);
    thrust::copy(
        thrust::host,
        row_nz->dense_val,
        row_nz->dense_val + fine_m + 1,
        out_U->csr_row);
    thrust::fill(
        thrust::host,
        row_nz->dense_val,
        row_nz->dense_val + fine_m + 1,
        0);

    for(mat_int_t b = 0; b < coarse_nnz; ++b)
    {
        const matrix_block * b_block = m_h_blocks->dense_val + b;

        const mat_int_t from_i = (*m_h_block_starts)[b_block->block_row];
        const mat_int_t to_i = (*m_h_block_starts)[b_block->block_row + 1];

        const mat_int_t from_j = (*m_h_block_starts)[b_block->block_col];
        const mat_int_t to_j = (*m_h_block_starts)[b_block->block_col + 1];

        if(b_block->format == BLOCK_DENSE)
        {
            const mat_int_t lda = to_j - from_j;
            const T * val_data = m_h_block_val_store->dense_val +
                b_block->val_ptr;

            /* exlude 0 entries by value comparison */
            for(mat_int_t i = 0; i < (to_i - from_i); ++i)
            {
                for(mat_int_t j = 0; j < (to_j - from_j); ++j)
                {
                    if(val_data[i * lda + j] != 0.0)
                    {
                        const mat_int_t row = from_i + i;
                        const mat_int_t col = from_j + j;

                        out_U->csr_col[
                            out_U->csr_row[row] +
                            (*row_nz)[row]] = col;
                        out_U->csr_val[
                            out_U->csr_row[row] +
                            (*row_nz)[row]] = val_data[i * lda + j];
                        ++(*row_nz)[row];
                    }
                }
            }
        }
        else
        {
            const mat_int_t lda = to_j - from_j;
            const mat_int_t * off_data = m_h_block_ix_store->dense_val +
                b_block->col_ptr;
            const T * val_data = m_h_block_val_store->dense_val +
                b_block->val_ptr;

            /* exclude 0 entries by value comparison */
            for(mat_int_t i = 0; i < b_block->nnz; ++i)
            {
                const mat_int_t row = from_i + off_data[i] / lda;
                const mat_int_t col = from_j + (off_data[i] % lda);

                out_U->csr_col[
                    out_U->csr_row[row] +
                    (*row_nz)[row]] = col;
                out_U->csr_val[
                    out_U->csr_row[row] +
                    (*row_nz)[row]] = val_data[i];
                ++(*row_nz)[row];
            }
        }
    }

    /* sort indices per row */
    #pragma omp parallel for
    for(mat_int_t i = 0; i < fine_m; ++i)
        thrust::sort_by_key(
            thrust::host,
            out_U->csr_col + out_U->csr_row[i],
            out_U->csr_col + out_U->csr_row[i + 1],
            out_U->csr_val + out_U->csr_row[i]);

    std::string U_path = std::string(folder) + std::string("/out_U.mtx");
    NS_TEST::Test<T>::write_csr_matrix(out_U.get(), U_path.c_str());
    printf("Exported U...\n");

    /**
     * Export D
     */
    std::vector<mat_int_t> D_i, D_j;
    std::vector<T> D_k;

    for(mat_int_t i = 0; i < fine_m; ++i)
    {
        const T sup_diag = (*m_h_tridiagonal)[3 * i];
        const T diag = (*m_h_tridiagonal)[3 * i + 1];
        const T sub_diag = (*m_h_tridiagonal)[3 * i + 2];

        if(i > 0 && sup_diag != 0)
        {
            D_i.push_back(i - 1);
            D_j.push_back(i);
            D_k.push_back(sup_diag);
        }

        D_i.push_back(i);
        D_j.push_back(i);
        D_k.push_back(diag);

        if(i < fine_m - 1 && sub_diag != 0)
        {
            D_i.push_back(i + 1);
            D_j.push_back(i);
            D_k.push_back(sub_diag);
        }
    }

    std::string D_path = std::string(folder) + std::string("/out_D.mtx");
    csr_matrix_ptr<T> out_D =
        NS_TEST::Test<T>::matrix_coo_to_csr(fine_m, fine_m, D_k.size(),
        D_i, D_j, D_k);
    NS_TEST::Test<T>::write_csr_matrix(out_D.get(), D_path.c_str());
    printf("Exported D...\n");

    /**
     * Export P
     */
    csr_matrix_ptr<T> out_P = make_csr_matrix_ptr<T>(fine_m, fine_m, fine_m,
        false);
    std::iota(out_P->csr_row, out_P->csr_row + fine_m + 1, 0);
    std::copy(m_h_permutation->dense_val, m_h_permutation->dense_val + fine_m,
        out_P->csr_col);
    std::fill(out_P->csr_val, out_P->csr_val + fine_m, 1.0);

    std::string P_path = std::string(folder) + std::string("/out_P.mtx");
    NS_TEST::Test<T>::write_csr_matrix(out_P.get(), P_path.c_str());
    printf("Exported P...\n");

    /* report fill value */
    printf("Achieved fill factor: %f\n", (2 * out_U->nnz + out_D->nnz) /
        (T) m_in_h_A->nnz);
}

/*
 * *****************************************************************************
 * **************************** PROTECTED / SOLVE ******************************
 * *****************************************************************************
 */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_analysis()
{
    switch(m_solve_algorithm)
    {
    case SOLVE_STRIPE:
        solve_analysis_stripe();
        break;

    case SOLVE_BLOCK:
        solve_analysis_block();
        break;

    case SOLVE_SCALAR:
        solve_analysis_scalar();
        break;

    default:
        break;
    }
}

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_analysis_stripe()
{
    const mat_int_t coarse_m = m_num_blocks;
    const mat_int_t coarse_nnz = m_h_levels_blocks.nnz;
    const mat_int_t num_levels = m_h_levels.m;

    /* block levels for U' - solve: create block list from CSC + levels */
    m_h_levels_blocks_solve = compressed_block_list(false, num_levels,
        coarse_nnz);

    /* create col -> level map */
    dense_vector_ptr<mat_int_t> col_level_map =
        make_managed_dense_vector_ptr<mat_int_t>(coarse_m, false);

    #pragma omp parallel for
    for(mat_int_t i = 0; i < num_levels; ++i)
    {
        for(mat_int_t j = (*m_h_levels.offsets)[i];
            j < (*m_h_levels.offsets)[i + 1]; ++j)
            (*col_level_map)[(*m_h_levels.indices)[j]] = i;
    }

    /* sort blocks by level (column-wise), pivs in front */
    thrust::copy(
        thrust::host,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(coarse_nnz),
        m_h_levels_blocks_solve.indices->dense_val);

    struct thrust_pivot_tuple_sort_col
    {
        const mat_int_t * col_level_map;
        const matrix_block * blocks;

        thrust_pivot_tuple_sort_col(
            const mat_int_t * in_level_map,
            const matrix_block * in_blocks)
        {
            col_level_map = in_level_map;
            blocks = in_blocks;
        }

        bool
        operator()(const mat_int_t blk0, const mat_int_t blk1)
        {
            const matrix_block * b_blk0 = blocks + blk0;
            const matrix_block * b_blk1 = blocks + blk1;

            if(col_level_map[b_blk0->block_col] !=
                col_level_map[b_blk1->block_col])
                return (col_level_map[b_blk0->block_col] <
                    col_level_map[b_blk1->block_col]);

            const bool blk0_piv = (blocks[blk0].block_row ==
                blocks[blk0].block_col);
            const bool blk1_piv = (blocks[blk1].block_row ==
                blocks[blk1].block_col);

            return (blk0_piv && !blk1_piv);
        }
    };
    thrust::sort(
        thrust::host,
        m_h_levels_blocks_solve.indices->dense_val,
        m_h_levels_blocks_solve.indices->dense_val + coarse_nnz,
        thrust_pivot_tuple_sort_col(
            col_level_map->dense_val,
            m_h_blocks->dense_val));

    /* count level sizes */
    thrust::fill(
        thrust::host,
        m_h_levels_blocks_solve.offsets->dense_val,
        m_h_levels_blocks_solve.offsets->dense_val + num_levels + 1,
        0);

    #pragma omp parallel for
    for(mat_int_t j = 0; j < coarse_m; ++j)
    {
        const mat_int_t j_size = (*m_h_coarse_csc.offsets)[j + 1] -
            (*m_h_coarse_csc.offsets)[j];
        mat_int_t * write_ptr = m_h_levels_blocks_solve.offsets->dense_val +
            (*col_level_map)[j];

        #pragma omp atomic
        (*write_ptr) += j_size;
    }

    /* create offsets for level/block list */
    thrust::exclusive_scan(
        thrust::host,
        m_h_levels_blocks_solve.offsets->dense_val,
        m_h_levels_blocks_solve.offsets->dense_val + num_levels + 1,
        m_h_levels_blocks_solve.offsets->dense_val);

    /* copy to GPU */
    m_d_levels_blocks_solve = m_h_levels_blocks_solve;
}

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_analysis_block()
{
    const mat_int_t coarse_m = m_num_blocks;
    const mat_int_t coarse_nnz = m_h_levels_blocks.nnz;
    const mat_int_t num_levels = m_h_levels.m;

    /* block levels for U' - solve: create block list from CSC + levels */
    m_h_levels_blocks_solve = compressed_block_list(false, num_levels,
        coarse_nnz);

    /* create col -> level map */
    dense_vector_ptr<mat_int_t> col_level_map =
        make_managed_dense_vector_ptr<mat_int_t>(coarse_m, false);

    #pragma omp parallel for
    for(mat_int_t i = 0; i < num_levels; ++i)
    {
        for(mat_int_t j = (*m_h_levels.offsets)[i];
            j < (*m_h_levels.offsets)[i + 1]; ++j)
            (*col_level_map)[(*m_h_levels.indices)[j]] = i;
    }

    /* sort blocks by level (column-wise), pivs in front */
    thrust::copy(
        thrust::host,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(coarse_nnz),
        m_h_levels_blocks_solve.indices->dense_val);

    struct thrust_pivot_tuple_sort_col
    {
        const mat_int_t * col_level_map;
        const matrix_block * blocks;

        thrust_pivot_tuple_sort_col(
            const mat_int_t * in_level_map,
            const matrix_block * in_blocks)
        {
            col_level_map = in_level_map;
            blocks = in_blocks;
        }

        bool
        operator()(const mat_int_t blk0, const mat_int_t blk1)
        {
            const matrix_block * b_blk0 = blocks + blk0;
            const matrix_block * b_blk1 = blocks + blk1;

            if(col_level_map[b_blk0->block_col] !=
                col_level_map[b_blk1->block_col])
                return (col_level_map[b_blk0->block_col] <
                    col_level_map[b_blk1->block_col]);

            const bool blk0_piv = (blocks[blk0].block_row ==
                blocks[blk0].block_col);
            const bool blk1_piv = (blocks[blk1].block_row ==
                blocks[blk1].block_col);

            return (blk0_piv && !blk1_piv);
        }
    };
    thrust::sort(
        thrust::host,
        m_h_levels_blocks_solve.indices->dense_val,
        m_h_levels_blocks_solve.indices->dense_val + coarse_nnz,
        thrust_pivot_tuple_sort_col(
            col_level_map->dense_val,
            m_h_blocks->dense_val));

    /* count level sizes */
    thrust::fill(
        thrust::host,
        m_h_levels_blocks_solve.offsets->dense_val,
        m_h_levels_blocks_solve.offsets->dense_val + num_levels + 1,
        0);

    #pragma omp parallel for
    for(mat_int_t j = 0; j < coarse_m; ++j)
    {
        const mat_int_t j_size = (*m_h_coarse_csc.offsets)[j + 1] -
            (*m_h_coarse_csc.offsets)[j];
        mat_int_t * write_ptr = m_h_levels_blocks_solve.offsets->dense_val +
            (*col_level_map)[j];

        #pragma omp atomic
        (*write_ptr) += j_size;
    }

    /* create offsets for level/block list */
    thrust::exclusive_scan(
        thrust::host,
        m_h_levels_blocks_solve.offsets->dense_val,
        m_h_levels_blocks_solve.offsets->dense_val + num_levels + 1,
        m_h_levels_blocks_solve.offsets->dense_val);

    /* copy to GPU */
    m_d_levels_blocks_solve = m_h_levels_blocks_solve;
}

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_analysis_scalar()
{
    const mat_int_t fine_m = m_in_h_A->m;
    const mat_int_t coarse_nnz = m_h_coarse_csr.nnz;

    /**
     * Transfer data back from from the GPU
     */
    m_h_coarse_csr = m_d_coarse_csr;

    *m_h_blocks = m_d_blocks.get();
    *m_h_block_ix_store = m_d_block_ix_store.get();
    *m_h_block_val_store = m_d_block_val_store.get();

    *m_h_tridiagonal = m_d_tridiagonal.get();

    /**
     * Export U
     */
    dense_vector_ptr<mat_int_t> row_nz =
        make_managed_dense_vector_ptr<mat_int_t>(fine_m + 1, false);
    std::fill(row_nz->dense_val, row_nz->dense_val + fine_m + 1, 0);

    for(mat_int_t b = 0; b < coarse_nnz; ++b)
    {
        const matrix_block * b_block = m_h_blocks->dense_val + b;

        const mat_int_t from_i = (*m_h_block_starts)[b_block->block_row];
        const mat_int_t to_i = (*m_h_block_starts)[b_block->block_row + 1];

        const mat_int_t from_j = (*m_h_block_starts)[b_block->block_col];
        const mat_int_t to_j = (*m_h_block_starts)[b_block->block_col + 1];

        if(b_block->format == BLOCK_DENSE)
        {
            const mat_int_t lda = to_j - from_j;
            const T * val_data = m_h_block_val_store->dense_val +
                b_block->val_ptr;

            /* exlude 0 entries by value comparison */
            for(mat_int_t i = 0; i < (to_i - from_i); ++i)
            {
                for(mat_int_t j = 0; j < (to_j - from_j); ++j)
                    (*row_nz)[from_i + i] += (val_data[i * lda + j] != 0.0);
            }
        }
        else
        {
            const mat_int_t lda = to_j - from_j;
            const mat_int_t * off_data = m_h_block_ix_store->dense_val +
                b_block->col_ptr;
            const T * val_data = m_h_block_val_store->dense_val +
                b_block->val_ptr;

            /* exclude 0 entries by value comparison */
            for(mat_int_t i = 0; i < b_block->nnz; ++i)
                (*row_nz)[from_i + (off_data[i] / lda)] +=
                    (val_data[i] != 0.0);
        }
    }

    /* compute row offsets */
    thrust::exclusive_scan(
        thrust::host,
        row_nz->dense_val,
        row_nz->dense_val + fine_m + 1,
        row_nz->dense_val);
    const mat_int_t fine_nnz = (*row_nz)[fine_m];

    csr_matrix_ptr<T> out_U = make_csr_matrix_ptr<T>(fine_m, fine_m,
        fine_nnz, false);
    thrust::copy(
        thrust::host,
        row_nz->dense_val,
        row_nz->dense_val + fine_m + 1,
        out_U->csr_row);
    thrust::fill(
        thrust::host,
        row_nz->dense_val,
        row_nz->dense_val + fine_m + 1,
        0);


    for(mat_int_t b = 0; b < coarse_nnz; ++b)
    {
        const matrix_block * b_block = m_h_blocks->dense_val + b;

        const mat_int_t from_i = (*m_h_block_starts)[b_block->block_row];
        const mat_int_t to_i = (*m_h_block_starts)[b_block->block_row + 1];

        const mat_int_t from_j = (*m_h_block_starts)[b_block->block_col];
        const mat_int_t to_j = (*m_h_block_starts)[b_block->block_col + 1];

        if(b_block->format == BLOCK_DENSE)
        {
            const mat_int_t lda = to_j - from_j;
            const T * val_data = m_h_block_val_store->dense_val +
                b_block->val_ptr;

            /* exlude 0 entries by value comparison */
            for(mat_int_t i = 0; i < (to_i - from_i); ++i)
            {
                for(mat_int_t j = 0; j < (to_j - from_j); ++j)
                {
                    if(val_data[i * lda + j] != 0.0)
                    {
                        const mat_int_t row = from_i + i;
                        const mat_int_t col = from_j + j;

                        out_U->csr_col[
                            out_U->csr_row[row] +
                            (*row_nz)[row]] = col;
                        out_U->csr_val[
                            out_U->csr_row[row] +
                            (*row_nz)[row]] = val_data[i * lda + j];
                        ++(*row_nz)[row];
                    }
                }
            }
        }
        else
        {
            const mat_int_t lda = to_j - from_j;
            const mat_int_t * off_data = m_h_block_ix_store->dense_val +
                b_block->col_ptr;
            const T * val_data = m_h_block_val_store->dense_val +
                b_block->val_ptr;

            /* exclude 0 entries by value comparison */
            for(mat_int_t i = 0; i < b_block->nnz; ++i)
            {
                const mat_int_t row = from_i + off_data[i] / lda;
                const mat_int_t col = from_j + (off_data[i] % lda);

                out_U->csr_col[
                    out_U->csr_row[row] +
                    (*row_nz)[row]] = col;
                out_U->csr_val[
                    out_U->csr_row[row] +
                    (*row_nz)[row]] = val_data[i];
                ++(*row_nz)[row];
            }
        }
    }

    /* sort indices per row */
    #pragma omp parallel for
    for(mat_int_t i = 0; i < fine_m; ++i)
        thrust::sort_by_key(
            thrust::host,
            out_U->csr_col + out_U->csr_row[i],
            out_U->csr_col + out_U->csr_row[i + 1],
            out_U->csr_val + out_U->csr_row[i]);

    /**
     * Upload U to GPU
     */
    m_d_csr_U = make_csr_matrix_ptr<T>(true);
    *m_d_csr_U = out_U.get();
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_U(
    dense_vector_t<T> * b)
const
{
    switch(m_solve_algorithm)
    {
    case SOLVE_STRIPE:
        solve_U_stripe(b);
        break;

    case SOLVE_BLOCK:
        solve_U_block(b);
        break;

    case SOLVE_SCALAR:
        solve_U_scalar(b);
        break;

    case SOLVE_JACOBI:
        solve_U_jacobi(b, m_jacobi_sweeps);
        break;
    }
}

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_U_stripe(
    dense_vector_t<T> * b)
const
{
    const mat_int_t num_levels = m_h_levels.m;

    for(mat_int_t lvl = num_levels - 1; lvl >= 0; --lvl)
    {
        /* block-row-wise version */
        const mat_int_t level_offset = (*m_h_levels.offsets)[lvl];
        const mat_int_t level_size = (*m_h_levels.offsets)[lvl + 1] -
            level_offset;
        const mat_int_t * level_row_start = m_d_levels.indices->dense_val
            + level_offset;

        k_solve_block_stripe_level<T, true, true, true>
            <<<level_size, 256, 0, this->m_handle->get_stream()>>>(
            m_d_block_starts->dense_val,
            level_row_start,
            m_d_coarse_csr.offsets->dense_val,
            m_d_coarse_csr.indices->dense_val,
            m_d_blocks->dense_val,
            m_d_block_ix_store->dense_val,
            m_d_block_val_store->dense_val,
            b->dense_val,
            b->dense_val,
            b->dense_val);
    }
}

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_U_block(
    dense_vector_t<T> * b)
const
{
    const mat_int_t num_levels = m_h_levels.m;

    for(mat_int_t lvl = num_levels - 1; lvl >= 0; --lvl)
    {
        /* block-wise version */
        const mat_int_t level_offset = (*m_h_levels_blocks.offsets)[lvl];
        const mat_int_t level_piv_size = (*m_h_levels.offsets)[lvl + 1] -
            (*m_h_levels.offsets)[lvl];
        const mat_int_t level_nopiv_size =
            ((*m_h_levels_blocks.offsets)[lvl + 1] -
            (*m_h_levels_blocks.offsets)[lvl]) - level_piv_size;
        const mat_int_t * level_piv_start =
            m_d_levels_blocks.indices->dense_val + level_offset;
        const mat_int_t * level_nopiv_start = level_piv_start +
            level_piv_size;

        const mat_int_t block_warp_size = 256 / 32;
        const mat_int_t grid_size = DIV_UP(level_nopiv_size,
            block_warp_size);
        if(level_nopiv_size > 0)
        {
            k_solve_blocks_offdiagonal_atomic<T, true>
                <<<grid_size, 256, 0, this->m_handle->get_stream()>>>(
                m_d_block_starts->dense_val,
                level_nopiv_start,
                m_d_blocks->dense_val,
                m_d_block_ix_store->dense_val,
                m_d_block_val_store->dense_val,
                b->dense_val,
                level_nopiv_size);
        }

        k_solve_blocks_diagonal_atomic<T, true>
            <<<level_piv_size, 256, 0, this->m_handle->get_stream()>>>(
            m_d_block_starts->dense_val,
            level_piv_start,
            m_d_blocks->dense_val,
            m_d_block_ix_store->dense_val,
            m_d_block_val_store->dense_val,
            b->dense_val);
    }
}

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_U_scalar(
    dense_vector_t<T> * b)
const
{
    const mat_int_t m = m_in_h_A->m;

    this->m_handle->push_scalar_mode();
    this->m_handle->set_scalar_mode(false);
    T one = 1.0;

    cudaMemcpyAsync(m_gpu_jacobi_a->dense_val, b->dense_val,
        m * sizeof(T), cudaMemcpyDeviceToDevice, this->m_handle->get_stream());

    T_triangular_solve<T>(
        this->m_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        m_d_csr_U.get(),
        m_gpu_jacobi_a.get(),
        b,
        &one);
    this->m_handle->pop_scalar_mode();
}

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_U_jacobi(
    dense_vector_t<T> * b,
    const mat_int_t num_sweeps)
const
{
    const mat_int_t coarse_m = m_h_coarse_csr.m;
    const mat_int_t fine_m = m_in_h_A->m;

    /* start with x_0 = 0 */
    const mat_int_t grid_size = DIV_UP(fine_m, 256);
    k_init_with_scalar<T><<<grid_size, 256, 0, this->m_handle->get_stream()>>>(
        m_gpu_jacobi_a->dense_val, 0.0, fine_m);

    dense_vector_t<T> * jacobi_in = m_gpu_jacobi_a.get();
    dense_vector_t<T> * jacobi_out = m_gpu_jacobi_b.get();
    for(mat_int_t s = 0; s < num_sweeps; ++s)
    {
        k_solve_block_stripe_level<T, true, true, false>
            <<<coarse_m, 256, 0, this->m_handle->get_stream()>>>(
            m_d_block_starts->dense_val,
            nullptr,
            m_d_coarse_csr.offsets->dense_val,
            m_d_coarse_csr.indices->dense_val,
            m_d_blocks->dense_val,
            m_d_block_ix_store->dense_val,
            m_d_block_val_store->dense_val,
            b->dense_val,
            jacobi_in->dense_val,
            jacobi_out->dense_val);

        std::swap(jacobi_in, jacobi_out);
    }

    cudaMemcpyAsync(b->dense_val, jacobi_in->dense_val,
        fine_m * sizeof(T), cudaMemcpyDeviceToDevice,
        this->m_handle->get_stream());
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_Ut(
    dense_vector_t<T> * b)
const
{
    switch(m_solve_algorithm)
    {
    case SOLVE_STRIPE:
        solve_Ut_stripe(b);
        break;

    case SOLVE_BLOCK:
        solve_Ut_block(b);
        break;

    case SOLVE_SCALAR:
        solve_Ut_scalar(b);
        break;

    case SOLVE_JACOBI:
        solve_Ut_jacobi(b, m_jacobi_sweeps);
        break;
    }
}

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_Ut_stripe(
    dense_vector_t<T> * b)
const
{
    /* multilevel solve - reuse factorization levelling */
    const mat_int_t num_levels = m_h_levels.m;

    for(mat_int_t lvl = 0; lvl < num_levels; ++lvl)
    {
        const mat_int_t level_offset = (*m_h_levels.offsets)[lvl];
        const mat_int_t level_size = (*m_h_levels.offsets)[lvl + 1]
            - level_offset;
        const mat_int_t * level_row_start =
            m_d_levels.indices->dense_val + level_offset;

        k_solve_block_stripe_level<T, false, true, true>
            <<<level_size, 256, 0, this->m_handle->get_stream()>>>(
            m_d_block_starts->dense_val,
            level_row_start,
            m_d_coarse_csc.offsets->dense_val,
            m_d_coarse_csc.indices->dense_val,
            m_d_blocks->dense_val,
            m_d_block_ix_store->dense_val,
            m_d_block_val_store->dense_val,
            b->dense_val,
            b->dense_val,
            b->dense_val);
    }
}

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_Ut_block(
    dense_vector_t<T> * b)
const
{
    /* multilevel solve - reuse factorization levelling */
    const mat_int_t num_levels = m_h_levels.m;

    for(mat_int_t lvl = 0; lvl < num_levels; ++lvl)
    {
        /* block-wise version */
        const mat_int_t level_offset = (*m_h_levels_blocks_solve.offsets)[lvl];
        const mat_int_t level_piv_size = (*m_h_levels.offsets)[lvl + 1] -
            (*m_h_levels.offsets)[lvl];
        const mat_int_t level_nopiv_size =
            ((*m_h_levels_blocks_solve.offsets)[lvl + 1] - level_offset) -
            level_piv_size;
        const mat_int_t * level_piv_start =
            m_d_levels_blocks_solve.indices->dense_val + level_offset;
        const mat_int_t * level_nopiv_start = level_piv_start +
            level_piv_size;

        const mat_int_t block_warp_size = 256 / 32;
        const mat_int_t grid_size = DIV_UP(level_nopiv_size,
            block_warp_size);
        if(level_nopiv_size > 0)
        {
            k_solve_blocks_offdiagonal_atomic<T, false>
                <<<grid_size, 256, 0, this->m_handle->get_stream()>>>(
                m_d_block_starts->dense_val,
                level_nopiv_start,
                m_d_blocks->dense_val,
                m_d_block_ix_store->dense_val,
                m_d_block_val_store->dense_val,
                b->dense_val,
                level_nopiv_size);
        }

        k_solve_blocks_diagonal_atomic<T, false>
        <<<level_piv_size, 256, 0, this->m_handle->get_stream()>>>(
            m_d_block_starts->dense_val,
            level_piv_start,
            m_d_blocks->dense_val,
            m_d_block_ix_store->dense_val,
            m_d_block_val_store->dense_val,
            b->dense_val);
    }
}

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_Ut_scalar(
    dense_vector_t<T> * b)
const
{
    const mat_int_t m = m_in_h_A->m;

    this->m_handle->push_scalar_mode();
    this->m_handle->set_scalar_mode(false);
    T one = 1.0;

    cudaMemcpyAsync(m_gpu_jacobi_a->dense_val, b->dense_val,
        m * sizeof(T), cudaMemcpyDeviceToDevice, this->m_handle->get_stream());

    T_triangular_solve<T>(
        this->m_handle,
        CUSPARSE_OPERATION_TRANSPOSE,
        m_d_csr_U.get(),
        m_gpu_jacobi_a.get(),
        b,
        &one);
    this->m_handle->pop_scalar_mode();
}

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_Ut_jacobi(
    dense_vector_t<T> * b,
    const mat_int_t num_sweeps)
const
{
    const mat_int_t coarse_m = m_h_coarse_csr.m;
    const mat_int_t fine_m = m_in_h_A->m;

    /* start with x_0 = 0 */
    const mat_int_t grid_size = DIV_UP(fine_m, 256);
    k_init_with_scalar<T><<<grid_size, 256, 0, this->m_handle->get_stream()>>>(
        m_gpu_jacobi_a->dense_val, 0.0, fine_m);

    dense_vector_t<T> * jacobi_in = m_gpu_jacobi_a.get();
    dense_vector_t<T> * jacobi_out = m_gpu_jacobi_b.get();
    for(mat_int_t s = 0; s < num_sweeps; ++s)
    {
        k_solve_block_stripe_level<T, false, true, false>
            <<<coarse_m, 256, 0, this->m_handle->get_stream()>>>(
            m_d_block_starts->dense_val,
            nullptr,
            m_d_coarse_csc.offsets->dense_val,
            m_d_coarse_csc.indices->dense_val,
            m_d_blocks->dense_val,
            m_d_block_ix_store->dense_val,
            m_d_block_val_store->dense_val,
            b->dense_val,
            jacobi_in->dense_val,
            jacobi_out->dense_val);

        std::swap(jacobi_in, jacobi_out);
    }

    cudaMemcpyAsync(b->dense_val, jacobi_in->dense_val,
        fine_m * sizeof(T), cudaMemcpyDeviceToDevice,
        this->m_handle->get_stream());
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_D(
    dense_vector_t<T> * b)
const
{
    const mat_int_t fine_m = m_in_h_A->m;
    const mat_int_t grid_size = DIV_UP(fine_m, 256);
    k_solve_D<<<grid_size, 256, 0, this->m_handle->get_stream()>>>(
        m_d_tridiagonal->dense_val,
        m_d_ispiv->dense_val,
        b->dense_val,
        fine_m);
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_P(
    const dense_vector_t<T> * b,
    dense_vector_t<T> * x)
const
{
    /* note: solve with P means multiply with Pt */
    const mat_int_t fine_m = m_in_h_A->m;

    const mat_int_t grid_size = DIV_UP(fine_m, 256);
    k_permute_P<T><<<grid_size, 256, 0, this->m_handle->get_stream()>>>(
        b->dense_val,
        x->dense_val,
        m_d_permutation->dense_val,
        fine_m);
}

/* ************************************************************************** */

template<typename T, bool opt_use_bk_pivoting, bool opt_use_rook_pivoting>
void
BlockiLDLt<T, opt_use_bk_pivoting, opt_use_rook_pivoting>::
solve_Pt(
    const dense_vector_t<T> * b,
    dense_vector_t<T> * x)
const
{
    /* note: solve with Pt means multiply with P */
    const mat_int_t fine_m = m_in_h_A->m;

    const mat_int_t grid_size = DIV_UP(fine_m, 256);
    k_permute_Pt<T><<<grid_size, 256, 0, this->m_handle->get_stream()>>>(
        b->dense_val,
        x->dense_val,
        m_d_permutation->dense_val,
        fine_m);
}

NS_LA_END
NS_CULIP_END
