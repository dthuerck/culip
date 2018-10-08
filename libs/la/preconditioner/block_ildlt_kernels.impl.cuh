/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/la/preconditioner/block_ildlt.cuh>

#include <libs/la/helper_kernels.cuh>

#include <cooperative_groups.h>
#include <cub/cub.cuh>

using namespace cooperative_groups;

#define NZ_EPS 1e-10

NS_CULIP_BEGIN
NS_LA_BEGIN

template<typename T, bool TRANSPOSE>
inline __device__
void
d_load_block(
    const matrix_block * blk,
    const mat_int_t * block_starts,
    const mat_int_t * ix_store,
    const T * val_store,
    T * dense_block_store,
    thread_block& tb)
{
    const mat_int_t tidx = tb.thread_rank();
    const mat_int_t widx = tidx / 32;
    const thread_block_tile<32> warp = tiled_partition<32>(tb);
    const mat_int_t lwidx = warp.thread_rank();

    const mat_int_t warps_per_block = tb.size() / 32;

    /**
     * Note: loading a transposed matrix in row-major equals loading
     *       the non-transposed matrix in col-major -> that's what's done
     *       here
     */

    /* load nonzero values */
    const mat_int_t block_height = block_starts[blk->block_row + 1] -
        block_starts[blk->block_row];
    const mat_int_t block_width = block_starts[blk->block_col + 1] -
        block_starts[blk->block_col];
    const mat_int_t * block_ix = ix_store + blk->col_ptr;
    const T * block_val = val_store + blk->val_ptr;

    /* initialize block with zeros to allow transposes later */
    for(mat_int_t i = widx; i < 32; i += warps_per_block)
        dense_block_store[32 * i + lwidx] = 0;
    tb.sync();

    if(blk->format == BLOCK_DENSE)
    {
        for(mat_int_t i = widx; i < block_height; i += warps_per_block)
        {
            const mat_int_t ptr = TRANSPOSE ?
                (32 * lwidx + i) : (32 * i + lwidx);
            dense_block_store[ptr] = (lwidx < block_width) ?
                block_val[i * block_width + lwidx] : 0;
        }
    }
    else
    {
        for(mat_int_t i = tidx; i < blk->nnz; i += tb.size())
        {
            const mat_int_t row = block_ix[i] / block_width;
            const mat_int_t col = block_ix[i] % block_width;

            const mat_int_t ptr = TRANSPOSE ?
                (32 * col + row) : (32 * row + col);
            dense_block_store[ptr] = block_val[i];
        }
    }
}

/* ************************************************************************** */

template<typename T>
inline __device__
void
d_save_block(
    matrix_block * blk,
    const mat_int_t * block_starts,
    const T * dense_block_store,
    mat_int_t * ix_store,
    T * val_store,
    thread_block& tb)
{
    const mat_int_t tidx = tb.thread_rank();
    const mat_int_t widx = tidx / 32;
    const thread_block_tile<32> warp = tiled_partition<32>(tb);
    const mat_int_t lwidx = warp.thread_rank();

    const mat_int_t warps_per_block = tb.size() / 32;

    const mat_int_t from_row = block_starts[blk->block_row];
    const mat_int_t to_row = block_starts[blk->block_row + 1];
    const mat_int_t block_height = to_row - from_row;

    const mat_int_t from_col = block_starts[blk->block_col];
    const mat_int_t to_col = block_starts[blk->block_col + 1];
    const mat_int_t block_width = to_col - from_col;

    if(blk->format == BLOCK_DENSE)
    {
        T * dense_blk = val_store + blk->val_ptr;

        /* dense format: just copy */
        for(mat_int_t i = widx; i < block_height; i += warps_per_block)
            if(lwidx < block_width)
                dense_blk[i * block_width + lwidx] =
                    dense_block_store[32 * i + lwidx];
    }
    else
    {
        /* sparse format: compress and copy */
        if(tidx == 0)
            blk->nnz = 0;
        tb.sync();

        mat_int_t * block_ix_store = ix_store + blk->col_ptr;
        T * block_val_store = val_store + blk->val_ptr;

        for(mat_int_t i = widx; i < block_height; i += warps_per_block)
        {
            const T i_val = (lwidx < block_width) ?
                dense_block_store[32 * i + lwidx] : 0;
            const bool is_nz = (fabs(i_val) > NZ_EPS);

            /* count how many nz there are */
            const mat_int_t warp_nnz = __popc(warp.ballot(is_nz));

            mat_int_t warp_offset;
            if(lwidx == 0)
                warp_offset = atomicAdd(&blk->nnz, warp_nnz);

            warp.sync();
            warp_offset = warp.shfl(warp_offset, 0);

            if(is_nz)
            {
                coalesced_group active = coalesced_threads();

                block_ix_store[warp_offset + active.thread_rank()] =
                    i * block_width + lwidx;
                block_val_store[warp_offset + active.thread_rank()] =
                    i_val;

                // if(block_ix_store[warp_offset + active.thread_rank()] >= (block_width * block_height))
                //     printf("Error (regular) in saving block %d (%d, %d) with %d | %d / %d from %d / %d, "\
                //         "%d/%d (val %g) - ix %d of max_nnz %d\n",
                //         blk->id,
                //         blk->block_row,
                //         blk->block_col,
                //         i * block_width + lwidx,
                //         block_ix_store[warp_offset + active.thread_rank()],
                //         block_width * block_height,
                //         i, block_height,
                //         lwidx, block_width,
                //         i_val,
                //         warp_offset + active.thread_rank(),
                //         blk->max_nnz);
            }
        }
    }
}

/* ************************************************************************** */

/**
 * Device functions for debugging - conveniently print sparse / dense blocks
 * from kernels
 */

template<typename T>
inline __device__
void
d_print_dense_block(
    const T * dense_block,
    const mat_int_t b_height = 32,
    const mat_int_t b_width = 32)
{
    for(mat_int_t i = 0; i < b_height; ++i)
    {
        printf("\t");
        for(mat_int_t j = 0; j < b_width; ++j)
            printf("%g ", dense_block[32 * i + j]);
        printf("\n");
    }
}

template<typename T>
inline __device__
void
d_print_block(
    const mat_int_t * block_starts,
    const matrix_block * blk,
    const mat_int_t * ix_store,
    const T * val_store,
    thread_block& tb,
    T * dense_block)
{
    const mat_int_t tidx = tb.thread_rank();

    /* read block and indices */
    const mat_int_t b_row = blk->block_row;
    const mat_int_t b_col = blk->block_col;

    const mat_int_t from_row = block_starts[b_row];
    const mat_int_t to_row = block_starts[b_row + 1];
    const mat_int_t from_col = block_starts[b_col];
    const mat_int_t to_col = block_starts[b_col + 1];

    /* print basic information */
    if(tidx == 0)
    {
        printf("Block %d: \n", blk->id);
        if(blk->format == BLOCK_SPARSE)
            printf("\tformat: SPARSE\n");
        else
            printf("\tformat: DENSE\n");
        printf("\tnnz: %d\n", blk->nnz);
        printf("\tmax nnz: %d\n", blk->max_nnz);
        printf("\tblock row: %d (%d -> %d)\n", blk->block_row, from_row,
            to_row - 1);
        printf("\tblock_col: %d (%d -> %d)\n", blk->block_col, from_col,
            to_col - 1);

        printf("\n\tData:\n");
    }

    /* read block into shared memory */
    d_load_block<T, false>(
        blk,
        block_starts,
        ix_store,
        val_store,
        dense_block,
        tb);
    tb.sync();

    /* use dense print function */
    if(tidx == 0)
    {
        if(blk->format == BLOCK_SPARSE)
        {
            const mat_int_t * ix = ix_store + blk->col_ptr;
            const T * val = val_store + blk->val_ptr;

            for(mat_int_t i = 0; i < blk->nnz; ++i)
                printf("\t\t%d: %d, %g\n", i, ix[i], val[i]);
        }

        d_print_dense_block<T>(dense_block, to_row - from_row,
            to_col - from_col);
    }
}

template<typename  T>
inline __device__
void
d_print_tridiag(
    const mat_int_t from_col,
    const mat_int_t to_col,
    const T * tridiag)
{
    printf("D[%d -> %d] = \n", from_col, to_col - 1);
    for(mat_int_t i = from_col; i < to_col; ++i)
        printf("\t%g %g %g\n",
            tridiag[3 * i],
            tridiag[3 * i + 1],
            tridiag[3 * i + 2]);
}

/**
 * Wrapper kernels for printing blocks on the device - always
 * use one CUDA block per kernel call!
 */

template<typename T>
__global__
void
k_print_dense_block(
    const T * dense_block,
    const mat_int_t b_height = 32,
    const mat_int_t b_width = 32)
{
    thread_block tb = this_thread_block();

    d_print_dense_block<T>(
        dense_block,
        b_height,
        b_width);
}

template<typename T>
__global__
void
k_print_block(
    const mat_int_t * block_starts,
    const matrix_block * blk,
    const mat_int_t * ix_store,
    const T * val_store)
{
    thread_block tb = this_thread_block();

    __shared__ T buf[32 * 32];

    d_print_block<T>(
        block_starts,
        blk,
        ix_store,
        val_store,
        tb,
        buf);
}

template<typename T>
__global__
void
k_print_tridiag(
    const mat_int_t from_col,
    const mat_int_t to_col,
    const T * tridiag)
{
    thread_block tb = this_thread_block();
    const mat_int_t tidx = tb.thread_rank();

    if(tidx == 0)
        d_print_tridiag<T>(
            from_col,
            to_col,
            tridiag);
}

/* ************************************************************************** */

template<typename T, bool TRANSPOSE>
inline __device__
void
d_copy_block(
    T * from_dense_block,
    T * to_dense_block,
    thread_block& tb,
    const mat_int_t b_height = 32,
    const mat_int_t b_width = 32)
{
    const mat_int_t widx = tb.thread_rank() / 32;
    const thread_block_tile<32> warp = tiled_partition<32>(tb);
    const mat_int_t lwidx = warp.thread_rank();

    const mat_int_t warps_per_block = tb.size() / 32;

    for(mat_int_t i = widx; i < b_height; i += warps_per_block)
    {
        if(TRANSPOSE)
            to_dense_block[32 * i + lwidx] = from_dense_block[32 * lwidx + i];
        else
            to_dense_block[32 * i + lwidx] = from_dense_block[32 * i + lwidx];
    }
}

/* ************************************************************************** */

template<typename T>
inline __device__
void
d_multiply_block_by_block_diagonal(
    T * dense_block_store,
    const T * block_diagonal,
    thread_block& tb,
    const mat_int_t b_height = 32,
    const mat_int_t b_width = 32)
{
    const mat_int_t widx = tb.thread_rank() / 32;
    const thread_block_tile<32> warp = tiled_partition<32>(tb);
    const mat_int_t lwidx = warp.thread_rank();

    const mat_int_t warps_per_block = tb.size() / 32;

    for(mat_int_t i = widx; i < b_height; i += warps_per_block)
    {
        /* multiply row with tridiagonal D */
        const T sup_diag_d = block_diagonal[3 * lwidx];
        const T diag_d = block_diagonal[3 * lwidx + 1];
        const T sub_diag_d = block_diagonal[3 * lwidx + 2];

        const T my_val = dense_block_store[32 * i + lwidx];
        warp.sync();

        T left_val = warp.shfl_up(my_val, 1);
        if(lwidx == 0) left_val = 0;
        T right_val = warp.shfl_down(my_val, 1);
        if(lwidx == 31) right_val = 0;

        if(lwidx < b_width)
        {
            dense_block_store[32 * i + lwidx] =
                sup_diag_d * left_val + diag_d * my_val +
                sub_diag_d * right_val;
        }
        warp.sync();
    }
}

/* ************************************************************************** */

template<typename T>
inline __device__
void
d_apply_left_right_update(
    const T * share_update_left,
    const T * share_update_right,
    T * share_piv_block,
    thread_block& tb,
    const mat_int_t l_b_height = 32,
    const mat_int_t b_width = 32,
    const mat_int_t r_b_height = 32)
{
    const mat_int_t widx = tb.thread_rank() / 32;
    const thread_block_tile<32> warp = tiled_partition<32>(tb);
    const mat_int_t lwidx = warp.thread_rank();

    const mat_int_t warps_per_block = tb.size() / 32;

    for(mat_int_t i = widx; i < l_b_height; i += warps_per_block)
    {
        /* load row from left update */
        const T t_val = share_update_left[32 * i + lwidx];
        warp.sync();

        /**
         * iterate over right update (row-major) per row and multiply each
         * thread's value with the kth value of the left side
         */
        T load_val = 0;
        T product = 0;
        for(mat_int_t k = 0; k < b_width; ++k)
        {
            load_val = share_update_right[32 * k + lwidx];
            product += load_val * warp.shfl(t_val, k);
        }
        warp.sync();

        /* add result to pivot block in shared memory */
        if(lwidx < r_b_height)
            share_piv_block[32 * i + lwidx] -= product;
    }
}

/* ************************************************************************** */

/**
 * applys all deferred updates for one pivot (diagonal) block; each block
 * works on one pivot, sharing the work among its warps -
 * needs 3 * 32 * 32 T's as shared memory for minimal bank conflicts
 */
template<typename T>
inline __device__
void
d_factor_pivot_apply_update(
    const mat_int_t * block_starts,
    const matrix_block * up_block,
    const mat_int_t * ix_store,
    const T * val_store,
    const T * up_block_diagonal,
    T * share_piv_block,
    T * share_update_left,
    T * share_update_right,
    thread_block& tb,
    thread_block_tile<32>& warp)
{
    /* index shorthands */
    const mat_int_t tidx = tb.thread_rank();
    const mat_int_t widx = tb.thread_rank() / 32;
    const mat_int_t lwidx = warp.thread_rank();
    const mat_int_t warps_per_block = tb.size() / 32;

    /**
     * call for one update per block
     */
    const mat_int_t up_from_row = block_starts[up_block->block_row];
    const mat_int_t up_to_row = block_starts[up_block->block_row + 1];

    const mat_int_t up_from_col = block_starts[up_block->block_col];
    const mat_int_t up_to_col = block_starts[up_block->block_col + 1];

    /* load update block for right (U_ki) */
    d_load_block<T, false>(
        up_block,
        block_starts,
        ix_store,
        val_store,
        share_update_right,
        tb);
    tb.sync();

    /* copy & transpose block for left update (U_ki') */
    d_copy_block<T, true>(
        share_update_right,
        share_update_left,
        tb);
    tb.sync();

    /* scale left update block with tridiagonal matrix (U_ki^T D_kk) */
    d_multiply_block_by_block_diagonal<T>(
        share_update_left,
        up_block_diagonal,
        tb,
        up_to_col - up_from_col,
        up_to_row - up_from_row);
    tb.sync();

    /**
     * Compute update - no sync needed since the warp -> row mapping is
     * constant.
     */
    d_apply_left_right_update<T>(
        share_update_left,
        share_update_right,
        share_piv_block,
        tb,
        up_to_col - up_from_col,
        up_to_row - up_from_row,
        up_to_col - up_from_col);
    tb.sync();
}

/* ************************************************************************** */

/* kernel for 1x1 step of right-looking LDLt, warp-centric */
template<typename T>
inline __device__
void
d_ldlt_1x1_step(
    T * warp_pivot,
    T * block_diagonal,
    const mat_int_t row,
    const mat_int_t len,
    thread_block_tile<32>& warp)
{
    const mat_int_t lwidx = warp.thread_rank();

    bool was_modified = false;

    /* 1st: read pivot */
    T d_11 = warp_pivot[32 * row + row];

    if(fabs(d_11) < 1e-8)
    {
        d_11 = 1e-6;
        was_modified = true;
    }

    /* 2nd: update pivot row */
    T a_12_j = (lwidx < row) ? 0 : warp_pivot[32 * row + lwidx];
    const T u_12_j = a_12_j / d_11;

    warp_pivot[32 * row + lwidx] = u_12_j;

    /* 3rd: rank-1 update in remainder */
    const T row_u_12_j = u_12_j * d_11;
    for(mat_int_t i = row + 1; i < len; ++i)
    {
        warp.sync();

        T left_val = warp.shfl(row_u_12_j, i);
        if(lwidx < len)
            warp_pivot[32 * i + lwidx] -= left_val * u_12_j;
    }

    if(was_modified && lwidx == 0)
        warp_pivot[32 * row + row] = 1;

    /* tridiagonal piv column: [0 d_11 0]' */
    if(lwidx < 3)
        block_diagonal[3 * row + lwidx] = (lwidx != 1 ? 0 : d_11);
}

/* ************************************************************************** */

/* kernel for 2x2 step of right-looking LDLt, warp-centric */
template<typename T>
inline __device__
void
d_ldlt_2x2_step(
    T * warp_pivot,
    T * block_diagonal,
    const mat_int_t row,
    const mat_int_t len,
    const thread_block_tile<32>& warp)
{
    const mat_int_t lwidx = warp.thread_rank();

    bool was_modified = false;

    /* 1st: read 2x2 pivot */
    T d_11 = warp_pivot[32 * row + row];
    T d_12 = warp_pivot[32 * row + row + 1];
    T d_22 = warp_pivot[32 * (row + 1) + (row + 1)];

    if(fabs(d_11 * d_22 - d_12 * d_12) < 1e-8)
    {
        d_11 = 1e-6;
        d_22 = 1e-6;

        was_modified = true;
    }

    /* 2nd: update pivot cols (use Givens transformation) */
    T a_12_j1 = (lwidx < row) ? 0 : warp_pivot[32 * row + lwidx];
    T a_12_j2 = (lwidx < row) ? 0 :
        ((lwidx == row) ? d_12 : warp_pivot[32 * (row + 1) + lwidx]);

    /* alternative: compute inverse directly */
    // const T det = d_11 * d_22 - d_12 * d_12;
    // const T u_12_j1 = (d_22 * a_12_j1 - d_12 * a_12_j2) / det;
    // const T u_12_j2 = (-d_12 * a_12_j1 + d_11 * a_12_j2) / det;

    const T giv_r = sqrt(d_11 * d_11 + d_12 * d_12);
    const T giv_c = d_11 / giv_r;
    const T giv_s = d_12 / giv_r;

    T u_12_j1 = giv_c * a_12_j1 + giv_s * a_12_j2;
    T u_12_j2 = -giv_s * a_12_j1 + giv_c * a_12_j2;

    const T tilde_d11 = giv_c * d_11 + giv_s * d_12;
    const T tilde_d12 = giv_c * d_12 + giv_s * d_22;
    const T tilde_d22 = -giv_s * d_12 + giv_c * d_22;

    u_12_j2 /= tilde_d22;
    u_12_j1 = (u_12_j1 - tilde_d12 * u_12_j2) / tilde_d11;

    warp_pivot[32 * row + lwidx] = (lwidx < row) ? 0 : u_12_j1;
    warp_pivot[32 * (row + 1) + lwidx] = (lwidx < row) ? 0 : u_12_j2;

    /* 3rd: rank-1 update in remainder */
    const T row_u_12_j1 = u_12_j1 * d_11 + u_12_j2 * d_12;
    const T row_u_12_j2 = u_12_j1 * d_12 + u_12_j2 * d_22;
    for(mat_int_t i = row + 2; i < len; ++i)
    {
        warp.sync();

        T left_val_j1 = warp.shfl(row_u_12_j1, i);
        T left_val_j2 = warp.shfl(row_u_12_j2, i);

        warp_pivot[32 * i + lwidx] -= (left_val_j1 * u_12_j1 +
            left_val_j2 * u_12_j2);
    }

    if(was_modified && lwidx == 0)
    {
        warp_pivot[32 * row + row] = 1;
        warp_pivot[32 * row + (row + 1)] = 0;
        warp_pivot[32 * (row + 1) + (row + 1)] = 1;
    }

    if(lwidx < 3)
    {
        /* tridiagonal first piv col: [0 d_11 d_12]' */
        block_diagonal[3 * row + lwidx] = (lwidx != 0) *
            ((lwidx == 1) ? d_11 : d_12);

        /* tridiagional second piv col: [d_12 d_22 0] */
        block_diagonal[3 * (row + 1) + lwidx] = (lwidx != 2) *
            ((lwidx == 0) ? d_12 : d_22);
    }
}

/* ************************************************************************** */

template<typename T>
inline __device__
void
d_dense_pivot_op(
    const mat_int_t block_len,
    T * share_piv_block,
    mat_int_t * permutation,
    const mat_int_t p1,
    const mat_int_t p2,
    thread_block_tile<32>& warp)
{
    const mat_int_t lwidx = warp.thread_rank();

    mat_int_t swap_ix_a, swap_ix_b;

    if(lwidx < p1)
    {
        /* col-swap */
        swap_ix_a = 32 * lwidx + p1;
        swap_ix_b = 32 * lwidx + p2;
    }
    else if(lwidx == p1)
    {
        /* pivot-swap */
        swap_ix_a = 32 * p1 + p1;
        swap_ix_b = 32 * p2 + p2;
    }
    else if(lwidx < p2)
    {
        /* row-to-col */
        swap_ix_a = 32 * p1 + lwidx;
        swap_ix_b = 32 * (p1 + (lwidx - p1)) + p2;
    }
    else if(lwidx == p2)
    {
        /* fix-point */
        swap_ix_a = 32 * p1 + p2;
        swap_ix_b = 32 * p1 + p2;
    }
    else
    {
        /* row-swap */
        swap_ix_a = 32 * p1 + lwidx;
        swap_ix_b = 32 * p2 + lwidx;
    }

    /* swap entries */
    if(lwidx < block_len)
    {
        const T tmp = share_piv_block[swap_ix_a];
        share_piv_block[swap_ix_a] = share_piv_block[swap_ix_b];
        share_piv_block[swap_ix_b] = tmp;
    }

    if(lwidx == 0)
    {
        const mat_int_t tmp = permutation[p1];
        permutation[p1] = permutation[p2];
        permutation[p2] = tmp;
    }
}

/* ************************************************************************** */

template<typename T>
inline __device__
void
d_select_pivot_bunch_kaufman(
    const mat_int_t block_len,
    const T * share_piv_block,
    const mat_int_t piv_ptr,
    thread_block_tile<32>& warp,
    mat_int_t * piv_order,
    mat_int_t * p0,
    mat_int_t * p1,
    const mat_int_t block_id)
{
    const mat_int_t lwidx = warp.thread_rank();

    const T alpha = (1 + sqrtf(17)) / 8;
    *piv_order = 1;

    /* select maximum in current row over diagonal */
    mat_int_t r = max(lwidx, piv_ptr + 1);
    T omega_1 = (lwidx > piv_ptr && lwidx < block_len) ?
        fabs(share_piv_block[32 * piv_ptr + lwidx]) : 0;

    for(mat_int_t s = 1; s < 32; s <<= 1)
    {
        const mat_int_t o_ix = warp.shfl_down(r, s);
        const T o_val = warp.shfl_down(omega_1, s);

        if(o_val > omega_1)
        {
            r = o_ix;
            omega_1 = o_val;
        }

        warp.sync();
    }

    r = warp.shfl(r, 0);
    omega_1 = warp.shfl(omega_1, 0);

    if(omega_1 > 0)
    {
        if(fabs(share_piv_block[32 * piv_ptr + piv_ptr]) >= alpha * omega_1)
        {
            /* use current 1x1 pivot */
            *p0 = piv_ptr;
        }
        else
        {
            /* select maximum magnitude in row r (Schur complement!) */
            T omega_r = (lwidx < r) ?
                fabs(share_piv_block[32 * lwidx + r]) :
                fabs(share_piv_block[32 * r + lwidx]);

            if(lwidx < piv_ptr || lwidx == r || lwidx >= block_len)
                omega_r = 0;

            for(mat_int_t s = 1; s < 32; s <<= 1)
            {
                const T o_val = warp.shfl_down(omega_r, s);
                omega_r = max(omega_r, o_val);
                warp.sync();
            }

            omega_r = warp.shfl(omega_r, 0);

            if(fabs(share_piv_block[32 * piv_ptr + piv_ptr]) * omega_r >=
                alpha * omega_1 * omega_1)
            {
                /* use current 1x1 pivot */
                *p0 = piv_ptr;
            }
            else if(fabs(share_piv_block[32 * r + r]) >= alpha * omega_r)
            {
                /* use col r as 1x1 pivot */
                *p0 = r;
            }
            else
            {
                *piv_order = 2;

                /* use cols r, p as 2x2 pivot */
                *p0 = piv_ptr;
                *p1 = r;
            }
        }
    }
    else
    {
        *p0 = piv_ptr;
    }
}

/* ************************************************************************** */

template<typename T>
inline __device__
void
d_select_pivot_rook(
    const mat_int_t block_len,
    const T * share_piv_block,
    const mat_int_t piv_ptr,
    thread_block_tile<32>& warp,
    mat_int_t * piv_order,
    mat_int_t * p0,
    mat_int_t * p1)
{
    const mat_int_t lwidx = warp.thread_rank();

    const T alpha = (1 + sqrtf(17)) / 8;
    *piv_order = 1;

    /* select maximum in current row */
    mat_int_t r = max(lwidx, piv_ptr + 1);
    T omega_i = (lwidx > piv_ptr && lwidx < block_len) ?
        fabs(share_piv_block[32 * piv_ptr + lwidx]) : 0;

    for(mat_int_t s = 1; s < 32; s <<= 1)
    {
        const mat_int_t o_ix = warp.shfl_down(r, s);
        const T o_val = warp.shfl_down(omega_i, s);

        if(o_val > omega_i)
        {
            r = o_ix;
            omega_i = o_val;
        }

        warp.sync();
    }

    r = warp.shfl(r, 0);
    omega_i = warp.shfl(omega_i, 0);

    if(omega_i > 0)
    {
        mat_int_t i = piv_ptr;

        if(fabs(share_piv_block[32 * piv_ptr + piv_ptr]) >= alpha * omega_i)
        {
            /* use current 1x1 pivot */
            *p0 = i;
        }
        else
        {
            while(true)
            {
                /* select index of entry of max. mag in row r */
                mat_int_t p = max(lwidx, piv_ptr);
                T omega_r = (lwidx >= piv_ptr && lwidx < block_len) ?
                    ((lwidx < r) ?
                        fabs(share_piv_block[32 * lwidx + r]) :
                        fabs(share_piv_block[32 * r + lwidx]))
                    : 0;

                if(lwidx == r)
                    omega_r = 0;

                for(mat_int_t s = 1; s < 32; s <<= 1)
                {
                    const mat_int_t o_ix = warp.shfl_down(p, s);
                    const T o_val = warp.shfl_down(omega_r, s);

                    if(o_val > omega_r)
                    {
                        p = o_ix;
                        omega_r = o_val;
                    }

                    warp.sync();
                }

                p = warp.shfl(p, 0);
                omega_r = warp.shfl(omega_r, 0);

                if(fabs(share_piv_block[32 * r + r]) >= alpha * omega_r)
                {
                    /* use col r as 1x1 pivot */
                    *p0 = r;

                    return;
                }
                else if(omega_i == omega_r)
                {
                    *piv_order = 2;

                    /* use cols r, p as 2x2 pivot */
                    *p0 = i;
                    *p1 = r;

                    return;
                }
                else
                {
                    i = r;
                    r = p;
                    omega_i = omega_r;
                }
            }
        }
    }
    else
    {
        *p0 = piv_ptr;
    }
}

/* ************************************************************************** */

/**
 * Factorize a 32 x 32 dense block by U' D U factorization using 1x1 and
 * 2x2 pivots selected by Bunch-Kaufmann pivoting.
 */
template<typename T, bool USE_BK_PIVOTING, bool USE_ROOK_PIVOTING>
inline __device__
void
d_factor_pivot_factorize(
    const mat_int_t block_len,
    mat_int_t * is_piv_this,
    T * block_diagonal_this,
    T * share_piv_block,
    mat_int_t * permutation_this,
    thread_block_tile<32>& warp,
    const mat_int_t block_id,
    const mat_int_t block_row)
{
    const bool use_pivoting = USE_BK_PIVOTING || USE_ROOK_PIVOTING;
    const mat_int_t lwidx = warp.thread_rank();
    mat_int_t piv_ptr = 0;

    while(piv_ptr < block_len)
    {
        /**
         * Select next pivot
         */
        mat_int_t piv_order, p0, p1;
        if(piv_ptr < block_len - 1 && USE_BK_PIVOTING)
        {
            d_select_pivot_bunch_kaufman<T>(
                block_len,
                share_piv_block,
                piv_ptr,
                warp,
                &piv_order,
                &p0,
                &p1,
                block_id);
        }
        else if(piv_ptr < block_len - 1 && USE_ROOK_PIVOTING)
        {
            d_select_pivot_rook<T>(
                block_len,
                share_piv_block,
                piv_ptr,
                warp,
                &piv_order,
                &p0,
                &p1);
        }
        else
        {
            piv_order = (piv_ptr < block_len - 1 && !is_piv_this[piv_ptr + 1])
                ? 2 : 1;
            p0 = piv_ptr;
            p1 = piv_ptr + 1;
        }
        warp.sync();

        /**
         * Permute matrix if necessary and execute factorization step
         */
        if(piv_order == 1)
        {
            if(use_pivoting && piv_ptr != p0)
            {
                d_dense_pivot_op<T>(
                    block_len,
                    share_piv_block,
                    permutation_this,
                    piv_ptr,
                    p0,
                    warp);
                warp.sync();
            }

            d_ldlt_1x1_step<T>(
                share_piv_block,
                block_diagonal_this,
                piv_ptr,
                block_len,
                warp);

            if(use_pivoting && lwidx == 0)
            {
                is_piv_this[piv_ptr] = 1;
            }
        }
        else
        {
            if(use_pivoting)
            {
                if(p0 != piv_ptr)
                {
                    d_dense_pivot_op<T>(
                        block_len,
                        share_piv_block,
                        permutation_this,
                        piv_ptr,
                        p0,
                        warp);
                    warp.sync();
                }

                d_dense_pivot_op<T>(
                    block_len,
                    share_piv_block,
                    permutation_this,
                    piv_ptr + 1,
                    p1,
                    warp);
                warp.sync();
            }

            d_ldlt_2x2_step<T>(
                share_piv_block,
                block_diagonal_this,
                piv_ptr,
                block_len,
                warp);

            if(use_pivoting && lwidx == 0)
            {
                is_piv_this[piv_ptr] = 1;
                is_piv_this[piv_ptr + 1] = 0;
            }
        }
        warp.sync();

        piv_ptr += piv_order;
    }
}

/* ************************************************************************** */

template<typename T, bool TRANSPOSE>
inline __device__
void
d_save_dense_block(
    const T * share_piv_block,
    T * dense_block,
    thread_block& tb)
{
    const mat_int_t widx = tb.thread_rank() / 32;
    const thread_block_tile<32> warp = tiled_partition<32>(tb);
    const mat_int_t lwidx = warp.thread_rank();

    const mat_int_t warps_per_block = tb.size() / 32;

    for(mat_int_t i = widx; i < 32; i += warps_per_block)
    {
        if(TRANSPOSE)
            dense_block[32 * lwidx + i] = share_piv_block[32 * i + lwidx];
        else
            dense_block[32 * i + lwidx] = share_piv_block[32 * i + lwidx];
    }
}

/* ************************************************************************** */

template<typename T, bool TRANSPOSE>
inline __device__
void
d_load_dense_block(
    const T * dense_block,
    T * share_piv_block,
    thread_block& tb)
{
    const mat_int_t widx = tb.thread_rank() / 32;
    const thread_block_tile<32> warp = tiled_partition<32>(tb);
    const mat_int_t lwidx = warp.thread_rank();

    const mat_int_t warps_per_block = tb.size() / 32;

    for(mat_int_t i = widx; i < 32; i += warps_per_block)
    {
        if(TRANSPOSE)
            share_piv_block[32 * lwidx + i] = dense_block[32 * i + lwidx];
        else
            share_piv_block[32 * i + lwidx] = dense_block[32 * i + lwidx];
    }
}

/* ************************************************************************** */

/**
 * LDL Pivot block factorization:
 * - no pivoting
 * - one block per pivot, iterate over updates
 * - use all warps for each update
 * - accumulate result in shared memory (distribute rows among warps)
 *
 * Shared memory required: (32 * 32 + #warps * 32) T's + 32 int
 */
template<typename T, bool USE_BK_PIVOTING, bool USE_ROOK_PIVOTING>
__global__
void
k_factor_pivot_blocks(
    const mat_int_t * block_starts,
    mat_int_t * is_piv,
    const mat_int_t * piv_level_block_list,
    const mat_int_t * block_csc_col,
    const mat_int_t * block_csc_row,
    const matrix_block * blox,
    const mat_int_t * __restrict__ ix_store,
    const T * __restrict__ val_store,
    mat_int_t * piv_block_location,
    T * inflight_piv_blocks,
    T * block_diagonal,
    T * row_norms,
    mat_int_t * permutation)
{
    /* shared memory: three 32 * 32 matrices = 24 kb */
    __shared__ T share_piv_block[32 * 32];
    __shared__ T share_update_left[32 * 32];
    __shared__ T share_update_right[32 * 32];

    thread_block tb = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(tb);

    /* shorthands for warp, thread index */
    const mat_int_t bidx = tb.group_index().x;
    const mat_int_t tidx = tb.thread_rank();
    const mat_int_t widx = tidx / 32;
    const mat_int_t lwidx = warp.thread_rank();

    const mat_int_t num_warps = tb.size() / 32;

    /**
     * Phase 0: select the block id we're working on - the level block list
     * contains all pivot blocks at the start of each level's indices
     */
    const mat_int_t piv_block_id = piv_level_block_list[bidx];
    const matrix_block * my_block = blox + piv_block_id;

    /* pivot block - from_col/to_col is same as rows */
    const mat_int_t from_row = block_starts[my_block->block_row];
    const mat_int_t to_row = block_starts[my_block->block_row + 1];

    /**
     * Phase 1: load A_ii into shared memory
     * - iterate through rows, each warp adopting one row
     * - fill the remainder with 0s
     */
    const mat_int_t block_size = to_row - from_row;

    d_load_block<T, false>(
        my_block,
        block_starts,
        ix_store,
        val_store,
        share_piv_block,
        tb);
    tb.sync();

    /**
     * Phase 2: apply deferred (symmetric) updates from all rows that have
     *          a nonzero block in the block column above this pivot
     * - load block once and write it into shared memory twice
     * - distribute rows in the update computation across warps
     *
     * Use (sorted) CSC to find these blocks, stop above the diagonal!
     */

    for(mat_int_t csc_ix = block_csc_col[my_block->block_col];
        csc_ix < block_csc_col[my_block->block_col + 1] - 1; ++csc_ix)
    {
        const mat_int_t dep_block_id = block_csc_row[csc_ix];
        const matrix_block * dep_block = blox + dep_block_id;
        const T * dep_block_diagonal = block_diagonal +
            3 * block_starts[dep_block->block_row];

        /* apply update collected from up_block */
        d_factor_pivot_apply_update<T>(
            block_starts,
            dep_block,
            ix_store,
            val_store,
            dep_block_diagonal,
            share_piv_block,
            share_update_left,
            share_update_right,
            tb,
            warp);
    }

    /**
     * Phase 3: factorize updated pivot block using 1 warp
     */

    if(widx == 0)
    {
        d_factor_pivot_factorize<T, USE_BK_PIVOTING, USE_ROOK_PIVOTING>(
            block_size,
            is_piv + from_row,
            block_diagonal + 3 * from_row,
            share_piv_block,
            permutation + from_row,
            warp,
            piv_block_id,
            my_block->block_row);
    }
    tb.sync();

    /**
     * Phase 4: add squares of entries to row norms
     */
    if(widx == 0)
    {
        T rowsum = 0;

        for(mat_int_t j = 0; j < to_row - from_row; ++j)
            rowsum += share_piv_block[32 * lwidx + j] *
                share_piv_block[32 * lwidx + j];

        if(lwidx < to_row - from_row)
            row_norms[from_row + lwidx] = rowsum;
    }
    tb.sync();

    /**
     * Phase 5: write out factorized pivot L into global memory (block-diagonal
     * D is already there)
     */

    T * my_dense_pivot = inflight_piv_blocks + (bidx * 32 * 32);
    d_save_dense_block<T, false>(share_piv_block, my_dense_pivot, tb);

    /* save an inverted dense map - i.e. where is this row's inflight block? */
    if(tidx == 0)
        piv_block_location[my_block->block_row] = bidx;
}

/* ************************************************************************** */

template<typename T>
inline __device__
void
d_col_pivot_block(
    const mat_int_t * block_starts,
    const matrix_block * blk,
    mat_int_t * ix_store,
    T * val_store,
    const mat_int_t * permutation,
    const mat_int_t * inv_permutation,
    thread_block_tile<32>& warp)
{
    const mat_int_t lwidx = warp.thread_rank();

    if(blk->format == BLOCK_SPARSE)
    {
        mat_int_t * ix_ptr = ix_store + blk->col_ptr;

        const mat_int_t from_col = block_starts[blk->block_col];
        const mat_int_t to_col = block_starts[blk->block_col + 1];
        const mat_int_t block_width = to_col - from_col;

        /* sparse: just change indices */
        for(mat_int_t ix = lwidx; ix < blk->nnz; ix += 32)
        {
            const mat_int_t in_ix = ix_ptr[ix];
            const mat_int_t in_row = in_ix / block_width;
            const mat_int_t in_col = in_ix % block_width;

            ix_ptr[ix] = in_row * block_width + inv_permutation[in_col];
        }
    }
    else
    {
        T * val_ptr = val_store + blk->val_ptr;

        const mat_int_t piv_from_row = block_starts[blk->block_row];
        const mat_int_t piv_to_row = block_starts[blk->block_row + 1];
        const mat_int_t j_height = piv_to_row - piv_from_row;

        const mat_int_t piv_from_col = block_starts[blk->block_col];
        const mat_int_t piv_to_col = block_starts[blk->block_col + 1];
        const mat_int_t j_width = piv_to_col - piv_from_col;

        const mat_int_t lwidx_inv_perm = (lwidx < j_width) ?
            permutation[lwidx] : lwidx;
        for(mat_int_t i = 0; i < j_height; ++i)
        {
            T my_val;

            if(lwidx < j_width)
                my_val = val_ptr[i * j_width + lwidx];

            my_val = warp.shfl(my_val, lwidx_inv_perm);

            if(lwidx < j_width)
                val_ptr[i * j_width + lwidx] = my_val;

            warp.sync();
        }
    }
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_pivot_col_blocks(
    const mat_int_t * block_starts,
    const mat_int_t * level_row_list,
    const mat_int_t * block_csc_col,
    const mat_int_t * block_csc_row,
    const matrix_block * blox,
    mat_int_t * ix_store,
    T * val_store,
    const mat_int_t * permutation)
{
    __shared__ mat_int_t inv_permutation[32];
    __shared__ mat_int_t lcl_permutation[32];

    thread_block tb = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(tb);

    /* shorthands for warp, thread index */
    const mat_int_t bidx = tb.group_index().x;
    const mat_int_t tidx = tb.thread_rank();
    const mat_int_t widx = tidx / 32;
    const mat_int_t lwidx = warp.thread_rank();

    const mat_int_t num_warps = tb.size() / 32;

    /* invert block permutation */
    const mat_int_t block_row = level_row_list[bidx];

    const mat_int_t from_row = block_starts[block_row];
    const mat_int_t to_row = block_starts[block_row + 1];
    const mat_int_t block_width = to_row - from_row;

    if(tidx < block_width)
    {
        const mat_int_t local_permutation = permutation[from_row + lwidx] -
            from_row;
        inv_permutation[local_permutation] = lwidx;
        lcl_permutation[lwidx] = local_permutation;
    }
    tb.sync();

    /* check whether permuting is at all necessary */
    const bool unchanged_piv = (lwidx < block_width) ?
        (inv_permutation[lwidx] == lwidx) : true;
    if(__popc(warp.ballot(unchanged_piv)) == 32)
        return;

    /* pivot all blocks above the block - diagonal (1 warp per block) */
    for(mat_int_t j = block_csc_col[block_row] + widx;
        j < block_csc_col[block_row + 1] - 1; j += num_warps)
    {
        const mat_int_t j_blk_id = block_csc_row[j];
        const matrix_block * j_blk = blox + j_blk_id;

        d_col_pivot_block<T>(
            block_starts,
            j_blk,
            ix_store,
            val_store,
            lcl_permutation,
            inv_permutation,
            warp);
        warp.sync();
    }
}

/* ************************************************************************** */

template<typename T>
inline __device__
void
d_apply_offdiagonal_update(
    const mat_int_t * block_starts,
    const mat_int_t * __restrict__ ix_store,
    const T * __restrict__ val_store,
    T * dense_nopiv_block,
    const matrix_block * left_block,
    T * dense_left_block,
    const matrix_block * right_block,
    T * dense_right_block,
    const T * tridiagonal,
    thread_block& tb)
{
    /* load left block (transposed) */
    d_load_block<T, true>(
        left_block,
        block_starts,
        ix_store,
        val_store,
        dense_left_block,
        tb);
    tb.sync();

    /* multiply left block by tridiagonal matrix */
    const mat_int_t left_from_row = block_starts[left_block->block_row];
    const mat_int_t left_to_row = block_starts[left_block->block_row + 1];

    const mat_int_t left_from_col = block_starts[left_block->block_col];
    const mat_int_t left_to_col = block_starts[left_block->block_col + 1];

    d_multiply_block_by_block_diagonal<T>(
        dense_left_block,
        tridiagonal + 3 * left_from_row,
        tb,
        left_to_col - left_from_col,
        left_to_row - left_from_row);
    tb.sync();

    /* load right block */
    const mat_int_t right_from_col = block_starts[right_block->block_col];
    const mat_int_t right_to_col = block_starts[right_block->block_col + 1];

    d_load_block<T, false>(
        right_block,
        block_starts,
        ix_store,
        val_store,
        dense_right_block,
        tb);
    tb.sync();

    /* apply update to A block */
    d_apply_left_right_update<T>(
        dense_left_block,
        dense_right_block,
        dense_nopiv_block,
        tb,
        left_to_col - left_from_col,
        left_to_row - left_from_row,
        right_to_col - right_from_col);
    tb.sync();
}

/* ************************************************************************** */

template<typename T>
inline __device__
void
d_left_solve_with_D_Ut(
    const T * UUt,
    const T * D,
    T * A,
    const mat_int_t * is_piv,
    thread_block& tb,
    const mat_int_t block_height,
    const mat_int_t block_width,
    const mat_int_t id)
{
    /**
     * Note: UUt contains U11 in row-major, i.e. U11^T in col-major
     */
    const mat_int_t widx = tb.thread_rank() / 32;
    thread_block_tile<32> warp = tiled_partition<32>(tb);
    const mat_int_t lwidx = warp.thread_rank();

    const mat_int_t warps_per_block = tb.size() / 32;

    const bool is_piv_1x1 = lwidx < block_height && is_piv[lwidx];
    const bool is_piv_2x2 = (lwidx + 1) < block_height && !is_piv[lwidx + 1];

    /* each warp processes one column in A_12 */
    for(mat_int_t j = widx; j < block_width; j += warps_per_block)
    {
        /* save a column of A in registers */
        T rhs = (lwidx < block_height) ? A[32 * lwidx + j] : 0;

        /* triangular forward solve with Ut equals backsolve with U */
        for(mat_int_t i = 0; i < block_height; ++i)
        {
            /* first: solve with pivot i by broadcast */
            T my_piv = rhs / UUt[32 * i + i];
            my_piv = warp.shfl(my_piv, i);

            /* update RHS */
            if(lwidx > i && lwidx < block_height)
                rhs -= UUt[32 * i + lwidx] * my_piv;
        }

        if(lwidx >= block_height)
            rhs = 0;
        warp.sync();

        /* left solve with block-diagonal matrix (always use 2x2) */
        T right_rhs = warp.shfl_down(rhs, 1);
        const T d_11 = (lwidx < block_height) ? D[3 * lwidx + 1] : 0;
        T d_12 = D[3 * lwidx + 2];
        T d_22 = warp.shfl_down(d_11, 1);

        if(lwidx == block_height)
            d_22 = 0;

        const T giv_r = sqrt(d_11 * d_11 + d_12 * d_12);
        const T giv_c = d_11 / giv_r;
        const T giv_s = d_12 / giv_r;

        T x_1 = giv_c * rhs + giv_s * right_rhs;
        T x_2 = -giv_s * rhs + giv_c * right_rhs;

        const T tilde_d11 = giv_c * d_11 + giv_s * d_12;
        const T tilde_d12 = giv_c * d_12 + giv_s * d_22;
        const T tilde_d22 = -giv_s * d_12 + giv_c * d_22;

        x_2 = (is_piv_2x2) ? (x_2 / tilde_d22) : 0;
        x_1 = (x_1 - tilde_d12 * x_2) / tilde_d11;

        /* alternative: ivert directly */
        // const T det = d_11 * d_22 - d_12 * d_12;
        // const T x_1 = (d_22 * rhs - d_12 * right_rhs) / det;
        // const T x_2 = (-d_12 * rhs + d_11 * right_rhs) / det;

        /* update A with the solved column */
        if(is_piv_1x1) /* 1x1 */
            A[32 * lwidx + j] = x_1;
        if(is_piv_2x2) /* 2x2 */
           A[32 * (lwidx + 1) + j] = x_2;
    }
}

/* ************************************************************************** */

/**
 * Allow atomic add for double on older CUDA architectures
 */

template<typename T>
inline __device__
T
real_t_atomic_add(
    T * ptr,
    const T val);

template<>
inline __device__
float
real_t_atomic_add(
    float * ptr,
    const float val)
{
    return atomicAdd(ptr, val);
}

template<>
inline __device__
double
real_t_atomic_add(
    double * ptr,
    const double val)
{
#if (__CUDA_ARCH__ >= 600)
    return atomicAdd(ptr, val);
#else
    /* proxy implementation using int64s */
    unsigned long long int* ptr_as_ull = (unsigned long long int*) ptr;
    unsigned long long int old = *ptr_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(ptr_as_ull, assumed, __double_as_longlong(val +
            __longlong_as_double(assumed)));
    }
    while (assumed != old);

    return __longlong_as_double(old);
#endif
}

/**
 * Off-diagonal block updates: processing block (i, j) needs blocks
 * (k, i) and (k, j) for all k < i. These dependencies are found by traversing
 * the block-CSC representation.
 *
 * Note: the pivot blocks are always the first blocks in a level's block list -
 * they are needed for processing offdiagonal blocks. The location of their
 * in-flight dense block needs to be saved, though.
 */
template<typename T, bool USE_PIVOTING>
__global__
void
k_update_scale_offdiagonal_blocks(
    const mat_int_t * block_starts,
    const mat_int_t * is_piv,
    const mat_int_t * nopiv_level_block_list,
    const mat_int_t * block_csc_col,
    const mat_int_t * block_csc_row,
    const matrix_block * blox,
    const mat_int_t * __restrict__ ix_store,
    const T * __restrict__ val_store,
    const mat_int_t * piv_block_location,
    const T * inflight_piv_blocks,
    T * inflight_nopiv_blocks,
    const T * tridiagonal,
    T * row_norms,
    const mat_int_t * permutation)
{
    __shared__ T base[32 * 32 * 3];

    thread_block tb = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(tb);

    /* shorthands for warp, thread index */
    const mat_int_t bidx = tb.group_index().x;
    const mat_int_t tidx = tb.thread_rank();
    const mat_int_t widx = tb.thread_rank() / 32;
    const mat_int_t lwidx = warp.thread_rank();

    const mat_int_t num_warps = tb.size() / 32;

    /* shared memory: three 32 * 32 matrices <= 24 kb */
    T * share_nopiv_block = base;
    T * share_update_left = share_nopiv_block + 32 * 32;
    T * share_update_right = share_update_left + 32 * 32;

    /* get current block */
    const mat_int_t nopiv_block_id = nopiv_level_block_list[bidx];
    const matrix_block * nopiv_block = blox + nopiv_block_id;

    const mat_int_t from_row = block_starts[nopiv_block->block_row];
    const mat_int_t to_row = block_starts[nopiv_block->block_row + 1];
    const mat_int_t block_height = to_row - from_row;

    const mat_int_t from_col = block_starts[nopiv_block->block_col];
    const mat_int_t to_col = block_starts[nopiv_block->block_col + 1];
    const mat_int_t block_width = to_col - from_col;

    /* load block into shared memory for processing */
    d_load_block<T, false>(
        nopiv_block,
        block_starts,
        ix_store,
        val_store,
        share_nopiv_block,
        tb);
    tb.sync();

    /**
     * Fan-in updates from previous rows
     */

    /* traverse over coarse columns and find dependencies */
    mat_int_t i_ptr = block_csc_col[nopiv_block->block_row];
    const mat_int_t i_end = block_csc_col[nopiv_block->block_row + 1] - 1;

    mat_int_t j_ptr = block_csc_col[nopiv_block->block_col];
    const mat_int_t j_end = block_csc_col[nopiv_block->block_col + 1] - 1;

    /* assumes both columns are ordered by rows */
    while(i_ptr < i_end && j_ptr < j_end)
    {
        const mat_int_t i_row = (blox + block_csc_row[i_ptr])->block_row;
        const mat_int_t j_row = (blox + block_csc_row[j_ptr])->block_row;

        if(i_row == j_row)
        {
            const matrix_block * i_blk = blox + block_csc_row[i_ptr];
            const matrix_block * j_blk = blox + block_csc_row[j_ptr];

            /* found a row with both blocks, update with i_ptr & j_ptr */
            d_apply_offdiagonal_update<T>(
                block_starts,
                ix_store,
                val_store,
                share_nopiv_block,
                i_blk,
                share_update_left,
                j_blk,
                share_update_right,
                tridiagonal,
                tb);
            tb.sync();
        }

        /* step to next row */
        i_ptr += (i_row <= j_row);
        j_ptr += (i_row >= j_row);
    }
    tb.sync();

    /**
     * If pivoting is enabled, permute the block's rows appropriately
     */
    if(USE_PIVOTING)
    {
        const mat_int_t * local_permute = permutation + from_row;
        T * tmp_holding = share_update_left;

        d_copy_block<T, false>(
            share_nopiv_block,
            tmp_holding,
            tb);
        tb.sync();

        for(mat_int_t i = widx; i < block_height; i += num_warps)
        {
            const mat_int_t p_i = local_permute[i] - from_row;
            share_nopiv_block[32 * i + lwidx] =
                tmp_holding[32 * p_i + lwidx];
        }
        tb.sync();
    }

    /**
     * Scale off-diagonal block, i.e. solve with piv block and triangular
     * part
     */
    T * share_piv_block = share_update_left;

    /* load dense pivot block */
    const mat_int_t piv_block_offset =
        piv_block_location[nopiv_block->block_row];
    d_load_dense_block<T, false>(
        inflight_piv_blocks + piv_block_offset * 32 * 32,
        share_piv_block,
        tb);
    tb.sync();

    /* left solve with U't and D */
    d_left_solve_with_D_Ut<T>(
        share_piv_block,
        tridiagonal + 3 * from_row,
        share_nopiv_block,
        is_piv + from_row,
        tb,
        block_height,
        block_width,
        nopiv_block_id);
    tb.sync();

    /**
     * Add squares of entries to row norms for dropping later
     */
    if(widx == 0)
    {
        T rowsum = 0;

        for(mat_int_t j = 0; j < to_col - from_col; ++j)
            rowsum += share_nopiv_block[32 * lwidx + j] *
                share_nopiv_block[32 * lwidx + j];

        if(lwidx < to_row - from_row)
            real_t_atomic_add(&row_norms[from_row + lwidx], rowsum);
    }
    tb.sync();

    /**
     * Write dense block to storage
     */
    d_save_dense_block<T, false>(
        share_nopiv_block,
        inflight_nopiv_blocks + bidx * 32 * 32,
        tb);
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_compute_row_offdiagonal_ssq(
    const mat_int_t * block_starts,
    const mat_int_t * level_row_list,
    const mat_int_t * block_csr_row,
    const mat_int_t * block_csr_col,
    const mat_int_t * row_first_block_offset,
    const T * inflight_nopiv_blocks,
    T * row_norms)
{
    __shared__ T base[32 * 32];

    thread_block tb = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(tb);

    /* shorthands for warp, thread index */
    const mat_int_t bidx = tb.group_index().x;
    const mat_int_t tidx = tb.thread_rank();
    const mat_int_t widx = tb.thread_rank() / 32;
    const mat_int_t lwidx = warp.thread_rank();

    const mat_int_t num_warps = tb.size() / 32;

    /* traverse block row, leave out pivot block */
    const mat_int_t coarse_i = level_row_list[bidx];
    const mat_int_t coarse_i_size = block_csr_row[coarse_i + 1] -
        block_csr_row[coarse_i] - 1;

    const mat_int_t from_row = block_starts[coarse_i];
    const mat_int_t to_row = block_starts[coarse_i + 1];

    for(mat_int_t j = 0; j < coarse_i_size; ++j)
    {
        const mat_int_t dense_blk_offset = row_first_block_offset[coarse_i] + j;
        d_load_dense_block<T, false>(
            inflight_nopiv_blocks + dense_blk_offset * 32 * 32,
            base,
            tb);
        tb.sync();

        /* use a warp to compute the row sums */
        if(widx == 0)
        {
            T rowsum = 0;

            for(mat_int_t j = 0; j < 32; ++j)
                rowsum += base[32 * lwidx + j] * base[32 * lwidx + j];

            if(lwidx < (to_row - from_row))
                row_norms[from_row + lwidx] += rowsum;
        }
        tb.sync();
    }
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_apply_block_dual_dropping(
    const mat_int_t * block_starts,
    const mat_int_t * level_block_list,
    matrix_block * blox,
    mat_int_t * ix_store,
    T * val_store,
    mat_int_t * leftover_space,
    mat_int_t * out_ix_offset,
    mat_int_t * out_val_offset,
    T * inflight_blocks,
    const T * row_norms,
    const T threshold)
{
    __shared__ T base_vals[32 * 32];

    thread_block tb = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(tb);

    /* shorthands for warp, thread index */
    const mat_int_t bidx = tb.group_index().x;
    const mat_int_t tidx = tb.thread_rank();
    const mat_int_t widx = tb.thread_rank() / 32;
    const mat_int_t lwidx = warp.thread_rank();

    const mat_int_t num_warps = tb.size() / 32;

    /* load in-flight block */
    const mat_int_t inflight_block_id = level_block_list[bidx];
    matrix_block * inflight_block = blox + inflight_block_id;

    const mat_int_t from_row = block_starts[inflight_block->block_row];
    const mat_int_t to_row = block_starts[inflight_block->block_row + 1];
    const mat_int_t block_height = to_row - from_row;

    const mat_int_t from_col = block_starts[inflight_block->block_col];
    const mat_int_t to_col = block_starts[inflight_block->block_col + 1];
    const mat_int_t block_width = to_col - from_col;

    d_load_dense_block<T, false>(
        inflight_blocks + bidx * 32 * 32,
        &base_vals[0],
        tb);

    if(tidx == 0)
    {
        inflight_block->nnz = 0;
    }

    tb.sync();

    /* there's no point in dropping stuff from a dense block, is there? */
    if(inflight_block->format == BLOCK_SPARSE)
    {
        /* execute threshold dropping and count nnz */
        for(mat_int_t i = widx; i < (to_row - from_row); i += num_warps)
        {
            const T i_norm = sqrtf(row_norms[from_row + i]);

            /* each warp tackles one row */
            const T i_val = (lwidx < block_width) ?
                base_vals[32 * i + lwidx] : 0;
            const bool do_drop = fabs(i_val) < threshold * i_norm;
            const bool is_nz = (fabs(i_val) > NZ_EPS);

            if(do_drop && is_nz)
            {
                base_vals[32 * i + lwidx] = 0;
            }

            /* count number of NZ in this warp's row */
            const mat_int_t i_nnz = __popc(warp.ballot(!do_drop && is_nz));

            if(lwidx == 0)
                atomicAdd(&inflight_block->nnz, i_nnz);

            warp.sync();
        }
        tb.sync();

        /* try to reuse leftover space from old blocks if possible */
        if(inflight_block->nnz > inflight_block->max_nnz)
        {
            const mat_int_t extra_req = inflight_block->nnz -
                inflight_block->max_nnz;

            /* try to use leftover space from earlier blocks */
            if(tidx == 0)
            {
                /* record free storage and request as much as we need */
                const mat_int_t was_free = atomicSub(leftover_space,
                    extra_req);

                /* can't have more than we wanted or than was available */
                const mat_int_t extra_granted =
                    max(min(extra_req, was_free), 0);

                /* in case we only need a part, set record straight */
                const mat_int_t giveback = extra_req - extra_granted;

                if(giveback > 0)
                    atomicAdd(leftover_space, giveback);

                /* got any space? great, record that! */
                if(extra_granted > 0)
                    inflight_block->max_nnz += extra_granted;

                inflight_block->got_extra = extra_granted;
            }
            tb.sync();
        }
    }

    /**
     * with the space requirements set, find new col_ptr / val_ptr
     */

    /* get enough storage for block */
    const mat_int_t storage_req_ix =
        (inflight_block->format == BLOCK_SPARSE) ?
        min(inflight_block->nnz, inflight_block->max_nnz) : 0;
    const mat_int_t storage_req_val =
        (inflight_block->format == BLOCK_SPARSE) ?
        min(inflight_block->nnz, inflight_block->max_nnz) :
        (block_height * block_width);

    /* get new pointer for data storage */
    if(tidx == 0)
    {
        inflight_block->col_ptr = atomicAdd(out_ix_offset, storage_req_ix);
        inflight_block->val_ptr = atomicAdd(out_val_offset,
            storage_req_val);

        /* add leftover space for sparse blocks */
        if(inflight_block->format == BLOCK_SPARSE)
        {
            const mat_int_t leftover_this = max(inflight_block->max_nnz -
                inflight_block->nnz, 0);
            atomicAdd(leftover_space, leftover_this);
        }
    }
    tb.sync();

    /* apply secondary dropping for sparse blocks, then store to mem directly */
    if(inflight_block->format == BLOCK_SPARSE &&
        inflight_block->nnz > inflight_block->max_nnz)
    {
        /* with max_nnz now fixed, continue to drop as usual */
        typedef cub::BlockRadixSort<T, 256, 4, mat_int_t> BlockRadixSort;
        __shared__ typename BlockRadixSort::TempStorage temp_storage;

        T thread_keys[4];
        mat_int_t thread_vals[4];

        /* collect keys / values (0 -> 1023) for sorting */
        for(mat_int_t i = 0; i < 4; ++i)
        {
            const mat_int_t i_ix = tidx * 4 + i;
            const mat_int_t i_row = i_ix / 32;
            const mat_int_t i_col = i_ix % 32;

            thread_keys[i] = (i_row < block_height && i_col < block_width) ?
                fabs(base_vals[tidx * 4 + i]) : 0;
            thread_vals[i] = i_ix;
        }
        tb.sync();

        /* sort keys after abs of value, descending */
        BlockRadixSort(temp_storage).SortDescending(
            thread_keys,
            thread_vals);
        tb.sync();

        /* write out the (sparse) block */
        if(tidx == 0)
            inflight_block->nnz = inflight_block->max_nnz;

        mat_int_t * block_ix_store = ix_store + inflight_block->col_ptr;
        T * block_val_store = val_store + inflight_block->val_ptr;

        for(mat_int_t i = 0; i < 4; ++i)
        {
            const mat_int_t i_ix = tidx * 4 + i;

            if(i_ix >= inflight_block->max_nnz)
                break;

            const mat_int_t i_key = thread_vals[i];
            const mat_int_t i_row = i_key / 32;
            const mat_int_t i_col = i_key % 32;

            block_ix_store[i_ix] = i_row * block_width + i_col;
            block_val_store[i_ix] = base_vals[i_key];
        }
    }
    else
    {
        /* finally, save the block to (compressed) storage */
        d_save_block<T>(
            inflight_block,
            block_starts,
            base_vals,
            ix_store,
            val_store,
            tb);
    }
}

/* ************************************************************************** */

/**
 * Solves a triangular system that is transposed in memory, i.e. solving with a
 * row-major upper triangular means storing a lower triangular col-major that
 * is processed from top to bottom.
 *
 * The rhs / solution vector is held by the warp and the each thread handles
 * his component in t_x.
 * Upon completion, the result is still held componentwise.
 */
template<typename T, bool UPPER_TRIANGULAR>
inline __device__
void
d_left_solve_row_major(
    const T * U,
    T& t_x,
    thread_block_tile<32>& warp,
    const mat_int_t block_size = 32)
{
    const mat_int_t lwidx = warp.thread_rank();

    for(mat_int_t i = 0; i < block_size; ++i)
    {
        const mat_int_t cur_i = UPPER_TRIANGULAR ? (block_size - i - 1) : i;

        t_x /= ((lwidx == cur_i) ? U[32 * cur_i + cur_i] : 1);
        warp.sync();

        const T piv = warp.shfl(t_x, cur_i);

        if((UPPER_TRIANGULAR && lwidx < cur_i) ||
            (!UPPER_TRIANGULAR && lwidx > cur_i))
            t_x -= U[32 * lwidx + cur_i] * piv;

        warp.sync();
    }
}

/* ************************************************************************** */

template<typename T, bool UPPER_TRIANGULAR, bool SUBTRACT, bool EXPLICIT_RANGE>
__global__
void
k_solve_block_stripe_level(
    const mat_int_t * block_starts,
    const mat_int_t * level_row_list,
    const mat_int_t * block_stripe_offset,
    const mat_int_t * block_stripe_index,
    const matrix_block * blox,
    const mat_int_t * block_ix_store,
    const T * block_val_store,
    const T * b,
    const T * x_in,
    T * x_out)
{
    __shared__ T local[32 * 32];

    thread_block tb = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(tb);

    const mat_int_t bidx = tb.group_index().x;
    const mat_int_t widx = tb.thread_rank() / 32;
    const mat_int_t tidx = tb.thread_rank();
    const mat_int_t lwidx = warp.thread_rank();
    const mat_int_t num_warps = tb.size() / 32;

    const mat_int_t block_stripe = EXPLICIT_RANGE ? level_row_list[bidx] : bidx;
    const mat_int_t from_major = block_starts[block_stripe];
    const mat_int_t to_major = block_starts[block_stripe + 1];
    const mat_int_t major_size = to_major - from_major;

    /* initialize storage */
    for(mat_int_t i = tidx; i < 32 * 32; i += tb.size())
        local[i] = 0;
    tb.sync();

    /* use one warp per block in stripe */
    for(mat_int_t dep_ix = block_stripe_offset[block_stripe] +
            (UPPER_TRIANGULAR ? 1 : 0) + widx;
        dep_ix < block_stripe_offset[block_stripe + 1] +
            (UPPER_TRIANGULAR ? 0 : -1);
        dep_ix += num_warps)
    {
        const matrix_block * dep_block = blox + block_stripe_index[dep_ix];

        const mat_int_t from_minor = block_starts[
            UPPER_TRIANGULAR ? dep_block->block_col : dep_block->block_row];
        const mat_int_t to_minor = block_starts[
            (UPPER_TRIANGULAR ? dep_block->block_col : dep_block->block_row)
            + 1];
        const mat_int_t minor_size = to_minor - from_minor;

        const mat_int_t * ix_ptr = block_ix_store + dep_block->col_ptr;
        const T * val_ptr = block_val_store + dep_block->val_ptr;

        /* load solution component */
        const T thread_solution = (lwidx < minor_size) ?
            x_in[from_minor + lwidx] : 0;

        if(dep_block->format == BLOCK_SPARSE)
        {
            const mat_int_t block_width = UPPER_TRIANGULAR ? minor_size :
                major_size;

            T my_vals[32];

            for(mat_int_t i = 0; i < 32; ++i)
                my_vals[i] = 0;

            const mat_int_t padded_nnz = DIV_UP(dep_block->nnz, 32) * 32;
            for(mat_int_t i = lwidx; i < padded_nnz; i += 32)
            {
                const mat_int_t i_ix = (i < dep_block->nnz) ? ix_ptr[i] : 0;
                const T i_val = (i < dep_block->nnz) ? val_ptr[i] : 0;

                const mat_int_t i_row = i_ix / block_width;
                const mat_int_t i_col = i_ix % block_width;

                if(UPPER_TRIANGULAR)
                {
                    const T o_thread_solution =
                        warp.shfl(thread_solution, i_col);
                    my_vals[i_row] += i_val * o_thread_solution;
                }
                else
                {
                    const T o_thread_solution =
                        warp.shfl(thread_solution, i_row);
                    my_vals[i_col] += i_val * o_thread_solution;
                }
            }

            /* reduce values and put them into the intermediate storage */
            for(mat_int_t i = 0; i < 32; i += 4)
            {
                T val0 = my_vals[i];
                T val1 = my_vals[i + 1];
                T val2 = my_vals[i + 2];
                T val3 = my_vals[i + 3];

                val0 += warp.shfl_down(val0, 16);
                val1 += warp.shfl_down(val1, 16);
                val2 += warp.shfl_down(val2, 16);
                val3 += warp.shfl_down(val3, 16);

                val0 += warp.shfl_down(val0, 8);
                val1 += warp.shfl_down(val1, 8);
                val2 += warp.shfl_down(val2, 8);
                val3 += warp.shfl_down(val3, 8);

                val0 += warp.shfl_down(val0, 4);
                val1 += warp.shfl_down(val1, 4);
                val2 += warp.shfl_down(val2, 4);
                val3 += warp.shfl_down(val3, 4);

                val0 += warp.shfl_down(val0, 2);
                val1 += warp.shfl_down(val1, 2);
                val2 += warp.shfl_down(val2, 2);
                val3 += warp.shfl_down(val3, 2);

                val0 += warp.shfl_down(val0, 1);
                val1 += warp.shfl_down(val1, 1);
                val2 += warp.shfl_down(val2, 1);
                val3 += warp.shfl_down(val3, 1);

                val0 = warp.shfl(val0, 0);
                val1 = warp.shfl(val1, 0);
                val2 = warp.shfl(val2, 0);
                val3 = warp.shfl(val3, 0);

                my_vals[i] = val0;
                my_vals[i + 1] = val1;
                my_vals[i + 2] = val2;
                my_vals[i + 3] = val3;
            }

            local[widx * 32 + lwidx] += my_vals[lwidx];
        }
        else
        {
            const mat_int_t row_size =
                block_starts[dep_block->block_row + 1] -
                block_starts[dep_block->block_row];
            const mat_int_t col_size =
                block_starts[dep_block->block_col + 1] -
                block_starts[dep_block->block_col];

            if(UPPER_TRIANGULAR)
            {
                /* use blocks directly from memory */
                for(mat_int_t i = 0; i < row_size; ++i)
                {
                    T i_val = (lwidx < col_size) ?
                        val_ptr[i * col_size + lwidx] : 0;
                    i_val = i_val * thread_solution;

                    /* reduce value */
                    i_val += warp.shfl_down(i_val, 16);
                    i_val += warp.shfl_down(i_val, 8);
                    i_val += warp.shfl_down(i_val, 4);
                    i_val += warp.shfl_down(i_val, 2);
                    i_val += warp.shfl_down(i_val, 1);

                    if(lwidx == 0)
                        local[widx * 32 + i] += i_val;

                    warp.sync();
                }
            }
            else
            {
                /* transpose blocks in-flight */
                for(mat_int_t i = 0; i < row_size; ++i)
                {
                    T i_val = (lwidx < col_size) ?
                        val_ptr[i * col_size + lwidx] : 0;

                    const T i_thread_solution =
                        warp.shfl(thread_solution, i);
                    i_val = i_val * i_thread_solution;

                    local[widx * 32 + lwidx] += i_val;

                    warp.sync();
                }
            }
        }
    }
    tb.sync();

    /* use one warp to add up results */
    T t_x;
    if(widx == 0)
    {
        t_x = (lwidx < major_size) ? b[from_major + lwidx] : 0;
        for(mat_int_t i = 0; i < num_warps; ++i)
            t_x += (SUBTRACT ? -1 : 1) * local[i * 32 + lwidx];
        warp.sync();
    }
    tb.sync();

    /* load pivot block into shared memory */
    const mat_int_t piv_block_id = block_stripe_index[
            UPPER_TRIANGULAR ?
                block_stripe_offset[block_stripe] :
                (block_stripe_offset[block_stripe + 1] - 1)
        ];
    const matrix_block * piv_block = blox + piv_block_id;

    d_load_block<T, !UPPER_TRIANGULAR>(
        piv_block,
        block_starts,
        block_ix_store,
        block_val_store,
        local,
        tb);
    tb.sync();

    /* solve with pivot block using a single warp */
    if(widx == 0)
    {
        d_left_solve_row_major<T, UPPER_TRIANGULAR>(
            local,
            t_x,
            warp,
            major_size);
        warp.sync();

        /* save result in global memory */
        if(lwidx < major_size)
            x_out[from_major + lwidx] = t_x;
    };
}

/* ************************************************************************** */

/* note: x must be initialized with b first! */
template<typename T, bool UPPER_TRIANGULAR>
__global__
void
k_solve_blocks_offdiagonal_atomic(
    const mat_int_t * block_starts,
    const mat_int_t * level_offdiagonal_block_list,
    const matrix_block * blox,
    const mat_int_t * block_ix_store,
    const T * block_val_store,
    T * x,
    const mat_int_t num_level_offdiagonal_blocks)
{
    thread_block tb = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(tb);

    const mat_int_t bidx = tb.group_index().x;
    const mat_int_t widx = tb.thread_rank() / 32;
    const mat_int_t lwidx = warp.thread_rank();
    const mat_int_t num_warps = tb.size() / 32;

    const mat_int_t warp_gidx = bidx * num_warps + widx;

    /* process one block per warp */
    if(warp_gidx >= num_level_offdiagonal_blocks)
        return;

    const mat_int_t warp_block_id = level_offdiagonal_block_list[warp_gidx];
    const matrix_block * warp_block = blox + warp_block_id;

    const mat_int_t from_major = block_starts[
        UPPER_TRIANGULAR ? warp_block->block_row : warp_block->block_col];
    const mat_int_t to_major = block_starts[
        (UPPER_TRIANGULAR ? warp_block->block_row : warp_block->block_col)
        + 1];
    const mat_int_t major_size = to_major - from_major;

    const mat_int_t from_minor = block_starts[
        UPPER_TRIANGULAR ? warp_block->block_col : warp_block->block_row];
    const mat_int_t to_minor = block_starts[
        (UPPER_TRIANGULAR ? warp_block->block_col : warp_block->block_row)
        + 1];
    const mat_int_t minor_size = to_minor - from_minor;

    const mat_int_t * ix_ptr = block_ix_store + warp_block->col_ptr;
    const T * val_ptr = block_val_store + warp_block->val_ptr;

    /* load solution component */
    const T thread_solution = (lwidx < minor_size) ?
        x[from_minor + lwidx] : 0;

    if(warp_block->format == BLOCK_SPARSE)
    {
        const mat_int_t block_width = UPPER_TRIANGULAR ? minor_size :
            major_size;

        T my_vals[32];

        for(mat_int_t i = 0; i < 32; ++i)
            my_vals[i] = 0;

        const mat_int_t padded_nnz = DIV_UP(warp_block->nnz, 32) * 32;
        for(mat_int_t i = lwidx; i < padded_nnz; i += 32)
        {
            const mat_int_t i_ix = (i < warp_block->nnz) ? ix_ptr[i] : 0;
            const T i_val = (i < warp_block->nnz) ? val_ptr[i] : 0;

            const mat_int_t i_row = i_ix / block_width;
            const mat_int_t i_col = i_ix % block_width;

            if(UPPER_TRIANGULAR)
            {
                const T o_thread_solution =
                    warp.shfl(thread_solution, i_col);
                my_vals[i_row] += i_val * o_thread_solution;
            }
            else
            {
                const T o_thread_solution =
                    warp.shfl(thread_solution, i_row);
                my_vals[i_col] += i_val * o_thread_solution;
            }
        }

        /* reduce partial contributions */
        for(mat_int_t i = 0; i < 32; i += 4)
        {
            T val0 = my_vals[i];
            T val1 = my_vals[i + 1];
            T val2 = my_vals[i + 2];
            T val3 = my_vals[i + 3];

            val0 += warp.shfl_down(val0, 16);
            val1 += warp.shfl_down(val1, 16);
            val2 += warp.shfl_down(val2, 16);
            val3 += warp.shfl_down(val3, 16);

            val0 += warp.shfl_down(val0, 8);
            val1 += warp.shfl_down(val1, 8);
            val2 += warp.shfl_down(val2, 8);
            val3 += warp.shfl_down(val3, 8);

            val0 += warp.shfl_down(val0, 4);
            val1 += warp.shfl_down(val1, 4);
            val2 += warp.shfl_down(val2, 4);
            val3 += warp.shfl_down(val3, 4);

            val0 += warp.shfl_down(val0, 2);
            val1 += warp.shfl_down(val1, 2);
            val2 += warp.shfl_down(val2, 2);
            val3 += warp.shfl_down(val3, 2);

            val0 += warp.shfl_down(val0, 1);
            val1 += warp.shfl_down(val1, 1);
            val2 += warp.shfl_down(val2, 1);
            val3 += warp.shfl_down(val3, 1);

            val0 = warp.shfl(val0, 0);
            val1 = warp.shfl(val1, 0);
            val2 = warp.shfl(val2, 0);
            val3 = warp.shfl(val3, 0);

            my_vals[i] = val0;
            my_vals[i + 1] = val1;
            my_vals[i + 2] = val2;
            my_vals[i + 3] = val3;

            warp.sync();
        }

        /* upate solution atomically */
        if(lwidx < major_size)
            real_t_atomic_add(x + from_major + lwidx, -my_vals[lwidx]);
    }
    else
    {
        const mat_int_t row_size =
            block_starts[warp_block->block_row + 1] -
            block_starts[warp_block->block_row];
        const mat_int_t col_size =
            block_starts[warp_block->block_col + 1] -
            block_starts[warp_block->block_col];

        if(UPPER_TRIANGULAR)
        {
            /* use blocks directly from memory */
            for(mat_int_t i = 0; i < row_size; ++i)
            {
                T i_val = (lwidx < col_size) ?
                    val_ptr[i * col_size + lwidx] : 0;
                i_val = i_val * thread_solution;

                /* reduce value */
                i_val += warp.shfl_down(i_val, 16);
                i_val += warp.shfl_down(i_val, 8);
                i_val += warp.shfl_down(i_val, 4);
                i_val += warp.shfl_down(i_val, 2);
                i_val += warp.shfl_down(i_val, 1);

                /* update solution atomically */
                if(lwidx == 0)
                    real_t_atomic_add(x + from_major + i, -i_val);

                warp.sync();
            }
        }
        else
        {
            /* transpose blocks in-flight */
            T contrib = 0;
            for(mat_int_t i = 0; i < row_size; ++i)
            {
                T i_val = (lwidx < col_size) ?
                    val_ptr[i * col_size + lwidx] : 0;

                const T i_thread_solution =
                    warp.shfl(thread_solution, i);
                i_val = i_val * i_thread_solution;

                contrib += i_val;

                warp.sync();
            }

            if(lwidx < major_size)
                real_t_atomic_add(x + from_major + lwidx, -contrib);
        }
    }
}

/* ************************************************************************** */

template<typename T, bool UPPER_TRIANGULAR>
__global__
void
k_solve_blocks_diagonal_atomic(
    const mat_int_t * block_starts,
    const mat_int_t * level_diagonal_block_list,
    const matrix_block * blox,
    const mat_int_t * block_ix_store,
    const T * block_val_store,
    T * x)
{
    __shared__ T local[32 * 32];

    thread_block tb = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(tb);

    const mat_int_t bidx = tb.group_index().x;
    const mat_int_t widx = tb.thread_rank() / 32;
    const mat_int_t tidx = tb.thread_rank();
    const mat_int_t lwidx = warp.thread_rank();
    const mat_int_t num_warps = tb.size() / 32;

    /* use one block per block row / block col */
    const mat_int_t piv_block_id = level_diagonal_block_list[bidx];
    const matrix_block * piv_block = blox + piv_block_id;

    const mat_int_t from_major = block_starts[
        UPPER_TRIANGULAR ? piv_block->block_row : piv_block->block_col];
    const mat_int_t to_major = block_starts[
        (UPPER_TRIANGULAR ? piv_block->block_row : piv_block->block_col)
        + 1];
    const mat_int_t major_size = to_major - from_major;

    /* load block using all threads cooperatively */
    d_load_block<T, !UPPER_TRIANGULAR>(
        piv_block,
        block_starts,
        block_ix_store,
        block_val_store,
        local,
        tb);
    tb.sync();

    /* solve with pivot block using a single warp */
    if(widx == 0)
    {
        /* cache solution in thread */
        T t_x = (lwidx < major_size) ? x[from_major + lwidx] : 0;
        warp.sync();

        d_left_solve_row_major<T, UPPER_TRIANGULAR>(
            local,
            t_x,
            warp,
            major_size);
        warp.sync();

        /* save result in global memory */
        if(lwidx < major_size)
            x[from_major + lwidx] = t_x;
    };
}

NS_LA_END
NS_CULIP_END