/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/la/sparse_la.cuh>
#include <libs/la/dense_la.cuh>
#include <libs/la/helper_kernels.cuh>

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cooperative_groups.h>

using namespace cooperative_groups;

NS_CULIP_BEGIN
NS_LA_BEGIN

template<typename T>
__global__
void
k_extract_diag(
    const mat_int_t * A_csr_row,
    const mat_int_t * A_csr_col,
    const T * A_csr_val,
    T * inv_diag,
    const mat_int_t n)
{
    const thread_block tb = this_thread_block();
    const mat_int_t tidx = tb.group_index().x * tb.size() + tb.thread_index().x;

    if(tidx >= n)
        return;

    inv_diag[tidx] = 0;

    const mat_int_t row_start = A_csr_row[tidx];
    const mat_int_t row_end = A_csr_row[tidx + 1] - 1;
    if(A_csr_col[row_start] == tidx)
        inv_diag[tidx] = 1.0 / A_csr_val[row_start];
    if(A_csr_col[row_end] == tidx)
        inv_diag[tidx] = 1.0 / A_csr_val[row_end];
}

template<typename T>
__host__
void
T_approx_triangular_step_analysis(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const csr_matrix_t<T> * gpu_A,
    T_approx_triangular_info_t<T>& info)
{
    /* retrieve diagonal */
    info.inv_diag = make_managed_dense_vector_ptr<T>(gpu_A->m, true);
    info.tmp = make_managed_dense_vector_ptr<T>(gpu_A->m, true);

    const mat_int_t block_size = 256;
    const mat_int_t grid_size = DIV_UP(gpu_A->m, block_size);
    k_extract_diag<T><<<grid_size, block_size, 0, gpu_handle->get_stream()>>>(
        gpu_A->csr_row, gpu_A->csr_col, gpu_A->csr_val,
        info.inv_diag->dense_val, gpu_A->m);
}

template<typename T>
__host__
void
T_approx_triangular_step_solve(
    const gpu_handle_ptr& gpu_handle,
    cusparseOperation_t transA,
    const T_approx_triangular_info_t<T>& info,
    const csr_matrix_t<T> * gpu_A,
    const dense_vector_t<T> * gpu_b,
    dense_vector_t<T> * gpu_x,
    const T * alpha,
    const mat_int_t sweeps)
{
    const mat_int_t n = gpu_A->n;

    T one = 1.0;
    T minus_one = -1.0;
    T zero = 0.0;

    gpu_handle->push_scalar_mode();
    gpu_handle->set_scalar_mode(false);

    /* initialize x = inv(D) * b */
    T_sbmv(gpu_handle, info.inv_diag.get(), gpu_b, gpu_x, &one, &zero, 1, n);

    gpu_handle->pop_scalar_mode();

    for(mat_int_t i = 0; i < sweeps; ++i)
    {
        gpu_handle->push_scalar_mode();
        gpu_handle->set_scalar_mode(false);

        /* tmp = A * x */
        T_csrmv(gpu_handle, transA, gpu_A, gpu_x, info.tmp.get(), &one,
            &zero);

        gpu_handle->pop_scalar_mode();

        /* x += inv(D) * (alpha * b) */
        T_sbmv(gpu_handle, info.inv_diag.get(), gpu_b, gpu_x, alpha,
            &one, 1, n);

        gpu_handle->push_scalar_mode();
        gpu_handle->set_scalar_mode(false);

        /* x += - inv(D) * tmp */
        T_sbmv(gpu_handle, info.inv_diag.get(), info.tmp.get(), gpu_x,
            &minus_one, &one, 1, n);

        gpu_handle->pop_scalar_mode();
    }
}

/* ************************************************************************** */

template<typename T>
__host__
void
T_symmetrize(
    const gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    csr_matrix_ptr<T>& gpu_C)
{
    const mat_int_t m = gpu_A->m;

    gpu_handle->push_scalar_mode();
    gpu_handle->set_scalar_mode(false);

    /* transpose A */
    csr_matrix_ptr<T> gpu_At;
    T_transpose_csr(gpu_handle, gpu_A, gpu_At);

    /* compute the NZ layout for C */
    dense_vector_ptr<mat_int_t> C_csr_row =
        make_managed_dense_vector_ptr<mat_int_t>(m + 1, true);
    mat_int_t C_nnz;
    gpu_handle->cusparse_status =
        cusparseXcsrgeamNnz(
            gpu_handle->cusparse_handle,
            m,
            m,
            gpu_A->get_description(),
            gpu_A->nnz,
            gpu_A->csr_row,
            gpu_A->csr_col,
            gpu_At->get_description(),
            gpu_At->nnz,
            gpu_At->csr_row,
            gpu_At->csr_col,
            gpu_A->get_description(),
            C_csr_row->dense_val,
            &C_nnz);
        CHECK_CUSPARSE(gpu_handle);

    /* compute values for C */
    T_symmetrize_compute(
        gpu_handle,
        gpu_A,
        gpu_At.get(),
        C_csr_row.get(),
        C_nnz,
        gpu_C);

    gpu_handle->pop_scalar_mode();
}

/* ************************************************************************** */

template<typename T>
__host__
void
T_SPNE(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    const SPNE_MODE mode,
    csr_matrix_ptr<T>& gpu_C)
{
    gpu_handle->push_scalar_mode();
    gpu_handle->set_scalar_mode(false);

    /* create vector for C's row indices */
    dense_vector_ptr<mat_int_t> C_csr_row =
        make_managed_dense_vector_ptr<mat_int_t>(gpu_A->m + 1, true);
    cusparseMatDescr_t C_descr = gpu_A->get_description();

    mat_int_t C_nnz;
    gpu_handle->cusparse_status =
        cusparseXcsrgemmNnz(
            gpu_handle->cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE,
            gpu_A->m,
            gpu_A->m,
            gpu_A->n,
            gpu_A->get_description(),
            gpu_A->nnz,
            gpu_A->csr_row,
            gpu_A->csr_col,
            gpu_A->get_description(),
            gpu_A->nnz,
            gpu_A->csr_row,
            gpu_A->csr_col,
            C_descr,
            C_csr_row->dense_val,
            &C_nnz);
    CHECK_CUSPARSE(gpu_handle);

    /* allocate index and value storage for C */
    gpu_C = make_csr_matrix_ptr<T>(gpu_A->m, gpu_A->m, C_nnz, true);

    /* copy row indices */
    CHECK_CUDA(cudaMemcpyAsync(gpu_C->csr_row, C_csr_row->dense_val,
        (gpu_A->m + 1) * sizeof(mat_int_t), cudaMemcpyDeviceToDevice,
        gpu_handle->get_stream()));

    /* perform computation */
    T_SPNE_compute(
        gpu_handle,
        gpu_A,
        gpu_C->csr_row,
        gpu_C->csr_col,
        gpu_C->csr_val);

    if(mode != SPNE_FULL)
    {
        csr_matrix_ptr<T> gpu_tri_C;

        /* extract lower / upper triangular part */
        T_SPNE_extract(
            gpu_handle,
            gpu_A->m,
            C_nnz,
            gpu_C->csr_row,
            gpu_C->csr_col,
            gpu_C->csr_val,
            mode,
            gpu_tri_C);

        /* swap pointers for return */
        gpu_C.swap(gpu_tri_C);
    }

    gpu_handle->pop_scalar_mode();
}

/* ************************************************************************** */

__global__
void
k_tri_row_count(
    const mat_int_t * C_csr_row,
    const mat_int_t * C_csr_col,
    const bool tril,
    const mat_int_t m,
    mat_int_t * tri_C_row_counts)
{
    /* one warp per row */
    thread_block tb = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(tb);

    const mat_int_t w_row = (tb.group_index().x *
        (tb.size() / 32) +
        (tb.thread_index().x / 32));
    const mat_int_t tidx = warp.thread_rank();

    if(w_row >= m)
        return;

    const mat_int_t row_start = C_csr_row[w_row];
    const mat_int_t row_end = C_csr_row[w_row + 1];

    mat_int_t t_row_ctr = 0;
    for(mat_int_t i = tidx; i < (row_end - row_start); i += 32)
    {
        const mat_int_t col = C_csr_col[row_start + i];

        t_row_ctr += ((col == w_row) || ((col < w_row) == tril));
    }

    /* warp reduction gets the row's nz count */
    t_row_ctr += warp.shfl_down(t_row_ctr, 16);
    t_row_ctr += warp.shfl_down(t_row_ctr, 8);
    t_row_ctr += warp.shfl_down(t_row_ctr, 4);
    t_row_ctr += warp.shfl_down(t_row_ctr, 2);
    t_row_ctr += warp.shfl_down(t_row_ctr, 1);

    /* save final result */
    if(warp.thread_rank() == 0)
        tri_C_row_counts[w_row] = t_row_ctr;
}

template<typename T>
__global__
void
k_tri_copy_indices(
    const mat_int_t * C_csr_row,
    const mat_int_t * C_csr_col,
    const T * C_csr_val,
    const bool tril,
    const mat_int_t m,
    const mat_int_t * tri_C_csr_row,
    mat_int_t * tri_C_csr_col,
    T * tri_C_csr_val)
{
    /* one warp per row */
    thread_block tb = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(tb);

    const mat_int_t w_row = (tb.group_index().x *
        (tb.size() / 32) +
        (tb.thread_index().x / 32));
    const mat_int_t tidx = warp.thread_rank();

    if(w_row >= m)
        return;

    const mat_int_t row_start = C_csr_row[w_row];
    const mat_int_t row_end = C_csr_row[w_row + 1];

    const mat_int_t tri_row_start = tri_C_csr_row[w_row];

    for(mat_int_t i = tidx; i < (row_end - row_start); i += 32)
    {
        const mat_int_t col = C_csr_col[row_start + i];
        const T val = C_csr_val[row_start + i];

        if(col == w_row || ((col < w_row) == tril))
        {
            tri_C_csr_col[tri_row_start + i] = col;
            tri_C_csr_val[tri_row_start + i] = val;
        }
    }
}

template<typename T>
__host__
void
T_SPNE_extract(
    gpu_handle_ptr& gpu_handle,
    const mat_int_t m,
    const mat_int_t nnz,
    const mat_int_t * C_csr_row,
    const mat_int_t * C_csr_col,
    const T * C_csr_val,
    const SPNE_MODE mode,
    csr_matrix_ptr<T>& gpu_tri_C)
{
    const mat_int_t block_size = 256;
    cudaStream_t stream = gpu_handle->get_stream();

    const bool tril = (mode == SPNE_TRIL);

    dense_vector_ptr<mat_int_t> tri_C_row_counts =
        make_managed_dense_vector_ptr<mat_int_t>(m + 1, true);
    CHECK_CUDA(cudaMemsetAsync(tri_C_row_counts->dense_val, 0, (m + 1) *
        sizeof(mat_int_t), stream));

    /* compute number of elements per row */
    const mat_int_t warps_per_block = block_size / 32;
    const mat_int_t num_blocks = DIV_UP(m, warps_per_block);
    k_tri_row_count<<<num_blocks, block_size>>>(C_csr_row, C_csr_col, tril, m,
        tri_C_row_counts->dense_val);

    /* scan to compute number of NNZs */
    thrust::device_ptr<mat_int_t> tri_C_row_counts_ptr(tri_C_row_counts->
        dense_val);
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        tri_C_row_counts_ptr, tri_C_row_counts_ptr +
        m + 1, tri_C_row_counts_ptr);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    mat_int_t tri_nnz;
    CHECK_CUDA(cudaMemcpy(&tri_nnz, tri_C_row_counts->dense_val + m,
        sizeof(mat_int_t), cudaMemcpyDeviceToHost));

    /* allocate tril_C and copy row indices */
    gpu_tri_C = make_csr_matrix_ptr<T>(m, m, tri_nnz, true);
    CHECK_CUDA(cudaMemcpyAsync(gpu_tri_C->csr_row, tri_C_row_counts->dense_val,
        (m + 1) * sizeof(mat_int_t), cudaMemcpyDeviceToDevice, stream));

    /* copy column indices and values */
    k_tri_copy_indices<T><<<num_blocks, block_size, 0, stream>>>(C_csr_row,
        C_csr_col, C_csr_val, tril, m, gpu_tri_C->csr_row, gpu_tri_C->csr_col,
        gpu_tri_C->csr_val);
    CHECK_CUDA(cudaDeviceSynchronize());
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_col_div_d(
    const mat_int_t * csr_cols,
    const T * d,
    T * csr_vals,
    const mat_int_t nnz)
{
    const mat_int_t gidx = blockIdx.x * blockDim.x + threadIdx.x;

    if(gidx >= nnz)
        return;

    const mat_int_t my_col = csr_cols[gidx];
    const T my_d = d[my_col];

    csr_vals[gidx] /= my_d * my_d;
}

template<typename T>
__host__
void
T_scale_AD(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    const dense_vector_t<T> * gpu_d,
    csr_matrix_ptr<T>& gpu_AD)
{
    cudaStream_t stream = gpu_handle->get_stream();

    /* copy matrix A */
    gpu_AD = make_csr_matrix_ptr<T>(true);
    *gpu_AD = gpu_A;

    /* multiply entries by d^-1/2 for their respective column */
    const mat_int_t block_size = 256;
    const mat_int_t grid_size_nnz = DIV_UP(gpu_A->nnz, block_size);
    k_col_div_d<T><<<grid_size_nnz, block_size, 0, stream>>>(gpu_AD->csr_col,
        gpu_d->dense_val, gpu_AD->csr_val, gpu_A->nnz);
}

/* ************************************************************************** */

template<typename T>
__host__
void
T_matrix_row_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    dense_vector_t<T> * gpu_norm)
{
    cudaStream_t stream = gpu_handle->get_stream();

    const mat_int_t block_size = 256;
    const mat_int_t grid_size_m = DIV_UP(gpu_A->m, block_size);

    k_row_norm<T><<<grid_size_m, block_size, 0, stream>>>(
        gpu_A->csr_row, gpu_A->csr_col, gpu_A->csr_val, gpu_norm->dense_val,
        gpu_A->m);
}

/* ************************************************************************** */

template<typename T>
__host__
void
T_matrix_col_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    dense_vector_t<T> * gpu_norm)
{
    /* transpose matrix */
    csr_matrix_ptr<T> At;
    T_transpose_csr(gpu_handle, gpu_A, At);

    T_matrix_row_norm(gpu_handle, At.get(), gpu_norm);
}

/* ************************************************************************** */

template<typename T>
__host__
void
T_matrix_augmented_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    const dense_vector_t<T> * gpu_d,
    dense_vector_t<T> * gpu_norm)
{
    cudaStream_t stream = gpu_handle->get_stream();

    const mat_int_t block_size = 256;
    const mat_int_t grid_size_n = DIV_UP(gpu_A->n, block_size);

    /* split norms into two raw vectors (n + m - sized) */
    dense_vector_ptr<T> n_norm = make_raw_dense_vector_ptr<T>(gpu_A->n,
        true, gpu_norm->dense_val);
    dense_vector_ptr<T> m_norm = make_raw_dense_vector_ptr<T>(gpu_A->m,
        true, gpu_norm->dense_val + gpu_A->n);

    /**
     * first: col-norm of [diag(D); A], which is
     * sqrt(norm(d)^2 + norm(A)^2)
     */
    T_matrix_col_norm(gpu_handle, gpu_A, n_norm.get());
    k_cwprod<T><<<grid_size_n, block_size, 0, stream>>>(n_norm->dense_val,
        n_norm->dense_val, n_norm->dense_val, gpu_A->n);

    k_add_cwprod<T><<<grid_size_n, block_size, 0, stream>>>(gpu_d->dense_val,
        gpu_d->dense_val, n_norm->dense_val, gpu_A->n);
    k_sqrt<T><<<grid_size_n, block_size, 0, stream>>>(n_norm->dense_val,
        n_norm->dense_val, gpu_A->n);

    /* second: col-norm of A' is just the row-norm of A */
    T_matrix_row_norm(gpu_handle, gpu_A, m_norm.get());
}

/* ************************************************************************** */

template<typename T>
__host__
void
T_matrix_normal_norm(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    const dense_vector_t<T> * gpu_d,
    dense_vector_t<T> * gpu_norm)
{
    /* copy matrix A and rescale column with D^-1/2 */
    csr_matrix_ptr<T> gpu_AD;
    T_scale_AD(gpu_handle, gpu_A, gpu_d, gpu_AD);

    /* first, compute the normal matrix for AD^-1/2 */
    csr_matrix_ptr<T> gpu_C;
    T_SPNE(gpu_handle, gpu_AD.get(), SPNE_FULL, gpu_C);

    /* second, compute row/col norm for C */
    T_matrix_row_norm(gpu_handle, gpu_C.get(), gpu_norm);
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_transform_coo(
    const T * row_scale,
    const T * col_scale,
    const mat_int_t * row_perm,
    const mat_int_t * col_perm,
    mat_int_t * coo_row,
    mat_int_t * coo_col,
    T * coo_val,
    const mat_int_t nnz,
    const bool scale_before_match)
{
    const mat_int_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

    if(tidx >= nnz)
        return;

    /* transform a COO triplet by permutation and scale */
    const mat_int_t old_row = coo_row[tidx];
    const mat_int_t old_col = coo_col[tidx];
    const mat_int_t new_row = row_perm[old_row];
    const mat_int_t new_col = col_perm[old_col];
    const T old_val = coo_val[tidx];

    const T scale = scale_before_match ?
        (row_scale[old_row] * col_scale[old_col]) :
        (row_scale[new_row] * col_scale[new_col]);

    coo_row[tidx] = new_row;
    coo_col[tidx] = new_col;
    coo_val[tidx] = old_val / scale;
}

__global__
void
k_inv_perm(
    const mat_int_t * perm,
    mat_int_t * inv_perm,
    const mat_int_t m)
{
    const mat_int_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

    if(tidx >= m)
        return;

    inv_perm[perm[tidx]] = tidx;
}

template<typename T>
__host__
void
T_matrix_permute_scale(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    const dense_vector_t<T> * gpu_row_scale,
    const dense_vector_t<T> * gpu_col_scale,
    const dense_vector_t<mat_int_t> * gpu_row_perm,
    const dense_vector_t<mat_int_t> * gpu_col_perm,
    csr_matrix_ptr<T>& gpu_PRACQ,
    const bool scale_before_match,
    const bool perm_is_old_to_new)
{
    coo_matrix_ptr<T> tmp;
    const mat_int_t block_size = 256;

    /* convert A: CSR to COO */
    T_csr2coo(gpu_handle, gpu_A, tmp);

    /* invert mapping if necessary */
    mat_int_t * row_perm = gpu_row_perm->dense_val;
    mat_int_t * col_perm = gpu_col_perm->dense_val;
    dense_vector_ptr<mat_int_t> inv_row_perm, inv_col_perm;
    if(!perm_is_old_to_new)
    {
        inv_row_perm = make_managed_dense_vector_ptr<mat_int_t>(gpu_A->m,
            true);

        const mat_int_t num_blocks_m = DIV_UP(gpu_A->m, block_size);
        k_inv_perm<<<num_blocks_m, block_size>>>(gpu_row_perm->dense_val,
            inv_row_perm->dense_val, gpu_A->m);
        row_perm = inv_row_perm->dense_val;
        col_perm = row_perm;

        if(gpu_row_perm != gpu_col_perm)
        {
            inv_col_perm = make_managed_dense_vector_ptr<mat_int_t>(gpu_A->m,
                true);

            /* nonsymmetric permutation */
            k_inv_perm<<<num_blocks_m, block_size>>>(gpu_col_perm->dense_val,
                inv_col_perm->dense_val, gpu_A->m);
            col_perm = inv_col_perm->dense_val;
        }
    }

    /* change indices by permutation and apply scale */
    CHECK_CUDA(cudaDeviceSynchronize());
    const mat_int_t num_blocks_nnz = DIV_UP(gpu_A->nnz, block_size);
    k_transform_coo<T><<<num_blocks_nnz, block_size>>>(gpu_row_scale->dense_val,
        gpu_col_scale->dense_val, row_perm, col_perm,
        tmp->coo_row, tmp->coo_col, tmp->coo_val,
        gpu_A->nnz, scale_before_match);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* convert A: COO to CSR */
    T_coo2csr(gpu_handle, tmp.get(), gpu_PRACQ);
}

/* ************************************************************************** */

template<typename T>
__global__
void
k_copy_permute_rows(
    const mat_int_t * __restrict__ csr_row,
    const mat_int_t * __restrict__ csr_col,
    const T * __restrict__ csr_val,
    const mat_int_t * __restrict__ old_to_new,
    const mat_int_t * __restrict__ permuted_csr_row,
    mat_int_t * permuted_csr_col,
    T * permuted_csr_val,
    const mat_int_t m)
{
}

template<typename T>
__host__
void
T_matrix_permute_row(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    const dense_vector_t<mat_int_t> * old_to_new,
    csr_matrix_ptr<T>& gpu_pA)
{
    /* compute row sizes and scatter */
    dense_vector_ptr<mat_int_t> new_csr_row =
        make_managed_dense_vector_ptr<mat_int_t>(gpu_A->m + 1, true);

    auto row_size =
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    gpu_A->csr_row_ptr(),
                    gpu_A->csr_row_ptr() + 1)),
            thrust_tuple_minus());
    auto permuted_row_size =
        thrust::make_permutation_iterator(
            row_size,
            old_to_new->dense_val_ptr());

    /* compute new row offsets */
    thrust::exclusive_scan(
        permuted_row_size,
        permuted_row_size + gpu_A->n,
        new_csr_row->dense_val_ptr());
    k_set_scalar<mat_int_t><<<1, 1>>>(new_csr_row->dense_val + gpu_A->n,
        gpu_A->nnz);

    /* create new matrix */
    gpu_pA = make_csr_matrix_ptr<T>(gpu_A->m, gpu_A->n, gpu_A->nnz,
        true);

    /* copy elements */
    const mat_int_t block_size = 256;
    const mat_int_t grid_size = DIV_UP(gpu_A->m, block_size / 32);
    k_copy_permute_rows<T><<<grid_size, block_size>>>(
        gpu_A->csr_row, gpu_A->csr_col, gpu_A->csr_val, old_to_new->dense_val,
        gpu_pA->csr_row, gpu_pA->csr_col, gpu_pA->csr_val, gpu_A->m);
    CHECK_CUDA(cudaDeviceSynchronize());
}

/* ************************************************************************** */

template<typename T>
struct thrust_sqrt
{
    __forceinline__ __device__
    T operator()(const T val)
    {
        return sqrt(val);
    }
};

template<typename T>
struct thrust_score
{
    __forceinline__ __device__
    T operator()(const T val)
    {
        return abs(1 - val);
    }
};

template<typename T>
__global__
void
k_scale_csr(
    const mat_int_t * __restrict__ csr_row,
    const mat_int_t * __restrict__ csr_col,
    T * csr_val,
    const T * __restrict__ row_scale,
    const T * __restrict__ col_scale,
    const mat_int_t m)
{
    /* one warp per row */
    thread_block tb = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(tb);

    const mat_int_t widx = tb.group_index().x * (tb.size() / 32) +
        (tb.thread_index().x / 32);

    if(widx >= m)
        return;

    const mat_int_t row_offset = csr_row[widx];
    const mat_int_t next_row_offset = csr_row[widx + 1];
    const T my_row_scale = row_scale[widx];

    for(mat_int_t j = row_offset + warp.thread_rank(); j < next_row_offset;
        j += 32)
    {
        const mat_int_t col = csr_col[j];
        const T j_col_scale = col_scale[col];

        csr_val[j] /= (my_row_scale * j_col_scale);
    }
}

template<typename T>
__host__
void
T_matrix_ruiz_scale(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_A,
    csr_matrix_ptr<T>& gpu_scaled_A,
    dense_vector_ptr<T>& gpu_row_scale,
    dense_vector_ptr<T>& gpu_col_scale,
    bool symmetric)
{
    dense_vector_ptr<T> row_norm =
        make_managed_dense_vector_ptr<T>(gpu_A->m, true);
    dense_vector_ptr<T> col_norm = symmetric ?
        make_raw_dense_vector_ptr<T>(gpu_A->n, true, row_norm->dense_val) :
        make_managed_dense_vector_ptr<T>(gpu_A->n, true);

    /* create vectors to save cumulated scales */
    gpu_row_scale = make_managed_dense_vector_ptr<T>(gpu_A->m, true);
    gpu_col_scale = make_managed_dense_vector_ptr<T>(gpu_A->n, true);

    thrust::fill(
        gpu_row_scale->dense_val_ptr(),
        gpu_row_scale->dense_val_ptr() + gpu_A->m,
        1);
    thrust::fill(
        gpu_col_scale->dense_val_ptr(),
        gpu_col_scale->dense_val_ptr() + gpu_A->n,
        1);

    /* initialize with input matrix */
    gpu_scaled_A = make_csr_matrix_ptr<T>(true);
    *gpu_scaled_A = gpu_A;

    T conv = 0;
    mat_int_t it = 0;
    while(true && it < 5)
    {
        /* compute sqrt of row and column norm */
        T_matrix_row_norm(gpu_handle, gpu_scaled_A.get(), row_norm.get());
        thrust::transform(
            row_norm->dense_val_ptr(),
            row_norm->dense_val_ptr() + gpu_A->m,
            row_norm->dense_val_ptr(),
            thrust_sqrt<T>());

        if(!symmetric)
        {
            T_matrix_col_norm(gpu_handle, gpu_scaled_A.get(), col_norm.get());
            thrust::transform(
                col_norm->dense_val_ptr(),
                col_norm->dense_val_ptr() + gpu_A->n,
                col_norm->dense_val_ptr(),
                thrust_sqrt<T>());
        }

        /* compute score (max abs(1 - norm)) */
        conv = max(
            thrust::transform_reduce(
                row_norm->dense_val_ptr(),
                row_norm->dense_val_ptr() + gpu_A->m,
                thrust_score<T>(),
                0,
                thrust::maximum<T>()),
            thrust::transform_reduce(
                col_norm->dense_val_ptr(),
                col_norm->dense_val_ptr() + gpu_A->n,
                thrust_score<T>(),
                0,
                thrust::maximum<T>()));

        printf("Scale conv: %g\n", conv);

        if(conv < 1e-6)
            break;

        /* scale matrix elements (one warp per row) */
        const mat_int_t block_size = 256;
        const mat_int_t grid_size_m_warps = DIV_UP(gpu_A->m, (block_size / 32));
        k_scale_csr<T><<<grid_size_m_warps, block_size>>>(
            gpu_scaled_A->csr_row, gpu_scaled_A->csr_col,
            gpu_scaled_A->csr_val, row_norm->dense_val,
            col_norm->dense_val, gpu_A->m);
        CHECK_CUDA(cudaDeviceSynchronize());

        /* cumulate scales */
        thrust::transform(
            gpu_row_scale->dense_val_ptr(),
            gpu_row_scale->dense_val_ptr() + gpu_A->m,
            row_norm->dense_val_ptr(),
            gpu_row_scale->dense_val_ptr(),
            thrust::multiplies<T>());
        thrust::transform(
            gpu_col_scale->dense_val_ptr(),
            gpu_col_scale->dense_val_ptr() + gpu_A->m,
            col_norm->dense_val_ptr(),
            gpu_col_scale->dense_val_ptr(),
            thrust::multiplies<T>());

        ++it;
    }
}

/* ************************************************************************** */

template<typename T>
__host__
void
T_csr2coo(
    gpu_handle_ptr& gpu_handle,
    const csr_matrix_t<T> * gpu_csr_A,
    coo_matrix_ptr<T>& gpu_coo_A)
{
    /* allocate COO matrix (same nnz) */
    gpu_coo_A = make_coo_matrix_ptr<T>(gpu_csr_A->m, gpu_csr_A->n,
        gpu_csr_A->nnz, true);

    /* expand CSR's row indices */
    cusparseXcsr2coo(
        gpu_handle->cusparse_handle,
        gpu_csr_A->csr_row,
        gpu_csr_A->nnz,
        gpu_csr_A->m,
        gpu_coo_A->coo_row,
        CUSPARSE_INDEX_BASE_ZERO);

    /* copy CSR's col & val arrays */
    CHECK_CUDA(cudaMemcpy(gpu_coo_A->coo_col, gpu_csr_A->csr_col,
        gpu_csr_A->nnz * sizeof(mat_int_t), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gpu_coo_A->coo_val, gpu_csr_A->csr_val,
        gpu_csr_A->nnz * sizeof(T), cudaMemcpyDeviceToDevice));
}

/* ************************************************************************** */

template<typename T>
struct
thrust_tuple_cmp
{
    __device__
    bool operator()(
        const thrust::tuple<mat_int_t, mat_int_t, T>& t0,
        const thrust::tuple<mat_int_t, mat_int_t, T>& t1)
    {
        if(thrust::get<0>(t0) < thrust::get<0>(t1))
            return true;
        if(thrust::get<0>(t0) > thrust::get<0>(t1))
            return false;

        return (thrust::get<1>(t0) < thrust::get<1>(t1));
    }
};

template<typename T>
__host__
void
T_coo2csr(
    gpu_handle_ptr& gpu_handle,
    const coo_matrix_t<T> * gpu_coo_A,
    csr_matrix_ptr<T>& gpu_csr_A,
    const bool sort)
{
    /* allocate CSR matrix (same nnz) */
    gpu_csr_A = make_csr_matrix_ptr<T>(gpu_coo_A->m, gpu_coo_A->n,
        gpu_coo_A->nnz, true);

    /* sort COO triplets by (row, col) by a zip iterator */
    coo_matrix_ptr<T> clone = make_coo_matrix_ptr<T>(true);
    if(sort)
    {
        *clone = gpu_coo_A;

        /* sort a copy */
        thrust::sort(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    clone->coo_row_ptr(),
                    clone->coo_col_ptr(),
                    clone->coo_val_ptr())),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    clone->coo_row_ptr() + clone->nnz,
                    clone->coo_col_ptr() + clone->nnz,
                    clone->coo_val_ptr() + clone->nnz)),
            thrust_tuple_cmp<T>());
    }

    /* compress row indices */
    cusparseXcoo2csr(
        gpu_handle->cusparse_handle,
        sort ? clone->coo_row : gpu_coo_A->coo_row,
        sort ? clone->nnz :gpu_coo_A->nnz,
        sort ? clone->m : gpu_coo_A->m,
        gpu_csr_A->csr_row,
        CUSPARSE_INDEX_BASE_ZERO);

    /* copy COO's (sorted) col & val arrays */
    CHECK_CUDA(cudaMemcpy(gpu_csr_A->csr_col,
        sort ? clone->coo_col : gpu_coo_A->coo_col,
        gpu_csr_A->nnz * sizeof(mat_int_t), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gpu_csr_A->csr_val,
        sort ? clone->coo_val : gpu_coo_A->coo_val,
        gpu_csr_A->nnz * sizeof(T), cudaMemcpyDeviceToDevice));
}

NS_LA_END
NS_CULIP_END
