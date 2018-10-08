/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/utils/types.cuh>

#include <libs/test/test.h>

#include <libs/la/helper_kernels.cuh>
#include <libs/staging/pattern_generator.h>

#include <numeric>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>

using namespace NS_CULIP;
using namespace NS_CULIP::NS_LA;
using namespace NS_CULIP::NS_STAGING;
using namespace NS_CULIP::NS_TEST;

template<typename T>
void
compute_blk_size_data(
    const Triangular<T> * mat,
    const mat_int_t * block_sizes,
    T& avg_blk_size,
    T& stddev_blk_size)
{
    avg_blk_size = 0;
    stddev_blk_size = 0;

    T avg_sum = 0;
    T avg_c = 0;

    T sq_sum = 0;
    T sq_c = 0;

    const mat_int_t m = mat->m();
    for(mat_int_t i = 0; i < m; ++i)
    {
        const mat_int_t i_len = mat->row_length(i);
        const mat_int_t * i_col = mat->row_col(i);

        for(mat_int_t j_ix = 0; j_ix < i_len; ++j_ix)
        {
            const mat_int_t j = i_col[j_ix];

            const mat_int_t blk_size = block_sizes[i] * block_sizes[j];

            const T avg_y = blk_size - avg_c;
            const T avg_t = avg_sum + avg_y;
            avg_c = (avg_t - avg_sum) - avg_y;
            avg_sum = avg_t;

            const T sq_y = blk_size * blk_size - sq_c;
            const T sq_t = sq_sum + sq_y;
            sq_c = (sq_t - sq_sum) - sq_y;
            sq_sum = sq_t;
        }
    }

    avg_blk_size = avg_sum / mat->nnz();
    stddev_blk_size = std::sqrt(sq_sum / mat->nnz() -
        avg_blk_size * avg_blk_size);
}

/* ************************************************************************** */

void
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

void
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

template<typename T>
void
coarse_matrix_structure(
    const csr_matrix_t<T> * fine,
    const dense_vector_t<mat_int_t> * permutation,
    const dense_vector_t<mat_int_t> * blocking,
    csr_matrix_ptr<T>& coarse)
{
    const mat_int_t fine_m = fine->m;
    const mat_int_t num_blocks = blocking->m - 1;

    /* create a row -> block map */
    dense_vector_ptr<mat_int_t> rowcol_in_block =
        make_managed_dense_vector_ptr<mat_int_t>(fine->m, false);
    for(mat_int_t i = 0; i < num_blocks; ++i)
    {
        const mat_int_t i_from = (*blocking)[i];
        const mat_int_t i_to = (*blocking)[i + 1];

        for(mat_int_t j = i_from; j < i_to; ++j)
            (*rowcol_in_block)[j] = i;
    }

    /**
     * convert matrix to COO
     */
    dense_vector_ptr<mat_int_t> coo_row =
        make_managed_dense_vector_ptr<mat_int_t>(fine->nnz, false);
    dense_vector_ptr<mat_int_t> coo_col =
        make_managed_dense_vector_ptr<mat_int_t>(fine->nnz, false);

    /* csr row offsets to indices */
    offsets_to_indices(fine->m, fine->nnz, fine->csr_row,
        coo_row->dense_val);

    /* copy csr indices */
    thrust::copy(
        thrust::host,
        fine->csr_col,
        fine->csr_col + fine->nnz,
        coo_col->dense_val);

    /**
     * permute matrix
     */
    dense_vector_ptr<mat_int_t> map_vec =
        make_managed_dense_vector_ptr<mat_int_t>(fine_m, false);
    for(mat_int_t i = 0; i < fine_m; ++i)
        (*map_vec)[(*permutation)[i]] = i;

    thrust::transform(
        thrust::host,
        coo_row->dense_val,
        coo_row->dense_val + fine->nnz,
        coo_row->dense_val,
        thrust_map_func<mat_int_t>(map_vec->dense_val));
    thrust::transform(
        thrust::host,
        coo_col->dense_val,
        coo_col->dense_val + fine->nnz,
        coo_col->dense_val,
        thrust_map_func<mat_int_t>(map_vec->dense_val));

    /**
     * map permuted elements to blocks and sort entries
     */
    thrust::transform(
        thrust::host,
        coo_row->dense_val,
        coo_row->dense_val + fine->nnz,
        coo_row->dense_val,
        thrust_map_func<mat_int_t>(rowcol_in_block->dense_val));
    thrust::transform(
        thrust::host,
        coo_col->dense_val,
        coo_col->dense_val + fine->nnz,
        coo_col->dense_val,
        thrust_map_func<mat_int_t>(rowcol_in_block->dense_val));

    thrust::sort(
        thrust::host,
        thrust::make_zip_iterator(thrust::make_tuple(
            coo_row->dense_val,
            coo_col->dense_val)),
        thrust::make_zip_iterator(thrust::make_tuple(
            coo_row->dense_val + fine->nnz,
            coo_col->dense_val + fine->nnz)),
        thrust_tuple_sort());

    /**
     * compress to CSR
     */
    dense_vector_ptr<mat_int_t> block_coo_row =
        make_managed_dense_vector_ptr<mat_int_t>(fine->nnz, false);
    dense_vector_ptr<mat_int_t> block_coo_col =
        make_managed_dense_vector_ptr<mat_int_t>(fine->nnz, false);
    dense_vector_ptr<mat_int_t> block_nnz =
        make_managed_dense_vector_ptr<mat_int_t>(fine->nnz, false);
    auto start_in_it = thrust::make_zip_iterator(thrust::make_tuple(
        coo_row->dense_val,
        coo_col->dense_val));
    auto start_out_it = thrust::make_zip_iterator(thrust::make_tuple(
        block_coo_row->dense_val,
        block_coo_col->dense_val));
    const auto start_out_end_tpl = thrust::reduce_by_key(
        thrust::host,
        start_in_it,
        start_in_it + fine->nnz,
        thrust::make_constant_iterator<mat_int_t>(1),
        start_out_it,
        block_nnz->dense_val,
        thrust_tuple_equal());
    const mat_int_t coarse_nnz = thrust::get<0>(start_out_end_tpl) -
        start_out_it;
    printf("Coarse nnz: %d\n", coarse_nnz);

    coarse = make_csr_matrix_ptr<T>(num_blocks, num_blocks, coarse_nnz, false);
    indices_to_offsets(num_blocks, coarse_nnz, block_coo_row->dense_val,
        coarse->csr_row);
    thrust::copy(
        thrust::host,
        block_coo_col->dense_val,
        block_coo_col->dense_val + coarse_nnz,
        coarse->csr_col);
    thrust::fill(
        thrust::host,
        coarse->csr_val,
        coarse->csr_val + coarse->nnz,
        (T) 1.0);
}

/* ************************************************************************** */

int
main(
    int argc,
    char * argv[])
{
    using real_t = double;

    if(argc != 5)
    {
        printf("Usage: culip-blocking-stats [path to matrix] " \
            "[path to permutation] [path to blocking] [max level]\n" \
            "\n" \
            "where\n" \
            "\n" \
            "[path to matrix] - path to matrix mtx file\n" \
            "[path to permutation] - path to a permutation/reordering, saved " \
            "as 0-based mtx vector\n" \
            "[path to blocking] - path to blocking, saved a list of first " \
            "rows in a block and number of rows as last element stored as " \
            "0-based mtx\n" \
            "[max level] - the maximum fill-in level l for which statistics " \
            "should be generated (0 <= l <= m)\n");
        std::exit(EXIT_FAILURE);
    }

    gpu_handle_ptr cu_handle(new gpu_handle_t);

    const char * matrix_path = argv[1];
    const char * permutation_path = argv[2];
    const char * blocking_path = argv[3];
    const mat_int_t max_level = std::atoi(argv[4]);

    printf("Input matrix: %s\n", matrix_path);
    printf("Max. level: %d\n", max_level);

    /* load input matrix */
    csr_matrix_ptr<real_t> in_fine =
        Test<real_t>::read_matrix_csr(matrix_path, false);

    /* load permutation */
    dense_vector_ptr<mat_int_t> in_permutation = Test<mat_int_t>::
        read_dense_vector(permutation_path);

    /* load block starts */
    dense_vector_ptr<mat_int_t> in_block_starts = Test<mat_int_t>::
        read_dense_vector(blocking_path);

    /* compute block sizes */
    dense_vector_ptr<mat_int_t> block_sizes =
        make_managed_dense_vector_ptr<mat_int_t>(in_block_starts->m - 1, false);
    for(mat_int_t i = 0; i < block_sizes->m; ++i)
        (*block_sizes)[i] = (*in_block_starts)[i + 1] -
            (*in_block_starts)[i];

    /* compute coarse matrix */
    csr_matrix_ptr<real_t> in_coarse;
    coarse_matrix_structure<real_t>(
        in_fine.get(),
        in_permutation.get(),
        in_block_starts.get(),
        in_coarse);

    /* generate piv vector (all 1x1 block) */
    dense_vector_ptr<mat_int_t> piv_starts =
        make_managed_dense_vector_ptr<mat_int_t>(in_coarse->m, false);
    std::iota(piv_starts->dense_val, piv_starts->dense_val + in_coarse->m,
        0);

    #pragma omp parallel for
    for(mat_int_t lvl = 0; lvl < max_level; ++lvl)
    {
        /* create level pattern of the coarse matrix */
        LevelPattern<real_t> lvl_pat_gen(lvl);
        Triangular_ptr<real_t> lvl_pattern = lvl_pat_gen.compute_pattern(
            in_coarse.get(), in_coarse->m, piv_starts->dense_val);

        /* pattern characteristics */
        const mat_int_t lvl_nnz = lvl_pattern->nnz();
        const mat_int_t lvl_levelsets = lvl_pattern->level_schedule();

        real_t avg_blk_size, stddev_blk_size;
        compute_blk_size_data<real_t>(lvl_pattern.get(),
            block_sizes->dense_val, avg_blk_size, stddev_blk_size);

        #pragma omp critical
        {
            printf("Level %d:\n" \
                "\tNumber of blocks: %d\n" \
                "\tNumber of level sets: %d\n" \
                "\tAverage block size: %g\n" \
                "\tStddev of block size: %g\n\n",
                lvl,
                lvl_nnz,
                lvl_levelsets,
                avg_blk_size,
                stddev_blk_size);
        }
    }

    return EXIT_SUCCESS;
}
