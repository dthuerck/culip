/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIB_TEST_H_
#define __CULIP_LIB_TEST_H_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>

NS_CULIP_BEGIN
NS_TEST_BEGIN

template <typename T>
class Test
{
public:
    /**
     * Reads matrix coordinate triplets (ii, jj, cc) for (row, column, value)
     * from a matrix market file.
     *
     * Returns number of nonzeros.
     */
    static size_t read_mm_triplets(
        mat_int_t& m,
        mat_int_t& n,
        std::vector<mat_int_t>& ii,
        std::vector<mat_int_t>& jj,
        std::vector<T>& cc,
        const char*  filename);

/* ************************************************************************** */

    /**
     * Create CSC matrix from COO (triplets).
     */
    static csr_matrix_ptr<T> matrix_coo_to_csr(
        const mat_int_t& m,
        const mat_int_t& n,
        const mat_int_t& nnz,
        const std::vector<mat_int_t>& ii,
        const std::vector<mat_int_t>& jj,
        const std::vector<T>& cc);

/* ************************************************************************** */

    /**
     * Extracts all COO triplets from a given host CSR matrix.
     */
    static void matrix_csr_to_coo(
        const csr_matrix_t<T> * A,
        std::vector<mat_int_t>& ii,
        std::vector<mat_int_t>& jj,
        std::vector<T>& cc);

/* ************************************************************************** */

    /**
     * Convenience function: reads sparse matrix provided in matrix market
     * format and converts it to CSR.
     */
    static csr_matrix_ptr<T> read_matrix_csr(
        const char* filename,
        const bool transpose = false);

/* ************************************************************************** */

    /**
     * Convenience function: read dense vector provided in matrix market file.
     */
    static dense_vector_ptr<T> read_dense_vector(
        const char * filename,
        const T shift = 0);

/* ************************************************************************** */

    /**
     * Convenience function: read dense vector provided in matrix market file
     * in sparse format.
     */
    static dense_vector_ptr<T> read_dense_vector_from_sparse(
        const char * filename,
        const T shift = 0);

/* ************************************************************************** */

    /**
     * Extracts the lower or upper triangular part for a given host CSR matrix.
     */
    static csr_matrix_ptr<T> extract_triangular_part(
        const csr_matrix_t<T> * A,
        const bool lower_triangular = true);

/* ************************************************************************** */

    /**
     * Compares two sparse matrices (CSR) up to a certain tolerance.
     */
    static bool
    compare_csr_matrix(
        const csr_matrix_t<T> * A1,
        const csr_matrix_t<T> * A2,
        const T tolerance = 1e-6);

/* ************************************************************************** */

    /**
     * Compares two dense vectors up to a certain tolerance.
     */
    static bool
    compare_dense_vector(
        const dense_vector_t<T> * a,
        const dense_vector_t<T> * b,
        const T tolerance = 1e-6);

/* ************************************************************************** */

    static bool
    write_csr_matrix(
        const csr_matrix_t<T> * A,
        const char * path);

/* ************************************************************************** */

    static bool
    write_col_major_matrix(
        const col_major_matrix_t<T> * A,
        const char * path);

/* ************************************************************************** */

    static bool
    write_dense_vector(
        const dense_vector_t<T> * v,
        const char * path);

/* ************************************************************************** */

protected:
    /**
     * Compare from matrix L to matrix R up to a tolerance.
     *
     * For each entry in the left matrix, finds an entry in the
     * right matrix.
     */
    static bool
    compare_sparse_matrix(
        const mat_int_t * L_p,
        const mat_int_t * L_i,
        const T * L_x,
        const mat_int_t * R_p,
        const mat_int_t * R_i,
        const T * R_x,
        const mat_int_t lead,
        const T tolerance = 1e-6,
        const bool R_right = true);
};

/* ************************************************************************** */

template<typename T>
class TestLA
{
public:
    /**
     * Convenience function: matrix-vector multiplication for operands
     * residing on host.
     */
    static dense_vector_ptr<T>
    mat_vec_multiply(
        const csr_matrix_t<T> * A,
        const dense_vector_t<T> * x,
        const gpu_handle_ptr& gpu_handle,
        const bool transpose = false);

    /**
     * Convenience function: compute 2-norm of a host vector
     */
    static T
    norm2(
        const dense_vector_t<T> * x);

    static T
    norm2(
        const mat_int_t v_len,
        const T * v_val);
};

NS_TEST_END
NS_CULIP_END

#endif /* __CULIP_LIB_TEST_H_ */