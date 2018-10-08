/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIBS_LA_IMPORT_LDU_PRECONDITIONER_CUH_
#define __CULIP_LIBS_LA_IMPORT_LDU_PRECONDITIONER_CUH_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>

#include <libs/la/sparse_la.cuh>
#include <libs/la/preconditioner.cuh>

NS_CULIP_BEGIN
NS_LA_BEGIN

/**
 * Most general case:
 * 1. scale input matrix
 * 2. permute input matrix for static pivoting (P_r, P_c)
 * 3. decompose scaled/permuted matrix (S_r, S_c)
 *
 * -> M = LDU = P_r S_r A S_c P_c
 * -> A = [inv(S_r) inv(P_r) L] [D] [U inv(P_c) inv(S_c)]
 *
 * [] [] [] = left, middle, right preconditioner parts
 * For symmetric matrices use P_c = P_r', S_c = S_r.
 *
 * For the permutation, multiplying with P means 'push', i.e. [0]: 4 in P means
 * element at 0 is pushed to 4; P' means 'pull', i.e. [0]: 4 means element on
 * 4 is pulled t0.
 * For the scaling, multiplying with S means multiplying componentwise with
 * S.
 */

template<typename T>
class ImportLDUPreconditioner : public Preconditioner<T>
{
public:

    /* LU case: M = L * U (nonsymmetric) */
    ImportLDUPreconditioner(gpu_handle_ptr& gpu_handle,
        const csr_matrix_t<T> * L, const csr_matrix_t<T> * U,
        const bool exact_solve = true);

    /* Cholesky-case: M = L * L' */
    ImportLDUPreconditioner(gpu_handle_ptr& gpu_handle,
        const csr_matrix_t<T> * L, const bool exact_solve = true);

    /**
     * Cholesky-like case: M = L * D * L', D is 1x1 / 2x2 - blockdiagonal,
     * served as tridiagonal vector or nullptr (symmetric)
     */
    ImportLDUPreconditioner(gpu_handle_ptr& gpu_handle,
        const csr_matrix_t<T> * L, const dense_vector_t<T> * D,
        const dense_vector_t<mat_int_t> * D_is_piv,
        const bool exact_solve = true);

    ~ImportLDUPreconditioner();

    /* set additional scaling (S) and permutation (P) */
    void set_permutation(const dense_vector_t<mat_int_t> * P_r,
        const dense_vector_t<mat_int_t> * P_c);
    void set_scaling(const dense_vector_t<T> * S_r,
        const dense_vector_t<T> * S_c);

    virtual mat_int_t n() const final;

    virtual bool is_left() const final;
    virtual bool is_middle() const final;
    virtual bool is_right() const final;

    virtual void solve_left(const dense_vector_t<T> * b, dense_vector_t<T> * x,
        const bool transpose = false) const final;
    virtual void solve_middle(const dense_vector_t<T> * b,
        dense_vector_t<T> * x, const bool transpose = false) const final;
    virtual void solve_right(const dense_vector_t<T> * b, dense_vector_t<T> * x,
        const bool transpose = false) const final;

protected:
    void analyze_L();
    void analyze_U();

    void solve_PS(const bool left_data, const bool transpose,
        const dense_vector_t<T> * b, dense_vector_t<T> * x) const;

protected:
    const mat_int_t m_block_size = 256;
    const bool m_exact_solve;

    const bool m_symmetric;
    const bool m_has_D;

    bool m_has_P;
    bool m_has_S;

    /* preconditioner parts */
    const csr_matrix_t<T> * m_L;
    const dense_vector_t<T> * m_D;
    const dense_vector_t<mat_int_t> * m_D_is_piv;
    const csr_matrix_t<T> * m_U;

    /* permutation & scaling */
    const dense_vector_t<mat_int_t> * m_P_r;
    const dense_vector_t<mat_int_t> * m_P_c;
    const dense_vector_t<T> * m_S_r;
    const dense_vector_t<T> * m_S_c;

    /* scratch space */
    mutable dense_vector_ptr<T> m_tmp;
    mutable dense_vector_ptr<T> m_tmp_ps;

    cusparseSolveAnalysisInfo_t m_info_L;
    cusparseSolveAnalysisInfo_t m_info_L_t;
    cusparseSolveAnalysisInfo_t m_info_U;
    cusparseSolveAnalysisInfo_t m_info_U_t;

    T_approx_triangular_info_t<T> m_approx_info_L;
    T_approx_triangular_info_t<T> m_approx_info_L_t;
    T_approx_triangular_info_t<T> m_approx_info_U;
    T_approx_triangular_info_t<T> m_approx_info_U_t;
};

NS_LA_END
NS_CULIP_END

#endif /* __CULIP_LIBS_LA_IMPORT_LDU_PRECONDITIONER_CUH_ */