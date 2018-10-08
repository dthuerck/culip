/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/staging/permutation.h>

#include <numeric>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

template<typename T>
Permutation<T>::
Permutation(
    const mat_int_t m)
: m_m(m)
{
    m_permutation.resize(m_m);
    std::iota(m_permutation.begin(), m_permutation.end(), 0);
}

/* ************************************************************************** */

template<typename T>
Permutation<T>::
Permutation(
    const mat_int_t m,
    const mat_int_t * permutation)
: m_m(m)
{
    m_permutation.resize(m_m);
    std::copy(permutation, permutation + m, m_permutation.data());
}

/* ************************************************************************** */

template<typename T>
Permutation<T>::
~Permutation()
{

}

/* ************************************************************************** */

template<typename T>
const mat_int_t
Permutation<T>::
m()
const
{
    return m_m;
}

/* ************************************************************************** */

template<typename T>
void
Permutation<T>::
pivot(
    const mat_int_t row_a,
    const mat_int_t row_b)
{
    std::swap(m_permutation[row_a], m_permutation[row_b]);
}

/* ************************************************************************** */

template<typename T>
void
Permutation<T>::
multiply(
    const T * x,
    T * b)
{
    for(mat_int_t i = 0; i < m_m; ++i)
        b[i] = x[m_permutation[i]];
}

/* ************************************************************************** */

template<typename T>
void
Permutation<T>::
multiply_t(
    const T * x,
    T * b)
{
    for(mat_int_t i = 0; i < m_m; ++i)
        b[m_permutation[i]] = x[i];
}

/* ************************************************************************** */

template<typename T>
csr_matrix_ptr<T>
Permutation<T>::
to_csr()
const
{
    csr_matrix_ptr<T> P = make_csr_matrix_ptr<T>(m_m, m_m, m_m, false);
    for(mat_int_t i = 0; i < m_m; ++i)
    {
        P->csr_row[i] = i;
        P->csr_col[i] = m_permutation[i];
        P->csr_val[i] = 1.0;
    }
    P->csr_row[m_m] = m_m;

    return P;
}

/* ************************************************************************** */

template<typename T>
Permutation_ptr<T>
Permutation<T>::
copy()
const
{
    return Permutation_ptr<T>(new Permutation<T>(m_m, m_permutation.data()));
}

/* ************************************************************************** */

template<typename T>
const mat_int_t *
Permutation<T>::
raw_permutation()
const
{
    return m_permutation.data();
}

NS_STAGING_END
NS_CULIP_END