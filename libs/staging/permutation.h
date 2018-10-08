/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_STAGING_PIVOT_PERMUTATION_H_
#define __CULIP_STAGING_PIVOT_PERMUTATION_H_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

template<typename T>
class Permutation;

template<typename T>
using Permutation_ptr = std::shared_ptr<Permutation<T>>;

template<typename T>
class Permutation
{
public:
    Permutation(const mat_int_t m);
    Permutation(const mat_int_t m, const mat_int_t * permutation);
    ~Permutation();

    const mat_int_t m() const;

    /* editing operations */
    void pivot(const mat_int_t row_a, const mat_int_t row_b);

    void multiply(const T * x, T * b);
    void multiply_t(const T * x, T * b);

    csr_matrix_ptr<T> to_csr() const;
    Permutation_ptr<T> copy() const;

    /* read raw data */
    const mat_int_t * raw_permutation() const;

protected:
    mat_int_t m_m;

    std::vector<mat_int_t> m_permutation;
};

NS_STAGING_END
NS_CULIP_END

#endif /* __CULIP_STAGING_PIVOT_PERMUTATION_H_ */