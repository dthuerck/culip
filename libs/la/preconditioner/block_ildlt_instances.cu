/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/la/preconditioner/block_ildlt.cuh>
#include <libs/la/preconditioner/block_ildlt.impl.cuh>

#include <libs/utils/types.impl.cuh>

NS_CULIP_BEGIN

/**
 * Explicitly have a dense vector that supports matrix_blocks
 * for alignment issues - need to overwrite the print function, though
 */

template<>
void
dense_vector_t<NS_LA::matrix_block>::
print(
    const char *s)
const
{
    printf("matrix_block vector is not printable!\n");
}

template class dense_vector_t<NS_LA::matrix_block>;

template
dense_vector_ptr<NS_LA::matrix_block>
make_raw_dense_vector_ptr();

template
dense_vector_ptr<NS_LA::matrix_block>
make_raw_dense_vector_ptr(
    const mat_size_t,
    const bool on_device,
    NS_LA::matrix_block * dense_val);

template
dense_vector_ptr<NS_LA::matrix_block>
make_managed_dense_vector_ptr(
    const mat_size_t m,
    const bool on_device);

template
dense_vector_ptr<NS_LA::matrix_block>
make_managed_dense_vector_ptr(
    const bool on_device);

NS_LA_BEGIN

template class BlockiLDLt<float, false, false>;
template class BlockiLDLt<float, true, false>;
template class BlockiLDLt<float, false, true>;

template class BlockiLDLt<double, false, false>;
template class BlockiLDLt<double, true, false>;
template class BlockiLDLt<double, false, true>;

NS_LA_END
NS_CULIP_END