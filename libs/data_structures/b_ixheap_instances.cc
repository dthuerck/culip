/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/data_structures/b_ixheap.h>
#include <libs/data_structures/b_ixheap.impl.h>

NS_CULIP_BEGIN
NS_DATA_STRUCTURES_BEGIN

template class b_ixheap<mat_int_t, std::less<mat_int_t>>;
template class b_ixheap<mat_int_t, std::greater<mat_int_t>>;

NS_DATA_STRUCTURES_END
NS_CULIP_END
