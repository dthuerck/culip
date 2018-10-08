/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/data_structures/b_kvheap.h>
#include <libs/data_structures/b_kvheap.impl.h>

NS_CULIP_BEGIN
NS_DATA_STRUCTURES_BEGIN

template class b_kvheap<mat_int_t, mat_int_t, std::less<mat_int_t>>;
template class b_kvheap<mat_int_t, mat_size_t, std::less<mat_size_t>>;
template class b_kvheap<mat_int_t, float, std::less<float>>;
template class b_kvheap<mat_int_t, double, std::less<double>>;

template class b_kvheap<mat_int_t, mat_int_t, std::greater<mat_int_t>>;
template class b_kvheap<mat_int_t, mat_size_t, std::greater<mat_size_t>>;
template class b_kvheap<mat_int_t, float, std::greater<float>>;
template class b_kvheap<mat_int_t, double, std::greater<double>>;

NS_DATA_STRUCTURES_END
NS_CULIP_END
