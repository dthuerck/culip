/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/data_structures/hgraph.h>
#include <libs/data_structures/hgraph.impl.h>

NS_CULIP_BEGIN
NS_DATA_STRUCTURES_BEGIN

template class HGraph<mat_int_t>;
template class HGraph<float>;
template class HGraph<double>;

NS_DATA_STRUCTURES_END
NS_CULIP_END
