/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/staging/block_diagonal.h>
#include <libs/staging/block_diagonal.impl.h>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

template class Block1x1<float>;
template class Block1x1<double>;

template class Block2x2<float>;
template class Block2x2<double>;

template class BlockFactory<float>;
template class BlockFactory<double>;

template class BlockDiagonal<float>;
template class BlockDiagonal<double>;

NS_STAGING_END
NS_CULIP_END