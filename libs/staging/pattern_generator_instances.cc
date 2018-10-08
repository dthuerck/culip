/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/staging/pattern_generator.h>
#include <libs/staging/pattern_generator.impl.h>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

template class ZeroFillInPattern<float>;
template class ZeroFillInPattern<double>;

template class ExactPattern<float>;
template class ExactPattern<double>;

template class LevelPattern<float>;
template class LevelPattern<double>;

template class BlockRestrictedPattern<float>;
template class BlockRestrictedPattern<double>;

template class BlockRestrictedExactPattern<float>;
template class BlockRestrictedExactPattern<double>;

NS_STAGING_END
NS_CULIP_END