/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/staging/triangular.h>
#include <libs/staging/triangular.impl.h>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

template class Triangular<float>;
template class Triangular<double>;

NS_STAGING_END
NS_CULIP_END