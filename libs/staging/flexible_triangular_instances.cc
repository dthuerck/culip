/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/staging/flexible_triangular.h>
#include <libs/staging/flexible_triangular.impl.h>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

template class FlexibleTriangular<float>;
template class FlexibleTriangular<double>;

NS_STAGING_END
NS_CULIP_END