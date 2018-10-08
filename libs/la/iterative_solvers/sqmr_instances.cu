/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/la/iterative_solvers/sqmr.cuh>
#include <libs/la/iterative_solvers/sqmr.impl.cuh>

NS_CULIP_BEGIN
NS_LA_BEGIN

template class SQMR<float>;
template class SQMR<double>;

NS_LA_END
NS_CULIP_END