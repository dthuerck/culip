/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/test/test.h>
#include <libs/test/test.impl.h>

NS_CULIP_BEGIN
NS_TEST_BEGIN

template class Test<float>;
template class Test<double>;
template class Test<mat_int_t>;

template class TestLA<float>;
template class TestLA<double>;

NS_TEST_END
NS_CULIP_END