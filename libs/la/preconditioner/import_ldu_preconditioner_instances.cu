/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/la/preconditioner/import_ldu_preconditioner.cuh>
#include <libs/la/preconditioner/import_ldu_preconditioner.impl.cuh>

NS_CULIP_BEGIN
NS_LA_BEGIN

template class ImportLDUPreconditioner<float>;
template class ImportLDUPreconditioner<double>;

NS_LA_END
NS_CULIP_END
