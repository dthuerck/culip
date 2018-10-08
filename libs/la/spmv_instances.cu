/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/la/spmv.cuh>
#include <libs/la/spmv.impl.cuh>

NS_CULIP_BEGIN
NS_LA_BEGIN

template class CSRMatrixSPMV<float>;
template class CSRMatrixSPMV<double>;

template class NormalMatrixSPMV<float>;
template class NormalMatrixSPMV<double>;

template class AugmentedMatrixSPMV<float>;
template class AugmentedMatrixSPMV<double>;

NS_LA_END
NS_CULIP_END
