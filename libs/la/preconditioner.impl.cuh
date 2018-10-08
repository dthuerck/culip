/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/la/preconditioner.cuh>

NS_CULIP_BEGIN
NS_LA_BEGIN

template<typename T>
Preconditioner<T>::
Preconditioner(
    gpu_handle_ptr& gpu_handle)
: m_handle(gpu_handle)
{

}

/* ************************************************************************** */

template<typename T>
Preconditioner<T>::
~Preconditioner()
{

}

NS_LA_END
NS_CULIP_END


