/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIBS_LA_PRECONDITIONER_CUH_
#define __CULIP_LIBS_LA_PRECONDITIONER_CUH_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>

NS_CULIP_BEGIN
NS_LA_BEGIN

template<typename T>
class Preconditioner
{
public:
    Preconditioner(gpu_handle_ptr& gpu_handle);
    virtual ~Preconditioner();

    virtual mat_int_t n() const = 0;

    virtual bool is_left() const = 0;
    virtual bool is_middle() const = 0;
    virtual bool is_right() const = 0;

    virtual void solve_left(const dense_vector_t<T> * b, dense_vector_t<T> * x,
        const bool transpose = false) const = 0;
    virtual void solve_middle(const dense_vector_t<T> * b,
        dense_vector_t<T> * x, const bool transpose = false) const = 0;
    virtual void solve_right(const dense_vector_t<T> * b, dense_vector_t<T> * x,
        const bool transpose = false) const = 0;

protected:
    gpu_handle_ptr m_handle;
};

template<typename T>
using preconditioner_ptr = std::unique_ptr<Preconditioner<T>>;

NS_LA_END
NS_CULIP_END

#include <libs/la/preconditioner.impl.cuh>

#endif /* __CULIP_LIBS_LA_PRECONDITIONER_CUH_ */
