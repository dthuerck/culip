/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIBS_UTILS_PARAMS_H_
#define __CULIP_LIBS_UTILS_PARAMS_H_

#include <memory>

#include <libs/utils/defines.h>

NS_CULIP_BEGIN

enum culip_graph_edge_mode_t
{
    UNIT,
    VAL
};

template<typename T>
struct params_t
{
    /* general parameters */
    T p_eps = 1e-8; //1e-12;

    /* parameters for IPM solver */
    T ipm_convergence_tolerance = 1e-6;
    mat_int_t ipm_max_iterations = 100;
    T ipm_sigma_relax_threshold = 0.2;
    T ipm_sigma_ub = 0.2;
    T ipm_tao_0 = 0.9995;
    T ipm_phi_0 = 1e-5;
    T ipm_alpha_reduction = 0.95;

    T ipm_infub_check_threshold = 1e-3;
    mat_int_t ipm_infub_check_iterations = 100;

    /* parameters for graph */
    culip_graph_edge_mode_t p_edge_mode;
};

template<typename T>
using params_ptr = std::shared_ptr<params_t<T>>;

NS_CULIP_END

#endif /* __CULIP_LIBS_UTILS_PARAMS_H_ */
