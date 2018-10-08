/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIB_UTILS_DEFINES_H_
#define __CULIP_LIB_UTILS_DEFINES_H_

#include <stdint.h>
#include <chrono>
#include <map>
#include <string>

#define NS_CULIP __culip
#define NS_LA __la
#define NS_TEST __test
#define NS_LP __lp
#define NS_UTILS __utils
#define NS_ALGORITHMS __algorithms
#define NS_DATA_STRUCTURES __data_structures
#define NS_STAGING __staging

#define NS_CULIP_BEGIN namespace NS_CULIP {
#define NS_CULIP_END }

#define NS_LA_BEGIN namespace NS_LA {
#define NS_LA_END }

#define NS_TEST_BEGIN namespace NS_TEST {
#define NS_TEST_END }

#define NS_LP_BEGIN namespace NS_LP {
#define NS_LP_END }

#define NS_UTILS_BEGIN namespace NS_UTILS {
#define NS_UTILS_END }

#define NS_ALGORITHMS_BEGIN namespace NS_ALGORITHMS {
#define NS_ALGORITHMS_END }

#define NS_DATA_STRUCTURES_BEGIN namespace NS_DATA_STRUCTURES {
#define NS_DATA_STRUCTURES_END }

#define NS_STAGING_BEGIN namespace NS_STAGING {
#define NS_STAGING_END }

#define EPS 1e-12
#define LA_TOL 1e-8

NS_CULIP_BEGIN

/* General type defines */
using index_t = int32_t;
using size_t = std::size_t;

#define invalid_index_t UINT32_MAX

/* Matrix index types */
using mat_int_t = index_t;
using mat_size_t = size_t;

/* timing utils */
using t_point = std::chrono::time_point<std::chrono::system_clock>;
struct _timer
{
    _timer();
    ~_timer();

    void start(const std::string& s);
    void stop(const std::string& s);
    double get_ms(const std::string& s);

    std::map<std::string, t_point> m_t_start;
    std::map<std::string, t_point> m_t_end;
};

extern _timer * __T;

#define START_TIMER(str) (__T->start(str));
#define STOP_TIMER(str) (__T->stop(str));
#define PRINT_TIMER(str) (printf("(Timing) %s: %f ms\n", str, __T->get_ms(str)));
#define GET_TIMER(str) (__T->get_ms(str))

NS_CULIP_END

#endif /* __CULIP_LIB_UTILS_DEFINES_H_ */
