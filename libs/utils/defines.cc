/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/utils/defines.h>

NS_CULIP_BEGIN

_timer::
_timer()
: m_t_start(),
  m_t_end()
{
}

_timer::
~_timer()
{

}

void
_timer::
start(
    const std::string& s)
{
    m_t_start[s] = std::chrono::system_clock::now();
}

void
_timer::
stop(
    const std::string& s)
{
    m_t_end[s] = std::chrono::system_clock::now();
}

double
_timer::
get_ms(
    const std::string& s)
{
     std::chrono::duration<double> elapsed_seconds = m_t_end[s] - m_t_start[s];
     return (elapsed_seconds.count() * 1000);
}

_timer * __T = new _timer();

NS_CULIP_END
