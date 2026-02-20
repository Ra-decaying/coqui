/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


#ifndef METHODS_MBPT_GRADIENT_DRIVERS_H
#define METHODS_MBPT_GRADIENT_DRIVERS_H

#include <string>
#include <boost/property_tree/ptree.hpp>

namespace methods
{

template<typename eri_t>
void mbpt_gradient(const std::string &solver_type, eri_t &eri,
                   const boost::property_tree::ptree &pt);

} // namespace methods

#endif
