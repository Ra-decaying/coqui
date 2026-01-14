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


#ifndef METHODS_GRADIENT_GRADIENT_DRIVER_H
#define METHODS_GRADIENT_GRADIENT_DRIVER_H

#include <string>

namespace imag_axes_ft
{
class IAFT;
} // namespace imag_axes_ft

namespace methods
{

struct MBState;

template<typename dyson_type, typename eri_grad_t>
void eval_gradient(MBState &mb_state, dyson_type &dyson, eri_grad_t &mb_eri_grad_t, const imag_axes_ft::IAFT& FT,
                   const std::string &solver_type,
                   const std::string &input, const std::string &input_grp, int input_iter,
                   const std::string &output, bool auxbasis_response);

} // namespace methods

#endif
