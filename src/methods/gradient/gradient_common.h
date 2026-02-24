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

#ifndef METHODS_GRADIENT_GRADIENT_COMMON_H
#define METHODS_GRADIENT_GRADIENT_COMMON_H

#include <string>

#include "nda/nda.hpp"

#include "mean_field/MF.hpp"

namespace methods {

template<typename data_type>
void print_mbpt_gradients(const nda::array<data_type, 2> &gradient,
                         std::shared_ptr<mf::MF> mf,
                         const std::string& str, bool bohr = true);

template<typename data_type>
void write_mbpt_gradients(const nda::array<data_type, 2> &gradient,
                          const std::string &output, long iter);

} // namespace methods

#endif
