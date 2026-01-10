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

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "IO/ptree/ptree_utilities.hpp"

#include "mean_field/MF.hpp"

namespace mpi3 = boost::mpi3;
namespace methods
{

template<typename eri_grad_t>
void mbpt_gradient(std::string solver_type, eri_grad_t &eri_grad, const ptree &pt);

template<typename data_type>
void print_mbpt_gradient(const nda::array<data_type, 2> &gradient,
                         std::shared_ptr<mf::MF> mf,
                         std::string str, bool bohr = true);

template<typename data_type>
void write_mbpt_gradient(const nda::array<data_type, 2> &gradient,
                         std::shared_ptr<mf::MF> mf,
                         std::string output, bool bohr = true);

} // namespace methods

#endif
