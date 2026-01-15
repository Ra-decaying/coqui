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


#include "MBPT_gradient_drivers.h"

#include "methods/ERI/chol_grad_reader_t.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/gradient/gradient_driver.h"
#include "methods/SCF/simple_dyson.h"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
#include "IO/ptree/ptree_utilities.hpp"

namespace methods
{

template<typename eri_grad_t>
void mbpt_gradient(const std::string &solver_type, eri_grad_t &eri_grad, const ptree &pt)
{

  auto mf = eri_grad.corr_eri->get().MF();
  auto& mpi = eri_grad.corr_eri->get().mpi();
  if (mpi->comm.size() % mpi->node_comm.size() !=0) {
    APP_ABORT("MBPT: number of processors on each node should be the same.");
  }

  auto input = io::get_value_with_default<std::string>(pt, "input", "bdft.mbpt");
  auto input_grp = io::get_value_with_default<std::string>(pt, "input_grp", "scf");
  auto input_iter = io::get_value_with_default<long>(pt, "input_iter", -1);
  auto output = io::get_value_with_default<std::string>(pt, "output", "mbpt.gradients");
  bool auxbasis_response = io::get_value_with_default<bool>(pt, "auxbasis_response", true);

  imag_axes_ft::IAFT ft = imag_axes_ft::read_iaft(input + ".mbpt.h5");

  if (solver_type == "hf_gradient") {
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi, ft, input);
    eval_gradient(mb_state, dyson, eri_grad, ft, solver_type, input, input_grp, input_iter,
                  output, auxbasis_response);
  } else {
    APP_ABORT("Only Hartree-Fock gradient is supported");
  }

}

template void mbpt_gradient(const std::string&,
                            mb_eri_t<chol_grad_reader_t, chol_grad_reader_t, chol_grad_reader_t, chol_grad_reader_t>&,
                            ptree const&);

template void mbpt_gradient(const std::string&,
                            mb_eri_t<chol_grad_reader_t, thc_reader_t, thc_reader_t, chol_grad_reader_t>&,
                            ptree const&);

} // namespace methods
