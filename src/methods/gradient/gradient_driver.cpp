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


#include "gradient_driver.h"

#include "methods/HF/hf_gradient_t.h"
#include "methods/mb_state/mb_state.hpp"
#include "methods/SCF/simple_dyson.h"
#include "numerics/imag_axes_ft/iaft_utils.hpp"

namespace methods
{

template<typename dyson_type, typename eri_grad_t, typename corr_solver_t>
void eval_gradient(MBState &mb_state, dyson_type &dyson, eri_grad_t &mb_eri_grad_t, const imag_axes_ft::IAFT& FT,
                   const std::string &solver_type,
                   const std::string &input, const std::string &input_grp, int input_iter,
                   const std::string &output, bool auxbasis_response)
{
  utils::TimerManager Timer;
  auto mpi = mb_eri_grad_t.corr_eri->get().mpi();
  auto mf = mb_eri_grad_t.corr_eri->get().MF();
  // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20gradient
  app_log(1, "\n"
             "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┬─┐┌─┐┌┬┐┬┌─┐┌┐┌┌┬┐\n"
             "║  ║ ║║═╬╗║ ║║  │ ┬├┬┘├─┤ │││├┤ │││ │ \n"
             "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┴└─┴ ┴─┴┘┴└─┘┘└┘ ┴ \n");
  app_log(1, "  - Input:                      {}", input + ".mbpt.h5");
  app_log(1, "    * input_grp, iteration:          {}, {}", input_grp, input_iter);
  app_log(1, "  - Total number processors:         {}", mpi->comm.size());
  app_log(1, "  - Number of nodes:                 {}\n", mpi->internode_comm.size());

  if (input_iter == -1) {
    h5::file file(input + ".mbpt.h5", 'r');
    auto scf_grp = h5::group(file).open_group(input_grp);
    h5::h5_read(scf_grp, "final_iter", input_iter);
  }
  using namespace solvers;
  using Array_view_4D_t = nda::array_view<ComplexType, 4>;
  using Array_view_5D_t = nda::array_view<ComplexType, 5>;

  // Initialize MBState
  mb_state.sF_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sDm_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sG_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {FT.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sSigma_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {FT.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  auto& sF_skij = mb_state.sF_skij.value();
  auto& sDm_skij = mb_state.sDm_skij.value();
  auto& sG_tskij = mb_state.sG_tskij.value();
  auto& sSigma_tskij = mb_state.sSigma_tskij.value();

  double mu = 0.0;
  long init_it = 0;
  init_it = read_scf(mpi->node_comm, sF_skij, sSigma_tskij, mu, input, input_grp, input_iter);
  sG_tskij = read_greens_function(*mpi, mf.get(), input + ".mbpt.h5", input_iter, input_grp);
  read_dm(mpi->node_comm, input, input_iter, sDm_skij);

  auto gradient_elec = nda::array<ComplexType, 2>::zeros({mf->natoms(), 3});
  auto gradient_nuc = nda::array<ComplexType, 2>::zeros({mf->natoms(), 3});
  auto gradient_total = nda::array<ComplexType, 2>::zeros({mf->natoms(), 3});

  if (solver_type == "hf_gradient") {
    hf_gradient_t hf_gradient(mf, auxbasis_response);
    hf_gradient.evaluate(sDm_skij.local(), sF_skij.local(), dyson.sS_skij().local(), dyson.sH0_skij().local(),
                         mb_eri_grad_t.hf_eri->get(), false);
    eval_hf_grand_potential(sDm_skij.local(), dyson.sS_skij().local(), mf, 0.0, FT.beta(), mu);
  }

  print_mbpt_gradient(gradient_elec, mf, "GRAD_ELEC");
  print_mbpt_gradient(gradient_nuc, mf, "GRAD_NUC");
  print_mbpt_gradient(gradient_total, mf, "GRAD_TOTAL");

  app_log(1, "####### Gradient routines end #######\n");

}

} // namespace methods
