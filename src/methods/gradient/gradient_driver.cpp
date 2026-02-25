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

#include <format>
#include <iostream>

#include "methods/ERI/chol_reader_t.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/gradient/gradient_common.h"
#include "methods/HF/hf_gradient_t.h"
#include "methods/HF/hf_grand_potential.h"
#include "methods/SCF/simple_dyson.h"


namespace methods
{

template<typename dyson_type, typename eri_t>
void eval_gradients(MBState &mb_state, dyson_type &dyson, eri_t &mb_eri_t, const imag_axes_ft::IAFT &FT,
                   const std::string &solver_type,
                   const std::string &input_grp, int input_iter)
{
  utils::TimerManager Timer;
  auto mpi = mb_eri_t.corr_eri->get().mpi();
  auto mf = mb_eri_t.corr_eri->get().MF();
  // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20gradient
  app_log(1, "\n"
             "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┬─┐┌─┐┌┬┐┬┌─┐┌┐┌┌┬┐\n"
             "║  ║ ║║═╬╗║ ║║  │ ┬├┬┘├─┤ │││├┤ │││ │ \n"
             "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┴└─┴ ┴─┴┘┴└─┘┘└┘ ┴ \n");
  app_log(1, "  Checkpoint HDF5          = {}", mb_state.coqui_prefix + ".mbpt.h5");
  app_log(1, "    - H5 group             = {}", input_grp);
  app_log(1, "    - Iteration            = {}", input_iter);
  app_log(1, "  - Total number processors:         {}", mpi->comm.size());
  app_log(1, "  Number of processors     = {} cores per node, {} nodes\n",
          mpi->comm.size(), mpi->internode_comm.size());

  if (input_iter == -1) {
    h5::file file(mb_state.coqui_prefix + ".mbpt.h5", 'r');
    auto scf_grp = h5::group(file).open_group(input_grp);
    h5::h5_read(scf_grp, "final_iter", input_iter);
  }
  using namespace chkpt;
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

  init_it = read_scf(mpi->node_comm, sF_skij, sSigma_tskij, mu, mb_state.coqui_prefix, input_grp, input_iter);
  sG_tskij = read_greens_function(*mpi, mf.get(), mb_state.coqui_prefix+ ".mbpt.h5", input_iter, input_grp);
  read_dm(mpi->node_comm, mb_state.coqui_prefix, input_iter, sDm_skij);

  auto gradient_1e = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
  auto gradient_2e = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
  auto gradient_pulay = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
  auto gradient_elec = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
  auto gradient_total = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});

  if (solver_type == "hf_gradient") {
    hf_gradient_t hf_gradient(mf);
    hf_gradient.evaluate(gradient_1e, gradient_2e, gradient_pulay,
                         sDm_skij.local(), sF_skij.local(), dyson.sS_skij().local(), dyson.sH0_skij().local(),
                         mb_eri_t.hf_eri->get(), false);
    eval_hf_grand_potential(sDm_skij.local(), dyson.sS_skij().local(), mf, 0.0, FT.beta(), mu);
  }

  gradient_elec = gradient_1e + gradient_2e + gradient_pulay;
  gradient_total =  gradient_elec + mf->nuclear_gradient();
  print_mbpt_gradients(mf->nuclear_gradient(), mf, "GRAD_NUC");
  print_mbpt_gradients(gradient_elec, mf, "GRAD_ELEC");
  print_mbpt_gradients(gradient_total, mf, "GRAD_TOTAL");
  write_mbpt_gradients(gradient_total, mb_state.coqui_prefix, input_iter);

  app_log(1, "####### Gradient routines end #######\n");

}

template void eval_gradients(MBState&, simple_dyson&,
                             mb_eri_t<chol_reader_t, chol_reader_t, chol_reader_t, chol_reader_t>&,
                             const imag_axes_ft::IAFT&, const std::string&, const std::string&, int);

template void eval_gradients(MBState&, simple_dyson&,
                             mb_eri_t<chol_reader_t, thc_reader_t, thc_reader_t, chol_reader_t>&,
                             const imag_axes_ft::IAFT&, const std::string&, const std::string&, int);

} // namespace methods
