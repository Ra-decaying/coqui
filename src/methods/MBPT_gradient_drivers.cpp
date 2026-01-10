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
#include "methods/HF/hf_gradient_t.h"
#include "methods/HF/hf_grand_potential.h"
#include "methods/SCF/scf_common.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/tools/chkpt_utils.h"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
#include "numerics/shared_array/nda.hpp"


namespace mpi3 = boost::mpi3;
namespace methods
{

template<typename eri_grad_t>
void mbpt_gradient(std::string solver_type, eri_grad_t &eri_grad, const ptree &pt)
{

  using namespace chkpt;
  using namespace solvers;
  using Array_view_4D_t = nda::array_view<ComplexType, 4>;
  using Array_view_5D_t = nda::array_view<ComplexType, 5>;

  auto mf = eri_grad.corr_eri->get().MF();
  auto& mpi = eri_grad.corr_eri->get().mpi();
  if (mpi->comm.size() % mpi->node_comm.size() !=0) {
    APP_ABORT("MBPT: number of processors on each node should be the same.");
  }

  // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20gradient
  app_log(1, "\n"
             "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┬─┐┌─┐┌┬┐┬┌─┐┌┐┌┌┬┐\n"
             "║  ║ ║║═╬╗║ ║║  │ ┬├┬┘├─┤ │││├┤ │││ │ \n"
             "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┴└─┴ ┴─┴┘┴└─┘┘└┘ ┴ \n");

  auto input = io::get_value_with_default<std::string>(pt, "input", "bdft.mbpt");
  auto input_grp = io::get_value_with_default<std::string>(pt, "input_grp", "scf");
  auto input_iter = io::get_value_with_default<long>(pt, "input_iter", -1);
  auto output = io::get_value_with_default<std::string>(pt, "output", "mbpt.gradients");
  bool auxbasis_response = io::get_value_with_default<bool>(pt, "auxbasis_response", true);

  app_log(1, "  - Input:                      {}", input + ".mbpt.h5");
  app_log(1, "    * input_grp, iteration:          {}, {}", input_grp, input_iter);
  app_log(1, "  - Total number processors:         {}", mpi->comm.size());
  app_log(1, "  - Number of nodes:                 {}\n", mpi->internode_comm.size());

  if (input_iter == -1) {
    h5::file file(input + ".mbpt.h5", 'r');
    auto scf_grp = h5::group(file).open_group(input_grp);
    h5::h5_read(scf_grp, "final_iter", input_iter);
  }

  imag_axes_ft::IAFT ft = imag_axes_ft::read_iaft(input + ".mbpt.h5");
  chkpt::sArray_t<Array_view_4D_t> sF_skij(math::shm::make_shared_array<Array_view_4D_t>(
      mpi->comm, mpi->internode_comm, mpi->node_comm, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  chkpt::sArray_t<Array_view_4D_t> sDm_skij(math::shm::make_shared_array<Array_view_4D_t>(
      mpi->comm, mpi->internode_comm, mpi->node_comm, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  chkpt::sArray_t<Array_view_5D_t> G_tskij(math::shm::make_shared_array<Array_view_5D_t>(
      mpi->comm, mpi->internode_comm, mpi->node_comm, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  chkpt::sArray_t<Array_view_5D_t> Sigma_tskij(math::shm::make_shared_array<Array_view_5D_t>(
      mpi->comm, mpi->internode_comm, mpi->node_comm, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));

  double mu = 0.0;
  long init_it = 0;

  init_it = read_scf(mpi->node_comm, sF_skij, Sigma_tskij, mu, input, input_grp, input_iter);
  G_tskij = read_greens_function(*mpi, mf.get(), input + ".mbpt.h5", input_iter, input_grp);
  read_dm(mpi->node_comm, input, input_iter, sDm_skij);

  if (solver_type == "hf_gradient") {
    simple_dyson dyson(mf.get(), &ft);
    hf_gradient_t hf_gradient(mf, auxbasis_response);
    hf_gradient.evaluate(sDm_skij.local(), sF_skij.local(), dyson.sS_skij().local(), dyson.sH0_skij().local(),
                         eri_grad.hf_eri->get(), false);
    print_mbpt_gradient(hf_gradient.electronic_gradient(), mf, "GRAD_ELEC");
    print_mbpt_gradient(hf_gradient.nuclear_gradient(), mf, "GRAD_NUC");
    print_mbpt_gradient(hf_gradient.total_gradient(), mf, "GRAD_TOTAL");
    eval_hf_grand_potential(sDm_skij.local(), dyson.sS_skij().local(), mf, 0.0, ft.beta(), mu);
  } else {
    APP_ABORT("Only Hartree-Fock gradient is supported");
  }

  app_log(1, "####### Gradient routines end #######\n");

}

template<typename data_type>
void print_mbpt_gradient(const nda::array<data_type, 2>& gradient, std::shared_ptr<mf::MF> mf,
                         std::string str, bool bohr)
{
    double factor;
    std::string unit;
    if (bohr) {
      factor = 1;
      unit = "(hartree/bohr)";
    } else {
      factor = 0.52917721054482;
      unit = "(hartree/angstrom)";
    }

    app_log(1, "  {}", str);
    app_log(1, "  -----------------------------------------------------------------------------------");
    app_log(1, "   id    nuc     X {0:<18}    Y {0:<18}    Z {0:<18}", unit);
    app_log(1, "  -----------------------------------------------------------------------------------");
    for (int iatom = 0; iatom < mf->number_of_atoms(); ++iatom) {
      app_log(1, "   {:<5} {:<5}   {:<+20.10f}    {:<+20.10f}    {:<+20.10f}",
        iatom, mf->atomic_id(iatom),
        nda::real(gradient(iatom, 0) * factor),
        nda::real(gradient(iatom, 1) * factor),
        nda::real(gradient(iatom, 2) * factor));
    }
    app_log(1, "\n");
}

using mpi3::communicator;
template void mbpt_gradient(const std::string,
                            mb_eri_t<chol_grad_reader_t, chol_grad_reader_t, chol_grad_reader_t, chol_grad_reader_t>&,
                            ptree const&);

template void print_mbpt_gradient(const nda::array<RealType, 2>&, std::shared_ptr<mf::MF>,
                                  std::string, bool);

template void print_mbpt_gradient(const nda::array<ComplexType,2> &, std::shared_ptr<mf::MF>,
                                  std::string, bool);

} // namespace methods
