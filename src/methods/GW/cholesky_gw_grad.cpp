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


#include "methods/ERI/chol_reader_t.hpp"
#include "methods/GW/gw_t.h"

namespace methods {

  namespace solvers {

    nda::array<ComplexType, 2> gw_t::eval_grad(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                               Cholesky_ERI auto &&chol)
    {
      // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20chol-gw-grad
      app_log(1, "\n"
                 "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┬ ┬┌─┐┬   ┌─┐┬ ┬   ┌─┐┬─┐┌─┐┌┬┐\n"
                 "║  ║ ║║═╬╗║ ║║  │  ├─┤│ ││───│ ┬│││───│ ┬├┬┘├─┤ ││\n"
                 "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┴ ┴└─┘┴─┘ └─┘└┴┘   └─┘┴└─┴ ┴─┴┘\n");

      utils::check(chol.MF()->nkpts() == chol.MF()->nkpts_ibz(),
                   "gw_t::cholesky_gw_grad::evaluate: Symmetry is not implemented yet.");
      utils::check(_ft->nt_f() == _ft->nt_b(),
                   "chol-gw:: we assume nt_f == nt_b at least for now \n"
                   "(will lift the restriction at some point...)");
      { // Check if tau_mesh is symmetric w.r.t. beta/2
        auto tau_mesh = _ft->tau_mesh();
        long nts = tau_mesh.shape(0);
        for (size_t it = 0; it < nts; ++it) {
          size_t imt = nts - it - 1;
          double diff = std::abs(tau_mesh(it)) - std::abs(tau_mesh(imt));
          utils::check(diff <= 1e-6, "cholesky-gw: IAFT grid is not compatible with particle-hole symmetry. {}, {}",
                       tau_mesh(it), tau_mesh(imt));
        }
      }

      for( auto& v: {"GRAD_TOTAL", "GRAD_EVALUATE_P0", "GRAD_DYSON_P"} ) {
        _Timer.add(v);
      }

      _Timer.start("GRAD_TOTAL");

      app_log(1, "  - nbnd:              {}", chol.MF()->nbnd());
      app_log(1, "  - nbnd_aux:          {}", chol.Np());
      app_log(1, "  - nspin:             {}", chol.MF()->nspin());
      app_log(1, "  - nkpts:             {}", chol.MF()->nkpts());
      app_log(1, "  - nkpts_ibz:         {}", chol.MF()->nkpts_ibz());
      app_log(1, "  - npol:              {}", chol.MF()->npol());
      app_log(1, "\n");

      nda::array<ComplexType, 2> tmp_grad_2e = eval_grad_2e(G_tskij, chol);

      _Timer.stop("GRAD_TOTAL");

      print_chol_gw_grad_timers();
      chol.print_timers();

      return tmp_grad_2e;
    }

    nda::array<ComplexType, 2> gw_t::eval_grad_2e(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                                  Cholesky_ERI auto &&chol)
    {
      size_t natoms = chol.MF()->number_of_atoms();

      auto tmp_grad_2e = nda::array<ComplexType, 2>::zeros({natoms, 3});

      auto mpi = chol.mpi();

      size_t nt_half = (_ft->nt_f() % 2 == 0) ? _ft->nt_f() / 2 : _ft->nt_f() / 2 + 1;
      size_t nw_half = (_ft->nw_b() % 2 == 0) ? _ft->nw_b() / 2 : _ft->nw_b() / 2 + 1;
      sArray_t<nda::array_view<ComplexType, 3>> sP0_tPQ(*mpi, {nt_half, chol.Np(), chol.Np()});
      sArray_t<nda::array_view<ComplexType, 3>> sP0_wPQ(*mpi, {nw_half, chol.Np(), chol.Np()});

      for (size_t iq = 0; iq < chol.MF()->nkpts(); ++iq) {
        _Timer.start("GRAD_EVALUATE_P0");
        evaluate_P0(iq, G_tskij, sP0_tPQ, chol, G_tskij.shape(3), (iq == 0) ? true : false);
        _Timer.stop("GRAD_EVALUATE_P0");

        _Timer.start("GRAD_DYSON_P");
        dyson_P(sP0_tPQ, sP0_wPQ);
        _Timer.stop("GRAD_DYSON_P");

      }
      return tmp_grad_2e;
    }

    ComplexType gw_t::eval_grad_2e(int iatom, int idirection,
                                   Cholesky_ERI auto && chol)
    {
      return ComplexType(0, 0);
    }

    using Arr2D = nda::array<ComplexType, 2>;
    using Arr5D = nda::array<ComplexType, 5>;
    using Arrv5D = nda::array_view<ComplexType, 5>;

    template Arr2D gw_t::eval_grad(const Arr5D&, chol_reader_t&);
    template Arr2D gw_t::eval_grad(const Arrv5D&, chol_reader_t&);

    template Arr2D gw_t::eval_grad_2e(const Arr5D&, chol_reader_t&);
    template Arr2D gw_t::eval_grad_2e(const Arrv5D&, chol_reader_t&);

    template ComplexType gw_t::eval_grad_2e(int, int, chol_reader_t&);

  } // namespace solvers

} // namespace methods

