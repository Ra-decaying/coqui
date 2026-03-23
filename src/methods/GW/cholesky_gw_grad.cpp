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
#include "methods/GW/gw_gradient_t.h"

namespace methods {

  namespace solvers {

    nda::array<ComplexType, 2> gw_gradient_t::evaluate(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                                       Cholesky_ERI auto && chol)
    {
      // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20chol-gw-grad
      app_log(1, "\n"
                 "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┬ ┬┌─┐┬   ┌─┐┬ ┬   ┌─┐┬─┐┌─┐┌┬┐\n"
                 "║  ║ ║║═╬╗║ ║║  │  ├─┤│ ││───│ ┬│││───│ ┬├┬┘├─┤ ││\n"
                 "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┴ ┴└─┘┴─┘ └─┘└┴┘   └─┘┴└─┴ ┴─┴┘\n");

      utils::check(_mf->nkpts() == _mf->nkpts_ibz(),
                   "hf_t::cholesky_hf_grad::evaluate: Symmetry not yet implemented.");

      for( auto& v: {"TOTAL", "ALLOC", "EVALUATE_P0", "DYSON_P"} ) {
        _Timer.add(v);
      }

      _Timer.start("TOTAL");

      app_log(1, "  - nbnd:              {}", _nbnd);
      app_log(1, "  - nbnd_aux:          {}", _nbnd_aux);
      app_log(1, "  - nspin:             {}", _nspin);
      app_log(1, "  - nkpts:             {}", _nkpts);
      app_log(1, "  - npol:              {}", _npol);
      app_log(1, "\n");

      nda::array<ComplexType, 2> tmp_grad_2e = eval_grad_2e(G_tskij, chol);

      _Timer.stop("TOTAL");

      print_chol_gw_grad_timers();
      chol.print_timers();

      return tmp_grad_2e;

    }

    nda::array<ComplexType, 2> gw_gradient_t::eval_grad_2e(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                                           Cholesky_ERI auto &&chol)
    {
      auto tmp_grad_2e = nda::array<ComplexType, 2>::zeros({_natoms, 3});

      auto mpi = chol.mpi();

      _Timer.start("ALLOC");

      size_t nt_half = (_ft->nt_f() % 2 == 0) ? _ft->nt_f() / 2 : _ft->nt_f() / 2 + 1;
      size_t nw_half = (_ft->nw_b() % 2 == 0) ? _ft->nw_b() / 2 : _ft->nw_b() / 2 + 1;
      sArray_t<nda::array_view<ComplexType, 3>> sP0_tPQ(*mpi, {nt_half, chol.Np(), chol.Np()});
      sArray_t<nda::array_view<ComplexType, 3>> sP0_wPQ(*mpi, {nw_half, chol.Np(), chol.Np()});

      _Timer.stop("ALLOC");

      return tmp_grad_2e;
    }

    ComplexType gw_gradient_t::eval_grad_2e(int iatom, int idirection,
                                            Cholesky_ERI auto && chol)
    {
      return ComplexType(0, 0);
    }

    using Arr2D = nda::array<ComplexType, 2>;
    using Arr5D = nda::array<ComplexType, 5>;
    using Arrv5D = nda::array_view<ComplexType, 5>;

    template Arr2D gw_gradient_t::evaluate(const Arr5D&, chol_reader_t&);
    template Arr2D gw_gradient_t::evaluate(const Arrv5D&, chol_reader_t&);

    template Arr2D gw_gradient_t::eval_grad_2e(const Arr5D&, chol_reader_t&);
    template Arr2D gw_gradient_t::eval_grad_2e(const Arrv5D&, chol_reader_t&);

    template ComplexType gw_gradient_t::eval_grad_2e(int, int, chol_reader_t&);

  } // namespace solvers

} // namespace methods