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


#include "hf_gradient_t.h"

namespace methods {

  namespace solvers {

    namespace mpi3 = boost::mpi3;

    hf_gradient_t::hf_gradient_t(std::shared_ptr<mf::MF> MF):
      _MF(MF),
      _Timer()
    {
    }

    ComplexType hf_gradient_t::evaluate_1e(int iatom, int direction,
                                           const nda::MemoryArrayOfRank<4> auto &D_skij)
    {
      ComplexType tmp_grad = 0;
      auto H0_grad = _MF->H0_grad();
      RealType spin_factor = (_nspin == 1 and _npol == 1) ? 2.0 : 1.0;
      for (int ispin = 0; ispin < _nspin; ++ispin) {
        for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
          tmp_grad += _k_weight(ikpt) * spin_factor *
                      nda::sum(D_skij(ispin, ikpt, nda::ellipsis{}) *
                               H0_grad(iatom, direction, ispin, ikpt, nda::ellipsis{}));

        }
      }
      return tmp_grad;
    }

    ComplexType hf_gradient_t::evaluate_pulay(int iatom, int direction,
                                              const nda::MemoryArrayOfRank<4> auto &DE_skij)
    {
      ComplexType tmp_grad = 0;
      auto S_grad = _MF->S_grad();
      RealType spin_factor = (_nspin == 1 and _npol == 1) ? 2.0 : 1.0;
      for (int ispin = 0; ispin < _nspin; ++ispin) {
        for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
          tmp_grad -= _k_weight(ikpt) * spin_factor *
                      nda::sum(DE_skij(ispin, ikpt, nda::ellipsis{}) *
                               S_grad(iatom, direction, ispin, ikpt, nda::ellipsis{}));

        }
      }
      return tmp_grad;
    }

    void hf_gradient_t::print_chol_hf_grad_timers()
    {
      app_log(2, "\n  CHOL-HF-GRAD timers");
      app_log(2, "  --------------");
      app_log(2, "    Total:                 {0:.3f} sec", _Timer.elapsed("TOTAL"));
      app_log(2, "    1-electron:            {0:.3f} sec", _Timer.elapsed("1-ELECTRON"));
      app_log(2, "    2-electron:            {0:.3f} sec", _Timer.elapsed("2-ELECTRON"));
      app_log(2, "    Pulay:                 {0:.3f} sec\n", _Timer.elapsed("PULAY"));
    }

    using Arr4D = nda::array<ComplexType, 4>;
    using Arrv4D = nda::array_view<ComplexType, 4>;
    template ComplexType hf_gradient_t::evaluate_1e(int, int, const Arr4D&);
    template ComplexType hf_gradient_t::evaluate_1e(int, int, const Arrv4D&);
    template ComplexType hf_gradient_t::evaluate_pulay(int, int, const Arr4D&);
    template ComplexType hf_gradient_t::evaluate_pulay(int, int, const Arrv4D&);

 } // namespace solvers

} // namespace method
