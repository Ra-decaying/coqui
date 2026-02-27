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
#include "methods/HF/hf_gradient_t.h"

namespace methods {

  namespace solvers {

    void hf_gradient_t::evaluate(nda::array<ComplexType, 2> &gradient_1e,
                                 nda::array<ComplexType, 2> &gradient_2e,
                                 nda::array<ComplexType, 2> &gradient_pulay,
                                 const nda::MemoryArrayOfRank<4> auto &D_skij,
                                 const nda::MemoryArrayOfRank<4> auto &F_skij,
                                 const nda::MemoryArrayOfRank<4> auto &S_skij,
                                 const nda::MemoryArrayOfRank<4> auto &H0_skij,
                                 Cholesky_ERI auto &&chol, bool F_has_H0)
    {
      utils::TimerManager Timer;

      // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20chol-hf-grad
      app_log(1, "\n"
                 "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┬ ┬┌─┐┬   ┬ ┬┌─┐  ┌─┐┬─┐┌─┐┌┬┐\n"
                 "║  ║ ║║═╬╗║ ║║  │  ├─┤│ ││───├─┤├┤───│ ┬├┬┘├─┤ ││\n"
                 "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┴ ┴└─┘┴─┘ ┴ ┴└    └─┘┴└─┴ ┴─┴┘\n");

      utils::check(_MF->nkpts() == _MF->nkpts_ibz(),
                   "hf_t::cholesky_hf_grad::evaluate: Symmetry not yet implemented.");

      decltype(nda::range::all) all;

      for( auto& v: {"TOTAL", "1-ELECTRON", "2-ELECTRON", "PULAY", "WRITE"} ) {
        _Timer.add(v);
      }

      _Timer.start("TOTAL");

      _natoms = _MF->number_of_atoms();
      _nbnd = _MF->nbnd();
      _nbnd_aux = _MF->nbnd_aux();
      _nspin = _MF->nspin();
      _nkpts = _MF->nkpts();
      _npol = _MF->npol();

      _k_weight = _MF->k_weight();

      app_log(1, "  - nbnd:              {}", _nbnd);
      app_log(1, "  - nbnd_aux:          {}", _nbnd_aux);
      app_log(1, "  - nspin:             {}", _nspin);
      app_log(1, "  - nkpts:             {}", _nkpts);
      app_log(1, "  - npol:              {}", _npol);
      app_log(1, "\n");

      gradient_1e = nda::array<ComplexType, 2>::zeros({_natoms, 3});
      gradient_2e = nda::array<ComplexType, 2>::zeros({_natoms, 3});
      gradient_pulay = nda::array<ComplexType, 2>::zeros({_natoms, 3});

      auto DE_skij = nda::array<ComplexType, 4>::zeros({_nspin, _nkpts, _nbnd, _nbnd});
      for (int ispin = 0; ispin < _nspin; ++ispin) {
        for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
          auto tmp = nda::array<ComplexType, 2>::zeros({_nbnd, _nbnd});
          auto SC = nda::array<ComplexType, 2>::zeros({_nbnd, _nbnd});
          auto F = nda::array<ComplexType, 2>::zeros({_nbnd, _nbnd});
          auto S = nda::array<ComplexType, 2>::zeros({_nbnd, _nbnd});
          if (F_has_H0) {
            F = F_skij(ispin, ikpt, all, all);
          } else {
            F = F_skij(ispin, ikpt, all, all) + H0_skij(ispin, ikpt, all, all);
          }
          S = S_skij(ispin, ikpt, all, all);
          auto [energies, coeffs] = nda::linalg::eigenelements(F, S);
          nda::blas::gemm(1.0, S_skij(ispin, ikpt, all, all), coeffs, 0.0, SC);
          nda::blas::gemm(1.0, nda::transpose(nda::conj(SC)), D_skij(ispin, ikpt, all, all), 0.0, tmp);
          nda::blas::gemm(1.0, tmp, SC, 0.0, DE_skij(ispin, ikpt, all, all));
          for (int ibnd = 0; ibnd < _nbnd; ++ibnd) {
            DE_skij(ispin, ikpt, ibnd, ibnd) = energies(ibnd) * DE_skij(ispin, ikpt, ibnd, ibnd);
          }
          nda::blas::gemm(1.0, coeffs, DE_skij(ispin, ikpt, all, all), 0.0, tmp);
          nda::blas::gemm(1.0, tmp, nda::transpose(nda::conj(coeffs)), 0.0, DE_skij(ispin, ikpt, all, all));
        }
      }

      for (int iatom = 0; iatom < _natoms; ++iatom) {
        for (int direction = 0; direction < 3; ++direction) {
          _Timer.start("1-ELECTRON");
          gradient_1e(iatom, direction) += evaluate_1e(iatom, direction, D_skij);
          _Timer.stop("1-ELECTRON");
        }
      }

      for (int iatom = 0; iatom < _natoms; ++iatom) {
        for (int direction = 0; direction < 3; ++direction) {
          _Timer.start("2-ELECTRON");
          gradient_2e(iatom, direction) += evaluate_2e(iatom, direction, D_skij, chol);
          _Timer.stop("2-ELECTRON");
        }
      }

      for (int iatom = 0; iatom < _natoms; ++iatom) {
        for (int direction = 0; direction < 3; ++direction) {
          _Timer.start("PULAY");
          gradient_pulay(iatom, direction) += evaluate_pulay(iatom, direction, DE_skij);
          _Timer.stop("PULAY");
        }
      }

      _Timer.stop("TOTAL");

      print_chol_hf_grad_timers();
      chol.print_timers();

    }

    ComplexType hf_gradient_t::evaluate_2e(int iatom, int idirection, const nda::MemoryArrayOfRank<4> auto &D_skij,
                                           Cholesky_ERI auto && chol)
    {
      decltype(nda::range::all) all;

      ComplexType tmp_grad = 0;

      // coulomb
      //   \sum D_{ji} D_{lk} [ d/dX (ij|kl) ]
      // = \sum D_{ji} D_{lk} ( V_{Qij} dV^{*}_{Qlk} + dV_{Qij} V^{*}_{Qlk} )
      // = \sum ( V_{Qij} D^{*}_{ij} ) (dV_{Qkl} D_{kl} )^{*} +
      //   \sum (dV_{Qij} D^{*}_{ij} ) ( V_{Qkl} D_{kl} )^{*}
      // TO-DO: check and confirm complex conjugate and transpose

      for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
        RealType spin_factor = (_nspin == 1 and _npol == 1) ? 2.0 : 1.0;
        auto tmp = nda::array<ComplexType, 2>::zeros({1, 1});
        auto V = chol.V(0, 0, ikpt);
        auto dV = chol.dV(0, iatom, idirection, 0, ikpt);
        auto dm_total = nda::array<ComplexType, 2>::zeros(_nbnd, _nbnd);
        for (int ispin = 0; ispin < _nspin; ++ispin) {
          dm_total += D_skij(ispin, ikpt, all, all) * spin_factor;
        }
        auto dm_total_conj = nda::make_regular(nda::conj(dm_total));
        auto dm_total_conj_ij_1 = nda::reshape(dm_total_conj, std::array<int, 2>({_nbnd * _nbnd, 1}));
        auto dm_total_kl_1 = nda::reshape(dm_total, std::array<int, 2>({_nbnd * _nbnd, 1}));

        {
          auto V_Pij_P_ij = nda::reshape(V, std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
          auto tmp_P_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
          nda::blas::gemm(V_Pij_P_ij, dm_total_conj_ij_1, tmp_P_1);
          auto dV_Qkl_Q_kl = nda::reshape(dV, std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
          auto tmp_Q_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
          nda::blas::gemm(dV_Qkl_Q_kl, dm_total_kl_1, tmp_Q_1);
          nda::blas::gemm(1, nda::conj(nda::transpose(tmp_Q_1)), tmp_P_1, 1, tmp);
        }

        {
          auto dV_Pij_P_ij = nda::reshape(dV, std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
          auto tmp_P_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
          nda::blas::gemm(dV_Pij_P_ij, dm_total_conj_ij_1, tmp_P_1);
          auto V_Qkl_Q_kl = nda::reshape(V, std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
          auto tmp_Q_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
          nda::blas::gemm(V_Qkl_Q_kl, dm_total_kl_1, tmp_Q_1);
          nda::blas::gemm(1, nda::conj(nda::transpose(tmp_Q_1)), tmp_P_1, 1, tmp);
        }

        tmp_grad += tmp(0, 0) * 0.5 * _k_weight(ikpt);
      }

      // exchange
      //   \sum D_{li} D_{jk} [ d/dX (ij|kl) ]
      // = \sum D_{li} D_{jk} ( V_{Qij} dV^{*}_{Qlk} + dV_{Qij} V^{*}_{Qlk} )
      // = \sum ( V_{Qij} D_{jk} ) (dV_{Qlk} D^{*}_{il} )^{*} +
      //   \sum (dV_{Qij} D_{jk} ) ( V_{Qlk} D^{*}_{il} )^{*}
      // TO-DO: check and confirm conjugate and transpose

        for (int ispin = 0; ispin < _nspin; ++ispin) {
          for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
            RealType spin_factor = (_nspin == 1 and _npol == 1) ? 1.0 : 0.5;
            auto V = chol.V(0, 0, ikpt);
            auto dV = chol.dV(0, iatom, idirection, 0, ikpt);
            auto dm_conj = nda::make_regular(nda::conj(D_skij(ispin, ikpt, all, all)));

            {
              auto V_Pij_Pi_j = nda::reshape(V, std::array<int, 2>({_nbnd_aux * _nbnd, _nbnd}));
              auto tmp_Pik_Pi_k = nda::array<ComplexType, 2>::zeros({_nbnd_aux * _nbnd, _nbnd});
              nda::blas::gemm(V_Pij_Pi_j, D_skij(ispin, ikpt, all, all), tmp_Pik_Pi_k);
              auto tmp_Qik_Q_i_k = nda::array<ComplexType, 3>::zeros({_nbnd_aux, _nbnd, _nbnd});
              for (int iQ = 0; iQ < _nbnd_aux; ++iQ) {
                nda::blas::gemm(dm_conj, dV(iQ, all, all), tmp_Qik_Q_i_k(iQ, all, all));
                tmp_Qik_Q_i_k(iQ, all, all) = nda::conj(tmp_Qik_Q_i_k(iQ, all, all));
              }
              auto tmp_Pik_P_i_k = nda::reshape(tmp_Pik_Pi_k, std::array<int, 3>({_nbnd_aux, _nbnd, _nbnd}));
              tmp_grad -= nda::sum(tmp_Pik_P_i_k * tmp_Qik_Q_i_k) * spin_factor * _k_weight(ikpt);
            }

            {
              auto dV_Pij_Pi_j = nda::reshape(dV, std::array<int, 2>({_nbnd_aux * _nbnd, _nbnd}));
              auto tmp_Pik_Pi_k = nda::array<ComplexType, 2>::zeros({_nbnd_aux * _nbnd, _nbnd});
              nda::blas::gemm(dV_Pij_Pi_j, D_skij(ispin, ikpt, all, all), tmp_Pik_Pi_k);
              auto tmp_Qik_Q_i_k = nda::array<ComplexType, 3>::zeros({_nbnd_aux, _nbnd, _nbnd});
              for (int iQ = 0; iQ < _nbnd_aux; ++iQ) {
                nda::blas::gemm(dm_conj, V(iQ, all, all), tmp_Qik_Q_i_k(iQ, all, all));
                tmp_Qik_Q_i_k(iQ, all, all) = nda::conj(tmp_Qik_Q_i_k(iQ, all, all));
              }
              auto tmp_Pik_P_i_k = nda::reshape(tmp_Pik_Pi_k, std::array<int, 3>({_nbnd_aux, _nbnd, _nbnd}));
              tmp_grad -= nda::sum(tmp_Pik_P_i_k * tmp_Qik_Q_i_k) * spin_factor * _k_weight(ikpt);
            }
        }
      }

      return tmp_grad;
    }

    using Arr2D = nda::array<ComplexType, 2>;
    using Arr4D = nda::array<ComplexType, 4>;
    using Arrv4D = nda::array_view<ComplexType, 4>;
    template void hf_gradient_t::evaluate(Arr2D&, Arr2D&, Arr2D&,
                                          const Arr4D&, const Arr4D&, const Arr4D&, const Arr4D&,
                                          chol_reader_t&, bool);
    template void hf_gradient_t::evaluate(Arr2D&, Arr2D&, Arr2D&,
                                          const Arrv4D&, const Arrv4D&, const Arrv4D&, const Arrv4D&,
                                          chol_reader_t&, bool);

  } // namespace solvers

} // namespace methods