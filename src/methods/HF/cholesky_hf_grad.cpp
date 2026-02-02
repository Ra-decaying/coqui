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


#include "methods/ERI/chol_grad_reader_t.hpp"
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
      app_log(1, "  - npol:              {}", _nkpts);
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

    ComplexType hf_gradient_t::evaluate_2e(int iatom, int direction, const nda::MemoryArrayOfRank<4> auto &D_skij,
                                           Cholesky_ERI auto && chol)
    {
      decltype(nda::range::all) all;

      auto bnd_slice = _MF->bnd_slice();
      auto bnd_slice_aux = _MF->bnd_slice_aux();

      chol.read_Vq_grad(0, 0);
      auto Vq0_k3Qij_di = chol.V_k3Qij_di(0, 0);
      auto Vq0_k3Qij_dQ = chol.V_k3Qij_dQ(0, 0);
      auto Vq0_kQij = chol.V_kQij(0, 0);
      auto Vq0_k3PQ_dP = chol.V_k3PQ_dP(0, 0);
      auto Vq0_kPQ = chol.V_kPQ(0, 0);
      auto Vq0_kPQ_inv = chol.V_kPQ_inv(0, 0);

      ComplexType tmp_grad = 0;

      int bnd_begin = bnd_slice(iatom, 0);
      int bnd_end = bnd_slice(iatom, 1);
      int bnd_begin_aux = bnd_slice_aux(iatom, 0);
      int bnd_end_aux = bnd_slice_aux(iatom, 1);

      // d/dX ( Q | i, j )
      auto Vq0_kQij_dQij = nda::array<ComplexType, 4>::zeros(_nkpts, _nbnd_aux, _nbnd, _nbnd);
      auto Vq0_kPQ_dPQ = nda::array<ComplexType, 3>::zeros( _nkpts, _nbnd_aux, _nbnd_aux);

      // related i and j
      // TO-DO: complex conj for index i or j while transpose?
      for (int iQ = 0; iQ < _nbnd_aux; ++iQ) {
        for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
          Vq0_kQij_dQij(ikpt, iQ, nda::range(bnd_begin, bnd_end), all)
            -= Vq0_k3Qij_di(ikpt, direction, iQ, nda::range(bnd_begin, bnd_end), all);
          Vq0_kQij_dQij(ikpt, iQ,  all, nda::range(bnd_begin, bnd_end))
            -= nda::transpose(Vq0_k3Qij_di(ikpt, direction, iQ, nda::range(bnd_begin, bnd_end), all));
        }
      }

      // related Q
      if (_auxbasis_response) {
        Vq0_kQij_dQij(all, nda::range(bnd_begin_aux, bnd_end_aux), all, all)
          -= Vq0_k3Qij_dQ(all, direction, nda::range(bnd_begin_aux, bnd_end_aux), all, all);
      }

      // d/dX ( P | Q )
      // TO-DO: complex conj for index Q while transpose?
      if (_auxbasis_response) {
        for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
          Vq0_kPQ_dPQ(ikpt, nda::range(bnd_begin_aux, bnd_end_aux), all)
            -= Vq0_k3PQ_dP(ikpt, direction, nda::range(bnd_begin_aux, bnd_end_aux), all);
          Vq0_kPQ_dPQ(ikpt, all, nda::range(bnd_begin_aux, bnd_end_aux))
            -= nda::transpose(Vq0_k3PQ_dP(ikpt, direction, nda::range(bnd_begin_aux, bnd_end_aux), all));
        }
      }

      // coulomb
      for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
        auto dm_total = nda::array<ComplexType, 2>::zeros(_nbnd, _nbnd);
        for (int ispin = 0; ispin < _nspin; ++ispin) {
          RealType spin_factor = (_nspin == 1 and _npol == 1) ? 2.0 : 1.0;
          dm_total += D_skij(ispin, ikpt, all, all) * spin_factor;
        }
        auto V_Pij_P_ij = nda::reshape(Vq0_kQij(ikpt, all, all, all), std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
        auto V_dQkl_Q_kl = nda::reshape(Vq0_kQij_dQij(ikpt, all, all, all), std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
        auto V_PQinv_P_Q = nda::reshape(Vq0_kPQ_inv(ikpt, all, all), std::array<int, 2>({_nbnd_aux, _nbnd_aux}));
        auto tmp_1_ij = nda::reshape(dm_total(all, all), std::array<int, 2>({1, _nbnd * _nbnd}));
        auto tmp_kl_1 = nda::reshape(dm_total(all, all), std::array<int, 2>({_nbnd * _nbnd, 1}));
        auto tmp_1_P = nda::array<ComplexType, 2>::zeros({1, _nbnd_aux});
        nda::blas::gemm(1, tmp_1_ij, nda::transpose(V_Pij_P_ij), 0, tmp_1_P);
        auto tmp_Q_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
        nda::blas::gemm(1, V_dQkl_Q_kl, tmp_kl_1, 0, tmp_Q_1);
        auto tmp_1_Q = nda::array<ComplexType, 2>::zeros({1, _nbnd_aux});
        nda::blas::gemm(1, tmp_1_P, V_PQinv_P_Q, 0, tmp_1_Q);
        auto tmp = nda::array<ComplexType, 2>::zeros({1, 1});
        nda::blas::gemm(1, tmp_1_Q, tmp_Q_1, 0, tmp);
        tmp_grad += tmp(0, 0) * _k_weight(ikpt);
        // (ij|P) (P|R)^{-1} d/dX(R|S) (S|Q)^{-1} (Q|kl)
        if (_auxbasis_response) {
          auto V_Qkl_Q_kl = nda::reshape(Vq0_kQij(ikpt, all, all, all), std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
          auto V_dRS_R_S = nda::reshape(Vq0_kPQ_dPQ(ikpt, all, all), std::array<int, 2>({_nbnd_aux, _nbnd_aux}));
          auto V_SQinv_S_Q = nda::reshape(Vq0_kPQ_inv(ikpt, all, all), std::array<int, 2>({_nbnd_aux, _nbnd_aux}));
          auto tmp_1_ij = nda::reshape(dm_total(all, all), std::array<int, 2>({1, _nbnd * _nbnd}));
          auto tmp_kl_1 = nda::reshape(dm_total(all, all), std::array<int, 2>({_nbnd * _nbnd, 1}));
          auto tmp_1_P = nda::array<ComplexType, 2>::zeros({1, _nbnd_aux});
          auto tmp_1_R = nda::array<ComplexType, 2>::zeros({1, _nbnd_aux});
          auto tmp_Q_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
          auto tmp_S_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
          nda::blas::gemm(1, tmp_1_ij, nda::transpose(V_Qkl_Q_kl), 0, tmp_1_P);
          nda::blas::gemm(1, tmp_1_P, V_SQinv_S_Q, 0, tmp_1_R);
          nda::blas::gemm(1, V_Qkl_Q_kl, tmp_kl_1, 0, tmp_Q_1);
          nda::blas::gemm(1, V_SQinv_S_Q, tmp_Q_1, 0, tmp_S_1);
          auto tmp_R_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
          nda::blas::gemm(1, V_dRS_R_S, tmp_S_1, 0, tmp_R_1);
          auto tmp = nda::array<ComplexType, 2>::zeros({1, 1});
          nda::blas::gemm(1, tmp_1_R, tmp_R_1, 0, tmp);
          tmp_grad -= tmp(0, 0) * 0.5 * _k_weight(ikpt);
        }
      }

      // exchange
      for (int ispin = 0; ispin < _nspin; ispin++) {
        for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
          auto V_Pij_Pi_j = nda::reshape(Vq0_kQij(ikpt, all, all, all), std::array<int, 2>({_nbnd_aux * _nbnd, _nbnd}));
          auto V_dQkl_Qk_l = nda::reshape(Vq0_kQij_dQij(ikpt, all, all, all), std::array<int, 2>({_nbnd_aux * _nbnd, _nbnd}));
          auto V_PQinv_P_Q = nda::reshape(Vq0_kPQ_inv(ikpt, all, all), std::array<int, 2>({_nbnd_aux, _nbnd_aux}));
          auto tmp_Pi_k = nda::array<ComplexType, 2>::zeros({_nbnd_aux * _nbnd, _nbnd});
          nda::blas::gemm(1, V_Pij_Pi_j, D_skij(ispin, ikpt, all, all), 0, tmp_Pi_k);
          auto tmp_Qk_i = nda::array<ComplexType, 2>::zeros({_nbnd_aux * _nbnd, _nbnd});
          nda::blas::gemm(1, V_dQkl_Qk_l, D_skij(ispin, ikpt, all, all), 0, tmp_Qk_i);
          auto tmp_Q_ki = nda::reshape(tmp_Qk_i, std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
          auto tmp_P_ki = nda::array<ComplexType, 2>::zeros({_nbnd_aux, _nbnd * _nbnd});
          nda::blas::gemm(1, V_PQinv_P_Q, tmp_Q_ki, 0, tmp_P_ki);
          auto tmp_P_i_k = nda::reshape(tmp_Pi_k, std::array<int, 3>({_nbnd_aux, _nbnd, _nbnd}));
          auto tmp_P_k_i = nda::reshape(tmp_P_ki, std::array<int, 3>({_nbnd_aux, _nbnd, _nbnd}));
          for (int iP = 0; iP < _nbnd_aux; ++iP) {
            auto tmp_i_k = nda::make_regular(nda::transpose(tmp_P_k_i(iP, all, all)));
            RealType spin_factor = (_nspin == 1 and _npol == 1) ? 2.0 : 1.0;
            tmp_grad -= nda::sum(tmp_P_i_k(iP, nda::ellipsis{}) * tmp_i_k) * spin_factor * _k_weight(ikpt);;
          }
          // (il|P) (P|Q)^{-1} d/dX(R|S) (S|Q)^{-1} (Q|kj)
          if (_auxbasis_response) {
            auto V_Qkl_Qk_l = nda::reshape(Vq0_kQij(ikpt, all, all, all), std::array<int, 2>({_nbnd_aux * _nbnd, _nbnd}));
            auto V_dRS_R_S = nda::reshape(Vq0_kPQ_dPQ(ikpt, all, all), std::array<int, 2>({_nbnd_aux, _nbnd_aux}));
            auto V_SQinv_S_Q = nda::reshape(Vq0_kPQ_inv(ikpt, all, all), std::array<int, 2>({_nbnd_aux, _nbnd_aux}));
            auto tmp_Qk_i = nda::array<ComplexType, 2>::zeros({_nbnd_aux * _nbnd, _nbnd});
            nda::blas::gemm(1, V_Qkl_Qk_l, D_skij(ispin, ikpt, all, all), 0, tmp_Qk_i);
            auto tmp_Q_ki = nda::reshape(tmp_Qk_i, std::array<int, 2>({_nbnd_aux , _nbnd * _nbnd}));
            auto tmp_S_ki = nda::array<ComplexType, 2>::zeros({_nbnd_aux, _nbnd * _nbnd});
            nda::blas::gemm(1, V_SQinv_S_Q, tmp_Q_ki, 0, tmp_S_ki);
            auto tmp_R_ki = nda::array<ComplexType, 2>::zeros({_nbnd_aux, _nbnd * _nbnd});
            nda::blas::gemm(1, V_dRS_R_S, tmp_S_ki, 0, tmp_R_ki);
            auto tmp_R_i_k = nda::reshape(tmp_S_ki, std::array<int, 3>({_nbnd_aux, _nbnd, _nbnd}));
            auto tmp_R_k_i = nda::reshape(tmp_R_ki, std::array<int, 3>({_nbnd_aux, _nbnd, _nbnd}));
            for (int iR = 0; iR < _nbnd_aux; ++iR) {
              auto tmp_i_k = nda::make_regular(nda::transpose(tmp_R_k_i(iR, all, all)));
              RealType spin_factor = (_nspin == 1 and _npol == 1) ? 1.0 : 0.5;
              tmp_grad += nda::sum(tmp_R_i_k(iR, nda::ellipsis{}) * tmp_i_k) * spin_factor * _k_weight(ikpt);
            }
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
                                          chol_grad_reader_t&, bool);
    template void hf_gradient_t::evaluate(Arr2D&, Arr2D&, Arr2D&,
                                          const Arrv4D&, const Arrv4D&, const Arrv4D&, const Arrv4D&,
                                          chol_grad_reader_t&, bool);

  } // namespace solvers

} // namespace methods