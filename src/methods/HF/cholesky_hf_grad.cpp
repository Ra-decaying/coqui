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
#include "methods/HF/hf_t.h"

namespace methods {

  namespace solvers {

    nda::array<ComplexType, 2> hf_t::eval_grad(const nda::MemoryArrayOfRank<4> auto &Dm_skij,
                                               Cholesky_ERI auto &&chol)
    {
      // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20chol-hf-grad
      app_log(1, "\n"
                 "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┬ ┬┌─┐┬   ┬ ┬┌─┐  ┌─┐┬─┐┌─┐┌┬┐\n"
                 "║  ║ ║║═╬╗║ ║║  │  ├─┤│ ││───├─┤├┤───│ ┬├┬┘├─┤ ││\n"
                 "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┴ ┴└─┘┴─┘ ┴ ┴└    └─┘┴└─┴ ┴─┴┘\n");

      utils::check(chol.MF()->nkpts() == chol.MF()->nkpts_ibz(),
                   "hf_t::cholesky_hf_grad::evaluate: Symmetry not yet implemented.");

      for( auto& v: {"GRAD_TOTAL", "GRAD_COULOMB", "GRAD_EXCHANGE", "GRAD_WRITE"} ) {
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

      nda::array<ComplexType, 2> tmp_grad_2e = eval_grad_2e(Dm_skij, chol);

      _Timer.stop("GRAD_TOTAL");

      print_chol_hf_grad_timers();
      chol.print_timers();

      return tmp_grad_2e;
    }

    nda::array<ComplexType, 2> hf_t::eval_grad_2e(const nda::MemoryArrayOfRank<4> auto &Dm_skij,
                                                  Cholesky_ERI auto && chol)
    {
      size_t natoms = chol.MF()->number_of_atoms();
      auto tmp_grad_2e = nda::array<ComplexType, 2>::zeros({natoms, 3});
      tmp_grad_2e = eval_grad_coulomb(Dm_skij, chol) + eval_grad_exchange(Dm_skij, chol);
      return tmp_grad_2e;
    }

    nda::array<ComplexType, 2> hf_t::eval_grad_coulomb(const nda::MemoryArrayOfRank<4> auto &Dm_skij,
                                                       Cholesky_ERI auto && chol)
    {
      size_t natoms = chol.MF()->number_of_atoms();
      auto tmp_grad_coulomb = nda::array<ComplexType, 2>::zeros({natoms, 3});
      for (size_t iatom = 0; iatom < natoms; ++iatom) {
        for (size_t direction = 0; direction < 3; ++direction) {
          tmp_grad_coulomb(iatom, direction) += eval_grad_coulomb(iatom, direction, Dm_skij, chol);
        }
      }
      return tmp_grad_coulomb;
    }

    nda::array<ComplexType, 2> hf_t::eval_grad_exchange(const nda::MemoryArrayOfRank<4> auto &Dm_skij,
                                                        Cholesky_ERI auto && chol)
    {
      auto natoms = chol.MF()->number_of_atoms();
      auto tmp_grad_exchange = nda::array<ComplexType, 2>::zeros({natoms, 3});
      for (size_t iatom = 0; iatom < natoms; ++iatom) {
        for (size_t direction = 0; direction < 3; ++direction) {
          tmp_grad_exchange(iatom, direction) += eval_grad_exchange(iatom, direction, Dm_skij, chol);
        }
      }
      return tmp_grad_exchange;
    }

    ComplexType hf_t::eval_grad_2e(int iatom, int direction,
                                   const nda::MemoryArrayOfRank<4> auto &Dm_skij,
                                   Cholesky_ERI auto && chol)
    {
      ComplexType tmp_grad(0.0, 0.0);
      tmp_grad += eval_grad_coulomb(iatom, direction, Dm_skij, chol);
      tmp_grad += eval_grad_exchange(iatom, direction, Dm_skij, chol);
      return tmp_grad;
    }

    ComplexType hf_t::eval_grad_coulomb(int iatom, int direction,
                                        const nda::MemoryArrayOfRank<4> auto &Dm_skij,
                                        Cholesky_ERI auto && chol)
    {
      decltype(nda::range::all) all;

      _Timer.start("GRAD_COULOMB");

      auto nbnd = chol.MF()->nbnd();
      auto nbnd_aux = chol.Np();
      auto nspin = chol.MF()->nspin();
      auto nkpts = chol.MF()->nkpts();
      auto npol = chol.MF()->npol();
      auto k_weight = chol.MF()->k_weight();

      ComplexType tmp_grad(0.0, 0.0);

      // coulomb
      //   \sum D_{ji} D_{lk} [ d/dX (ij|kl) ]
      // = \sum D_{ji} D_{lk} ( V_{Qij} dV^{*}_{Qlk} + dV_{Qij} V^{*}_{Qlk} )
      // = \sum ( V_{Qij} D^{*}_{ij} ) (dV_{Qkl} D_{kl} )^{*} +
      //   \sum (dV_{Qij} D^{*}_{ij} ) ( V_{Qkl} D_{kl} )^{*}
      // TO-DO: check and confirm complex conjugate and transpose

      for (size_t ikpt = 0; ikpt < nkpts; ++ikpt) {
        RealType spin_factor = (nspin == 1 and npol == 1) ? 2.0 : 1.0;
        auto tmp = nda::array<ComplexType, 2>::zeros({1, 1});
        auto dm_total = nda::array<ComplexType, 2>::zeros({nbnd, nbnd});
        auto V = chol.V(0, 0, ikpt);
        auto dV = chol.dV(0, iatom, direction, 0, ikpt);

        for (size_t ispin = 0; ispin < nspin; ++ispin) {
          dm_total += Dm_skij(ispin, ikpt, all, all);
        }
        dm_total *= spin_factor;

        auto dm_total_conj = nda::make_regular(nda::conj(dm_total));
        auto dm_total_conj_ij_1 = nda::reshape(dm_total_conj, std::array<int, 2>({nbnd * nbnd, 1}));
        auto dm_total_kl_1 = nda::reshape(dm_total, std::array<int, 2>({nbnd * nbnd, 1}));

        {
          auto V_Pij_P_ij = nda::reshape(V, std::array<int, 2>({nbnd_aux, nbnd * nbnd}));
          auto tmp_P_1 = nda::array<ComplexType, 2>::zeros({nbnd_aux, 1});
          nda::blas::gemm(V_Pij_P_ij, dm_total_conj_ij_1, tmp_P_1);
          auto dV_Qkl_Q_kl = nda::reshape(dV, std::array<int, 2>({nbnd_aux, nbnd * nbnd}));
          auto tmp_Q_1 = nda::array<ComplexType, 2>::zeros({nbnd_aux, 1});
          nda::blas::gemm(dV_Qkl_Q_kl, dm_total_kl_1, tmp_Q_1);
          nda::blas::gemm(1, nda::conj(nda::transpose(tmp_Q_1)), tmp_P_1, 1, tmp);
        }

        {
          auto dV_Pij_P_ij = nda::reshape(dV, std::array<int, 2>({nbnd_aux, nbnd * nbnd}));
          auto tmp_P_1 = nda::array<ComplexType, 2>::zeros({nbnd_aux, 1});
          nda::blas::gemm(dV_Pij_P_ij, dm_total_conj_ij_1, tmp_P_1);
          auto V_Qkl_Q_kl = nda::reshape(V, std::array<int, 2>({nbnd_aux, nbnd * nbnd}));
          auto tmp_Q_1 = nda::array<ComplexType, 2>::zeros({nbnd_aux, 1});
          nda::blas::gemm(V_Qkl_Q_kl, dm_total_kl_1, tmp_Q_1);
          nda::blas::gemm(1, nda::conj(nda::transpose(tmp_Q_1)), tmp_P_1, 1, tmp);
        }

        tmp_grad += tmp(0, 0) * 0.5 * k_weight(ikpt);
      }

      _Timer.stop("GRAD_COULOMB");

      return tmp_grad;
    }

    ComplexType hf_t::eval_grad_exchange(int iatom, int direction,
                                         const nda::MemoryArrayOfRank<4> auto &Dm_skij,
                                         Cholesky_ERI auto && chol)
    {
      decltype(nda::range::all) all;

      _Timer.start("GRAD_EXCHANGE");

      auto nbnd = chol.MF()->nbnd();
      auto nbnd_aux = chol.Np();
      auto nspin = chol.MF()->nspin();
      auto nkpts = chol.MF()->nkpts();
      auto npol = chol.MF()->npol();
      auto k_weight = chol.MF()->k_weight();


      ComplexType tmp_grad(0.0, 0.0);

      // exchange
      //   \sum D_{li} D_{jk} [ d/dX (ij|kl) ]
      // = \sum D_{li} D_{jk} ( V_{Qij} dV^{*}_{Qlk} + dV_{Qij} V^{*}_{Qlk} )
      // = \sum ( V_{Qij} D_{jk} ) (dV_{Qlk} D^{*}_{il} )^{*} +
      //   \sum (dV_{Qij} D_{jk} ) ( V_{Qlk} D^{*}_{il} )^{*}
      // TO-DO: check and confirm conjugate and transpose

      for (size_t ispin = 0; ispin < nspin; ++ispin) {
        for (size_t ikpt = 0; ikpt < nkpts; ++ikpt) {
          RealType spin_factor = (nspin == 1 and npol == 1) ? 1.0 : 0.5;
          auto V = chol.V(0, 0, ikpt);
          auto dV = chol.dV(0, iatom, direction, 0, ikpt);
          auto dm_conj = nda::make_regular(nda::conj(Dm_skij(ispin, ikpt, all, all)));

          {
            auto V_Pij_Pi_j = nda::reshape(V, std::array<int, 2>({nbnd_aux * nbnd, nbnd}));
            auto tmp_Pik_Pi_k = nda::array<ComplexType, 2>::zeros({nbnd_aux * nbnd, nbnd});
            nda::blas::gemm(V_Pij_Pi_j, Dm_skij(ispin, ikpt, all, all), tmp_Pik_Pi_k);
            auto tmp_Qik_Q_i_k = nda::array<ComplexType, 3>::zeros({nbnd_aux, nbnd, nbnd});
            for (size_t iQ = 0; iQ < nbnd_aux; ++iQ) {
              nda::blas::gemm(dm_conj, dV(iQ, all, all), tmp_Qik_Q_i_k(iQ, all, all));
              tmp_Qik_Q_i_k(iQ, all, all) = nda::make_regular(nda::conj(tmp_Qik_Q_i_k(iQ, all, all)));
            }
            auto tmp_Pik_P_i_k = nda::reshape(tmp_Pik_Pi_k, std::array<int, 3>({nbnd_aux, nbnd, nbnd}));
            tmp_grad -= nda::sum(tmp_Pik_P_i_k * tmp_Qik_Q_i_k) * spin_factor * k_weight(ikpt);
          }

          {
            auto dV_Pij_Pi_j = nda::reshape(dV, std::array<int, 2>({nbnd_aux * nbnd, nbnd}));
            auto tmp_Pik_Pi_k = nda::array<ComplexType, 2>::zeros({nbnd_aux * nbnd, nbnd});
            nda::blas::gemm(dV_Pij_Pi_j, Dm_skij(ispin, ikpt, all, all), tmp_Pik_Pi_k);
            auto tmp_Qik_Q_i_k = nda::array<ComplexType, 3>::zeros({nbnd_aux, nbnd, nbnd});
            for (size_t iQ = 0; iQ < nbnd_aux; ++iQ) {
              nda::blas::gemm(dm_conj, V(iQ, all, all), tmp_Qik_Q_i_k(iQ, all, all));
              tmp_Qik_Q_i_k(iQ, all, all) = nda::make_regular(nda::conj(tmp_Qik_Q_i_k(iQ, all, all)));
            }
            auto tmp_Pik_P_i_k = nda::reshape(tmp_Pik_Pi_k, std::array<int, 3>({nbnd_aux, nbnd, nbnd}));
            tmp_grad -= nda::sum(tmp_Pik_P_i_k * tmp_Qik_Q_i_k) * spin_factor * k_weight(ikpt);
          }
        }
      }

      _Timer.stop("GRAD_EXCHANGE");

      return tmp_grad;
    }

    using Arr2D = nda::array<ComplexType, 2>;
    using Arr3D = nda::array<ComplexType, 2>;
    using Arr4D = nda::array<ComplexType, 4>;
    using Arrv4D = nda::array_view<ComplexType, 4>;

    template Arr2D hf_t::eval_grad(const Arr4D&, chol_reader_t&);
    template Arr2D hf_t::eval_grad(const Arrv4D&, chol_reader_t&);

    template Arr2D hf_t::eval_grad_2e(const Arr4D&, chol_reader_t&);
    template Arr2D hf_t::eval_grad_2e(const Arrv4D&, chol_reader_t&);

    template Arr2D hf_t::eval_grad_coulomb(const Arr4D&, chol_reader_t&);
    template Arr2D hf_t::eval_grad_coulomb(const Arrv4D&, chol_reader_t&);

    template Arr2D hf_t::eval_grad_exchange(const Arr4D&, chol_reader_t&);
    template Arr2D hf_t::eval_grad_exchange(const Arrv4D&, chol_reader_t&);

    template ComplexType hf_t::eval_grad_2e(int, int, const Arr4D&, chol_reader_t&);
    template ComplexType hf_t::eval_grad_2e(int, int, const Arrv4D&, chol_reader_t&);

    template ComplexType hf_t::eval_grad_coulomb(int, int, const Arr4D&, chol_reader_t&);
    template ComplexType hf_t::eval_grad_coulomb(int, int, const Arrv4D&, chol_reader_t&);

    template ComplexType hf_t::eval_grad_exchange(int, int, const Arr4D&, chol_reader_t&);
    template ComplexType hf_t::eval_grad_exchange(int, int, const Arrv4D&, chol_reader_t&);

  } // namespace solvers

} // namespace methods