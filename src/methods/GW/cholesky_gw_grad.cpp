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

      for( auto& v: {"TOTAL", "ALLOC", "COMM", "EVAL_DYSON_P", "EVAL_P0", "IMAG_FT"} ) {
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
      sArray_t<nda::array_view<ComplexType, 3>> sP0_tPQ(*mpi, {nt_half, _nbnd_aux, _nbnd_aux});
      sArray_t<nda::array_view<ComplexType, 3>> sP0_wPQ(*mpi, {nw_half, _nbnd_aux, _nbnd_aux});

      _Timer.stop("ALLOC");

      // TODO these should be input parameters
      int nbnd_aux_batch = sP0_tPQ.local().shape(1);

      for (size_t iq = 0; iq < chol.MF()->nkpts(); ++iq) {
        _Timer.start("EVAL_P0");
        eval_P0(iq, G_tskij, sP0_tPQ, chol, nbnd_aux_batch, (iq == 0)? true : false);
        _Timer.stop("EVAL_P0");

        _Timer.start("EVAL_DYSON_P");
        eval_dyson_P(sP0_tPQ, sP0_wPQ);
        _Timer.stop("EVAL_DYSON_P");
      }

      return tmp_grad_2e;
    }

    // same as gw_t::evaluate_P0
    template<nda::MemoryArray Array_3D_t>
    void gw_gradient_t::eval_P0(size_t iq, const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                sArray_t<Array_3D_t> &sP0_tPQ, Cholesky_ERI auto &chol,
                                int batch_size, bool print_mpi)
    {
      decltype(nda::range::all) all;
      sP0_tPQ.set_zero();
      size_t nt_half  = sP0_tPQ.local().shape(0);
      size_t nts   = G_tskij.shape(0);
      size_t ns    = G_tskij.shape(1);
      size_t nkpts = chol.MF()->nkpts();
      size_t Np    = chol.Np();
      size_t nbnd  = chol.MF()->nbnd();

      if (batch_size < 0) batch_size = Np;
      utils::check(Np % batch_size == 0, "gw_t::evaluate_P0: Np % batch_size != 0");
      size_t n_batch = Np / batch_size;
      auto[dim0_rank, dim0_comm_size, dim1_rank, dim1_comm_size] =
          utils::setup_two_layer_mpi(sP0_tPQ.communicator(), nt_half, n_batch);
      if (print_mpi) {
        app_log(2, "    - evaluate_P0: batch size = {}", batch_size);
        app_log(2, "    - MPI processors along nt_half axis = {}", dim0_comm_size);
        app_log(2, "    - MPI processors along Np axis = {}", dim1_comm_size);
      }

      _Timer.start("ALLOC");
      nda::array<ComplexType, 3> L_Pab_conj(batch_size, nbnd, nbnd);
      auto L_Pa_b_conj = nda::reshape(L_Pab_conj, shape_t<2>{batch_size*nbnd, nbnd});

      nda::array<ComplexType, 3> X_Pac(batch_size, nbnd, nbnd);
      auto X_Pa_c = nda::reshape(X_Pac, shape_t<2>{batch_size*nbnd, nbnd});
      auto X_P_ac = nda::reshape(X_Pac, shape_t<2>{batch_size, nbnd*nbnd});

      nda::array<ComplexType, 3> X2_acP(nbnd, nbnd, batch_size);
      auto X2_ac_P = nda::reshape(X2_acP, shape_t<2>{nbnd*nbnd, batch_size});
      auto X2_a_cP = nda::reshape(X2_acP, shape_t<2>{nbnd, nbnd*batch_size});

      nda::array<ComplexType, 3> Y_dcP(nbnd, nbnd, batch_size);
      auto Y_d_cP = nda::reshape(Y_dcP, shape_t<2>{nbnd, nbnd*batch_size});
      auto Y_dc_P = nda::reshape(Y_dcP, shape_t<2>{nbnd*nbnd, batch_size});

      nda::matrix<ComplexType> Z_QP(Np, batch_size);
      _Timer.stop("ALLOC");

      double spin_factor = (ns == 1)? -2.0/nkpts : -1.0/nkpts; // FIXME keep 1/nkpts just for debug

      sP0_tPQ.win().fence();
      for (size_t it = dim0_rank; it < nt_half; it += dim0_comm_size) { // MPI
        size_t itt = nts - it - 1;
        for (size_t is = 0; is < ns; ++is) {
          for (size_t ik = 0; ik < nkpts; ++ik) {
            long ikmq = chol.MF()->qk_to_k2(iq, ik); // K(ikmq) = K(ik) - Q(iq) + G
            auto L_Pab = chol.V(iq, is, ik);
            auto Gmt_bc = nda::transpose(G_tskij(itt, is, ikmq, all, all));
            auto Gt_da  = nda::transpose(G_tskij(it, is, ik, all, all));
            for (size_t PP = dim1_rank; PP < n_batch; PP += dim1_comm_size) { // MPI
              nda::range P_range(PP*batch_size, (PP+1)*batch_size);
              // X_Pac = L_Pab_conj * Gmt_bc
              L_Pab_conj = nda::conj(L_Pab(P_range, nda::ellipsis{}));
              nda::blas::gemm(L_Pa_b_conj, Gmt_bc, X_Pa_c);

              // X2_acP = X_Pac
              X2_ac_P = nda::transpose(X_P_ac);

              // Y_dcP = Gt_da * X2_acP
              nda::blas::gemm(Gt_da, X2_a_cP, Y_d_cP);

              // P0_PQ = Y_dcP * L_Qdc (L_Pab)
              auto L_Q_dc = nda::reshape(L_Pab, shape_t<2>{Np, nbnd*nbnd});
              nda::blas::gemm(ComplexType(spin_factor), L_Q_dc, Y_dc_P, ComplexType(0.0), Z_QP);
              sP0_tPQ.local()(it, P_range, all) += nda::transpose(Z_QP);
            }
          }
        }
      }
      _Timer.start("COMM");
      sP0_tPQ.win().fence();
      sP0_tPQ.all_reduce();
      _Timer.stop("COMM");
    }

    // same as gw_t::dyson_P
    template<nda::MemoryArray Array_3D_t>
    void gw_gradient_t::eval_dyson_P(sArray_t<Array_3D_t> &sP0_tPQ, sArray_t<Array_3D_t> &sP0_wPQ)
    {
      size_t Np = sP0_tPQ.local().shape(1);
      size_t nw_half = sP0_wPQ.local().shape(0);
      _Timer.start("ALLOC");
      auto I = nda::eye<ComplexType>(Np);
      nda::matrix<ComplexType> X(Np, Np);
      nda::matrix<ComplexType> Y(Np, Np);
      _Timer.stop("ALLOC");

      _Timer.start("IMAG_FT");
      sP0_wPQ.win().fence();
      if (sP0_wPQ.node_comm()->root())
        _ft->tau_to_w_PHsym(sP0_tPQ.local(), sP0_wPQ.local());
      sP0_wPQ.win().fence();
      _Timer.stop("IMAG_FT");

      int rank = sP0_wPQ.communicator()->rank();
      int comm_size = sP0_wPQ.communicator()->size();
      int node_rank = sP0_wPQ.internode_comm()->rank();
      int num_nodes = sP0_wPQ.internode_comm()->size();
      int node_size = sP0_wPQ.node_comm()->size();
      sP0_wPQ.win().fence();
      for (size_t iw = 0; iw < nw_half; ++iw) {
        int iw_node = (iw / node_size) % num_nodes;
        if (iw % comm_size == rank) {
          auto P0w_PQ = sP0_wPQ.local()(iw, nda::ellipsis{});
          // Y = [I - P0w_PQ]^{-1}
          Y = nda::inverse(I - P0w_PQ);
          // [I - P0w_PQ]^{-1} * P0w_PQ
          nda::blas::gemm(Y, P0w_PQ, X);
          P0w_PQ = X;
        } else if (iw_node != node_rank) {
          auto P0w_PQ = sP0_wPQ.local()(iw, nda::ellipsis{});
          P0w_PQ() = 0.0;
        }
      }
      sP0_wPQ.communicator()->barrier();
      _Timer.start("COMM");
      sP0_wPQ.win().fence();
      sP0_wPQ.all_reduce();
      _Timer.stop("COMM");

      _Timer.start("IMAG_FT");
      sP0_tPQ.win().fence();
      if (sP0_tPQ.node_comm()->root())
        _ft->w_to_tau_PHsym(sP0_wPQ.local(), sP0_tPQ.local());
      sP0_tPQ.win().fence();
      _Timer.stop("IMAG_FT");
    }

    ComplexType gw_gradient_t::eval_grad_2e(int iatom, int idirection,
                                            Cholesky_ERI auto && chol)
    {
      return ComplexType(0, 0);
    }

    using Arr2D = nda::array<ComplexType, 2>;
    using Arrv2D = nda::array_view<ComplexType, 2>;
    using Arr3D = nda::array<ComplexType, 3>;
    using Arrv3D = nda::array_view<ComplexType, 3>;
    using Arr5D = nda::array<ComplexType, 5>;
    using Arrv5D = nda::array_view<ComplexType, 5>;
    using Arrv5D2 = nda::array_view<ComplexType, 5, nda::C_layout>;

    template Arr2D gw_gradient_t::evaluate(const Arr5D&, chol_reader_t&);
    template Arr2D gw_gradient_t::evaluate(const Arrv5D&, chol_reader_t&);

    template Arr2D gw_gradient_t::eval_grad_2e(const Arr5D&, chol_reader_t&);
    template Arr2D gw_gradient_t::eval_grad_2e(const Arrv5D&, chol_reader_t&);

    template ComplexType gw_gradient_t::eval_grad_2e(int, int, chol_reader_t&);

    template void gw_gradient_t::eval_P0(size_t, const Arr5D &, sArray_t<Arrv3D>&, chol_reader_t&, int , bool);
    template void gw_gradient_t::eval_P0(size_t, const Arrv5D &, sArray_t<Arrv3D>&, chol_reader_t&, int , bool);
    template void gw_gradient_t::eval_P0(size_t, const Arrv5D2 &, sArray_t<Arrv3D>&, chol_reader_t&, int , bool);


  } // namespace solvers

} // namespace methods