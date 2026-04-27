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

      for( auto& v: {"TOTAL", "ALLOC", "COMM", "EVAL_2BDM_INTERMEDIATE",
                     "EVAL_DYSON_P", "EVAL_GRAD_INTERMEDIATE1", "EVAL_GRAD_INTERMEDIATE2",
                     "EVAL_P0", "IMAG_FT"} ) {
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
      auto mpi = chol.mpi();

      utils::check(_mf->nkpts() == 1,
                   "gw_gradient_t::eval_grad_2e: Only works for molecular system yet.");

      _Timer.start("ALLOC");
      sArray_t<nda::array_view<ComplexType, 3>> sP_tPQ(*mpi, {_ft->nt_f(), _nbnd_aux, _nbnd_aux});
      sArray_t<nda::array_view<ComplexType, 3>> sP_wPQ(*mpi, {_ft->nw_b(), _nbnd_aux, _nbnd_aux});
      sArray_t<nda::array_view<ComplexType, 4>> sInter1_tsPQ(*mpi, {_ft->nt_f(), _nspin, _nbnd_aux, _nbnd_aux});
      sArray_t<nda::array_view<ComplexType, 4>> sInter1_wsPQ(*mpi, {_ft->nw_b(), _nspin, _nbnd_aux, _nbnd_aux});
      sArray_t<nda::array_view<ComplexType, 4>> sInter2_tsPQ(*mpi, {_ft->nt_f(), _nspin, _nbnd_aux, _nbnd_aux});
      sArray_t<nda::array_view<ComplexType, 4>> sInter2_wsPQ(*mpi, {_ft->nw_b(), _nspin, _nbnd_aux, _nbnd_aux});
      _Timer.stop("ALLOC");

      // TODO these should be input parameters
      int nbnd_aux_batch = _nbnd_aux;

      auto tmp_grad_2e = nda::array<ComplexType, 2>::zeros({_natoms, 3});

      for (size_t iq = 0; iq < _mf->nkpts(); ++iq) {

        _Timer.start("EVAL_P0");
        eval_P0(iq, G_tskij, sP_tPQ, chol, nbnd_aux_batch, (iq == 0)? true : false);
        _Timer.stop("EVAL_P0");

        _Timer.start("IMAG_FT");
        sP_wPQ.win().fence();
        if (sP_wPQ.node_comm()->root()) {
          _ft->tau_to_w(sP_tPQ.local(), sP_wPQ.local(), imag_axes_ft::boson);
        }
        sP_wPQ.win().fence();
        _Timer.stop("IMAG_FT");

        _Timer.start("EVAL_DYSON_P");
        eval_dyson_P(sP_tPQ, sP_wPQ, false);
        _Timer.stop("EVAL_DYSON_P");

        _Timer.start("EVAL_GRAD_INTERMEDIATE2");
        eval_grad_intermediate2(iq, G_tskij, sInter2_tsPQ, sInter2_wsPQ, chol, nbnd_aux_batch, (iq == 0)? true : false);
        _Timer.stop("EVAL_GRAD_INTERMEDIATE2");

        for (size_t iatom = 0; iatom < _natoms; ++iatom) {
          for (size_t direction = 0; direction < 3; ++direction){
            tmp_grad_2e(iatom, direction) += eval_grad_2e(iq, iatom, direction, G_tskij,
                                                          sP_wPQ, sInter1_tsPQ, sInter1_wsPQ,
                                                          sInter2_wsPQ, chol);
          }
        }

      }


      return tmp_grad_2e;
    }

    nda::array<ComplexType, 6> gw_gradient_t::eval_2bdm(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                                        Cholesky_ERI auto &&chol, bool PHsym)
    {
      decltype(nda::range::all) all;

      auto mpi = chol.mpi();

      size_t nt = 0;
      size_t nw = 0;
      if (PHsym) {
        nt = (_ft->nt_b()%2==0)? _ft->nt_b()/2 : _ft->nt_b()/2 + 1;
        nw = (_ft->nw_b()%2==0)? _ft->nw_b()/2 : _ft->nw_b()/2 + 1;
      } else {
        nt = _ft->nt_f();
        nw = _ft->nw_b();
      }

      _Timer.start("ALLOC");

      sArray_t<nda::array_view<ComplexType, 3>> sP0_tPQ(*mpi, {nt, _nbnd_aux, _nbnd_aux});
      sArray_t<nda::array_view<ComplexType, 3>> sP0_wPQ(*mpi, {nw, _nbnd_aux, _nbnd_aux});
      sArray_t<nda::array_view<ComplexType, 5>> sInter_tsijQ(*mpi, {nt, _nspin, _nbnd, _nbnd, _nbnd_aux});

      auto tbdm_sspqrs = nda::array<ComplexType, 6>::zeros({_nspin, _nspin, _nbnd, _nbnd, _nbnd, _nbnd});

      _Timer.stop("ALLOC");

      // TODO these should be input parameters
      int nbnd_aux_batch = _nbnd_aux;

      for (size_t iq = 0; iq < _mf->nkpts(); ++iq) {
        _Timer.start("EVAL_P0");
        eval_P0(iq, G_tskij, sP0_tPQ, chol, nbnd_aux_batch, (iq == 0)? true : false);
        _Timer.stop("EVAL_P0");

        _Timer.start("EVAL_DYSON_P");
        eval_dyson_P(sP0_tPQ, sP0_wPQ, PHsym);
        _Timer.stop("EVAL_DYSON_P");

        _Timer.start("EVAL_2BDM_INTERMEDIATE");
        _Timer.start("IMAG_FT");
        eval_2bdm_intermediate(iq, G_tskij, sInter_tsijQ, chol, nbnd_aux_batch, (iq == 0)? true : false);
        auto inter_2bdm_wsijQ = nda::array<ComplexType, 5>::zeros({nw, _nspin, _nbnd, _nbnd, _nbnd_aux});
        if (PHsym) {
          _ft->tau_to_w_PHsym(sInter_tsijQ.local(), inter_2bdm_wsijQ);
        } else {
          _ft->tau_to_w(sInter_tsijQ.local(), imag_axes_ft::fermi, inter_2bdm_wsijQ, imag_axes_ft::boson);
        }
        _Timer.stop("IMAG_FT");
        _Timer.stop("EVAL_2BDM_INTERMEDIATE");

        auto tbdm_tijkl = nda::array<ComplexType, 5>::zeros({nt, _nbnd, _nbnd, _nbnd, _nbnd});
        auto tbdm_wijkl = nda::array<ComplexType, 5>::zeros({nw, _nbnd, _nbnd, _nbnd, _nbnd});
        auto tbdm_klij = nda::array<ComplexType, 4>::zeros({_nbnd, _nbnd, _nbnd, _nbnd});
        auto tbdm_kl_ij = nda::reshape(tbdm_klij, shape_t<2>{_nbnd*_nbnd, _nbnd*_nbnd});

        for (size_t is1 = 0; is1 < _nspin; ++is1) {
          for (size_t is2 = 0; is2 < _nspin; ++is2) {

            for (size_t iwn = 0; iwn < nw; ++iwn) {
              auto inter_ijQ = nda::array<ComplexType, 3>::zeros({_nbnd, _nbnd, _nbnd_aux});
              auto inter_klP = nda::array<ComplexType, 3>::zeros({_nbnd, _nbnd, _nbnd_aux});
              auto inter_ij_Q = nda::reshape(inter_ijQ, shape_t<2>{_nbnd*_nbnd, _nbnd_aux});
              auto inter_kl_P = nda::reshape(inter_klP, shape_t<2>{_nbnd*_nbnd, _nbnd_aux});
              inter_ijQ = inter_2bdm_wsijQ(iwn, is1, all, all, all);
              inter_klP = inter_2bdm_wsijQ(iwn, is2, all, all, all);

              auto tmp_klQ = nda::array<ComplexType, 3>::zeros({_nbnd, _nbnd, _nbnd_aux});
              auto tmp_kl_Q = nda::reshape(tmp_klQ, shape_t<2>{_nbnd*_nbnd, _nbnd_aux});

              auto eye = nda::eye<ComplexType>(_nbnd_aux);
              auto tmp_PQ = nda::array<ComplexType, 2>::zeros({_nbnd_aux, _nbnd_aux});
              tmp_PQ = eye + sP0_wPQ.local()(iwn, all, all);
              nda::blas::gemm(inter_kl_P, tmp_PQ, tmp_kl_Q);
              nda::blas::gemm(-1.0, tmp_kl_Q, nda::transpose(inter_ij_Q), 0.0, tbdm_kl_ij);
              tbdm_wijkl(iwn, all, all, all, all) = tbdm_klij;
            }

            if (PHsym) {
              _ft->w_to_tau_PHsym(tbdm_wijkl, tbdm_tijkl);
              _ft->tau_to_beta_PHsym(tbdm_tijkl, tbdm_klij);
              tbdm_sspqrs(is1, is2, nda::ellipsis{}) = tbdm_klij;
            } else {
              _ft->w_to_tau(tbdm_wijkl, imag_axes_ft::boson, tbdm_tijkl, imag_axes_ft::fermi);
              _ft->tau_to_beta(tbdm_tijkl, tbdm_sspqrs(is1, is2, nda::ellipsis{}));
            }

          }
        }

      }

      return tbdm_sspqrs;
    }

    template<nda::MemoryArray Array_3D_t, nda::MemoryArray Array_4D_t>
    ComplexType gw_gradient_t::eval_grad_2e(size_t iq, size_t iatom, size_t direction,
                                            const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                            sArray_t<Array_3D_t> &sP_wPQ,
                                            sArray_t<Array_4D_t> &sInter1_tsPQ,
                                            sArray_t<Array_4D_t> &sInter1_wsPQ,
                                            sArray_t<Array_4D_t> &sInter2_wsPQ,
                                            Cholesky_ERI auto &&chol)
    {
      decltype(nda::range::all) all;

      // TODO these should be input parameters
      int nbnd_aux_batch = _nbnd_aux;

      _Timer.start("EVAL_GRAD_INTERMEDIATE1");
      eval_grad_intermediate1(iq, iatom, direction, G_tskij, sInter1_tsPQ, sInter1_wsPQ, chol, nbnd_aux_batch,
                              (iq == 0)? true : false);
      _Timer.stop("EVAL_GRAD_INTERMEDIATE1");

      auto trace_w = nda::array<ComplexType, 2>::zeros({sInter1_tsPQ.local().shape(0), 1});
      auto trace_t = nda::array<ComplexType, 2>::zeros({sInter1_tsPQ.local().shape(0), 1});
      auto trace_beta = nda::array<ComplexType, 1>::zeros({1});
      auto tmp1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, _nbnd_aux});
      auto tmp2 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, _nbnd_aux});

      RealType spin_factor = (_nspin == 1 and _npol == 1) ? 2.0 : 0.5;

      for (size_t iwn = 0; iwn < sP_wPQ.local().shape(0); ++iwn) {
        for (size_t is1 = 0; is1 < _nspin; ++is1) {
          for (size_t is2 = 0; is2 < _nspin; ++is2) {
            nda::blas::gemm(nda::transpose(sInter1_wsPQ.local()(iwn, is2, all, all)), sP_wPQ.local()(iwn, all, all), tmp1);
            nda::blas::gemm(1.0, tmp1, sInter2_wsPQ.local()(iwn, is1, all, all), 0.0, tmp2);
            nda::blas::gemm(nda::transpose(sInter2_wsPQ.local()(iwn, is2, all, all)), sP_wPQ.local()(iwn, all, all), tmp1);
            nda::blas::gemm(1.0, tmp1, sInter1_wsPQ.local()(iwn, is1, all, all), 1.0, tmp2);
            trace_w(iwn, 0) += nda::trace(tmp2) * spin_factor;
          }
        }
      }

      _ft->w_to_tau(trace_w, imag_axes_ft::boson, trace_t, imag_axes_ft::fermi);
      _ft->tau_to_beta(trace_t, trace_beta);

      return trace_beta(0);
    }

    // similar to gw_t::evaluate_P0
    // JHL: for the evaluation of 2-RDM, should we sum over all k-points in this step?
    template<nda::MemoryArray Array_3D_t>
    void gw_gradient_t::eval_P0(size_t iq, const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                sArray_t<Array_3D_t> &sP0_tPQ, Cholesky_ERI auto &chol,
                                int batch_size, bool print_mpi)
    {
      decltype(nda::range::all) all;
      sP0_tPQ.set_zero();

      if (batch_size < 0) batch_size = _nbnd_aux;
      utils::check(_nbnd_aux % batch_size == 0, "gw_gradient_t::eval_P0: _nbnd_aux % batch_size != 0");
      size_t n_batch = _nbnd_aux / batch_size;
      auto[dim0_rank, dim0_comm_size, dim1_rank, dim1_comm_size] =
          utils::setup_two_layer_mpi(sP0_tPQ.communicator(), sP0_tPQ.local().shape(0), n_batch);
      if (print_mpi) {
        app_log(2, "    - evaluate_P0: batch size = {}", batch_size);
        app_log(2, "    - MPI processors along nt axis = {}", dim0_comm_size);
        app_log(2, "    - MPI processors along nbnd_aux axis = {}", dim1_comm_size);
        app_log(2, "\n");
      }

      _Timer.start("ALLOC");
      nda::array<ComplexType, 3> L_Pab_conj(batch_size, _nbnd, _nbnd);
      auto L_Pa_b_conj = nda::reshape(L_Pab_conj, shape_t<2>{batch_size*_nbnd, _nbnd});

      nda::array<ComplexType, 3> X_Pac(batch_size, _nbnd, _nbnd);
      auto X_Pa_c = nda::reshape(X_Pac, shape_t<2>{batch_size*_nbnd, _nbnd});
      auto X_P_ac = nda::reshape(X_Pac, shape_t<2>{batch_size, _nbnd*_nbnd});

      nda::array<ComplexType, 3> X2_acP(_nbnd, _nbnd, batch_size);
      auto X2_ac_P = nda::reshape(X2_acP, shape_t<2>{_nbnd*_nbnd, batch_size});
      auto X2_a_cP = nda::reshape(X2_acP, shape_t<2>{_nbnd, _nbnd*batch_size});

      nda::array<ComplexType, 3> Y_dcP(_nbnd, _nbnd, batch_size);
      auto Y_d_cP = nda::reshape(Y_dcP, shape_t<2>{_nbnd, _nbnd*batch_size});
      auto Y_dc_P = nda::reshape(Y_dcP, shape_t<2>{_nbnd*_nbnd, batch_size});

      nda::matrix<ComplexType> Z_QP(_nbnd_aux, batch_size);
      _Timer.stop("ALLOC");

      double spin_factor = (_nspin == 1)? -2.0/_nkpts : -1.0/_nkpts; // FIXME keep 1/nkpts just for debug

      sP0_tPQ.win().fence();
      for (size_t it = dim0_rank; it < sP0_tPQ.local().shape(0); it += dim0_comm_size) { // MPI
        size_t itt = _ft->nt_f() - it - 1;
        for (size_t is = 0; is < _nspin; ++is) {
          for (size_t ik = 0; ik < _nkpts; ++ik) {
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
              auto L_Q_dc = nda::reshape(L_Pab, shape_t<2>{_nbnd_aux, _nbnd*_nbnd});
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

    // It might be wrong for periodic system
    // G is hermitian (for molecule, G should be real)
    // No need to sum over spin
    template<nda::MemoryArray Array_4D_t>
    void gw_gradient_t::eval_grad_intermediate1(size_t iq, size_t iatom, size_t idirection,
                                                const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                                sArray_t<Array_4D_t> &sInter1_tsPQ, sArray_t<Array_4D_t> &sInter1_wsPQ,
                                                Cholesky_ERI auto &chol, int batch_size, bool print_mpi)
    {
      decltype(nda::range::all) all;
      sInter1_tsPQ.set_zero();

      if (batch_size < 0) batch_size = _nbnd_aux;
      utils::check(_nbnd_aux % batch_size == 0, "gw_gradient_t::eval_intermediate1: _nbnd_aux % batch_size != 0");
      size_t n_batch = _nbnd_aux / batch_size;
      auto[dim0_rank, dim0_comm_size, dim1_rank, dim1_comm_size] =
          utils::setup_two_layer_mpi(sInter1_tsPQ.communicator(), sInter1_tsPQ.local().shape(0), n_batch);
      if (print_mpi) {
        app_log(2, "    - eval_grad_intermediate1: batch size = {}", batch_size);
        app_log(2, "    - MPI processors along nt_f axis = {}", dim0_comm_size);
        app_log(2, "    - MPI processors along nbnd_aux axis = {}", dim1_comm_size);
        app_log(2, "\n");
      }

      _Timer.start("ALLOC");
      nda::array<ComplexType, 3> dL_Pab_conj(batch_size, _nbnd, _nbnd);
      auto dL_Pa_b_conj = nda::reshape(dL_Pab_conj, shape_t<2>{batch_size*_nbnd, _nbnd});

      nda::array<ComplexType, 3> X_Pac(batch_size, _nbnd, _nbnd);
      auto X_Pa_c = nda::reshape(X_Pac, shape_t<2>{batch_size*_nbnd, _nbnd});
      auto X_P_ac = nda::reshape(X_Pac, shape_t<2>{batch_size, _nbnd*_nbnd});

      nda::array<ComplexType, 3> X2_acP(_nbnd, _nbnd, batch_size);
      auto X2_ac_P = nda::reshape(X2_acP, shape_t<2>{_nbnd*_nbnd, batch_size});
      auto X2_a_cP = nda::reshape(X2_acP, shape_t<2>{_nbnd, _nbnd*batch_size});

      nda::array<ComplexType, 3> Y_dcP(_nbnd, _nbnd, batch_size);
      auto Y_d_cP = nda::reshape(Y_dcP, shape_t<2>{_nbnd, _nbnd*batch_size});
      auto Y_dc_P = nda::reshape(Y_dcP, shape_t<2>{_nbnd*_nbnd, batch_size});

      nda::matrix<ComplexType> Z_QP(_nbnd_aux, batch_size);
      _Timer.stop("ALLOC");

      sInter1_tsPQ.win().fence();

      for (size_t it = dim0_rank; it < sInter1_tsPQ.local().shape(0); it += dim0_comm_size) { // MPI
        size_t itt = _ft->nt_f() - it - 1;
        for (size_t is = 0; is < _nspin; ++is) {
          for (size_t ik = 0; ik < _nkpts; ++ik) {
            long ikmq = chol.MF()->qk_to_k2(iq, ik); // K(ikmq) = K(ik) - Q(iq) + G
            auto dL_Pab = chol.dV(iq, iatom, idirection, is, ik);
            auto L_Qdc = chol.V(iq, is, ik);
            auto Gmt_bc = nda::transpose(G_tskij(itt, is, ikmq, all, all));
            auto Gt_da  = nda::transpose(G_tskij(it, is, ik, all, all));
            for (size_t PP = dim1_rank; PP < n_batch; PP += dim1_comm_size) { // MPI
              nda::range P_range(PP*batch_size, (PP+1)*batch_size);
              // X_Pac = dL_Pab_conj * Gmt_bc
              dL_Pab_conj = nda::conj(dL_Pab(P_range, nda::ellipsis{}));
              nda::blas::gemm(dL_Pa_b_conj, Gmt_bc, X_Pa_c);

              // X2_acP = X_Pac
              X2_ac_P = nda::transpose(X_P_ac);

              // Y_dcP = Gt_da * X2_acP
              nda::blas::gemm(Gt_da, X2_a_cP, Y_d_cP);

              // Inter1_PQ = Y_dcP * dL_Qdc
              auto L_Q_dc = nda::reshape(L_Qdc, shape_t<2>{_nbnd_aux, _nbnd*_nbnd});
              nda::blas::gemm(L_Q_dc, Y_dc_P, Z_QP);
              sInter1_tsPQ.local()(it, is, P_range, all) += nda::transpose(Z_QP);
            }
          }
        }
      }

      _Timer.start("COMM");
      sInter1_tsPQ.win().fence();
      sInter1_tsPQ.all_reduce();
      _Timer.stop("COMM");

      _Timer.start("IMAG_FT");
      sInter1_wsPQ.win().fence();
      if (sInter1_wsPQ.node_comm()->root()) {
        _ft->tau_to_w(sInter1_tsPQ.local(), sInter1_wsPQ.local(), imag_axes_ft::boson);
      }
      sInter1_wsPQ.win().fence();
      _Timer.stop("IMAG_FT");
    }


    // It might be wrong for periodic system
    // G is hermitian (for molecule, G should be real)
    // No need to sum over spin
    template<nda::MemoryArray Array_4D_t>
    void gw_gradient_t::eval_grad_intermediate2(size_t iq, const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                                sArray_t<Array_4D_t> &sInter2_tsPQ, sArray_t<Array_4D_t> &sInter2_wsPQ,
                                                Cholesky_ERI auto &chol, int batch_size, bool print_mpi)
    {
      decltype(nda::range::all) all;
      sInter2_tsPQ.set_zero();

      if (batch_size < 0) batch_size = _nbnd_aux;
      utils::check(_nbnd_aux % batch_size == 0, "gw_gradient_t::eval_intermediate2: _nbnd_aux % batch_size != 0");
      size_t n_batch = _nbnd_aux / batch_size;
      auto[dim0_rank, dim0_comm_size, dim1_rank, dim1_comm_size] =
          utils::setup_two_layer_mpi(sInter2_tsPQ.communicator(), sInter2_tsPQ.local().shape(0), n_batch);
      if (print_mpi) {
        app_log(2, "    - eval_grad_intermediate2: batch size = {}", batch_size);
        app_log(2, "    - MPI processors along nt_f axis = {}", dim0_comm_size);
        app_log(2, "    - MPI processors along nbnd_aux axis = {}", dim1_comm_size);
        app_log(2, "\n");
      }

      _Timer.start("ALLOC");
      nda::array<ComplexType, 3> L_Pab_conj(batch_size, _nbnd, _nbnd);
      auto L_Pa_b_conj = nda::reshape(L_Pab_conj, shape_t<2>{batch_size*_nbnd, _nbnd});

      nda::array<ComplexType, 3> X_Pac(batch_size, _nbnd, _nbnd);
      auto X_Pa_c = nda::reshape(X_Pac, shape_t<2>{batch_size*_nbnd, _nbnd});
      auto X_P_ac = nda::reshape(X_Pac, shape_t<2>{batch_size, _nbnd*_nbnd});

      nda::array<ComplexType, 3> X2_acP(_nbnd, _nbnd, batch_size);
      auto X2_ac_P = nda::reshape(X2_acP, shape_t<2>{_nbnd*_nbnd, batch_size});
      auto X2_a_cP = nda::reshape(X2_acP, shape_t<2>{_nbnd, _nbnd*batch_size});

      nda::array<ComplexType, 3> Y_dcP(_nbnd, _nbnd, batch_size);
      auto Y_d_cP = nda::reshape(Y_dcP, shape_t<2>{_nbnd, _nbnd*batch_size});
      auto Y_dc_P = nda::reshape(Y_dcP, shape_t<2>{_nbnd*_nbnd, batch_size});

      nda::matrix<ComplexType> Z_QP(_nbnd_aux, batch_size);
      _Timer.stop("ALLOC");

      sInter2_tsPQ.win().fence();
      for (size_t it = dim0_rank; it < sInter2_tsPQ.local().shape(0); it += dim0_comm_size) { // MPI
        size_t itt = _ft->nt_f() - it - 1;
        for (size_t is = 0; is < _nspin; ++is) {
          for (size_t ik = 0; ik < _nkpts; ++ik) {
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

              // Inter2_PQ = Y_dcP * L_Qdc (L_Pab)
              auto L_Q_dc = nda::reshape(L_Pab, shape_t<2>{_nbnd_aux, _nbnd*_nbnd});
              nda::blas::gemm(L_Q_dc, Y_dc_P, Z_QP);
              sInter2_tsPQ.local()(it, is, P_range, all) += nda::transpose(Z_QP);
            }
          }
        }
      }
      _Timer.start("COMM");
      sInter2_tsPQ.win().fence();
      sInter2_tsPQ.all_reduce();
      _Timer.stop("COMM");

      _Timer.start("IMAG_FT");
      sInter2_wsPQ.win().fence();
      if (sInter2_wsPQ.node_comm()->root()) {
        _ft->tau_to_w(sInter2_tsPQ.local(), sInter2_wsPQ.local(), imag_axes_ft::boson);
      }
      sInter2_wsPQ.win().fence();
      _Timer.stop("IMAG_FT");

    }

  template<nda::MemoryArray Array_5D_t>
  void gw_gradient_t::eval_2bdm_intermediate(size_t iq, const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                             sArray_t<Array_5D_t> &sInter_tsijQ,
                                             Cholesky_ERI auto &chol, int batch_size, bool print_mpi)
    {
      decltype(nda::range::all) all;
      sInter_tsijQ.set_zero();

      if (batch_size < 0) batch_size = _nbnd_aux;
      utils::check(_nbnd_aux % batch_size == 0, "gw_gradient_t::eval_2bdm_intermediate: _nbnd_aux % batch_size != 0");
      size_t n_batch = _nbnd_aux / batch_size;
      auto[dim0_rank, dim0_comm_size, dim1_rank, dim1_comm_size] =
          utils::setup_two_layer_mpi(sInter_tsijQ.communicator(), sInter_tsijQ.local().shape(0), n_batch);
      if (print_mpi) {
        app_log(2, "    - evaluate_eval_2bdm_intermediate: batch size = {}", batch_size);
        app_log(2, "    - MPI processors along nt_f axis = {}", dim0_comm_size);
        app_log(2, "    - MPI processors along Np axis = {}", dim1_comm_size);
      }

      std::cout << "dim0_rank = " << dim0_rank << std::endl;
      std::cout << "dim0_comm_size = " << dim0_comm_size<< std::endl;
      std::cout << "dim1_rank = " << dim1_rank << std::endl;
      std::cout << "dim1_comm_size= " << dim1_comm_size << std::endl;


      _Timer.start("ALLOC");
      nda::array<ComplexType, 3> L_Pab_conj(_nbnd_aux, _nbnd, _nbnd);
      auto L_Pa_b_conj = nda::reshape(L_Pab_conj, shape_t<2>{_nbnd_aux*_nbnd, _nbnd});

      nda::array<ComplexType, 3> X_Pac(_nbnd_aux, _nbnd, _nbnd);
      auto X_Pa_c = nda::reshape(X_Pac, shape_t<2>{_nbnd_aux*_nbnd, _nbnd});
      auto X_P_ac = nda::reshape(X_Pac, shape_t<2>{_nbnd_aux, _nbnd*_nbnd});

      nda::array<ComplexType, 3> X2_acP(_nbnd, _nbnd, _nbnd_aux);
      auto X2_ac_P = nda::reshape(X2_acP, shape_t<2>{_nbnd*_nbnd, _nbnd_aux});
      auto X2_a_cP = nda::reshape(X2_acP, shape_t<2>{_nbnd, _nbnd*_nbnd_aux});

      nda::array<ComplexType, 3> Y_dcP(_nbnd, _nbnd, _nbnd_aux);
      auto Y_d_cP = nda::reshape(Y_dcP, shape_t<2>{_nbnd, _nbnd*_nbnd_aux});
      auto Y_dc_P = nda::reshape(Y_dcP, shape_t<2>{_nbnd*_nbnd, _nbnd_aux});

      _Timer.stop("ALLOC");

      sInter_tsijQ.win().fence();
      for (size_t it = dim0_rank; it < sInter_tsijQ.local().shape(0); it += dim0_comm_size) { // MPI
        size_t itt = _ft->nt_f() - it - 1;
        for (size_t is = 0; is < _nspin; ++is) {
          for (size_t ik = 0; ik < _nkpts; ++ik) {
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

              sInter_tsijQ.local()(it, is, all, all, P_range) += Y_dcP;
            }
          }
        }
      }
      _Timer.start("COMM");
      sInter_tsijQ.win().fence();
      sInter_tsijQ.all_reduce();
      _Timer.stop("COMM");

    }

    // similar gw_t::dyson_P, but with options with or without particle-hole symmetry
    template<nda::MemoryArray Array_3D_t>
    void gw_gradient_t::eval_dyson_P(sArray_t<Array_3D_t> &sP0_tPQ, sArray_t<Array_3D_t> &sP0_wPQ, bool PHsym)
    {
      size_t Np = sP0_tPQ.local().shape(1);
      size_t nw = sP0_wPQ.local().shape(0);
      _Timer.start("ALLOC");
      auto I = nda::eye<ComplexType>(Np);
      nda::matrix<ComplexType> X(Np, Np);
      nda::matrix<ComplexType> Y(Np, Np);
      _Timer.stop("ALLOC");

      _Timer.start("IMAG_FT");
      sP0_wPQ.win().fence();
      if (sP0_wPQ.node_comm()->root()) {
        if (PHsym) {
          _ft->tau_to_w_PHsym(sP0_tPQ.local(), sP0_wPQ.local());
        } else {
          _ft->tau_to_w(sP0_tPQ.local(), sP0_wPQ.local(), imag_axes_ft::boson);
        }
      }
      sP0_wPQ.win().fence();
      _Timer.stop("IMAG_FT");

      int rank = sP0_wPQ.communicator()->rank();
      int comm_size = sP0_wPQ.communicator()->size();
      int node_rank = sP0_wPQ.internode_comm()->rank();
      int num_nodes = sP0_wPQ.internode_comm()->size();
      int node_size = sP0_wPQ.node_comm()->size();
      sP0_wPQ.win().fence();
      for (size_t iw = 0; iw < nw; ++iw) {
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
      if (sP0_tPQ.node_comm()->root()) {
        if (PHsym) {
          _ft->w_to_tau_PHsym(sP0_wPQ.local(), sP0_tPQ.local());
        } else {
          _ft->w_to_tau(sP0_wPQ.local(), sP0_tPQ.local(), imag_axes_ft::boson);
        }
      }
      sP0_tPQ.win().fence();
      _Timer.stop("IMAG_FT");
    }

    using Arr2D = nda::array<ComplexType, 2>;
    using Arr3D = nda::array<ComplexType, 3>;
    using Arr4D = nda::array<ComplexType, 4>;
    using Arr5D = nda::array<ComplexType, 5>;
    using Arr6D = nda::array<ComplexType, 6>;
    using Arrv2D = nda::array_view<ComplexType, 2>;
    using Arrv3D = nda::array_view<ComplexType, 3>;
    using Arrv4D = nda::array_view<ComplexType, 4>;
    using Arrv5D = nda::array_view<ComplexType, 5>;
    using Arrv5D2 = nda::array_view<ComplexType, 5, nda::C_layout>;

    template Arr2D gw_gradient_t::evaluate(const Arr5D&, chol_reader_t&);
    template Arr2D gw_gradient_t::evaluate(const Arrv5D&, chol_reader_t&);

    template Arr2D gw_gradient_t::eval_grad_2e(const Arr5D&, chol_reader_t&);
    template Arr2D gw_gradient_t::eval_grad_2e(const Arrv5D&, chol_reader_t&);

    template Arr6D gw_gradient_t::eval_2bdm(const Arr5D&, chol_reader_t&, bool);
    template Arr6D gw_gradient_t::eval_2bdm(const Arrv5D&, chol_reader_t&, bool);

    template ComplexType gw_gradient_t::eval_grad_2e(size_t, size_t, size_t, const Arr5D&, sArray_t<Arr3D>&,
                                                     sArray_t<Arr4D>&, sArray_t<Arr4D>&, sArray_t<Arr4D>&, chol_reader_t&);
    template ComplexType gw_gradient_t::eval_grad_2e(size_t, size_t, size_t, const Arr5D&,  sArray_t<Arr3D>&,
                                                     sArray_t<Arrv4D>&, sArray_t<Arrv4D>&, sArray_t<Arrv4D>&, chol_reader_t&);
    template ComplexType gw_gradient_t::eval_grad_2e(size_t, size_t, size_t, const Arr5D&, sArray_t<Arrv3D>&,
                                                     sArray_t<Arr4D>&, sArray_t<Arr4D>&, sArray_t<Arr4D>&, chol_reader_t&);
    template ComplexType gw_gradient_t::eval_grad_2e(size_t, size_t, size_t, const Arr5D&, sArray_t<Arrv3D>&,
                                                     sArray_t<Arrv4D>&, sArray_t<Arrv4D>&, sArray_t<Arrv4D>&, chol_reader_t&);

    template void gw_gradient_t::eval_P0(size_t, const Arr5D&, sArray_t<Arrv3D>&, chol_reader_t&, int, bool);
    template void gw_gradient_t::eval_P0(size_t, const Arrv5D&, sArray_t<Arrv3D>&, chol_reader_t&, int, bool);
    template void gw_gradient_t::eval_P0(size_t, const Arrv5D2&, sArray_t<Arrv3D>&, chol_reader_t&, int, bool);

    template void gw_gradient_t::eval_grad_intermediate1(size_t, size_t, size_t, const Arr5D&, sArray_t<Arr4D>&, sArray_t<Arr4D>&,
                                                         chol_reader_t&, int, bool);
    template void gw_gradient_t::eval_grad_intermediate1(size_t, size_t, size_t, const Arr5D&, sArray_t<Arrv4D>&, sArray_t<Arrv4D>&,
                                                         chol_reader_t&, int, bool);
    template void gw_gradient_t::eval_grad_intermediate1(size_t, size_t, size_t, const Arrv5D&, sArray_t<Arr4D>&, sArray_t<Arr4D>&,
                                                         chol_reader_t&, int, bool);
    template void gw_gradient_t::eval_grad_intermediate1(size_t, size_t, size_t, const Arrv5D&, sArray_t<Arrv4D>&, sArray_t<Arrv4D>&,
                                                         chol_reader_t&, int, bool);

    template void gw_gradient_t::eval_grad_intermediate2(size_t, const Arr5D&, sArray_t<Arr4D>&, sArray_t<Arr4D>&,
                                                         chol_reader_t&, int, bool);
    template void gw_gradient_t::eval_grad_intermediate2(size_t, const Arr5D&, sArray_t<Arrv4D>&, sArray_t<Arrv4D>&,
                                                         chol_reader_t&, int, bool);
    template void gw_gradient_t::eval_grad_intermediate2(size_t, const Arrv5D&, sArray_t<Arr4D>&, sArray_t<Arr4D>&,
                                                         chol_reader_t&, int, bool);
    template void gw_gradient_t::eval_grad_intermediate2(size_t, const Arrv5D&, sArray_t<Arrv4D>&, sArray_t<Arrv4D>&,
                                                         chol_reader_t&, int, bool);

    template void gw_gradient_t::eval_2bdm_intermediate(size_t, const Arr5D&, sArray_t<Arr5D>&,
                                                        chol_reader_t&, int, bool);
    template void gw_gradient_t::eval_2bdm_intermediate(size_t, const Arr5D&, sArray_t<Arrv5D>&,
                                                        chol_reader_t&, int, bool);
    template void gw_gradient_t::eval_2bdm_intermediate(size_t, const Arrv5D&, sArray_t<Arr5D>&,
                                                        chol_reader_t&, int, bool);
    template void gw_gradient_t::eval_2bdm_intermediate(size_t, const Arrv5D&, sArray_t<Arrv5D>&,
                                                        chol_reader_t&, int, bool);
    template void gw_gradient_t::eval_2bdm_intermediate(size_t, const Arrv5D2&, sArray_t<Arr5D>&,
                                                        chol_reader_t&, int, bool);
    template void gw_gradient_t::eval_2bdm_intermediate(size_t, const Arrv5D2&, sArray_t<Arrv5D>&,
                                                        chol_reader_t&, int, bool);


  } // namespace solvers

} // namespace methods