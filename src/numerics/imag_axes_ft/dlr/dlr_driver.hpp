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

#ifndef COQUI_DLR_GENERATOR_HPP
#define COQUI_DLR_GENERATOR_HPP

#ifdef ENABLE_DLR

#include <string>

#include "nda/nda.hpp"
#include "nda/lapack.hpp"

#include "numerics/nda_functions.hpp"
// FIXME hacky solution to updates nda namespace until cppdlr's nda dependency is updated. 
#include "numerics/imag_axes_ft/dlr/nda_linalg_compat.hpp"
// FIXME avoid including full cppdlr header until cppdlr's nda dependency is updated. 
//#include "cppdlr/cppdlr.hpp"
#include "cppdlr/dlr_build.hpp"
#include "cppdlr/dlr_imtime.hpp"
#include "cppdlr/dlr_imfreq.hpp"

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

namespace imag_axes_ft {
  namespace dlr {

    inline nda::matrix<ComplexType> build_if2it(cppdlr::imfreq_ops const &ifops, cppdlr::imtime_ops const &itops) {
      auto nw = ifops.get_ifnodes().size();
      auto dlr_rank = itops.rank();

      auto if2it = nda::matrix<ComplexType>(dlr_rank, nw);

      // Solve cf2if^T C^T = cf2it^T to obtain C = cf2it * cf2if^{-1}
      
      // Copy cf2it into output matrix (cast to complex)
      if2it(nda::range::all, nda::range(dlr_rank)) = itops.get_cf2it();

      if (nw == dlr_rank) {
        // Square system---use getrs
        auto lu  = nda::matrix<ComplexType>(ifops.get_if2cf_lu());
        auto piv = nda::vector<int>(ifops.get_if2cf_piv());
        nda::lapack::getrs(nda::transpose(lu), nda::transpose(if2it), piv);
      } else {
        // Non-square system---use least squares solver
        auto s       = nda::vector<double>(dlr_rank); // Not needed
        double rcond = 0;                             // Not needed
        int rank     = 0;                             // Not needed
        auto cf2if = nda::matrix<ComplexType>(ifops.get_cf2if());
        nda::lapack::gelss(nda::transpose(cf2if), nda::transpose(if2it), s, rcond, rank);
      }

      return if2it;
    }

    struct DLR {
    public:
      DLR() = default;

      // TODO overload constructor to parse eps directly
      DLR(double beta_, double wmax_, std::string prec_ = "high", bool print_meta_log = false):
          beta(beta_), wmax(wmax_), prec(prec_) {

        lambda = beta * wmax;
        utils::check(lambda > 0.0, "Invalid value of lambda, {}, in imag_axes_ft::dlr::DLR.", lambda);
        utils::check(beta > 0.0, "Invalid value of beta, {}, in imag_axes_ft::dlr::DLR.", beta);
        utils::check(wmax > 0.0, "Invalid value of wmax, {}, in imag_axes_ft::dlr::DLR.", wmax);

        if (prec == "high") {
          eps = 1e-13;
        } else if (prec == "medium") {
          eps = 1e-10;
        } else if (prec == "low") {
          eps = 1e-06;
        } else {
          utils::check(false, 
            "imag_axes_ft::dlr: prec = {} is not acceptable. Acceptable list = \"high\", \"medium\", \"low\"", 
            prec);
        }

        // Construct DLR basis by obtaining DLR real frequencies 
        auto dlr_rf = cppdlr::build_dlr_rf(lambda, eps, cppdlr::SYM);
        auto dlr_rank = dlr_rf.size();

        // get DLR imaginary time object
        auto it_ops = cppdlr::imtime_ops(lambda, dlr_rf, cppdlr::SYM);
        utils::check(dlr_rank == it_ops.rank(), 
        "imag_axes_ft::dlr::DLR: size of DLR real frequency basis, {}, does not match the rank of DLR imaginary time object, {}", 
        dlr_rank, it_ops.rank());
        // dlr imagianry time interpolation nodes. 
        // TODO keep only one set of nt and tau_mesh for both fermions and bosons
        nt_f = it_ops.rank();
        nt_b = it_ops.rank();
        tau_mesh_f = nda::array<double, 1>(nt_f);
        tau_mesh_b = nda::array<double, 1>(nt_b);
        // The native DLR tau nodes live in [0, 1], rather than [0, beta] nor [-1, 1]
        // Convert to the IR basis notation which lives in [-1, 1] instead...
        for (int i = 0; i < nt_f; ++i) {
          tau_mesh_f(i) = 2.0 * cppdlr::rel2abs(it_ops.get_itnodes(i)) - 1.0;
        }
        tau_mesh_b() = tau_mesh_f;

        // get DLR imaginary frequency objects
        auto if_ops_f = cppdlr::imfreq_ops(lambda, dlr_rf, cppdlr::Fermion, cppdlr::SYM);
        auto if_ops_b = cppdlr::imfreq_ops(lambda, dlr_rf, cppdlr::Boson, cppdlr::SYM);

        nw_f = if_ops_f.get_ifnodes().size();
        nw_b = if_ops_b.get_ifnodes().size();
        wn_mesh_f = nda::array<long, 1>(nw_f);
        wn_mesh_b = nda::array<long, 1>(nw_b);
        // Convert to IR basis notation
        for (int i = 0; i < nw_f; ++i) wn_mesh_f(i) = long(2.0 * if_ops_f.get_ifnodes(i) + 1);
        for (int i = 0; i < nw_b; ++i) wn_mesh_b(i) = long(2.0 * if_ops_b.get_ifnodes(i));

        // Matsubara frequency to imaginary time
        auto if2it_f = cppdlr::build_if2it(if_ops_f, it_ops) / beta;
        auto if2it_b = cppdlr::build_if2it(if_ops_b, it_ops) / beta;
        Ttw_ff = nda::matrix<ComplexType>(nt_f, nw_f);
        Ttw_ff() = if2it_f;
        Ttw_bb = nda::matrix<ComplexType>(nt_b, nw_b);
        Ttw_bb() = if2it_b;
        // imaginary time to Matsubara frequency
        auto it2if_f = cppdlr::build_it2if(it_ops, if_ops_f) * beta;
        auto it2if_b = cppdlr::build_it2if(it_ops, if_ops_b) * beta;
        Twt_ff = nda::matrix<ComplexType>(nw_f, nt_f);
        Twt_ff() = it2if_f;
        Twt_bb = nda::matrix<ComplexType>(nw_b, nt_b);
        Twt_bb() = it2if_b;

        // fermion <-> boson tau interpolation (same DLR tau nodes)
        Ttt_bf = nda::eye<ComplexType>(nt_b);
        Ttt_fb = nda::eye<ComplexType>(nt_f); 

        // tau -> coefficients
        auto it2cf  = it_ops.vals2coefs(nda::eye<ComplexType>(nt_f));
        Tct_ff = it2cf;
        Tct_bb = it2cf;

        std::tie(T_zero_t_ff, T_beta_t_ff) = construct_tau_to_zero_and_beta_matrices(it_ops);

        if (print_meta_log) metadata_log();
      }

      DLR(const DLR &other) = default;
      DLR(DLR &&other) = default;

      DLR &operator=(const DLR &other) = default;
      DLR &operator=(DLR &&other) = default;

      ~DLR() = default;

      void metadata_log() const {
        std::string prec_prefix;
        if (prec == "high") {
          prec_prefix = "1e-13";
        } else if (prec == "medium") {
          prec_prefix = "1e-10";
        } else if (prec == "low") {
          prec_prefix = "1e-06";
        } else {
          utils::check(false,
                       "imag_axes_ft::dlr: prec = {} is not acceptable. Acceptable list = \"high\", \"medium\", \"low\"",
                       prec);
        }

        app_log(1, "  Mesh details on the imaginary axis");
        app_log(1, "  ----------------------------------");
        app_log(1, "  Discrete Lehmann Representation");
        app_log(1, "  Beta                   = {} a.u.", beta);
        app_log(1, "  Frequency cutoff       = {} a.u.", wmax);
        app_log(1, "  Lambda                 = {}", lambda);
        app_log(1, "  Precision              = {}", prec_prefix);
        app_log(1, "  nt_f, nt_b, nw_f, nw_b = {}, {}, {}, {}\n", nt_f, nt_b, nw_f, nw_b);
      }

    auto construct_tau_to_zero_and_beta_matrices(cppdlr::imtime_ops const &it_ops) 
    -> std::pair<nda::array<ComplexType, 1>, nda::array<ComplexType, 1>> {
      // Matrix that interpolate to tau = 0^{+} and beta^{-}
      // In cppdlr, t = [0, 0.5] + [-0.5, 0.0]
      auto nts = it_ops.rank();
      auto cf2zero = it_ops.coefs2eval(nda::eye<ComplexType>(nts), 0.0);
      auto cf2beta = it_ops.coefs2eval(nda::eye<ComplexType>(nts), -1e-12);

      // Build cf2it and solve cf2it^T * x = cf2eval with LU factors, avoiding explicit (cf2it)^{-1}.
      auto cf2it_t_lu = nda::matrix<ComplexType>(nda::transpose(it_ops.get_cf2it()));

      auto rhs_f = nda::matrix<ComplexType, nda::F_layout>(nts, 2);
      for (int it = 0; it < nts; ++it) {
        rhs_f(it, 0) = cf2zero(it);
        rhs_f(it, 1) = cf2beta(it);
      }

      nda::array<int, 1> ipiv(nts);
      auto info = nda::lapack::getrf(cf2it_t_lu, ipiv);
      utils::check(info == 0,
                   "dlr_driver.hpp::DLR: LU factorization failed for cf2it^T (getrf info = {}).",
                   info);
      info = nda::lapack::getrs(cf2it_t_lu, rhs_f, ipiv);
      utils::check(info == 0,
                   "dlr_driver.hpp::DLR: Linear solve failed for cf2it^T x = cf2eval (getrs info = {}).",
                   info);

      auto T_zero_t = nda::array<ComplexType, 1>(nts);
      auto T_beta_t = nda::array<ComplexType, 1>(nts);
      for (int it = 0; it < nts; ++it) {
        T_zero_t(it) = rhs_f(it, 0);
        T_beta_t(it) = rhs_f(it, 1);
      }

      return std::make_pair(T_zero_t, T_beta_t);
    }

    public:
      double beta = 0.0;
      double wmax = 0.0;
      std::string prec = "high";
      double eps = 0.0;
      double lambda = 0.0;

      int nt_f = 0;
      int nt_b = 0;
      int nw_f = 0;
      int nw_b = 0;

      // Matsubara frequency mesh
      nda::array<long, 1> wn_mesh_f;
      nda::array<long, 1> wn_mesh_b;
      // tau mesh: relative format (same convention as IR backend)
      nda::array<double, 1> tau_mesh_f;
      // TODO tau_mesh_b as an array view of tau_mesh_f since they are the same. 
      nda::array<double, 1> tau_mesh_b;

      // transformation from w_f to t_f
      nda::matrix<ComplexType> Ttw_ff;
      // transformation from t_f to w_f
      nda::matrix<ComplexType> Twt_ff;
      // transformation from w_b to t_b
      nda::matrix<ComplexType> Ttw_bb;
      // transformation from t_b to w_b
      nda::matrix<ComplexType> Twt_bb;
      // interpolation from t_f to t_b
      nda::matrix<ComplexType> Ttt_bf;
      // interpolation from t_b to t_f
      nda::matrix<ComplexType> Ttt_fb;
      // interpolation matrix to t_f = beta^{-}
      nda::array<ComplexType, 1> T_beta_t_ff;
      // interpolation matrix to t_f = 0^{+}
      nda::array<ComplexType, 1> T_zero_t_ff;
      // tau to DLR coefficients
      nda::matrix<ComplexType> Tct_ff;
      nda::matrix<ComplexType> Tct_bb;
    };
  } // dlr
} // imag_axes_ft

#endif // ENABLE_DLR
#endif // COQUI_DLR_GENERATOR_HPP
