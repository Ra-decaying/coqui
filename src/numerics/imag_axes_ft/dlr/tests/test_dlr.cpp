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

#undef NDEBUG

#include "catch2/catch.hpp"
#include "configuration.hpp"
#include "nda/nda.hpp"

#include "utilities/test_common.hpp"
#include "numerics/imag_axes_ft/dlr/dlr_driver.hpp"

namespace bdft_tests {

  using utils::ARRAY_EQUAL;
  template<int N>
  using shape_t = std::array<long, N>;
  
  // Same utility function pattern as cppdlr/test/c++/imfreq_ops.cpp
  // G_ij(t) = sum_l c_ijl K(t, om_ijl)
  nda::matrix<ComplexType> gfun(int norb, double beta, double t) {
    int npeak = 5;

    auto g    = nda::matrix<ComplexType>(norb, norb);
    g         = ComplexType(0.0, 0.0);
    auto c    = nda::vector<double>(npeak);
    double om = 0.0;

    for (int i = 0; i < norb; ++i) {
      for (int j = 0; j < norb; ++j) {
        for (int l = 0; l < npeak; ++l) {
          c(l) = (std::sin(1000.0 * (i + 2 * j + 3 * l + 7)) + 1.0) / 2.0;
        }
        c = c / nda::sum(c);
        
        for (int l = 0; l < npeak; ++l) {
          om = std::sin(2000.0 * (3 * i + 2 * j + l + 6));
          auto k_it = (om >= 0.0)? 
            -std::exp(-(t + 1) * 0.5 * beta * om) / (1.0 + std::exp(-beta * om)) : 
            -std::exp((-t + 1) * 0.5 * beta * om) / (1.0 + std::exp(beta * om));

          g(i, j) += c(l) * k_it;
        }
      }
    }
    return g;
  }

  // Same utility function pattern as cppdlr/test/c++/imfreq_ops.cpp
  // G_ij(iw_n) = sum_l c_ijl K(iw_n, om_ijl)
  nda::matrix<ComplexType> gfun(int norb, double beta, int n) {
    int npeak = 5;

    auto g    = nda::matrix<ComplexType>(norb, norb);
    g         = ComplexType(0.0, 0.0);
    auto c    = nda::vector<double>(npeak);
    double om = 0.0;

    for (int i = 0; i < norb; ++i) {
      for (int j = 0; j < norb; ++j) {
        for (int l = 0; l < npeak; ++l) {
          c(l) = (std::sin(1000.0 * (i + 2 * j + 3 * l + 7)) + 1.0) / 2.0;
        }
        c = c / nda::sum(c);

        for (int l = 0; l < npeak; ++l) {
          om = std::sin(2000.0 * (3 * i + 2 * j + l + 6));
          auto iw_n = ComplexType(0.0, n * M_PI);
          g(i, j) += beta * c(l) / (iw_n - beta * om);
        }
      }
    }
    return g;
  }

  TEST_CASE("cppdlr", "[dlr]") {
    auto test_gfun_roundtrip = [&](double tol) {

      double beta = 100.0;
      double wmax = 1.0;
      double lambda = wmax * beta;
      double eps = 1e-12;
      int norb = 2;

      auto dlr_rf = cppdlr::build_dlr_rf(lambda, eps, cppdlr::SYM);
      
      // Get DLR imaginary time object
      auto it_ops = cppdlr::imtime_ops(lambda, dlr_rf, cppdlr::SYM);
      auto dlr_it = cppdlr::rel2abs(it_ops.get_itnodes());
      auto nts = it_ops.rank();
      auto G_t_ref       = nda::array<ComplexType, 3>(nts, norb, norb);
      // convert from [0, 1] notation to [-1, 1]
      for (int i = 0; i < nts; ++i) { 
        G_t_ref(i, nda::range::all, nda::range::all) = gfun(norb, beta, 2.0 * dlr_it(i) - 1.0); 
      }

      // DLR Matsubara frequency object
      auto if_ops_f = cppdlr::imfreq_ops(lambda, dlr_rf, cppdlr::Fermion, cppdlr::SYM);      
      auto nw_f = if_ops_f.get_ifnodes().size();

      nda::array<ComplexType, 3> G_w_ref(nw_f, norb, norb);
      for (int iw = 0; iw < nw_f; ++iw) {
        G_w_ref(iw, nda::range::all, nda::range::all) = gfun(norb, beta, 2 * if_ops_f.get_ifnodes(iw) + 1);
      }
      app_log(2, "G_t_ref(0, 0, 0) = {}", G_t_ref(0, 0, 0));
      app_log(2, "G_w_ref(0, 0, 0) = {}", G_w_ref(0, 0, 0));

      // Verify tau -> iw recovers G(iw)
      {
        auto G_c = it_ops.vals2coefs(G_t_ref);
        auto G_w = if_ops_f.coefs2vals(beta, G_c);

        app_log(2, "G_w(0, 0, 0) = {}", G_w(0, 0, 0));
        ARRAY_EQUAL(G_w, G_w_ref, tol);

        auto it2if = cppdlr::build_it2if(it_ops, if_ops_f);
        auto G_t_ref_2D = nda::reshape(G_t_ref, std::array<long, 2>{nts, norb * norb});
        auto G_w_2D = nda::reshape(G_w, std::array<long, 2>{nw_f, norb * norb});
        G_w_2D = it2if * nda::matrix_const_view<ComplexType>(G_t_ref_2D) * beta;
        app_log(2, "G_w_2(0, 0, 0) = {}", G_w(0, 0, 0));
        ARRAY_EQUAL(G_w, G_w_ref, tol);
      }

      // Verify iw -> tau recovers G(t)
      {
        auto G_c = if_ops_f.vals2coefs(beta, G_w_ref);
        auto G_t = it_ops.coefs2vals(G_c);
        app_log(2, "G_t(0, 0, 0) = {}", G_t(0, 0, 0));
        ARRAY_EQUAL(G_t, G_t_ref, tol);

        auto if2it = cppdlr::build_if2it(if_ops_f, it_ops);
        auto G_w_ref_2D = nda::reshape(G_w_ref, std::array<long, 2>{nw_f, norb * norb});
        auto G_t_2D = nda::reshape(G_t, std::array<long, 2>{nts, norb * norb});
        G_t_2D = if2it * nda::matrix_const_view<ComplexType>(G_w_ref_2D) / beta;
        app_log(2, "G_t_2(0, 0, 0) = {}", G_t(0, 0, 0));
        ARRAY_EQUAL(G_t, G_t_ref, tol);

        auto it2if = cppdlr::build_it2if(it_ops, if_ops_f);
        nda::array<ComplexType, 3> G_w(nw_f, norb, norb);
        auto G_w_2D = nda::reshape(G_w, std::array<long, 2>{nw_f, norb * norb});
        G_w_2D = it2if * nda::matrix_const_view<ComplexType>(G_t_2D) * beta;
        app_log(2, "G_w_2(0, 0, 0) = {}", G_w(0, 0, 0));
        ARRAY_EQUAL(G_w, G_w_ref, tol);
      }

      // it2if * if2it is not necessary identity in DLR basis. 
      // But the back and forth transformation does not accumulate much error somehow. 
      /*{
        auto it2if = cppdlr::build_it2if(it_ops, if_ops_f);
        auto if2it = cppdlr::build_if2it(if_ops_f, it_ops); 
        auto eye1 = if2it * it2if;
        ARRAY_EQUAL(eye1, nda::eye<ComplexType>(nts), tol);
      }*/

    };

    test_gfun_roundtrip(1e-10);
  }

  TEST_CASE("dlr_dimensions", "[dlr_dimensions]") {
    auto check_dimensions = [&](double beta, double wmax, std::string prec) { 
      
      imag_axes_ft::dlr::DLR mydlr(beta, wmax, prec, true);

      REQUIRE(mydlr.nt_f == mydlr.nt_b);
      REQUIRE(mydlr.nw_f > 0);
      REQUIRE(mydlr.nw_b > 0);

      REQUIRE(mydlr.Ttw_ff.shape() == shape_t<2>{mydlr.nt_f, mydlr.nw_f});
      REQUIRE(mydlr.Twt_ff.shape() == shape_t<2>{mydlr.nw_f, mydlr.nt_f});
      REQUIRE(mydlr.Ttt_bf.shape() == shape_t<2>{mydlr.nt_b, mydlr.nt_f});
      REQUIRE(mydlr.Ttw_bb.shape() == shape_t<2>{mydlr.nt_b, mydlr.nw_b});
      REQUIRE(mydlr.Twt_bb.shape() == shape_t<2>{mydlr.nw_b, mydlr.nt_b});
      REQUIRE(mydlr.Ttt_fb.shape() == shape_t<2>{mydlr.nt_f, mydlr.nt_b});
      REQUIRE(mydlr.T_beta_t_ff.shape() == shape_t<1>{mydlr.nt_f});

    };

    check_dimensions(100.0, 1.2, "high");
    check_dimensions(100.0, 1.2, "medium");
    check_dimensions(100.0, 1.2, "low");
  }

} // bdft_tests
