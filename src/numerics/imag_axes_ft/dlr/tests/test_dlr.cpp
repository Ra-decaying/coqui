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
#include "numerics/imag_axes_ft/iaft_utils.hpp"

namespace bdft_tests {

  using utils::ARRAY_EQUAL;
  template<int N>
  using shape_t = std::array<long, N>;

  TEST_CASE("cppdlr", "[iaft][dlr]") {

    auto test_gfun_roundtrip = [&](double tol, cppdlr::statistic_t stat, bool ph_sym = false) {
      double beta = 100.0;
      double wmax = 1.0;
      double lambda = wmax * beta;
      double eps = 1e-12;
      int norb = 2;

      auto dlr_rf = cppdlr::build_dlr_rf(lambda, eps, cppdlr::SYM);

      struct test_mesh_t {
        int nt_f = 0;
        int nt_b = 0;
        int nw_f = 0;
        int nw_b = 0;
        nda::array<double, 1> tau_mesh_f;
        nda::array<double, 1> tau_mesh_b;
        nda::array<long, 1> wn_mesh_f;
        nda::array<long, 1> wn_mesh_b;
      };
      
      // Get DLR imaginary time object
      auto it_ops = cppdlr::imtime_ops(lambda, dlr_rf, cppdlr::SYM);
      auto nts = it_ops.rank();

      // DLR Matsubara frequency object
      auto if_ops_f = cppdlr::imfreq_ops(lambda, dlr_rf, cppdlr::Fermion, cppdlr::SYM);
      auto if_ops_b = cppdlr::imfreq_ops(lambda, dlr_rf, cppdlr::Boson, cppdlr::SYM);
      auto nw_f = if_ops_f.get_ifnodes().size();
      auto nw_b = if_ops_b.get_ifnodes().size();

      auto mesh = test_mesh_t{
        nts, nts, nw_f, nw_b, 
        nda::array<double, 1>(nts), nda::array<double, 1>(nts), 
        nda::array<long, 1>(nw_f), nda::array<long, 1>(nw_b)
      };
      auto dlr_it = cppdlr::rel2abs(it_ops.get_itnodes());
      for (int it = 0; it < nts; ++it) mesh.tau_mesh_f(it) = 2.0 * dlr_it(it) - 1.0;
      mesh.tau_mesh_b() = mesh.tau_mesh_f;
      for (int iw = 0; iw < nw_f; ++iw) mesh.wn_mesh_f(iw) = long(2 * if_ops_f.get_ifnodes(iw) + 1);
      for (int iw = 0; iw < nw_b; ++iw) mesh.wn_mesh_b(iw) = long(2 * if_ops_b.get_ifnodes(iw));

      auto const iaft_stat = (stat == cppdlr::Fermion) ? imag_axes_ft::fermi : imag_axes_ft::boson;
      auto& if_ops = (stat == cppdlr::Fermion) ? if_ops_f : if_ops_b;
      auto nw = (stat == cppdlr::Fermion) ? nw_f : nw_b;
      auto G_t_ref = imag_axes_ft::test_utils::build_g_tau(mesh, norb, beta, iaft_stat, ph_sym);
      auto G_w_ref = imag_axes_ft::test_utils::build_g_iw(mesh, norb, beta, iaft_stat, ph_sym);
      app_log(2, "G_t_ref(0, 0, 0) = {}", G_t_ref(0, 0, 0));
      app_log(2, "G_w_ref(0, 0, 0) = {}", G_w_ref(0, 0, 0));

      // Verify tau -> iw recovers G(iw)
      {
        auto G_c = it_ops.vals2coefs(G_t_ref);
        auto G_w = if_ops.coefs2vals(beta, G_c);

        app_log(2, "G_w(0, 0, 0) = {}", G_w(0, 0, 0));
        ARRAY_EQUAL(G_w, G_w_ref, tol);

        auto it2if = cppdlr::build_it2if(it_ops, if_ops);
        auto G_t_ref_2D = nda::reshape(G_t_ref, std::array<long, 2>{nts, norb * norb});
        auto G_w_2D = nda::reshape(G_w, std::array<long, 2>{nw, norb * norb});
        G_w_2D = it2if * nda::matrix_const_view<ComplexType>(G_t_ref_2D) * beta;
        app_log(2, "G_w_2(0, 0, 0) = {}", G_w(0, 0, 0));
        ARRAY_EQUAL(G_w, G_w_ref, tol);
      }

      // Verify iw -> tau recovers G(t)
      {
        auto G_c = if_ops.vals2coefs(beta, G_w_ref);
        auto G_t = it_ops.coefs2vals(G_c);
        app_log(2, "G_t(0, 0, 0) = {}", G_t(0, 0, 0));
        ARRAY_EQUAL(G_t, G_t_ref, tol);

        auto if2it = cppdlr::build_if2it(if_ops, it_ops);
        auto G_w_ref_2D = nda::reshape(G_w_ref, std::array<long, 2>{nw, norb * norb});
        auto G_t_2D = nda::reshape(G_t, std::array<long, 2>{nts, norb * norb});
        G_t_2D = if2it * nda::matrix_const_view<ComplexType>(G_w_ref_2D) / beta;
        app_log(2, "G_t_2(0, 0, 0) = {}", G_t(0, 0, 0));
        ARRAY_EQUAL(G_t, G_t_ref, tol);

        auto it2if = cppdlr::build_it2if(it_ops, if_ops);
        nda::array<ComplexType, 3> G_w(nw, norb, norb);
        auto G_w_2D = nda::reshape(G_w, std::array<long, 2>{nw, norb * norb});
        G_w_2D = it2if * nda::matrix_const_view<ComplexType>(G_t_2D) * beta;
        app_log(2, "G_w_2(0, 0, 0) = {}\n", G_w(0, 0, 0));
        ARRAY_EQUAL(G_w, G_w_ref, tol);
      }

      // it2if * if2it is not necessary identity in DLR basis. 
      // But the back and forth transformation does not accumulate much error somehow. 
      /*{
        auto it2if = cppdlr::build_it2if(it_ops, if_ops);
        auto if2it = cppdlr::build_if2it(if_ops, it_ops); 
        auto eye1 = if2it * it2if;
        ARRAY_EQUAL(eye1, nda::eye<ComplexType>(nts), tol);
      }*/
    };

    SECTION("fermion") {
      test_gfun_roundtrip(1e-10, cppdlr::Fermion);
    }

    SECTION("boson") {
      test_gfun_roundtrip(1e-10, cppdlr::Boson); 
    }

  }

  TEST_CASE("dlr_dimensions", "[iaft][dlr]") {
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
