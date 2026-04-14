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
#include "nda/h5.hpp"

#include <cmath>

#include "utilities/mpi_context.h"
#include "utilities/test_common.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"

namespace bdft_tests {

  using utils::ARRAY_EQUAL;
  template<int N>
  using shape_t = std::array<long,N>;

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

  TEST_CASE("iaft_ir_read", "[iaft_ir_read]") {
    double beta = 1000;
    double wmax = 12.0;
    {
      imag_axes_ft::ir::IR myir(beta, wmax);
      imag_axes_ft::IAFT iaft(myir);
    }
    imag_axes_ft::IAFT iaft(beta, wmax, imag_axes_ft::ir_source);

    REQUIRE(iaft.beta() == beta);
    REQUIRE(iaft.nt_f() == 137);
    REQUIRE(iaft.nw_f() == 138);
    REQUIRE(iaft.nt_b() == 137);
    REQUIRE(iaft.nw_b() == 137);

    REQUIRE(iaft.Ttw_ff().shape() == shape_t<2>{iaft.nt_f(), iaft.nw_f()});
    REQUIRE(iaft.Twt_ff().shape() == shape_t<2>{iaft.nw_f(), iaft.nt_f()});
    REQUIRE(iaft.Ttt_bf().shape() == shape_t<2>{iaft.nt_b(), iaft.nt_f()});
    REQUIRE(iaft.Ttw_bb().shape() == shape_t<2>{iaft.nt_b(), iaft.nw_b()});
    REQUIRE(iaft.Twt_bb().shape() == shape_t<2>{iaft.nw_b(), iaft.nt_b()});
    REQUIRE(iaft.Ttt_fb().shape() == shape_t<2>{iaft.nt_f(), iaft.nt_b()});
    REQUIRE(iaft.T_beta_t_ff().shape() == shape_t<1>{iaft.nt_f()});

    auto eye1 = iaft.Ttw_ff() * iaft.Twt_ff();
    auto eye2 = iaft.Ttt_bf() * iaft.Ttt_fb();
    ARRAY_EQUAL(eye1, nda::eye<ComplexType>(iaft.nt_f()), 1e-10);
    ARRAY_EQUAL(eye2, nda::eye<RealType>(iaft.nt_b()), 1e-10);
  }

  TEST_CASE("iaft_ir_ft", "[iaft_ir_ft]") {
    decltype(nda::range::all) all;

    std::string source_path = std::string(PROJECT_SOURCE_DIR)+"/tests/unit_test_files/pyscf/si_kp222_krhf/";

    auto test_iaft = [&](double beta, double wmax, std::string prec, double tol) {

      imag_axes_ft::IAFT myft(beta, wmax, imag_axes_ft::ir_source, prec, true);

      std::string filename = source_path+"/gw_Gw_Gt_beta"+std::to_string(int(beta))+"_wmax"+std::format("{:.1f}", wmax)+"_"+prec+".h5";
      h5::file file(filename, 'r');
      h5::group grp(file);

      nda::array<std::complex<double>, 5> G_tskij_ref;
      nda::array<std::complex<double>, 5> G_wskij_ref;
      nda::array<std::complex<double>, 4> Dm_skij_ref;
      nda::h5_read(grp, "Gt", G_tskij_ref);
      nda::h5_read(grp, "Gw", G_wskij_ref);
      nda::h5_read(grp, "Dm", Dm_skij_ref);

      size_t nts   = G_tskij_ref.shape(0);
      size_t nw    = G_wskij_ref.shape(0);
      size_t ns    = G_tskij_ref.shape(1);
      size_t nkpts = G_tskij_ref.shape(2);
      size_t nbnd  = G_tskij_ref.shape(3);
      // Fourier transform between tau and iwn
      {
        nda::array<std::complex<double>, 5> G_tskij(nts, ns, nkpts, nbnd, nbnd);
        nda::array<std::complex<double>, 5> G_wskij(nw, ns, nkpts, nbnd, nbnd);
        myft.tau_to_w(G_tskij_ref, G_wskij, imag_axes_ft::fermi);
        myft.w_to_tau(G_wskij_ref, G_tskij, imag_axes_ft::fermi);

        ARRAY_EQUAL(G_tskij, G_tskij_ref, tol);
        ARRAY_EQUAL(G_wskij, G_wskij_ref, tol);
      }
      // tau to a specific w
      {
        nda::array<std::complex<double>, 5> G_wskij(nw, ns, nkpts, nbnd, nbnd);
        nda::array<std::complex<double>, 4> G_skij(ns, nkpts, nbnd, nbnd);
        for (size_t n = 0; n < nw; ++n) {
          nda::array_view<std::complex<double>, 4> Gw_skij({ns, nkpts, nbnd, nbnd},
                                                           G_wskij.data() + n*ns*nkpts*nbnd*nbnd);
          myft.tau_to_w(G_tskij_ref, G_skij, imag_axes_ft::fermi, n);
          Gw_skij = G_skij;
        }
        ARRAY_EQUAL(G_wskij, G_wskij_ref, tol);
      }
      // Partial Fourier transform
      {
        nda::array<std::complex<double>, 5> G_tskij(nts, ns, nkpts, nbnd, nbnd);
        for (size_t n = 0; n < nw; ++n) {
          auto Gw_skij = G_wskij_ref(n, all, all, all, all);
          myft.w_to_tau_partial(Gw_skij, G_tskij, imag_axes_ft::fermi, n);
        }
        ARRAY_EQUAL(G_tskij, G_tskij_ref, tol);
      }
      // tau = beta^{-} via the interpolation at sparse sampling nodes
      {
        nda::array<std::complex<double>, 4> Dm_skij(ns, nkpts, nbnd, nbnd);
        myft.tau_to_beta(G_tskij_ref, Dm_skij);
        Dm_skij *= -1.0;
        ARRAY_EQUAL(Dm_skij, Dm_skij_ref, tol);
      }
    };

    double beta = 1000;
    double wmax = 1.2;
    SECTION("prec_high") {
      test_iaft(beta, wmax, "high", 1e-11);
    }
    SECTION("prec_medium") {
      test_iaft(beta, wmax, "medium", 1e-9);
    }
  }

  TEST_CASE("iaft_gfun_tau_w_roundtrip", "[iaft_gfun_tau_w_roundtrip]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto test_gfun_roundtrip = [&](imag_axes_ft::source_e source, double tol) {
      double beta = 1000.0;
      double wmax = 1.0;
      int norb = 2;

      imag_axes_ft::IAFT myft(beta, wmax, source, "high", true);

      auto tau_mesh = myft.tau_mesh();
      auto wn_mesh = myft.wn_mesh();

      nda::array<ComplexType, 3> G_t_ref(myft.nt_f(), norb, norb);
      nda::array<ComplexType, 3> G_w_ref(myft.nw_f(), norb, norb);

      for (int it = 0; it < myft.nt_f(); ++it) {
        G_t_ref(it, nda::range::all, nda::range::all) = gfun(norb, beta, tau_mesh(it));
      }
      for (int iw = 0; iw < myft.nw_f(); ++iw) {
        G_w_ref(iw, nda::range::all, nda::range::all) = gfun(norb, beta, int(wn_mesh(iw)));
      }
      myft.check_leakage(G_t_ref, imag_axes_ft::fermi, std::addressof(mpi_context->comm), "G_t_ref");

      // Verify tau -> iw recovers G(iw)
      {
        nda::array<ComplexType, 3> G_w(myft.nw_f(), norb, norb);
        myft.tau_to_w(G_t_ref, G_w, imag_axes_ft::fermi);
        ARRAY_EQUAL(G_w, G_w_ref, tol);
      }

      // Verify iw -> tau recovers G(t)
      {
        nda::array<ComplexType, 3> G_t(myft.nt_f(), norb, norb);
        myft.w_to_tau(G_w_ref, G_t, imag_axes_ft::fermi);
        ARRAY_EQUAL(G_t, G_t_ref, tol);
      }

      // Verify tau -> tau = beta^- and 0^+
      {
        nda::array<ComplexType, 2> Dm_skij(norb, norb);
        myft.tau_to_beta(G_t_ref, Dm_skij);
        auto Dm_skij_ref = gfun(norb, beta, 1.0);
        app_log(2, "Dm_skij_ref(0, 0) = {}", Dm_skij_ref(0, 0));
        app_log(2, "Dm_skij(0, 0) = {}", Dm_skij(0, 0));
        ARRAY_EQUAL(Dm_skij, Dm_skij_ref, tol);

        myft.tau_to_zero(G_t_ref, Dm_skij);
        Dm_skij_ref() = gfun(norb, beta, -1.0);
        app_log(2, "I - Dm_skij_ref(0, 0) = {}", Dm_skij_ref(0, 0));
        app_log(2, "I - Dm_skij(0, 0) = {}", Dm_skij(0, 0));
        ARRAY_EQUAL(Dm_skij, Dm_skij_ref, tol);
      }
    };

    SECTION("ir_backend") {
      test_gfun_roundtrip(imag_axes_ft::ir_source, 1e-10);
    }

#ifdef ENABLE_DLR
    SECTION("dlr_backend") {
      test_gfun_roundtrip(imag_axes_ft::dlr_source, 1e-10);
    }
#endif
  }

} // bdft_tests
