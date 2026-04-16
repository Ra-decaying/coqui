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


#ifndef COQUI_IAFT_UTILS_HPP
#define COQUI_IAFT_UTILS_HPP

#include <string>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "numerics/imag_axes_ft/iaft_enum_e.hpp"

namespace imag_axes_ft {

  class IAFT; // forward declaration

  namespace test_utils {

    /**
     * Construct a Green's function at a given imaginary time with a known analytical form for testing IAFT. 
     * The convention of t is [-1, 1] which maps to the physical imaginary time [0, beta]. 
     */
    nda::matrix<ComplexType> gfun_tau(int norb, double beta, double t, bool ph_sym = false);

    /**
     * Construct a Green's function at a given Matsubara frequency with a known analytical form for testing IAFT.
     * The convention for n is that n = (2k + 1) for fermions and n = 2k for bosons, where k is an integer. 
     */
    nda::matrix<ComplexType> gfun_iw(int norb, double beta, int n, imag_axes_ft::stats_e stat, bool ph_sym = false);

    namespace {
      // CNY: helper functions to handle grid types with different interfaces for testing purposes. 
      //      These are a bit too clunky though...

      template<typename grid_t>
      int nt_f(grid_t const &grid) {
        if constexpr (requires { grid.nt_f(); }) {
          return grid.nt_f();
        } else {
          return grid.nt_f;
        }
      }

      template<typename grid_t>
      int nt_b(grid_t const &grid) {
        if constexpr (requires { grid.nt_b(); }) {
          return grid.nt_b();
        } else {
          return grid.nt_b;
        }
      }

      template<typename grid_t>
      int nw_f(grid_t const &grid) {
        if constexpr (requires { grid.nw_f(); }) {
          return grid.nw_f();
        } else {
          return grid.nw_f;
        }
      }

      template<typename grid_t>
      int nw_b(grid_t const &grid) {
        if constexpr (requires { grid.nw_b(); }) {
          return grid.nw_b();
        } else {
          return grid.nw_b;
        }
      }

      template<typename grid_t>
      decltype(auto) tau_mesh_f(grid_t const &grid) {
        if constexpr (requires { grid.tau_mesh(); }) {
          return grid.tau_mesh();
        } else if constexpr (requires { grid.tau_mesh_f(); }) {
          return grid.tau_mesh_f();
        } else {
          return grid.tau_mesh_f;
        }
      }

      template<typename grid_t>
      decltype(auto) tau_mesh_b(grid_t const &grid) {
        if constexpr (requires { grid.tau_mesh_b(); }) {
          return grid.tau_mesh_b();
        } else {
          return grid.tau_mesh_b;
        }
      }

      template<typename grid_t>
      decltype(auto) wn_mesh_f(grid_t const &grid) {
        if constexpr (requires { grid.wn_mesh(); }) {
          return grid.wn_mesh();
        } else if constexpr (requires { grid.wn_mesh_f(); }) {
          return grid.wn_mesh_f();
        } else {
          return grid.wn_mesh_f;
        }
      }

      template<typename grid_t>
      decltype(auto) wn_mesh_b(grid_t const &grid) {
        if constexpr (requires { grid.wn_mesh_b(); }) {
          return grid.wn_mesh_b();
        } else {
          return grid.wn_mesh_b;
        }
      }

    } // namespace

    /**
     * Build Green's function on the imaginary time mesh with a known analytical form for testing IAFT.
     * The convention of t is [-1, 1] which maps to the physical imaginary time [0, beta].
     */
    template<typename grid_t>
    nda::array<ComplexType, 3> build_g_tau(
      grid_t const &grid, int norb, double beta, imag_axes_ft::stats_e stat, bool ph_sym = false) {
      auto nts = (stat == imag_axes_ft::fermi) ? nt_f(grid) : nt_b(grid);
      auto tau_mesh = (stat == imag_axes_ft::fermi) ? tau_mesh_f(grid) : tau_mesh_b(grid);

      // only construct the first half of the tau points for particle-hole symmetry. 
      if (ph_sym) nts = (nts % 2 == 0) ? nts / 2 : nts / 2 + 1; 

      auto G_t_ref = nda::array<ComplexType, 3>(nts, norb, norb);
      for (int it = 0; it < nts; ++it) {
        G_t_ref(it, nda::range::all, nda::range::all) = gfun_tau(norb, beta, tau_mesh(it), ph_sym);
      }

      return G_t_ref;
    }

    template<typename grid_t>
    nda::array<ComplexType, 3> build_g_iw(
      grid_t const &grid, int norb, double beta, imag_axes_ft::stats_e stat, bool ph_sym = false) {

      auto nw = (stat == imag_axes_ft::fermi) ? nw_f(grid) : nw_b(grid);
      auto wn_mesh = (stat == imag_axes_ft::fermi) ? wn_mesh_f(grid) : wn_mesh_b(grid);

      // only construct positive Matsubara frequencies for particle-hole symmetry. 
      int iw_shift = (ph_sym)? nw / 2 : 0;
      if (ph_sym) nw = (nw % 2 == 0) ? nw / 2 : nw / 2 + 1; 

      auto G_w_ref = nda::array<ComplexType, 3>(nw, norb, norb);
      for (int iw = 0; iw < nw; ++iw) {
        G_w_ref(iw, nda::range::all, nda::range::all) =
          gfun_iw(norb, beta, int(wn_mesh(iw+iw_shift)), stat, ph_sym);
      }

      return G_w_ref;
    }
  } // namespace test_utils

  /**
   * Reconstruct IAFT object from the metadata in bdft scf output
   * @return IAFT
   */
  IAFT read_iaft(std::string scf_file, bool print_meta_log = true);

} // imag_axes_ft



#endif //COQUI_IAFT_UTILS_HPP
