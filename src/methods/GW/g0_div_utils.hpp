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


#ifndef COQUI_G0_DIV_UTILS_HPP
#define COQUI_G0_DIV_UTILS_HPP

#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/h5.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/distributed_array/nda.hpp"

#include "IO/app_loggers.h"
#include "utilities/check.hpp"

#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/ERI/div_treatment_e.hpp"
#include "mean_field/MF.hpp"

namespace methods {
  namespace solvers {
    /**
     * Utility functions for GW divergence treatments
     */
    struct div_utils {

      template<nda::ArrayOfRank<2> Array_q_t>
      static auto filter_qpts(Array_q_t &&Qpts, double threshold, int order, bool two_dim=false) {
        std::vector<int> q_indices;
        std::set<double> q_set{};
        for (int i=0; i<Qpts.shape(0); ++i) {
          auto qpt = Qpts(i,nda::range::all);
          double norm = std::sqrt( qpt(0)*qpt(0) + qpt(1)*qpt(1) + qpt(2)*qpt(2) );
          norm = std::pow(norm, order);
          if (norm > 0.0 && norm <= threshold && (!two_dim or qpt(2)==0.0)) {
            auto insert_pair = q_set.insert(norm);
            if (insert_pair.second) {
              q_indices.push_back(i);
            }
          }
        }
        return q_indices;
      }

      /**
       * Estimate the inverse of symmetric dielectric function at q=0.  
       * @param eps_inv_wq    - [INPUT] inverse dielectric function on frequency axis, shape [niw, nqpts_ibz]
       * @param mf            - [INPUT] mean-field object
       * @param div_treatment - [INPUT] divergence treatment method
       * @param poly_in_q2    - [INPUT] whether to use polynomial in q^2 (default: false)
       * @param q_max         - [INPUT] maximum q magnitude for filtering (default: 0.8)
       * @param fit_order     - [INPUT] polynomial fit order (default: 2)
       * @return - inverse dielectric function at q=0, shape [niw]
       */
      template<nda::ArrayOfRank<2> Array_axis_t>
      static auto extrapolate_eps_inv_q0(const Array_axis_t &eps_inv_wq,
                                          mf::MF &mf,
                                          div_treatment_e div_treatment=gygi_extrplt,
                                          bool poly_in_q2=false, double q_max=0.8, int fit_order=2)
      -> nda::array<ComplexType, 1> {
        
        auto [niw, nqpts_ibz] = eps_inv_wq.shape();
        nda::array<ComplexType, 1> eps_inv_q0_w(niw);
        eps_inv_q0_w() = 0.0;

        nda::array<ComplexType, 1> Q_abs2(nqpts_ibz);
        nda::array<ComplexType, 1> Q_abs(nqpts_ibz);
        for (int iq = 0; iq < nqpts_ibz; ++iq) {
          auto qpt = mf.Qpts_ibz(iq);
          Q_abs2(iq) = ComplexType( qpt(0)*qpt(0) + qpt(1)*qpt(1) + qpt(2)*qpt(2) );
          Q_abs(iq) = ComplexType( std::sqrt( qpt(0)*qpt(0) + qpt(1)*qpt(1) + qpt(2)*qpt(2) ) );
        }

        if (div_treatment==gygi_extrplt or div_treatment==gygi_extrplt_2d) {
          utils::check(mf.nkpts_ibz() > 1, "extrapolate_eps_inv_q0: nkpts_ibz <= 1 while div_treatment==gygi_extrplt");
          bool two_dim = (div_treatment==gygi_extrplt_2d)? true : false;
          
          auto q_indices = (poly_in_q2)? filter_qpts(mf.Qpts_ibz(), q_max, 2, two_dim)
                           : filter_qpts(mf.Qpts_ibz(), q_max, 1, two_dim);
          nda::array<ComplexType, 1> Q_filtered(q_indices.size());
          nda::array<ComplexType, 2> eps_inv_filtered(niw, q_indices.size());
          
          for (size_t i = 0; i < q_indices.size(); ++i) {
            Q_filtered(i) = (poly_in_q2)? Q_abs2( q_indices[i] ) : Q_abs( q_indices[i] );
            eps_inv_filtered(nda::range::all, i) = eps_inv_wq(nda::range::all, q_indices[i] );
          }
          
          if ( q_indices.size() < 3) {
            // Choose the closest point to the gamma instead
            eps_inv_q0_w = eps_inv_wq(nda::range::all, q_indices[0]);
          } else {
            if (two_dim)
              app_log(2, "  Extrapolate head of the inverse of the dielectric function from {} q-points on the xy-plane", q_indices.size());
            else
              app_log(2, "  Extrapolate head of the inverse of the dielectric function from {} q-points", q_indices.size());
            
            for (int n = 0; n < niw; ++n) {
              eps_inv_q0_w(n) = extrapolate_to_q0(Q_filtered, eps_inv_filtered(n, nda::range::all), fit_order,
                                                  (n==0)? true : false);
            }
          }
        } else if (div_treatment==gygi_linear or div_treatment==gygi_linear_2d) {
          // Linear extrapolation to q=0 as A + B*q^2 using the two closest q-points to Gamma
          auto closest_two_indices = find_two_closest_per_direction(mf.Qpts_ibz(), mf.recv());
          
          bool two_dim = (div_treatment==gygi_linear_2d)? true : false;
          if (two_dim)
            app_log(2, "  Linear extrapolate head of the inverse of the dielectric function (A + B*q^2)\n"
                       "  using the closet two points along b1 and b2 directions");
          else
            app_log(2, "  Linear extrapolate head of the inverse of the dielectric function (A + B*q^2)\n"
                       "  using the closet two points along b1, b2, and b3 directions");
          
          int dim = 0;
          for (int dir = 0; dir < 3; ++dir) {
            
            if (two_dim and dir == 2) continue; // skip b3 direction for 2D materials
            
            nda::array<ComplexType, 1> Q_filtered(closest_two_indices[dir].size());
            nda::array<ComplexType, 2> eps_inv_filtered(niw, closest_two_indices[dir].size());
            
            for (size_t i = 0; i < closest_two_indices[dir].size(); ++i) {
              Q_filtered(i) = Q_abs2( closest_two_indices[dir][i] );
              eps_inv_filtered(nda::range::all, i) = eps_inv_wq(nda::range::all, closest_two_indices[dir][i] );
            }
            
            if (closest_two_indices[dir].size() == 0) {
              continue; // no point found along this direction, skip extrapolation
            } else if (closest_two_indices[dir].size() == 1) {
              // Choose the closest point to the gamma instead of linear extrapolation
              app_log(2, "Only found {} point along b{} direction --> use it directly without extrapolation.", 
                      closest_two_indices[dir].size(), dir+1);
              eps_inv_q0_w += eps_inv_filtered(nda::range::all, 0);
            } else {
              for (int n = 0; n < niw; ++n) {
                eps_inv_q0_w(n) += extrapolate_to_q0(Q_filtered, eps_inv_filtered(n, nda::range::all), 1,
                                                     (n==0)? true : false);
              }
            }
            dim += 1;
          }
          utils::check(dim > 0, "extrapolate_eps_inv_q0: no valid q-point found for extrapolation in any direction");
          eps_inv_q0_w /= static_cast<double>(dim); // average over the number of dimensions extrapolated

        } else if (div_treatment==gygi_average) {
          // Average over all q-points with smallest |q|
          auto smallest_indices = find_smallest_qabs_indices(mf.Qpts_ibz(), false, false);
          eps_inv_q0_w = 0.0;
          for (int idx : smallest_indices) {
            eps_inv_q0_w += eps_inv_wq(nda::range::all, idx);
          }
          eps_inv_q0_w /= smallest_indices.size();
        } else {
          int smallest_idx = find_smallest_qabs(mf.Qpts_ibz(), false);
          eps_inv_q0_w = eps_inv_wq(nda::range::all, smallest_idx);
        }

        return eps_inv_q0_w;
      }

      /**
       * Estimate the inverse of symmetric dielectric function in plane-wave basis at G=G'=0
       * on the imaginary-time axis
       * @tparam extrapolate - perform extrapolation or not
       * @param dW_wqPQ - [INPUT] screened interaction in the THC auxiliary basis on
       *                          the imaginary-time or Matsubara frequency axis.
       * @param thc - [INPUT] THC-ERI
       * @param mf - [INPUT] mean-field object
       * @return - inverse of symmetric dielectric function at finite q and q = 0
       */
      template<nda::MemoryArrayOfRank<4> Array_w_t, THC_ERI thc_t, typename communicator_t>
      static auto eps_inv_head_t(memory::darray_t<Array_w_t, communicator_t> &dW_tqPQ, thc_t &thc,
                                 mf::MF &mf, const imag_axes_ft::IAFT *ft,
                                 div_treatment_e div_treatment=gygi_extrplt,
                                 bool poly_in_q2=false, double q_max=0.8, int fit_order=2)
      -> std::tuple<nda::array<ComplexType, 2>, nda::array<ComplexType, 1> > {

        // Compute the head (i.e. G=G'=0 component) of the inverse dielectric function at finite q
        auto eps_inv_t = eval_eps_inv_q(dW_tqPQ, thc, mf);
        auto [nts, nqpts_ibz] = eps_inv_t.shape();

        if (thc.MF()->nqpts_ibz() == 1 and div_treatment != ignore_g0) {
          app_log(2, "eps_inv_head_t: nqpts_ibz == 1 while div_treatment != ignore. "
                     "CoQui will take div_treatment = ignore_g0 anyway!");
          div_treatment = ignore_g0;
        }

        // Convert to Matsubara frequency axis for extrapolation
        long nw = (ft->nw_b()%2==0)? ft->nw_b()/2 : ft->nw_b()/2 + 1;
        nda::array<ComplexType, 2> eps_inv_w(nw, nqpts_ibz);
        ft->tau_to_w_PHsym(eps_inv_t, eps_inv_w);

        // Perform q=0 extrapolation on frequency domain data
        nda::array<ComplexType, 1> eps_inv_q0_w = 
          extrapolate_eps_inv_q0(eps_inv_w, mf, div_treatment, poly_in_q2, q_max, fit_order);

        // Convert result back to time domain
        nda::array<ComplexType, 1> eps_inv_q0_t(nts);
        auto eps_inv_q0_w_2D = nda::reshape(eps_inv_q0_w, std::array<long,2>{nw, 1});
        auto eps_inv_q0_t_2D = nda::reshape(eps_inv_q0_t, std::array<long,2>{nts,1});
        ft->w_to_tau_PHsym(eps_inv_q0_w_2D, eps_inv_q0_t_2D);

        return std::make_tuple(std::move(eps_inv_t), std::move(eps_inv_q0_t));
      }

      /**
       * Estimate the inverse of symmetric dielectric function in plane-wave basis at G=G'=0
       * on the Matsubara frequency axis
       * @tparam extrapolate - perform extrapolation or not
       * @param dW_wqPQ - [INPUT] screened interaction in the THC auxiliary basis on
       *                          the imaginary-time or Matsubara frequency axis.
       * @param thc - [INPUT] THC-ERI
       * @param mf - [INPUT] mean-field object
       * @return - inverse of symmetric dielectric function at finite q and q = 0
       */
      template<nda::MemoryArrayOfRank<4> Array_w_t, THC_ERI thc_t, typename communicator_t>
      static auto eps_inv_head_w(memory::darray_t<Array_w_t, communicator_t> &dW_wqPQ, thc_t &thc,
                                 mf::MF &mf, div_treatment_e div_treatment=gygi_extrplt,
                                 bool poly_in_q2=false, double q_max=0.8, int fit_order=2)
      -> std::tuple<nda::array<ComplexType, 2>, nda::array<ComplexType, 1> > {

        // Compute the head (i.e. G=G'=0 component) of the inverse dielectric function at finite q
        auto eps_inv_w = eval_eps_inv_q(dW_wqPQ, thc, mf);
        auto [nw, nqpts_ibz] = eps_inv_w.shape();

        if (thc.MF()->nqpts_ibz() == 1 and div_treatment != ignore_g0) {
          app_log(2, "eps_inv_head_w: nqpts_ibz == 1 while div_treatment != ignore. "
                     "CoQui will take div_treatment = ignore_g0 anyway!");
          div_treatment = ignore_g0;
        }

        // Perform q=0 extrapolation on frequency domain data
        nda::array<ComplexType, 1> eps_inv_q0_w = 
          extrapolate_eps_inv_q0(eps_inv_w, mf, div_treatment, poly_in_q2, q_max, fit_order);

        return std::make_tuple(std::move(eps_inv_w), std::move(eps_inv_q0_w));
      }

      /**
       * Extract the head (G=G'=0 component) of the inverse symmetric dielectric function at finite q
       * 
       * Computes \epsilon^{q,-1}_{G=0,G'=0} from the screened interaction W(x,q,P,Q) in the THC product basis.
       * Works for W on both imaginary-time and Matsubara frequency axes.
       * 
       * @param dW_xqPQ    - [INPUT] screened interaction in the THC product basis
       *                             with shape [nx, nqpts_ibz, NP, NQ], where x is the axis index
       *                             (x can be time τ or Matsubara frequency iω)
       * @param thc        - [INPUT] THC-ERI instance
       * @param mf         - [INPUT] mean-field instance
       * @return - head of inverse dielectric function with shape [nx, nqpts_ibz],
       *           indexed as eps_inv_x(x_idx, q_idx)
       */
      template<nda::MemoryArrayOfRank<4> Array_w_t, THC_ERI thc_t, typename communicator_t>
      static auto eval_eps_inv_q(memory::darray_t<Array_w_t, communicator_t> &dW_xqPQ,
                                 thc_t &thc, mf::MF &mf) {
        auto [nx_loc, nq_loc, NP_loc, NQ_loc] = dW_xqPQ.local_shape();
        auto x_rng = dW_xqPQ.local_range(0);
        auto qpt_rng = dW_xqPQ.local_range(1);
        auto P_rng = dW_xqPQ.local_range(2);
        auto Q_rng = dW_xqPQ.local_range(3);
        auto [nx, nqpts_ibz, NP, NQ] = dW_xqPQ.global_shape();

        nda::array<ComplexType, 2> eps_inv_x(nx, nqpts_ibz);
        nda::array<ComplexType, 1> Chi_bar_Q_conj(NQ_loc);
        nda::array<ComplexType, 1> buffer_P(NP_loc);
        const double fpi = 4.0*3.14159265358979323846;
        auto Chi_bar_qu = thc.basis_bar_head();
        auto Wloc = dW_xqPQ.local();
        for (auto [iq, q] : itertools::enumerate(qpt_rng)) {

          auto qpts = mf.Qpts_ibz(q);
          double q_abs2 = qpts(0)*qpts(0) + qpts(1)*qpts(1) + qpts(2)*qpts(2);

          Chi_bar_Q_conj = nda::conj(Chi_bar_qu(q, Q_rng));
          double factor = (q_abs2 / fpi) * mf.volume();
          for (auto [ix, x_idx] : itertools::enumerate(x_rng)) {
            nda::blas::gemv(Wloc(ix, iq, nda::ellipsis{}), Chi_bar_Q_conj, buffer_P);
            eps_inv_x(x_idx, q) += factor * nda::blas::dot(Chi_bar_qu(q, P_rng), buffer_P);
          }
        }
        dW_xqPQ.communicator()->all_reduce_in_place_n(eps_inv_x.data(), eps_inv_x.size(), std::plus<>{});

        return eps_inv_x;
      }

      /**
       * Evaluate the head of a matrix, M_{G=0,G'=0}(t,q), for an arbitrary tensor, M(t,q,P,Q), in the thc product basis. 
       * @param dM_tqPQ - [INPUT] input tensor in the thc product basis
       * @param thc - [INPUT] thc-eri instance
       * @param mf - [INPUT] mean-field instance
       * @param bar_basis - [INPUT] bool (default = false). If true, use the "bar" basis, otherwise use the direct basis.
       * @return
       */
      template<nda::MemoryArrayOfRank<4> Array_w_t, THC_ERI thc_t, typename communicator_t>
      static auto head_from_prod_basis(memory::darray_t<Array_w_t, communicator_t> &dM_tqPQ,
                                 thc_t &thc, bool bar_basis = false) {
        auto [nt_loc, nq_loc, NP_loc, NQ_loc] = dM_tqPQ.local_shape();
        auto t_rng = dM_tqPQ.local_range(0);
        auto qpt_rng = dM_tqPQ.local_range(1);
        auto P_rng = dM_tqPQ.local_range(2);
        auto Q_rng = dM_tqPQ.local_range(3);
        auto [nts, nqpts_ibz, NP, NQ] = dM_tqPQ.global_shape();

        nda::array<ComplexType, 2> m_t(nts, nqpts_ibz);
        nda::array<ComplexType, 1> buffer_P(NP_loc);
        auto Mloc = dM_tqPQ.local();
        if(bar_basis) {
          // M(t,q) = sum_PQ B_bar(q,P) * M(t,q,P,Q) conj(B_bar(q,Q))
          auto Chi_bar_qu = thc.basis_bar_head();
          nda::array<ComplexType, 1> Chi_bar_Q_conj(NQ_loc);
          for (auto [iq, q] : itertools::enumerate(qpt_rng)) {
            Chi_bar_Q_conj = nda::conj(Chi_bar_qu(q, Q_rng));
            for (auto [it, t] : itertools::enumerate(t_rng)) {
              nda::blas::gemv(Mloc(it, iq, nda::ellipsis{}), Chi_bar_Q_conj, buffer_P);
              m_t(t, q) += nda::blas::dot(Chi_bar_qu(q, P_rng), buffer_P);
            }
          }
        } else {
          // M(t,q) = sum_PQ conj(B(q,P)) * M(t,q,P,Q) B(q,Q)
          auto Chi_qu = thc.basis_head();
          for (auto [iq, q] : itertools::enumerate(qpt_rng)) {
            for (auto [it, t] : itertools::enumerate(t_rng)) {
              nda::blas::gemv(Mloc(it, iq, nda::ellipsis{}), Chi_qu(q, Q_rng), buffer_P);
              m_t(t, q) += nda::blas::dotc(Chi_qu(q, P_rng), buffer_P);
            }
          }
        }
        dM_tqPQ.communicator()->all_reduce_in_place_n(m_t.data(), m_t.size(), std::plus<>{});

        return m_t;
      }

      /**
       * Find the two closest points to origin along each Cartesian direction
       * For a uniform 3D grid centered at origin spanning [-0.5, 0.5] in each direction,
       * this function finds the two points with smallest positive coordinate values
       * along x, y, and z directions. Only searches on the positive side (accounting for symmetry).
       * @param Qpts      - [INPUT] array of q-points (shape: [nqpts, 3])
       * @param recv      - [INPUT] reciprocal lattice vectors (shape: [3, 3])
       * @param tolerance - [INPUT] threshold to consider a coordinate as zero (default: 1e-10)
       * @return array of 3 vectors containing indices for x, y, z directions respectively
       */
      template<nda::ArrayOfRank<2> Array_2D_t, nda::ArrayOfRank<2> Array_2D_recv_t>
      static std::array<std::vector<int>, 3> find_two_closest_per_direction(
        Array_2D_t &&Qpts, Array_2D_recv_t &&recv, double tolerance=1e-10) {

        if (Qpts.shape(0) == 1) return {{{0}, {0}, {0}}};
        
        // Convert to crystal coordinates using the reciprocal lattice vectors
        nda::array<double, 2> Qpts_crys(Qpts.shape(0), 3);
        for (int iq = 0; iq < Qpts.shape(0); ++iq) {
          nda::blas::gemv(1.0, recv, Qpts(iq,nda::range::all), 0.0, Qpts_crys(iq,nda::range::all));
        }
        
        std::array<std::vector<int>, 3> result_indices;
        
        // For each direction (x=0, y=1, z=2)
        for (int dir = 0; dir < 3; ++dir) {
          // Collect all positive values in this direction with their indices
          std::vector<std::pair<double, int>> coord_idx_pairs;
          
          for (int i = 0; i < Qpts_crys.shape(0); ++i) {
            double coord = Qpts_crys(i, dir);
            
            // Check if other dimensions are zero
            bool other_dims_zero = true;
            for (int other_dir = 0; other_dir < 3; ++other_dir) {
              if (other_dir != dir && std::abs(Qpts_crys(i, other_dir)) > tolerance) {
                other_dims_zero = false;
                break;
              }
            }
            
            // Only consider positive coordinates with other dimensions zero
            if (coord > tolerance && other_dims_zero) {
              coord_idx_pairs.push_back({coord, i});
            }
          }
          
          // Sort by coordinate value
          std::sort(coord_idx_pairs.begin(), coord_idx_pairs.end());
          
          // Collect the two smallest (or fewer if not enough points)
          int count = std::min(2, static_cast<int>(coord_idx_pairs.size()));
          for (int j = 0; j < count; ++j) {
            result_indices[dir].push_back(coord_idx_pairs[j].second);
          }
        }
        
        return result_indices;
      }

      template<nda::ArrayOfRank<2> Array_q_t>
      static int find_smallest_qabs(Array_q_t &&Qpts, bool two_dim=false) {
        if (Qpts.shape(0) == 1) return 0;

        double min = -1;
        int idx = -1;
        for (int i=0; i<Qpts.shape(0); ++i) {
          auto qpt = Qpts(i,nda::range::all);
          double norm = std::sqrt( qpt(0)*qpt(0) + qpt(1)*qpt(1) + qpt(2)*qpt(2) );
          if (norm > 0.0 and min==-1 and (!two_dim or qpt(2)!=0.0)) {
            min = norm;
            idx = i;
          }
          else if (norm > 0.0 and norm<min and (!two_dim or qpt(2)==0.0)) {
            min = norm;
            idx = i;
          }
        }
        return idx;
      }

      /**
       * Find all indices of q-points with the smallest absolute value
       * Returns all q-points that share the same minimum |q| (accounting for symmetry)
       * @param Qpts - [INPUT] array of q-points (shape: [nqpts, 3])
       * @param two_dim - [INPUT] if true, only consider q-points with qpt(2)==0
       * @param include_gamma - [INPUT] if true, include gamma point (q=0)
       * @return vector of all indices with the minimum |q| value
       */
      template<nda::ArrayOfRank<2> Array_q_t>
      static std::vector<int> find_smallest_qabs_indices(Array_q_t &&Qpts, bool two_dim=false, 
                                                          bool include_gamma=false) {
        if (Qpts.shape(0) == 1) return {0};
        
        double min_norm = -1.0;
        constexpr double tolerance = 1e-10;
        
        // First pass: find the minimum norm
        for (int i=0; i<Qpts.shape(0); ++i) {
          auto qpt = Qpts(i,nda::range::all);
          double norm = std::sqrt( qpt(0)*qpt(0) + qpt(1)*qpt(1) + qpt(2)*qpt(2) );
          
          // Skip gamma point if not requested
          if (!include_gamma && norm < tolerance) continue;
          
          // For 2D, only consider points with qpt(2)==0
          if (two_dim && std::abs(qpt(2)) > tolerance) continue;
          
          if (min_norm < 0.0 || norm < min_norm) {
            min_norm = norm;
          }
        }
        
        // Second pass: collect all indices with the minimum norm
        std::vector<int> indices;
        for (int i=0; i<Qpts.shape(0); ++i) {
          auto qpt = Qpts(i,nda::range::all);
          double norm = std::sqrt( qpt(0)*qpt(0) + qpt(1)*qpt(1) + qpt(2)*qpt(2) );
          
          // Skip gamma point if not requested
          if (!include_gamma && norm < tolerance) continue;
          
          // For 2D, only consider points with qpt(2)==0
          if (two_dim && std::abs(qpt(2)) > tolerance) continue;
          
          if (std::abs(norm - min_norm) < tolerance) {
            indices.push_back(i);
          }
        }
        
        return indices;
      }

      template<nda::ArrayOfRank<1> Array_q_t, nda::ArrayOfRank<1> Array_eps_t>
      static auto extrapolate_to_q0(const Array_q_t &Qpts_abs2, const Array_eps_t &eps_inv_n, int fit_order,
                                    bool print=false) {

        long num_sample = Qpts_abs2.shape(0);
        nda::matrix<ComplexType> A(num_sample, fit_order+1);
        for (int i = 0; i < num_sample; ++i) {
          A(i, 0) = 1.0;
          for (int j = 1; j <= fit_order; ++j) {
            A(i, j) = std::pow(Qpts_abs2(i), j);
          }
        }

        auto AT = nda::make_regular(nda::transpose(A));
        nda::array<ComplexType, 1> AT_b(fit_order+1);
        nda::blas::gemv(AT, eps_inv_n, AT_b);
        nda::matrix<ComplexType> ATA_inv(fit_order+1, fit_order+1);
        nda::blas::gemm(AT, A, ATA_inv);
        ATA_inv = nda::inverse(ATA_inv);

        nda::array<ComplexType, 1> x(fit_order+1);
        nda::blas::gemv(ATA_inv, AT_b, x);

        if (print) {
          for (size_t n=0; n<=fit_order; ++n)
            app_log(2, "    x({}) = {}", n, x(n));
          app_log(2, "");
        }

        return x(0);
      }

    }; // div_utils
  } // solvers
} // methods



#endif //COQUI_G0_DIV_UTILS_HPP
