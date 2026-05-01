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


#ifndef COQUI_QP_COM_DIIS_RESIDUAL_H
#define COQUI_QP_COM_DIIS_RESIDUAL_H

#include "numerics/iter_scf/diis/vspace.h"
#include "numerics/iter_scf/diis/vspace_heff.hpp"
#include "numerics/iter_scf/diis/diis_residual.h"

// TODO refactor memory usage for S_overlap and Dm_incoming. This is not necessary. 

namespace iter_scf {

class qp_com_diis_residual : public diis_residual<Heff> {
  using Array_4D = nda::array<ComplexType, 4>;

protected:
  using diis_residual<Heff>::is_initialized;
  using diis_residual<Heff>::current_state;

  bool com_initialized = false;

  Array_4D Dm_incoming;
  Array_4D S_overlap;
  std::string mbpt_output;
  long residual_iter = -1;

public:
  qp_com_diis_residual() = default;

  template<nda::MemoryArrayOfRank<4> S_t>
  void initialize(opt_state<Heff>* current_state_, S_t &&S, std::string mbpt_output_) {
    if(!com_initialized) {
      current_state = current_state_;
      S_overlap = nda::make_regular(S);
      mbpt_output = mbpt_output_;
    }
    is_initialized = true;
    com_initialized = true;
  }

  void set_iteration(long iter_) {
    residual_iter = iter_;
  }

  void upload_dm() {
    utils::check(residual_iter >= 0, "qp_com_diis_residual: residual iteration must be non-negative.");
    std::string filename = mbpt_output + ".mbpt.h5";
    h5::file file(filename, 'r');
    h5::group grp(file);
    utils::check(grp.has_subgroup("scf"),
                 "qp_com_diis_residual::upload_dm: Simulation HDF5 file does not have an scf group");
    auto scf_grp = grp.open_group("scf");
    std::string iter_grp_name = "iter" + std::to_string(residual_iter);
    utils::check(scf_grp.has_subgroup(iter_grp_name),
                 "qp_com_diis_residual::upload_dm: {} does not exist in {}",
                 "scf/" + iter_grp_name, filename);
    auto iter_grp = scf_grp.open_group(iter_grp_name);
    nda::h5_read(iter_grp, "Dm_skij", Dm_incoming);
  }

  bool get_diis_residual(Heff& res) override {
    utils::check(com_initialized, "QP commutator DIIS residual is not initialized");
    upload_dm();

    auto H = current_state->get().get_heff();
    auto [ns, nk, nao, nao2] = H.shape();
    utils::check(nao == nao2,
                 "qp_com_diis_residual::get_diis_residual: Heff must be square matrices");

    auto [d_ns, d_nk, d_nao, d_nao2] = Dm_incoming.shape();
    utils::check(ns == d_ns and nk == d_nk and nao == d_nao and nao2 == d_nao2,
                 "qp_com_diis_residual::get_diis_residual: Heff and Dm shapes do not match");

    auto [s_ns, s_nk, s_nao, s_nao2] = S_overlap.shape();
    utils::check(ns == s_ns and nk == s_nk and nao == s_nao and nao2 == s_nao2,
                 "qp_com_diis_residual::get_diis_residual: Heff and S shapes do not match");

    Array_4D C(ns, nk, nao, nao);
    C() = 0;
    decltype(nda::range::all) all;
    nda::array<ComplexType, 2> I1(nao, nao);
    nda::array<ComplexType, 2> I2(nao, nao);
    nda::array<ComplexType, 2> I3(nao, nao);
    nda::array<ComplexType, 2> I4(nao, nao);
    for(size_t s = 0; s < ns; s++)
    for(size_t k = 0; k < nk; k++) {
      auto H_sk = H(s, k, all, all);
      auto D_sk = Dm_incoming(s, k, all, all);
      auto S_sk = S_overlap(s, k, all, all);
      I1() = 0;
      I2() = 0;
      I3() = 0;
      I4() = 0;
      nda::blas::gemm(H_sk, D_sk, I1);
      nda::blas::gemm(I1, S_sk, I2);
      nda::blas::gemm(S_sk, D_sk, I3);
      nda::blas::gemm(I3, H_sk, I4);
      C(s, k, all, all) = nda::make_regular(I2 - I4);
    }

    res.set_heff(C);
    return true;
  }
};

}

#endif // COQUI_QP_COM_DIIS_RESIDUAL_H
