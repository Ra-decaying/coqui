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


#ifndef COQUI_HF_GRADIENT_T_H
#define COQUI_HF_GRADIENT_T_H

#include "mpi3/communicator.hpp"
#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"

namespace methods {

  namespace solvers {

    namespace mpi3 = boost::mpi3;

    class hf_gradient_t {

    template<nda::Array Array_base_t>
    using sArray_t = math::shm::shared_array<Array_base_t>;

    using Array_view_4D_t = nda::array_view<ComplexType, 4>;

    public:

    hf_gradient_t(std::shared_ptr<mf::MF> MF, bool auxbasis_response = true);

    ~hf_gradient_t() = default;

    void evaluate(const nda::MemoryArrayOfRank<4> auto &D_skij,
                  const nda::MemoryArrayOfRank<4> auto &F_skij,
                  const nda::MemoryArrayOfRank<4> auto &S_skij,
                  const nda::MemoryArrayOfRank<4> auto &H0_skij,
                  Cholesky_ERI auto &&chol, bool F_has_H0);

    ComplexType evaluate_1e(int iatom, int direction, const nda::MemoryArrayOfRank<4> auto &D_skij);
    ComplexType evaluate_2e(int iatom, int direction, const nda::MemoryArrayOfRank<4> auto &D_skij,
                            Cholesky_ERI auto && chol);
    ComplexType evaluate_pulay(int iatom, int direction, const nda::MemoryArrayOfRank<4> auto& DE_skij);

    const nda::array<ComplexType, 2> & gradient_elec() const;
    const nda::array<ComplexType, 2> & gradient_nuc() const;
    const nda::array<ComplexType, 2> & gradient_total() const;

    private:

    std::shared_ptr<mf::MF> _MF = nullptr;

    utils::TimerManager _Timer;

    bool _auxbasis_response = true;

    int _natoms = 0;
    int _nbnd = 0;
    int _nbnd_aux = 0;
    int _nspin = 0;
    int _nkpts = 0;
    int _npol = 0;

    nda::array<RealType, 1> _k_weight;

    nda::array<ComplexType, 2> _gradient_total;
    nda::array<ComplexType, 2> _gradient_elec;
    nda::array<ComplexType, 2> _gradient_nuc;

  };

 } // namespace solvers

} // namespace method

#endif
