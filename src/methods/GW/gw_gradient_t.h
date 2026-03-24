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


#ifndef COQUI_GW_GRADIENT_T_H
#define COQUI_GW_GRADIENT_T_H

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "mpi3/communicator.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"

namespace methods {

  namespace solvers {

    namespace mpi3 = boost::mpi3;

    class gw_gradient_t {

    template<nda::Array Array_base_t>
    using sArray_t = math::shm::shared_array<Array_base_t>;
    template<int N>
    using shape_t = std::array<long,N>;

    public:

    gw_gradient_t(std::shared_ptr<mf::MF> MF, const imag_axes_ft::IAFT *FT);

    ~gw_gradient_t() = default;

    nda::array<ComplexType, 2> evaluate(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                        Cholesky_ERI auto && chol);

    nda::array<ComplexType, 2> eval_grad_2e(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                            Cholesky_ERI auto && chol);

    ComplexType eval_grad_2e(int iatom, int idirection, Cholesky_ERI auto && chol);

    template<nda::MemoryArray Array_3D_t>
    void eval_P0(size_t iq, const nda::MemoryArrayOfRank<5> auto &G_tskij,
                 sArray_t<Array_3D_t> &sP0_tPQ, Cholesky_ERI auto &chol,
                 int batch_size, bool print_mpi);

    template<nda::MemoryArray Array_3D_t>
    void eval_dyson_P(sArray_t<Array_3D_t> &sP0_tPQ, sArray_t<Array_3D_t> &sP0_wPQ);


    void print_chol_gw_grad_timers();

    private:

    std::shared_ptr<mf::MF> _mf = nullptr;

    const imag_axes_ft::IAFT* _ft = nullptr;

    utils::TimerManager _Timer;

    int _natoms = 0;
    int _nbnd = 0;
    int _nbnd_aux = 0;
    int _nspin = 0;
    int _nkpts = 0;
    int _npol = 0;

    nda::array<RealType, 1> _k_weight;

  };

 } // namespace solvers

} // namespace method

#endif
