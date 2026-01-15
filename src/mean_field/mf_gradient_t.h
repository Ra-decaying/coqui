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


#ifndef COQUI_MEANFIELD_MF_GRADIENT_T_H
#define COQUI_MEANFIELD_MF_GRADIENT_T_H

#include <cmath>

#include "mean_field/MF.hpp"
#include "methods/ERI/chol_grad_reader_t.hpp"
#include "methods/ERI/chol_reader_t.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "nda/nda.hpp"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"

namespace mf {

  /**
   * The zero-temperature Hartree gradient calculator.
   * This is not optimal and it is for testing.
   */

   // TODO Pass mf::MF and mpi_context_t via ERI

   class mf_gradient_t {

    public:

      using mpi_context_t = utils::mpi_context_t<mpi3::communicator>;

      template<nda::MemoryArray local_Array_t>
      using dArray_t = math::nda::distributed_array<local_Array_t, mpi3::communicator>;

      template<nda::Array Array_base_t>
      using sArray_t = math::shm::shared_array<Array_base_t>;

      template<int N>
      using shape_t = std::array<long, N>;

      mf_gradient_t(const mf::MF *MF, const ptree &pt);
      ~mf_gradient_t() = default;

      // assume that all elements are real
      void evaluate(methods::chol_reader_t &chol, methods::chol_grad_reader_t &chol_gradient);
      void evaluate(methods::THC_ERI auto &&thc, methods::THC_ERI auto &&thc_grad);

      void print_gradients(const nda::array<ComplexType, 2> &gradients, const std::string &str, bool bohr = true);
      void print_timers();
      void write_output();

      ComplexType evaluate_1e(int iatom, int direction);
      ComplexType evaluate_2e(int iatom, int direction);
      ComplexType evaluate_pulay(int iatom, int direction);

    private:

    std::shared_ptr<mpi_context_t> _context;

    const mf::MF *_MF = nullptr;

    utils::TimerManager _Timer;

    std::string _output;

    const double _bohr_to_angstrom = 0.52917721054482;

    bool _auxbasis_response = true;

    int _natoms = 0;
    int _nbnd = 0;
    int _nbnd_aux = 0;
    int _nspin = 0;
    int _nkpts = 0;
    int _npol = 0;

    nda::array<RealType, 1> _k_weight;

    nda::array<ComplexType, 3> _occ;
    nda::array<ComplexType, 3> _eigval;
    nda::array<ComplexType, 4> _mo_coeff;

    nda::array<ComplexType, 4> _dm;
    nda::array<ComplexType, 4> _dme;

    nda::array<ComplexType, 2> _gradient_total;
    nda::array<ComplexType, 2> _gradient_elec;
    nda::array<ComplexType, 2> _gradient_nuc;
  };

} // namespace mf


#endif // MEANFIELD_MF_GRAD_H
