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


#ifndef COQUI_GRADIENT_COMMON_H
#define COQUI_GRADIENT_COMMON_H

#include "gradient_common.h"

#include "IO/app_loggers.h"
#include "methods/HF/hf_grand_potential.h"

namespace methods {

template<typename data_type>
void print_mbpt_gradient(const nda::array<data_type, 2>& gradient, std::shared_ptr<mf::MF> mf,
                         const std::string &str, bool bohr)
{
    double factor;
    std::string unit;
    if (bohr) {
      factor = 1;
      unit = "(hartree/bohr)";
    } else {
      factor = 0.52917721054482;
      unit = "(hartree/angstrom)";
    }

    app_log(1, "  {}", str);
    app_log(1, "  --------------------------------------------------------------------------------------");
    app_log(1, "   {:<5}{:<5}{:>20}{:>20}{:>20}", "id", "nuc", "X" + unit, "Y" + unit, "Z" + unit);
    app_log(1, "  --------------------------------------------------------------------------------------");
    for (int iatom = 0; iatom < mf->number_of_atoms(); ++iatom) {
      app_log(1, "   {:<5}{:<5}{:>+20.10f}{:>+20.10f}{:>+20.10f}",
        iatom, mf->atomic_id(iatom),
        nda::real(gradient(iatom, 0) * factor),
        nda::real(gradient(iatom, 1) * factor),
        nda::real(gradient(iatom, 2) * factor));
    }
    app_log(1, "\n");
}


template void print_mbpt_gradient(const nda::array<RealType, 2>&, std::shared_ptr<mf::MF>,
                                  const std::string&, bool);

template void print_mbpt_gradient(const nda::array<ComplexType, 2> &, std::shared_ptr<mf::MF>,
                                  const std::string&, bool);

} // namespace methods

#endif
