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


#ifndef COQUI_HF_GRAND_POTENTIAL_H
#define COQUI_HF_GRAND_POTENTIAL_H

#include "mean_field/MF.hpp"
#include "nda/nda.hpp"

namespace methods {

  double eval_hf_grand_potential(const nda::MemoryArrayOfRank<4> auto &D_skij,
                                 const nda::MemoryArrayOfRank<4> auto &S_skij,
                                 const mf::MF &MF, double e_hf, double beta, double mu);

}

#endif
