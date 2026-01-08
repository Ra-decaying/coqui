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
