#include "hf_grand_potential.h"

namespace methods {

  double eval_hf_grand_potential(const nda::MemoryArrayOfRank<4> auto &D_skij,
                                 const nda::MemoryArrayOfRank<4> auto &S_skij,
                                 const mf::MF &MF, double e_hf, double beta, double mu)
  {
    decltype(nda::range::all) all;

    int nbnd = MF.nbnd();
    int nkpts = MF.nkpts();
    int npol = MF.npol();
    int nspin = MF.nspin();

    double entropy = 0.0;
    double nelec = 0.0;
    for (int ispin = 0; ispin < nspin; ++ispin) {
      for (int ikpt = 0; ikpt < nkpts; ++ikpt) {
        double tmp_entropy = 0.0;
        double tmp_nelec = 0.0;
        auto C = nda::array<ComplexType, 2>::zeros({nbnd, nbnd});
        auto D = nda::array<ComplexType, 2>::zeros({nbnd, nbnd});
        auto SC = nda::array<ComplexType, 2>::zeros({nbnd, nbnd});
        auto tmp = nda::array<ComplexType, 2>::zeros({nbnd, nbnd});
        auto [lambda, U] = nda::linalg::eigenelements(S_skij(ispin, ikpt, all, all));
        for (int ibnd = 0; ibnd < nbnd; ++ibnd) {
            C(ibnd, ibnd) = std::pow(lambda(ibnd), -0.5);
        }
        nda::blas::gemm(1.0, U, C, 0.0, tmp);
        nda::blas::gemm(1.0, tmp, nda::transpose(nda::conj(U)), 0.0, C);
        nda::blas::gemm(1.0, S_skij(ispin, ikpt, all, all), C, 0.0, SC);
        nda::blas::gemm(1.0, nda::transpose(nda::conj(SC)), D_skij(ispin, ikpt, all, all), 0.0, tmp);
        nda::blas::gemm(1.0, tmp, SC, 0.0, D);
        auto [occ, _] = nda::linalg::eigenelements(D);
        for (int ibnd = 0; ibnd < nbnd; ++ibnd) {
          nelec += occ(ibnd);
          if (occ(ibnd) < -1e-10 or occ(ibnd) > 1.0 + 1e-10) {
            app_log(2, " Warning: an eigenvalue of D is not between 0 and 1: {}", occ(ibnd));
          } else {
            double num = std::clamp(occ(ibnd), 1e-10, 1.0 - 1e-10);
            tmp_entropy += num * std::log(num) + (1.0 - num) * std::log(1.0 - num);
          }
        }
        entropy += tmp_entropy;
        nelec += tmp_nelec;
      }
    }
    double spin_factor = (nspin == 1 and npol == 1) ? 2.0 : 1.0;
    entropy *= spin_factor;
    nelec *= spin_factor;
    return e_hf - beta * entropy - mu * nelec;
  }

    using Arr4D = nda::array<ComplexType, 4>;
    using Arrv4D = nda::array_view<ComplexType, 4>;
    template double eval_hf_grand_potential(const Arr4D&, const Arr4D&,  const mf::MF&, double, double, double);
    template double eval_hf_grand_potential(const Arrv4D&, const Arrv4D&,  const mf::MF&, double, double, double);

}
