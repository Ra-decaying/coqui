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

    const nda::array<ComplexType, 2> & electronic_gradient() const;
    const nda::array<ComplexType, 2> & nuclear_gradient() const;
    const nda::array<ComplexType, 2> & total_gradient() const;

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
