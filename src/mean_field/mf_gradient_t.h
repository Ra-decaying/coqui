#ifndef COQUI_MEANFIELD_MF_GRADIENT_T_H
#define COQUI_MEANFIELD_MF_GRADIENT_T_H

#include <cmath>

#include "mean_field/MF.hpp"
#include "methods/ERI/chol_grad_reader_t.hpp"
#include "methods/ERI/chol_reader_t.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "nda/nda.hpp"
#include "utilities/Timer.hpp"

namespace mf {

  /**
   * The zero-temperature Hartree gradient calculator.
   * This is not optimal and it is for testing.
   */
  class mf_gradient_t {

    public:

      template<nda::MemoryArray local_Array_t>
      using dArray_t = math::nda::distributed_array<local_Array_t, mpi3::communicator>;

      template<nda::Array Array_base_t>
      using sArray_t = math::shm::shared_array<Array_base_t>;

      template<int N>
      using shape_t = std::array<long, N>;

      mf_gradient_t(boost::mpi3::communicator gcomm, const mf::MF *MF, const ptree &pt);
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

    mpi3::communicator _gcomm;
    mpi3::shared_communicator _node_comm;
    mpi3::communicator _internode_comm;

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
