#include "mean_field/mf_gradient_t.h"

namespace mf {

  mf_gradient_t::mf_gradient_t(mpi3::communicator gcomm, const mf::MF *MF, const ptree &pt):
    _gcomm(gcomm),
    _node_comm(_gcomm.split_shared()),
    _MF(MF),
    _Timer() {
    if (_gcomm.size()%_node_comm.size()!=0) {
      APP_ABORT("Zero temperature MF gradient: number of processors on each node should be the same.");
    }
    // Setup internode communicator
    int node_size = _node_comm.size();
    int color =_gcomm.rank() % node_size;
    int key = _gcomm.rank() / node_size;
    _internode_comm = _gcomm.split(color, key);

    bool auxbasis_response = io::get_value_with_default<bool>(pt, "auxbasis_response", true);
    _auxbasis_response = auxbasis_response;

    std::string output = io::get_value_with_default<std::string>(pt, "output", "mf.gradients");
    _output = output;
  }

  // assume that all elements are real
  void mf_gradient_t::evaluate(methods::chol_reader_t &chol,
                               methods::chol_grad_reader_t &chol_gradient) {

    app_log(1, "\n"
                "╔═╗╔═╗╔═╗ ╦ ╦╦  ┬┬┬┌─┐   ┌─┐┬─┐┌─┐┌┬┐\n"
                "║  ║ ║║═╬╗║ ║║  │││├┤────│ ┬├┬┘├─┤ ││\n"
                "╚═╝╚═╝╚═╝╚╚═╝╩  ┴ ┴└     └─┘┴└─┴ ┴─┴┘\n");

    decltype(nda::range::all) all;

   for( auto& v: {"TOTAL", "1ELECTRON", "2ELECTRON", "PULAY", "NUCLEI", "WRITE"} ) {
     _Timer.add(v);
   }

    _Timer.start("TOTAL");

    _natoms = _MF->number_of_atoms();
    _nbnd = _MF->nbnd();
    _nbnd_aux = _MF->nbnd_aux();
    _nspin = _MF->nspin();
    _nkpts = _MF->nkpts();
    _npol = _MF->npol();

    _k_weight = _MF->k_weight();

    _occ = _MF->occ();
    _eigval = _MF->eigval();
    _mo_coeff = _MF->mo_coeff();

    app_log(1, "  - nbnd:              {}", _nbnd);
    app_log(1, "  - nbnd_aux:          {}", _nbnd_aux);
    app_log(1, "  - nspin:             {}", _nspin);
    app_log(1, "  - nkpts:             {}", _nkpts);
    app_log(1, "\n");

    // density matrix
    _dm = nda::array<ComplexType, 4>::zeros({_nspin, _nkpts, _nbnd, _nbnd});
    for (int ispin = 0; ispin < _nspin; ++ispin) {
      for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
        auto occ_diag = nda::array<ComplexType, 2>::zeros({_nbnd, _nbnd});
        auto tmp = nda::array<ComplexType, 2>::zeros({_nbnd, _nbnd});
        for (int j = 0; j < _nbnd; j++) {
          occ_diag(j, j) = _occ(ispin, ikpt, j);
        }
        nda::blas::gemm(1.0, _mo_coeff(ispin, ikpt, all, all), occ_diag, 0.0, tmp);
        nda::blas::gemm(1.0, tmp, nda::transpose(_mo_coeff(ispin, ikpt, all, all)), 0.0, _dm(ispin, ikpt, all, all));
        }
    }

    // energy-weighted density matrix
    _dme = nda::array<ComplexType, 4>::zeros({_nspin, _nkpts, _nbnd, _nbnd});
    for (int ispin = 0; ispin < _nspin; ++ispin) {
      for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
        auto energy_occ_diag = nda::array<ComplexType, 2>::zeros({_nbnd, _nbnd});
        auto tmp = nda::array<ComplexType, 2>::zeros({_nbnd, _nbnd});
        for (int j = 0; j < _nbnd; j++) {
          energy_occ_diag(j, j) = _occ(ispin, ikpt, j) * _eigval(ispin, ikpt, j);
        }
        nda::blas::gemm(1.0, _mo_coeff(ispin, ikpt, all, all), energy_occ_diag, 0.0, tmp);
        nda::blas::gemm(1.0, tmp, nda::transpose(_mo_coeff(ispin, ikpt, all, all)), 0.0, _dme(ispin, ikpt, all, all));
        }
    }

    _gradient_total = nda::array<ComplexType, 2>::zeros({_natoms, 3});
    _gradient_elec = nda::array<ComplexType, 2>::zeros({_natoms, 3});
    _gradient_nuc = nda::array<ComplexType, 2>::zeros({_natoms, 3});

    for (int iatom = 0; iatom < _natoms; ++iatom) {
      for (int direction = 0; direction < 3; ++direction) {
        _Timer.start("1-ELECTRON");
        _gradient_elec(iatom, direction) += evaluate_1e(iatom, direction);
        _Timer.stop("1-ELECTRON");
        _Timer.start("2-ELECTRON");
        _gradient_elec(iatom, direction) += evaluate_2e(iatom, direction);
        _Timer.stop("2-ELECTRON");
        _Timer.start("PULAY");
        _gradient_elec(iatom, direction) += evaluate_pulay(iatom, direction);
        _Timer.stop("PULAY");
      }
    }

    _gradient_nuc = _MF->nuclear_gradient();

    _gradient_total = _gradient_elec + _gradient_nuc;

    _Timer.stop("TOTAL");

    print_timers();

    print_gradients(_gradient_elec, "GRAD_ELEC");
    print_gradients(_gradient_nuc, "GRAD_NUC");
    print_gradients(_gradient_total, "GRAD_TOTAL");

    write_output();
  }

  void mf_gradient_t::evaluate(methods::THC_ERI auto &&thc, methods::THC_ERI auto &&thc_gradient) {
    APP_ABORT("Error: {} for zero-temperature Hartree-Fock not implemented yet \n ", "THC");
  }

  ComplexType mf_gradient_t::evaluate_1e(int iatom, int direction) {
    ComplexType tmp_grad = 0;
    auto H0_grad = _MF->H0_grad();
    for (int ispin = 0; ispin < _nspin; ++ispin) {
      for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
        tmp_grad += _k_weight(ikpt) * nda::sum(_dm(ispin, ikpt, nda::ellipsis{}) *
                                               H0_grad(iatom, direction, ispin, ikpt, nda::ellipsis{}));
      }
    }
    return tmp_grad;
  }

  ComplexType mf_gradient_t::evaluate_2e(int iatom, int direction) {
    decltype(nda::range::all) all;

    auto bnd_slice = _MF->bnd_slice();
    auto bnd_slice_aux = _MF->bnd_slice_aux();

    nda::array<ComplexType, 5> Vq0_3Qkij_di(3, _nbnd_aux, _nkpts, _nbnd, _nbnd);
    nda::array<ComplexType, 5> Vq0_3Qkij_dQ(3, _nbnd_aux, _nkpts, _nbnd, _nbnd);
    nda::array<ComplexType, 4> Vq0_Qkij(_nbnd_aux, _nkpts, _nbnd, _nbnd);
    nda::array<ComplexType, 4> Vq0_3PQk_dP(3, _nbnd_aux, _nbnd_aux, _nkpts);
    nda::array<ComplexType, 3> Vq0_PQk(_nbnd_aux, _nbnd_aux, _nkpts);
    nda::array<ComplexType, 3> Vq0_PQk_inv(_nbnd_aux, _nbnd_aux, _nkpts);

    h5::file file = h5::file("Vq0_grad.h5", 'r');
    h5::group grp(file);
    h5::group grad_grp = grp.open_group("Interaction_Gradient");
    nda::h5_read(grad_grp, "Vq0_3Qskij_di", Vq0_3Qkij_di, std::tuple{all, all, 0, all, all, all});
    nda::h5_read(grad_grp, "Vq0_3Qskij_dQ", Vq0_3Qkij_dQ, std::tuple{all, all, 0, all, all, all});
    nda::h5_read(grad_grp, "Vq0_Qskij", Vq0_Qkij, std::tuple{all, 0, all, all, all});
    nda::h5_read(grad_grp, "Vq0_3PQsk_dP", Vq0_3PQk_dP, std::tuple{all, all, all, 0, all});
    nda::h5_read(grad_grp, "Vq0_PQsk", Vq0_PQk, std::tuple{all, all, 0, all});
    nda::h5_read(grad_grp, "Vq0_PQsk_inv", Vq0_PQk_inv, std::tuple{all, all, 0, all});

    ComplexType tmp_grad = 0;

    int bnd_begin = bnd_slice(iatom, 0);
    int bnd_end = bnd_slice(iatom, 1);
    int bnd_begin_aux = bnd_slice_aux(iatom, 0);
    int bnd_end_aux = bnd_slice_aux(iatom, 1);

    // d/dX ( Q | i, j )
    auto Vq0_Qkij_dQij = nda::array<ComplexType, 4>::zeros(_nbnd_aux, _nkpts, _nbnd, _nbnd);
    auto Vq0_PQk_dPQ = nda::array<ComplexType, 3>::zeros(_nbnd_aux, _nbnd_aux, _nkpts);

    // related i and j
    for (int iQ = 0; iQ < _nbnd_aux; ++iQ) {
      for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
        Vq0_Qkij_dQij(iQ, ikpt, nda::range(bnd_begin, bnd_end), all)
          -= Vq0_3Qkij_di(direction, iQ, ikpt, nda::range(bnd_begin, bnd_end), all);
        Vq0_Qkij_dQij(iQ, ikpt, all, nda::range(bnd_begin, bnd_end))
          -= nda::transpose(Vq0_3Qkij_di(direction, iQ, ikpt, nda::range(bnd_begin, bnd_end), all));
      }
    }

    // related Q
    if (_auxbasis_response) {
      Vq0_Qkij_dQij(nda::range(bnd_begin_aux, bnd_end_aux), all, all, all)
        -= Vq0_3Qkij_dQ(direction, nda::range(bnd_begin_aux, bnd_end_aux),  all, all, all);
    }

    // d/dX ( P | Q )
    if (_auxbasis_response) {
      for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
        Vq0_PQk_dPQ(nda::range(bnd_begin_aux, bnd_end_aux), all, ikpt)
          -= Vq0_3PQk_dP(direction, nda::range(bnd_begin_aux, bnd_end_aux), all, ikpt);
        Vq0_PQk_dPQ(all, nda::range(bnd_begin_aux, bnd_end_aux), ikpt)
          -= nda::transpose(Vq0_3PQk_dP(direction, nda::range(bnd_begin_aux, bnd_end_aux), all, ikpt));
      }
    }

    // coulomb
    for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
      auto dm_total = nda::array<ComplexType, 2>::zeros(_nbnd, _nbnd);
      for (int ispin = 0; ispin < _nspin; ++ispin) {
        dm_total += _dm(ispin, ikpt, all, all);
      }
      auto V_Pij_P_ij = nda::reshape(Vq0_Qkij(all, ikpt, all, all), std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
      auto V_dQkl_Q_kl = nda::reshape(Vq0_Qkij_dQij(all, ikpt, all, all), std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
      auto V_PQinv_P_Q = nda::reshape(Vq0_PQk_inv(all, all, ikpt), std::array<int, 2>({_nbnd_aux, _nbnd_aux}));
      auto tmp_1_ij = nda::reshape(dm_total(all, all), std::array<int, 2>({1, _nbnd * _nbnd}));
      auto tmp_kl_1 = nda::reshape(dm_total(all, all), std::array<int, 2>({_nbnd * _nbnd, 1}));
      auto tmp_1_P = nda::array<ComplexType, 2>::zeros({1, _nbnd_aux});
      nda::blas::gemm(1, tmp_1_ij, nda::transpose(V_Pij_P_ij), 0, tmp_1_P);
      auto tmp_Q_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
      nda::blas::gemm(1, V_dQkl_Q_kl, tmp_kl_1, 0, tmp_Q_1);
      auto tmp_1_Q = nda::array<ComplexType, 2>::zeros({1, _nbnd_aux});
      nda::blas::gemm(1, tmp_1_P, V_PQinv_P_Q, 0, tmp_1_Q);
      auto tmp = nda::array<ComplexType, 2>::zeros({1, 1});
      nda::blas::gemm(1, tmp_1_Q, tmp_Q_1, 0, tmp);
      tmp_grad += tmp(0, 0) * _k_weight(ikpt);
      // (ij|P) (P|R)^{-1} d/dX(R|S) (S|Q)^{-1} (Q|kl)
      if (_auxbasis_response) {
        auto V_Qkl_Q_kl = nda::reshape(Vq0_Qkij(all, ikpt, all, all), std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
        auto V_dRS_R_S = nda::reshape(Vq0_PQk_dPQ(all, all, ikpt), std::array<int, 2>({_nbnd_aux, _nbnd_aux}));
        auto V_SQinv_S_Q = nda::reshape(Vq0_PQk_inv(all, all, ikpt), std::array<int, 2>({_nbnd_aux, _nbnd_aux}));
        auto tmp_1_ij = nda::reshape(dm_total(all, all), std::array<int, 2>({1, _nbnd * _nbnd}));
        auto tmp_kl_1 = nda::reshape(dm_total(all, all), std::array<int, 2>({_nbnd * _nbnd, 1}));
        auto tmp_1_P = nda::array<ComplexType, 2>::zeros({1, _nbnd_aux});
        auto tmp_1_R = nda::array<ComplexType, 2>::zeros({1, _nbnd_aux});
        auto tmp_Q_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
        auto tmp_S_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
        nda::blas::gemm(1, tmp_1_ij, nda::transpose(V_Qkl_Q_kl), 0, tmp_1_P);
        nda::blas::gemm(1, tmp_1_P, V_SQinv_S_Q, 0, tmp_1_R);
        nda::blas::gemm(1, V_Qkl_Q_kl, tmp_kl_1, 0, tmp_Q_1);
        nda::blas::gemm(1, V_SQinv_S_Q, tmp_Q_1, 0, tmp_S_1);
        auto tmp_R_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
        nda::blas::gemm(1, V_dRS_R_S, tmp_S_1, 0, tmp_R_1);
        auto tmp = nda::array<ComplexType, 2>::zeros({1, 1});
        nda::blas::gemm(1, tmp_1_R, tmp_R_1, 0, tmp);
        tmp_grad -= tmp(0, 0) * 0.5 * _k_weight(ikpt);
      }
    }
    // exchange
    for (int ispin = 0; ispin < _nspin; ispin++) {
      for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
        auto V_Pij_Pi_j = nda::reshape(Vq0_Qkij(all, ikpt, all, all), std::array<int, 2>({_nbnd_aux * _nbnd, _nbnd}));
        auto V_dQkl_Qk_l = nda::reshape(Vq0_Qkij_dQij(all, ikpt, all, all), std::array<int, 2>({_nbnd_aux * _nbnd, _nbnd}));
        auto V_PQinv_P_Q = nda::reshape(Vq0_PQk_inv(all, all, ikpt), std::array<int, 2>({_nbnd_aux, _nbnd_aux}));
        auto tmp_Pi_k = nda::array<ComplexType, 2>::zeros({_nbnd_aux * _nbnd, _nbnd});
        nda::blas::gemm(1, V_Pij_Pi_j, _dm(ispin, ikpt, all, all), 0, tmp_Pi_k);
        auto tmp_Qk_i = nda::array<ComplexType, 2>::zeros({_nbnd_aux * _nbnd, _nbnd});
        nda::blas::gemm(1, V_dQkl_Qk_l, _dm(ispin, ikpt, all, all), 0, tmp_Qk_i);
        auto tmp_Q_ki = nda::reshape(tmp_Qk_i, std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
        auto tmp_P_ki = nda::array<ComplexType, 2>::zeros({_nbnd_aux, _nbnd * _nbnd});
        nda::blas::gemm(1, V_PQinv_P_Q, tmp_Q_ki, 0, tmp_P_ki);
        auto tmp_P_i_k = nda::reshape(tmp_Pi_k, std::array<int, 3>({_nbnd_aux, _nbnd, _nbnd}));
        auto tmp_P_k_i = nda::reshape(tmp_P_ki, std::array<int, 3>({_nbnd_aux, _nbnd, _nbnd}));
        for (int iP = 0; iP < _nbnd_aux; ++iP) {
          auto tmp_i_k = nda::make_regular(nda::transpose(tmp_P_k_i(iP, all, all)));
          double spin_factor = (_nspin == 1 and _npol == 1) ? 0.5 : 1.0;
          tmp_grad -= nda::sum(tmp_P_i_k(iP, nda::ellipsis{}) * tmp_i_k) * spin_factor * _k_weight(ikpt);
        }
        // (il|P) (P|Q)^{-1} d/dX(R|S) (S|Q)^{-1} (Q|kj)
        if (_auxbasis_response) {
          auto V_Qkl_Qk_l = nda::reshape(Vq0_Qkij(all, ikpt, all, all), std::array<int, 2>({_nbnd_aux * _nbnd, _nbnd}));
          auto V_dRS_R_S = nda::reshape(Vq0_PQk_dPQ(all, all, ikpt), std::array<int, 2>({_nbnd_aux, _nbnd_aux}));
          auto V_SQinv_S_Q = nda::reshape(Vq0_PQk_inv(all, all, ikpt), std::array<int, 2>({_nbnd_aux, _nbnd_aux}));
          auto tmp_Qk_i = nda::array<ComplexType, 2>::zeros({_nbnd_aux * _nbnd, _nbnd});
          nda::blas::gemm(1, V_Qkl_Qk_l, _dm(ispin, ikpt, all, all), 0, tmp_Qk_i);
          auto tmp_Q_ki = nda::reshape(tmp_Qk_i, std::array<int, 2>({_nbnd_aux , _nbnd * _nbnd}));
          auto tmp_S_ki = nda::array<ComplexType, 2>::zeros({_nbnd_aux, _nbnd * _nbnd});
          nda::blas::gemm(1, V_SQinv_S_Q, tmp_Q_ki, 0, tmp_S_ki);
          auto tmp_R_ki = nda::array<ComplexType, 2>::zeros({_nbnd_aux, _nbnd * _nbnd});
          nda::blas::gemm(1, V_dRS_R_S, tmp_S_ki, 0, tmp_R_ki);
          auto tmp_R_i_k = nda::reshape(tmp_S_ki, std::array<int, 3>({_nbnd_aux, _nbnd, _nbnd}));
          auto tmp_R_k_i = nda::reshape(tmp_R_ki, std::array<int, 3>({_nbnd_aux, _nbnd, _nbnd}));
          for (int iR = 0; iR < _nbnd_aux; ++iR) {
            auto tmp_i_k = nda::make_regular(nda::transpose(tmp_R_k_i(iR, all, all)));
            double spin_factor = (_nspin == 1 and _npol == 1) ? 0.25 : 0.5;
            tmp_grad += nda::sum(tmp_R_i_k(iR, nda::ellipsis{}) * tmp_i_k) * spin_factor * _k_weight(ikpt);
          }
        }
      }
    }
    return tmp_grad;
  }

  ComplexType mf_gradient_t::evaluate_pulay(int iatom, int direction) {
    ComplexType tmp_grad = 0;
    auto S_grad = _MF->S_grad();
    for (int ispin = 0; ispin < _nspin; ++ispin) {
      for (int ikpt = 0; ikpt < _nkpts; ++ikpt) {
        tmp_grad -= _k_weight(ikpt) * nda::sum(_dme(ispin, ikpt, nda::ellipsis{}) *
                                               S_grad(iatom, direction, ispin, ikpt, nda::ellipsis{}));
      }
    }
    return tmp_grad;
  }

  void mf_gradient_t::print_gradients(const nda::array<ComplexType, 2>& gradients, const std::string& str, bool bohr) {
    double factor;
    std::string unit;
    if (bohr) {
      factor = 1;
      unit = "(hartree/bohr)";
    } else {
      factor = _bohr_to_angstrom;
      unit = "(hartree/angstrom)";
    }

    app_log(1, "  {}", str);
    app_log(1, "  -----------------------------------------------------------------------------------");
    app_log(1, "   id    nuc     X {0:<18}    Y {0:<18}    Z {0:<18}", unit);
    app_log(1, "  -----------------------------------------------------------------------------------");
    for (int iatom = 0; iatom < _natoms; ++iatom) {
      app_log(1, "   {:<5} {:<5}   {:<+20.10f}    {:<+20.10f}    {:<+20.10f}",
        iatom, _MF->atomic_id(iatom),
        nda::real(gradients(iatom, 0) * factor),
        nda::real(gradients(iatom, 1) * factor),
        nda::real(gradients(iatom, 2) * factor));
    }
    app_log(1, "\n");
  }

  void mf_gradient_t::print_timers() {
    app_log(1, "  MF-GRAD timers");
    app_log(1, "  --------------");
    app_log(1, "    Total:                 {0:.3f} sec", _Timer.elapsed("TOTAL"));
    app_log(1, "    1-electron:            {0:.3f} sec", _Timer.elapsed("1-ELECTRON"));
    app_log(1, "    2-electron:            {0:.3f} sec", _Timer.elapsed("2-ELECTRON"));
    app_log(1, "    pulay:                 {0:.3f} sec", _Timer.elapsed("PULAY"));
    app_log(1, "    nuclei:                {0:.3f} sec", _Timer.elapsed("NUCLEI"));
    app_log(1, "\n");
  }

  void mf_gradient_t::write_output() {
    std::ofstream fout(_output + ".txt", std::ios::app);
    for (int iatom = 0; iatom < _natoms; ++iatom) {
      for (int direction = 0; direction < 3; ++direction) {
        fout << std::scientific << std::setprecision(10) << std::setw(20) << nda::real(_gradient_total(iatom, direction));
      }
      fout << std::endl;
    }
    fout.close();
    app_log(1, "  {} {}", "Gradients written to", _output + ".txt");
    app_log(1, "\n");
  }

} // namespace mf
