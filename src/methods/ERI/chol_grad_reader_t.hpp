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


#ifndef COQUI_CHOLESKY_GRADIENT_READER_HPP
#define COQUI_CHOLESKY_GRADIENT_READER_HPP

#include <string>

#include "mpi3/communicator.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"

#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"

#include "utilities/Timer.hpp"

#include "methods/ERI/chol_reader_t.hpp"

namespace methods
{

  namespace mpi3 = boost::mpi3;

  class chol_grad_reader_t {

    template<nda::MemoryArray local_Array_t>
    using dArray_t = math::nda::distributed_array<local_Array_t,mpi3::communicator>;

  private:

    std::shared_ptr<mf::MF> _MF;
    std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi;
    std::string _eri_grad_dir;
    std::string _eri_grad_filename;
    eri_storage_e _storage;
    chol_reading_type_e _read_type;
    chol_writing_type_e _write_type;

    int _ns = 0;
    int _ns_in_basis = 0;
    int _nkpts = 0;
    int _Np = 0;
    int _nbnd = 0;
    int _naux = 0;
    double _tol = 0.0;

    int _natoms = 0;

    int _is = -1;
    int _ik = -1;
    int _iq = -1;

    std::optional<nda::array<ComplexType, 4>> _V_3Qij_di;
    std::optional<nda::array<ComplexType, 4>> _V_3Qij_dQ;
    std::optional<nda::array<ComplexType, 3>> _V_Qij;
    std::optional<nda::array<ComplexType, 3>> _V_3PQ_dP;
    std::optional<nda::array<ComplexType, 2>> _V_PQ;
    std::optional<nda::array<ComplexType, 2>> _V_PQ_inv;

    std::optional<nda::array<ComplexType, 5>> _Vq_k3Qij_di;
    std::optional<nda::array<ComplexType, 5>> _Vq_k3Qij_dQ;
    std::optional<nda::array<ComplexType, 4>> _Vq_kQij;
    std::optional<nda::array<ComplexType, 4>> _Vq_k3PQ_dP;
    std::optional<nda::array<ComplexType, 3>> _Vq_kPQ;
    std::optional<nda::array<ComplexType, 3>> _Vq_kPQ_inv;

    mutable utils::TimerManager _Timer;

    void init()
    {
      _Np = read_Np();
      std::string filename = _eri_grad_dir + "/" + _eri_grad_filename;
      h5::file file = h5::file(filename, 'r');
      h5::group grp(file);
      h5::group sgrp = grp.open_group("Interaction_Gradient");
      h5::h5_read(sgrp, "natoms", _natoms);
      h5::h5_read(sgrp, "tol", _tol);

      std::string rtype = (_read_type == single_kpair)? "single_kpair" : "each_q";
      app_log(1, "*******************************");
      app_log(1, " Cholesky ERI Gradient Reader: ");
      app_log(1, "*******************************");
      app_log(1, "    - natoms = {}", _natoms);
      app_log(1, "    - Np max  = {}", _Np);
      app_log(1, "    - accuracy = {}", _tol);
      app_log(1, "    - read mode = {}", rtype);
      app_log(1, "    - eri storage: {}", eriform_enum_to_string(_storage));
      app_log(1, "    - ERI gradient dir = {}", _eri_grad_dir);
      app_log(1, "    - ERI gardient output = {}\n", _eri_grad_filename);
    }

    public:

    /**
     * Read V_grad^{K(ik), K(ik)-Q(iq)}
     * @param iq
     * @param ik
     */
    void read_V_grad(size_t iq, size_t is, size_t ik)
    {
      decltype(nda::range::all) all;
      utils::check(_read_type == single_kpair, "Error: read_V() can only be called in \"single_kpair\" read mode");
      utils::check( is < _ns, "Error in chol_reader_grad_t::read_Vq_grad: is out of bounds: is:{}", is);

      _is = std::min(is, size_t(_ns_in_basis-1));
      _ik = ik;
      _iq = iq;

      if (_Vq_k3Qij_di) {
        _Vq_k3Qij_di = std::nullopt;
      }
      if (_Vq_k3Qij_dQ) {
        _Vq_k3Qij_dQ = std::nullopt;
      }
      if (_Vq_kQij) {
        _Vq_kQij = std::nullopt;
      }
      if (_Vq_k3PQ_dP) {
        _Vq_k3PQ_dP = std::nullopt;
      }
      if (_Vq_kPQ) {
        _Vq_kPQ = std::nullopt;
      }
      if (_Vq_kPQ_inv) {
        _Vq_kPQ_inv = std::nullopt;
      }

      _Timer.start("READ");

      // ( d/dX i, j | Q )
      if (!_V_3Qij_di) {
        _V_3Qij_di.emplace(nda::array<ComplexType, 4>(3, _Np, _nbnd, _nbnd));
      } else {
        _V_3Qij_di.value()() = 0.0;
      }
      // ( i, j | d/dX Q )
      if (!_V_3Qij_dQ) {
        _V_3Qij_dQ.emplace(nda::array<ComplexType, 4>(3, _Np, _nbnd, _nbnd));
      } else {
        _V_3Qij_dQ.value()() = 0.0;
      }
      // ( i, j | P )
      if (!_V_Qij) {
        _V_Qij.emplace(nda::array<ComplexType, 3>(_Np, _nbnd, _nbnd));
      } else {
        _V_Qij.value()() = 0.0;
      }
      // ( d/dX P | Q )
      if (!_V_3PQ_dP) {
        _V_3PQ_dP.emplace(nda::array<ComplexType, 3>(3, _Np, _Np));
      } else {
        _V_3PQ_dP.value()() = 0.0;
      }
      // ( P | Q )
      if (!_V_PQ) {
        _V_PQ.emplace(nda::array<ComplexType, 2>(_Np, _Np));
      } else {
        _V_PQ.value()() = 0.0;
      }
      // ( P | Q )^{-1}
      if (!_V_PQ_inv) {
        _V_PQ_inv.emplace(nda::array<ComplexType, 2>(_Np, _Np));
      } else {
        _V_PQ_inv.value()() = 0.0;
      }

      std::string dataset;
      std::string filename = _eri_grad_dir + "/" + (_write_type == multi_file ?
                                                    "Vq" + std::to_string(iq) + "_grad.h5" :
                                                    _eri_grad_filename);
      h5::file file = h5::file(filename, 'r');
      h5::group grp(file);
      h5::group sgrp = grp.open_group("Interaction_Gradient");
      // ( Q | d/dX i, j )
      dataset = "Vq" + std::to_string(iq) + "_3Qskij_di";
      auto V_3Qij_di = _V_3Qij_di.value()(nda::range(3), nda::range(_naux), nda::ellipsis{});
      nda::h5_read(sgrp, dataset, V_3Qij_di,
        std::tuple{all, all, std::min(is, size_t(_ns_in_basis-1)), ik, nda::range(_nbnd), nda::range(_nbnd)});
      // ( Q | d/dX i, j )
      dataset = "Vq" + std::to_string(iq) + "_3Qskij_dQ";
      auto V_3Qij_dQ = _V_3Qij_dQ.value()(nda::range(3), nda::range(_naux), nda::ellipsis{});
      nda::h5_read(sgrp, dataset, V_3Qij_dQ,
        std::tuple{all, all, std::min(is, size_t(_ns_in_basis-1)), ik, nda::range(_nbnd), nda::range(_nbnd)});
      // ( P | i, j )
      dataset = "Vq" + std::to_string(iq) + "_Qskij";
      auto V_Qij = _V_Qij.value()(nda::range(_naux), nda::ellipsis{});
      nda::h5_read(sgrp, dataset, V_Qij,
        std::tuple{all, std::min(is, size_t(_ns_in_basis-1)), ik, nda::range(_nbnd), nda::range(_nbnd)});
      // ( d/dX P | Q )
      dataset = "Vq" + std::to_string(iq) + "_3PQsk_dP";
      auto V_3PQ_dP = _V_3PQ_dP.value()(nda::range(3), nda::range(_naux), nda::range(_naux), nda::ellipsis{});
      nda::h5_read(sgrp, dataset, V_3PQ_dP,
        std::tuple{all, all, all, std::min(is, size_t(_ns_in_basis-1)), ik});
      // ( P | Q )
      dataset = "Vq" + std::to_string(iq) + "_PQsk";
      auto V_PQ =  _V_PQ.value()(nda::range(_naux), nda::range(_naux), nda::ellipsis{});
      nda::h5_read(sgrp, dataset, V_PQ,
        std::tuple{all, all, std::min(is, size_t(_ns_in_basis-1)), ik});
      // ( P | Q ) ^{-1}
      dataset = "Vq" + std::to_string(iq) + "_PQsk_inv";
      auto V_PQ_inv =  _V_PQ_inv.value()(nda::range(_naux), nda::range(_naux), nda::ellipsis{});
      nda::h5_read(sgrp, dataset, V_PQ_inv,
        std::tuple{all, all, std::min(is, size_t(_ns_in_basis-1)), ik});

      _Timer.stop("END");

    }

    /**
     * Read V_grad^{K(ik), K(ik)-Q(iq)} for all ik
     * @param iq
     */
    void read_Vq_grad(size_t iq, size_t is)
    {
      decltype(nda::range::all) all;
      utils::check( is < _ns, "Error in chol_reader_grad_t::read_Vq_grad: is out of bounds: is:{}", is);
      utils::check(_read_type == each_q, "Error: read_V() can only be called in \"each_q\" read mode");

      _is = std::min(is, size_t(_ns_in_basis-1));
      _iq = iq;

      if (_V_3Qij_di) {
       _V_3Qij_di = std::nullopt;
      }
      if (_V_3Qij_dQ) {
        _V_3Qij_dQ = std::nullopt;
      }
      if (_V_Qij) {
        _V_Qij = std::nullopt;
      }
      if (_V_3PQ_dP) {
        _V_3PQ_dP = std::nullopt;
      }
      if (_V_PQ) {
        _V_PQ = std::nullopt;
      }
      if (_V_PQ_inv) {
        _V_PQ_inv = std::nullopt;
      }

      _Timer.start("READ");

      // ( Q | d/dX i, j )
      if (!_Vq_k3Qij_di) {
        _Vq_k3Qij_di.emplace(nda::array<ComplexType, 5>(_nkpts, 3, _Np, _nbnd, _nbnd));
      } else {
        _Vq_k3Qij_di.value()() = 0.0;
      }
      // ( d/dX Q | i, j )
      if (!_Vq_k3Qij_dQ) {
        _Vq_k3Qij_dQ.emplace(nda::array<ComplexType, 5>(_nkpts, 3, _Np, _nbnd, _nbnd));
      } else {
        _Vq_k3Qij_dQ.value()() = 0.0;
      }
      // ( i , j | P )
      if (!_Vq_kQij) {
        _Vq_kQij.emplace(nda::array<ComplexType, 4>(_nkpts, _Np, _nbnd, _nbnd));
      } else {
        _Vq_kQij.value()() = 0.0;
      }
      // ( d/dX P | Q )
      if (!_Vq_k3PQ_dP) {
        _Vq_k3PQ_dP.emplace(nda::array<ComplexType, 4>(_nkpts, 3, _Np, _Np));
      } else {
        _Vq_k3PQ_dP.value()() = 0.0;
      }
      // ( P | Q )
      if (!_Vq_kPQ) {
        _Vq_kPQ.emplace(nda::array<ComplexType, 3>(_nkpts, _Np, _Np));
      } else {
        _Vq_kPQ.value()() = 0.0;
      }
      // ( P | Q )^{-1}
      if (!_Vq_kPQ_inv) {
        _Vq_kPQ_inv.emplace(nda::array<ComplexType, 3>(_nkpts, _Np, _Np));
      } else {
        _Vq_kPQ_inv.value()() = 0.0;
      }

      std::string dataset;
      std::string filename = _eri_grad_dir + "/" + (_write_type == multi_file ?
                                                    "Vq" + std::to_string(iq) + "_grad.h5" :
                                                    _eri_grad_filename);
      h5::file file = h5::file(filename, 'r');
      h5::group grp(file);
      h5::group sgrp = grp.open_group("Interaction_Gradient");
      for (size_t ik = 0; ik < _nkpts; ++ik) {
        // (P | d/dX i, j)
        dataset = "Vq" + std::to_string(iq) + "_3Qskij_di";
        auto Vq_k3Qij_di = _Vq_k3Qij_di.value()(ik, nda::range(3), nda::range(_naux), nda::ellipsis{});
        nda::h5_read(sgrp, dataset, Vq_k3Qij_di,
          std::tuple{all, all, std::min(is, size_t(_ns_in_basis-1)), ik, nda::range(_nbnd), nda::range(_nbnd)});
        // (d/dX P | i, j)
        dataset = "Vq" + std::to_string(iq) + "_3Qskij_dQ";
        auto Vq_k3Qij_dQ = _Vq_k3Qij_dQ.value()(ik, nda::range(3), nda::range(_naux), nda::ellipsis{});
        nda::h5_read(sgrp, dataset, Vq_k3Qij_dQ,
          std::tuple{all, all, std::min(is, size_t(_ns_in_basis-1)), ik, nda::range(_nbnd), nda::range(_nbnd)});
        // ( i, j | P )
        dataset = "Vq" + std::to_string(iq) + "_Qskij";
        auto Vq_kQij = _Vq_kQij.value()(ik, nda::range(_naux), nda::ellipsis{});
        nda::h5_read(sgrp, dataset, Vq_kQij,
          std::tuple{all, std::min(is, size_t(_ns_in_basis-1)), ik, nda::range(_nbnd), nda::range(_nbnd)});
        // ( d/dX P | Q )
        dataset = "Vq" + std::to_string(iq) + "_3PQsk_dP";
        auto Vq_k3PQ_dP = _Vq_k3PQ_dP.value()(ik, nda::range(3), nda::range(_naux), nda::range(_naux), nda::ellipsis{});
        nda::h5_read(sgrp, dataset, Vq_k3PQ_dP,
          std::tuple{all, all, all, std::min(is, size_t(_ns_in_basis-1)), ik});
        // ( P | Q )
        dataset = "Vq" + std::to_string(iq) + "_PQsk";
        auto Vq_kPQ =  _Vq_kPQ.value()(ik, nda::range(_naux), nda::range(_naux), nda::ellipsis{});
        nda::h5_read(sgrp, dataset, Vq_kPQ,
        std::tuple{all, all, std::min(is, size_t(_ns_in_basis-1)), ik});
        // ( P | Q )^{-1}
        dataset = "Vq" + std::to_string(iq) + "_PQsk_inv";
        auto Vq_kPQ_inv =  _Vq_kPQ_inv.value()(ik, nda::range(_naux), nda::range(_naux), nda::ellipsis{});
        nda::h5_read(sgrp, dataset, Vq_kPQ_inv,
          std::tuple{all, all, std::min(is, size_t(_ns_in_basis-1)), ik});
      }

      _Timer.stop("END");

    }

  chol_grad_reader_t(std::shared_ptr<mf::MF> MF, ptree const& pt):
    _MF(std::move(MF)),
    _mpi(_MF->mpi()),
    _eri_grad_dir(io::get_value_with_default<std::string>(pt, "path","./")),
    _eri_grad_filename(io::get_value_with_default<std::string>(pt, "output", "chol_grad_info.h5")),
    _storage((_eri_grad_dir == "") ? incore : outcore),
    _read_type(io::tolower_copy(io::get_value_with_default<std::string>(pt,"read_type","all")) == "all" ? each_q : single_kpair),
    _write_type(io::tolower_copy(io::get_value_with_default<std::string>(pt,"write_type","multi")) == "multi" ? multi_file : single_file),
    _ns(_MF->nspin()),
    _ns_in_basis(_MF->nspin_in_basis()),
    _nkpts(_MF->nkpts()),
    _Np(0),
    _nbnd(_MF->nbnd()),
    _naux(_MF->nbnd_aux()),
    _tol(io::get_value_with_default<double>(pt, "tol", 0.0001)),
    _Timer()
  {
    utils::check(_storage != incore, "chol_grad_rader_t: incore version is not implemented yet!");

    for( auto& v: {"BUILD", "READ"}) {
      _Timer.add(v);
    }

    init();
  }

  // read from existing CD/GDF integral gradient
    chol_grad_reader_t(std::shared_ptr<mf::MF> MF,
                       std::string eri_grad_dir = "./",
                       std::string eri_grad_filename = "chol_grad_info.h5",
                       chol_reading_type_e read_type = each_q,
                       chol_writing_type_e write_type = multi_file):
      _MF(std::move(MF)),
      _mpi(_MF->mpi()),
      _eri_grad_dir(eri_grad_dir),
      _eri_grad_filename(eri_grad_filename),
      _storage((eri_grad_dir == "")? incore : outcore),
      _read_type(read_type),
      _write_type(write_type),
      _ns(_MF->nspin()),
      _ns_in_basis(_MF->nspin_in_basis()),
      _nkpts(_MF->nkpts()),
      _Np(0),
      _nbnd(_MF->nbnd()),
      _naux(_MF->nbnd_aux()),
      _tol(-1.0),
      _Timer() {

      utils::check(_storage != incore, "chol_grad_reader_t: incore version is not implemented yet!");
      if (!std::filesystem::exists(_eri_grad_dir + "/" + _eri_grad_filename)) {
          utils::check(false, "chol_grad_reader_t: Cholesky ERI gradients not found!");
        }

      for (auto& v: {"BUILD", "READ"}) {
        _Timer.add(v);
      }

      init();
    }

    int read_Np(long iq = -1)
    {
      int Np;
      if (iq == -1) { // read from meta_data
        std::string filename = _eri_grad_dir + "/" + _eri_grad_filename;
        h5::file file = h5::file(filename, 'r');
        h5::group grp(file);
        auto sgrp = grp.open_group("Interaction_Gradient");
        h5::h5_read(sgrp, "Np", Np);
      } else {
        std::string filename = _eri_grad_dir + "/" + (_write_type == multi_file ?
                                                      "Vq" + std::to_string(iq) + ".h5" :
                                                      _eri_grad_filename);
        h5::file file = h5::file(filename, 'r');
        h5::group grp(file);
        auto sgrp = grp.open_group("Interaction_Gradient");
        std::string dataset = "Vq" + std::to_string(iq) + "_grad";
        auto l = h5::array_interface::get_dataset_info(sgrp, dataset);
        Np = l.lengths[0];
      }
      return Np;
    }

    auto& mpi() const { return _mpi; }
    auto& MF() const { return _MF; }
    int nspin() const { return _ns; }
    int nkpts() const { return _nkpts; }
    int Np() const { return _Np; }
    int nbnd() const { return _nbnd; }
    int nbnd_aux() const { return _naux; }
    chol_reading_type_e& set_read_type() { return _read_type; }
    chol_reading_type_e chol_read_type() const { return _read_type; }
    chol_writing_type_e chol_write_type() const { return _write_type; }

    auto V_3Qij_di(size_t iq, size_t is, size_t ik)
    {
      if (_read_type == single_kpair) {
        return _V_3Qij_di.value()();
      } else {
        return _Vq_k3Qij_di.value()(ik, nda::range::all, nda::range::all, nda::range::all, nda::range::all);
      }
    }

    auto V_3Qij_dQ(size_t iq, size_t is, size_t ik)
    {
      if (_read_type == single_kpair) {
        return _V_3Qij_dQ.value()();
      } else {
        return _Vq_k3Qij_dQ.value()(ik, nda::range::all, nda::range::all, nda::range::all, nda::range::all);
      }
    }

    auto V_Qij(size_t iq, size_t is, size_t ik)
    {
      if (_read_type == single_kpair) {
        return _V_Qij.value()();
      } else {
        return _Vq_kQij.value()(ik, nda::range::all, nda::range::all, nda::range::all);
      }
    }

    auto V_3PQ_dP(size_t iq, size_t is, size_t ik)
    {
      if (_read_type == single_kpair) {
        return _V_3PQ_dP.value()();
      } else {
        return _Vq_k3PQ_dP.value()(ik, nda::range::all, nda::range::all, nda::range::all);
      }
    }

    auto V_PQ(size_t iq, size_t is, size_t ik)
    {
      if (_read_type == single_kpair) {
        return _V_PQ.value()();
      } else {
        return _Vq_kPQ.value()(ik, nda::range::all, nda::range::all);
      }
    }

    auto V_PQ_inv(size_t iq, size_t is, size_t ik)
    {
      if (_read_type == single_kpair) {
        return _V_PQ_inv.value()();
      } else {
        return _Vq_kPQ_inv.value()(ik, nda::range::all, nda::range::all);
      }
    }

     /**
      * Checks if the output file can be used to initializing an object
      * @param path       - [INPUT] Directory to store ERIs
      * @param output     - [INPUT] ERI h5 file
      * @param nq         - [INPUT] Number of q-points
      * @param write_type - [INPUT] Write type: multi_file or single_file
      * @return - Whether the Cholesky ERI gradients are initialized already.
      */
    static bool check_init(std::string path, std::string output, int nq, chol_writing_type_e write_type)
    {
      if (!std::filesystem::exists(path + "/" + output)) {
        return false;
      }
      std::string filename = path + "/" + output;
      h5::file file = h5::file(filename, 'r');
      h5::group grp(file);
      if (not grp.has_subgroup("Interaction_Gradient")) { return false; }
      h5::group sgrp = grp.open_group("Interaction_Gradient");
      if (not (sgrp.has_key("tol") and sgrp.has_key("Np"))) { return false; }
      if (write_type == single_file) {
        for (size_t iq = 0; iq < nq; ++iq) {
          // (d/dX i,j|Q)
          if (not sgrp.has_dataset("Vq" + std::to_string(iq) + "_3Qskij_di")) { return false; }
          // (i,j|d/dX Q)
          if (not sgrp.has_dataset("Vq" + std::to_string(iq) + "_3Qskij_dQ")) { return false; }
          // (i,j|Q)
          if (not sgrp.has_dataset("Vq" + std::to_string(iq) + "_Qskij")) { return false; }
          // (d/dX P|Q)
          if (not sgrp.has_dataset("Vq" + std::to_string(iq) + "_3PQsk_dP")) { return false; }
          // (P|Q)
          if (not sgrp.has_dataset("Vq" + std::to_string(iq) + "_PQsk")) { return false; }
          // (P|Q)^{-1}
          if (not sgrp.has_dataset("Vq" + std::to_string(iq) + "_PQsk_inv")) { return false; }
        }
      } else {
        for (size_t iq = 0; iq < nq; ++iq) {
          std::string Vq_file = path + "/Vq" + std::to_string(iq) + "_grad.h5";
          if (!std::filesystem::exists(Vq_file)) { return false; }
          h5::file f = h5::file(Vq_file, 'r');
          h5::group g(f);
          if (not g.has_subgroup("Interaction_Gradient")) { return false; }
          h5::group sg = g.open_group("Interaction_Gradient");
          // (d/dX i,j|Q)
          if (not sg.has_dataset("Vq" + std::to_string(iq) + "_3Qskij_di")) { return false; }
          // (i,j|d/dX Q)
          if (not sg.has_dataset("Vq" + std::to_string(iq) + "_3Qskij_dQ")) { return false; }
          // (i,j|P)
          if (not sg.has_dataset("Vq" + std::to_string(iq) + "_Qskij")) { return false; }
          // (d/dX P|Q)
          if (not sg.has_dataset("Vq" + std::to_string(iq) + "_3PQsk_dP")) { return false; }
          // (P|Q)
          if (not sg.has_dataset("Vq" + std::to_string(iq) + "_PQsk")) { return false; }
          // (P|Q)^{-1}
          if (not sg.has_dataset("Vq" + std::to_string(iq) + "_PQsk_inv")) { return false; }
        }
      }
      return true;
    }

  };

} // namespace methods

#endif
