"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==========================================================================
"""

import sys
import numpy as np


def set_epsilon(prec_str: str):
    if isinstance(prec_str, str):
        if prec_str == "high":
            return 1e-15
        elif prec_str == "medium":
            return 1e-10
        elif prec_str == "low":
            return 1e-6
        else:
            raise ValueError(f"Unknown precision string: {prec_str}. Acceptable options are 'high', 'medium', and 'low'.")
    return -1.0


def set_lambda(lmbda, coqui_cxx_style=False):
    if coqui_cxx_style:
        if lmbda > 0 and lmbda <= 100:
            return 100
        elif lmbda > 100 and lmbda <= 1000:
            return 1000
        elif lmbda > 1000 and lmbda <= 10000:
            return 10000
        elif lmbda > 10000 and lmbda <= 100000:
            return 100000
        elif lmbda > 100000 and lmbda <= 1000000:
            return 1000000
        else:
            raise ValueError(f"Invalid lambda value: {lmbda}. Acceptable range is (0, 1000000]")
    return lmbda


def _normalize_stats(stats_input: str) -> str:
    """
    Normalize stats input to short form ('f' or 'b').
    Accepts both short forms ('f', 'b') and long forms ('fermion', 'boson').

    :param stats_input: str
    Statistics string ('f', 'fermion', 'b', or 'boson')
    :return: str
    Normalized short form ('f' or 'b')
    :raises ValueError: if stats_input is not a recognized statistics format
    """
    if stats_input == 'f' or stats_input == 'fermion':
        return 'f'
    elif stats_input == 'b' or stats_input == 'boson':
        return 'b'
    else:
        raise ValueError(
            "Unknown statistics '{}'. Acceptable options are 'f'/'fermion' for fermions "
            "and 'b'/'boson' for bosons.".format(stats_input)
        )


class _IAFTIRAdapter(object):
    """
    IR-based adapter using sparse_ir backend.
    Internal implementation class; use IAFT() with basis='ir' for public API.

    Dependency:
        Requires sparse-ir (https://sparse-ir.readthedocs.io/en/latest/) version 1.1.7 with xprec support.
        To install: "pip install sparse-ir[xprec]==1.1.7".
        Note: Versions 2.0 and above of sparse-ir have not yet been tested with this code and may not be compatible.
    """
    def __init__(self, beta: float, wmax: float, *, prec: str=None, eps: float=None, verbose: bool=True):
        """
        :param beta: float
            Inverse temperature (a.u.)
        :param wmax: float
            Frequency cutoff (a.u.)
        :param prec: float
            Precision for IR basis
        """
        import sparse_ir

        self.beta = beta

        if eps is not None and (prec is None or prec == "custom"):
            if not isinstance(eps, (float, int)):
                raise TypeError("eps should be a float or integer value, got type {}".format(type(eps)))
            self.prec = "custom"
            self.eps = float(eps)
        else:
            if prec is None:
                raise ValueError("prec must be provided when eps is not provided")
            self.prec = prec
            self.eps = set_epsilon(prec)

        if prec == "custom" or prec is None:
            self.lmbda = wmax * beta
        else:
            self.lmbda = set_lambda(wmax * beta, coqui_cxx_style=True)
        self.wmax = self.lmbda / self.beta

        self.statistics = {'f', 'b'}

        self._bases = sparse_ir.FiniteTempBasisSet(beta=self.beta, wmax=self.wmax, eps=self.eps)
        self._tau_mesh_f = self._bases.smpl_tau_f.sampling_points
        self._tau_mesh_b = self._bases.smpl_tau_b.sampling_points
        self._wn_mesh_f = self._bases.smpl_wn_f.sampling_points
        self._wn_mesh_b = self._bases.smpl_wn_b.sampling_points
        self.nt_f, self.nw_f = self._tau_mesh_f.shape[0], self._wn_mesh_f.shape[0]
        self.nt_b, self.nw_b = self._tau_mesh_b.shape[0], self._wn_mesh_b.shape[0]

        Ttl_ff = self._bases.basis_f.u(self._tau_mesh_f).T
        Twl_ff = self._bases.basis_f.uhat(self._wn_mesh_f).T
        Ttl_bb = self._bases.basis_b.u(self._tau_mesh_b).T
        Twl_bb = self._bases.basis_b.uhat(self._wn_mesh_b).T

        self.Tlt_ff = np.linalg.pinv(Ttl_ff)
        self.Tlt_bb = np.linalg.pinv(Ttl_bb)
        self.Tlw_ff = np.linalg.pinv(Twl_ff)
        self.Tlw_bb = np.linalg.pinv(Twl_bb)

        # Ttw_ff = Ttl_ff * [Twl_ff]^{-1}
        self.Ttw_ff = np.dot(Ttl_ff, self.Tlw_ff)
        self.Twt_ff = np.dot(Twl_ff, self.Tlt_ff)
        self.Ttw_bb = np.dot(Ttl_bb, self.Tlw_bb)
        self.Twt_bb = np.dot(Twl_bb, self.Tlt_bb)

        # convert tau_mesh to [-1, 1] notation
        self._tau_mesh_f = 2.0 * (self._tau_mesh_f) / self.beta - 1.0
        self._tau_mesh_b = 2.0 * (self._tau_mesh_b) / self.beta - 1.0

        if verbose:
            print(self)
            sys.stdout.flush()

    def save(self, h5_grp):
        if 'imaginary_fourier_transform' in h5_grp:
            return
        h5_grp.create_group('imaginary_fourier_transform')
        h5_grp['imaginary_fourier_transform']['beta'] = self.beta
        h5_grp['imaginary_fourier_transform']['wmax'] = self.wmax
        h5_grp['imaginary_fourier_transform']['prec'] = self.prec
        h5_grp['imaginary_fourier_transform']['eps'] = self.eps
        h5_grp['imaginary_fourier_transform']['basis'] = "ir"

    def __str__(self):
        return ("Mesh details on the imaginary axis\n"
                "----------------------------------\n"
                "Intermediate Representation\n"
                "epsilon = {}\n"
                "beta = {}\n"
                "frequency cutoff = {}\n"
                "lambda = {}\n"
                "nt_f, nw_f = {}, {}\n"
                "nt_b, nw_b = {}, {}\n".format(
            self.eps, self.beta, self.wmax, self.lmbda,
            self.nt_f, self.nw_f, self.nt_b, self.nw_b))

    def __eq__(self, other):
        return (
            isinstance(other, _IAFTIRAdapter) and 
            self.beta == other.beta and
            self.wmax == other.wmax and
            self.lmbda == other.lmbda and
            self.eps == other.eps
        )

    def tau_mesh(self, stats: str, rel_notation: bool=False):
        """
        Return imaginary time mesh points, following the convention t = [-1, 1].
        :param stats: str
            statistics: 'f' for fermions and 'b' for bosons
        :param rel_notation: bool
            Whether to return relative notation (t in [-1, 1]) or absolute notation (t in [0, beta))
        :return: numpy.ndarray(dim=1)
            imaginary time mesh points in [0, beta)
        """
        stats = _normalize_stats(stats)
        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        tau_mesh = np.array(self._tau_mesh_f, dtype=float) if stats == 'f' else np.array(self._tau_mesh_b, dtype=float)
        if not rel_notation:
            tau_mesh = (tau_mesh + 1.0) * self.beta / 2.0
        return tau_mesh

    def wn_mesh(self, stats: str, ir_notation: bool=True, *, positive_only=False):
        """
        Return Matsubara frequency indices.
        :param stats: str
            statistics: 'f' for fermions and 'b' for bosons
        :param ir_notation: bool
            Whether wn_mesh_interp is in sparse_ir notation where iwn = n*pi/beta for both fermions and bosons.
            Otherwise, iwn = (2n+1)*pi/beta  for fermions and 2n*pi/beta for bosons.

        :return: numpy.ndarray(dim=1)
            Matsubara frequency indices
        """
        stats = _normalize_stats(stats)
        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        wn_mesh = np.array(self._wn_mesh_f, dtype=int) if stats == 'f' else np.array(self._wn_mesh_b, dtype=int)
        if not ir_notation:
            wn_mesh = (wn_mesh - 1) // 2 if stats == 'f' else wn_mesh // 2

        if positive_only:
            nw_half = wn_mesh.shape[0] // 2
            return wn_mesh[nw_half:]
        else:
            return wn_mesh

    def tau_to_w(self, Ot, stats: str):
        stats = _normalize_stats(stats)
        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        Twt = self.Twt_ff if stats == 'f' else self.Twt_bb
        if Ot.shape[0] != Twt.shape[1]:
            raise ValueError(
                "tau_to_w: Number of tau points are inconsistent: {} and {}".format(Ot.shape[0], Twt.shape[1]))

        Ot_shape = Ot.shape
        Ot = Ot.reshape(Ot.shape[0], -1)
        Ow = np.dot(Twt, Ot)

        Ot = Ot.reshape(Ot_shape)
        Ow = Ow.reshape((Twt.shape[0],) + Ot_shape[1:])
        return Ow

    def tau_to_w_phsym(self, Ot, stats: str):
        stats = _normalize_stats(stats)
        if stats != 'b':
            raise ValueError("FT w/ particle-hole symmetry only support bosonic correlation functions")

        nw_half = self.nw_b // 2 if self.nw_b % 2 == 0 else self.nw_b // 2 + 1
        nt_half = self.nt_b // 2 if self.nt_b % 2 == 0 else self.nt_b // 2 + 1
        if Ot.shape[0] != nt_half:
            raise ValueError(
                "tau_to_w_phsym: Number of tau points are inconsistent: {} and {}".format(Ot.shape[0], nt_half))

        Twt_pos = np.zeros((nw_half, nt_half), dtype=self.Twt_bb.dtype)
        for n in range(nw_half):
            iw = self.nw_b // 2 + n
            for it in range(nt_half):
                imt = self.nt_b - it - 1
                Twt_pos[n, it] = self.Twt_bb[iw, it] if it == imt else self.Twt_bb[iw, it] + self.Twt_bb[iw, imt]

        Ot_shape = Ot.shape
        Ot = Ot.reshape(Ot.shape[0], -1)
        Ow = np.dot(Twt_pos, Ot)

        Ot = Ot.reshape(Ot_shape)
        Ow = Ow.reshape((Twt_pos.shape[0],) + Ot_shape[1:])
        return Ow

    def w_to_tau(self, Ow, stats):
        stats = _normalize_stats(stats)
        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        Ttw = self.Ttw_ff if stats == 'f' else self.Ttw_bb
        if Ow.shape[0] != Ttw.shape[1]:
            raise ValueError(
                "w_to_tau: Number of w points are inconsistent: {} and {}".format(Ow.shape[0], Ttw.shape[1]))

        Ow_shape = Ow.shape
        Ow = Ow.reshape(Ow.shape[0], -1)
        Ot = np.dot(Ttw, Ow)

        Ow = Ow.reshape(Ow_shape)
        Ot = Ot.reshape((Ttw.shape[0],) + Ow_shape[1:])
        return Ot

    def w_to_tau_phsym(self, Ow, stats):
        stats = _normalize_stats(stats)
        if stats != 'b':
            raise ValueError("FT w/ particle-hole symmetry only support bosonic correlation functions")

        nw_half = self.nw_b // 2 if self.nw_b % 2 == 0 else self.nw_b // 2 + 1
        nt_half = self.nt_b // 2 if self.nt_b % 2 == 0 else self.nt_b // 2 + 1
        if Ow.shape[0] != nw_half:
            raise ValueError(
                "w_to_tau_phsym: Number of w points are inconsistent: {} and {}".format(Ow.shape[0], nw_half))

        Ttw_pos = np.zeros((nt_half, nw_half), dtype=self.Ttw_bb.dtype)
        for it in range(nt_half):
            for n in range(nw_half):
                iw = self.nw_b // 2 + n
                imw = self.nw_b // 2 - n
                Ttw_pos[it, n] = self.Ttw_bb[it, iw] if iw == imw else self.Ttw_bb[it, iw] + self.Ttw_bb[it, imw]

        Ow_shape = Ow.shape
        Ow = Ow.reshape(Ow.shape[0], -1)
        Ot = np.dot(Ttw_pos, Ow)

        Ow = Ow.reshape(Ow_shape)
        Ot = Ot.reshape((Ttw_pos.shape[0],) + Ow_shape[1:])
        return Ot

    def w_interpolate(self, Ow, target, stats: str, ir_notation: bool=True, ph_sym: bool=False):
        if isinstance(target, _IAFTIRAdapter):
            return self._w_interpolate(Ow, target.wn_mesh(stats, ir_notation, positive_only=ph_sym), 
                                       stats, ir_notation=ir_notation, ph_sym=ph_sym)
        else:
            return self._w_interpolate(Ow, target, stats, ir_notation=ir_notation, ph_sym=ph_sym)

    def w_interpolate_phsym(self, Ow, target, stats: str, ir_notation: bool=True):
        return self.w_interpolate(Ow, target, stats, ir_notation, ph_sym=True)

    def _w_interpolate(self, Ow, wn_mesh_interp, stats: str, *, ir_notation: bool=True, ph_sym: bool=False):
        stats = _normalize_stats(stats)
        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))

        if isinstance(wn_mesh_interp, int):
            wn_mesh_interp = np.array([wn_mesh_interp], dtype=int)
        
        if ir_notation:
            wn_indices = np.asarray(wn_mesh_interp, dtype=int)
        if not ir_notation:
            wn_indices = np.array([2 * n + 1 if stats == 'f' else 2 * n for n in wn_mesh_interp], dtype=int)

        Tlw = self.Tlw_ff if stats == 'f' else self.Tlw_bb

        if ph_sym:
            nw_half = self.nw_b // 2 if self.nw_b % 2 == 0 else self.nw_b // 2 + 1
            nw_zero_offset = self.nw_b // 2
            Tlw_pos = np.zeros((Tlw.shape[0], nw_half), dtype=Tlw.dtype)
            for l in range(Tlw.shape[0]):
                for n in range(nw_half):
                    iw = nw_zero_offset + n
                    imw = nw_zero_offset - n
                    Tlw_pos[l, n] = Tlw[l, iw] if iw == imw else Tlw[l, iw] + Tlw[l, imw]
            Tlw = Tlw_pos

        if Ow.shape[0] != Tlw.shape[1]:
            raise ValueError(
                "w_interpolate: Number of w points are inconsistent: {} and {}".format(Ow.shape[0], Tlw.shape[1]))

        Twl_interp = self._bases.basis_f.uhat(wn_indices).T if stats == 'f' else self._bases.basis_b.uhat(wn_indices).T
        Tww = np.dot(Twl_interp, Tlw)

        Ow_shape = Ow.shape
        Ow = Ow.reshape(Ow.shape[0], -1)
        Ow_interp = np.dot(Tww, Ow)

        Ow = Ow.reshape(Ow_shape)
        Ow_interp = Ow_interp.reshape((wn_indices.shape[0],) + Ow_shape[1:])
        return Ow_interp

    def tau_interpolate(self, Ot, target, stats: str, *, rel_notation: bool=False, ph_sym: bool=False):
        stats = _normalize_stats(stats)
        if isinstance(target, _IAFTIRAdapter):
            #tau_mesh = target._tau_mesh_f if stats == 'f' else target._tau_mesh_b
            tau_mesh = target.tau_mesh(stats, rel_notation=rel_notation)
            if ph_sym:
                nt_half = tau_mesh.shape[0] // 2 + tau_mesh.shape[0] % 2
                tau_mesh = tau_mesh[:nt_half]
            return self._tau_interpolate(Ot, tau_mesh, stats, rel_notation=rel_notation, ph_sym=ph_sym)
        else:
            return self._tau_interpolate(Ot, target, stats, rel_notation=rel_notation, ph_sym=ph_sym)

    def tau_interpolate_phsym(self, Ot, target, stats: str, *, rel_notation: bool=False):
        return self.tau_interpolate(Ot, target, stats, rel_notation=rel_notation, ph_sym=True)

    def _tau_interpolate(self, Ot, tau_mesh_interp, stats: str, *, rel_notation: bool=False, ph_sym: bool=False):
        stats = _normalize_stats(stats)
        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        
        if np.isscalar(tau_mesh_interp):
            try:
                tau_mesh_interp = np.array([tau_mesh_interp], dtype=float)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "tau_mesh_interp must be a number, a list of numbers, or a 1D numpy array of numbers."
                ) from exc
        elif isinstance(tau_mesh_interp, (list, tuple, np.ndarray)):
            try:
                tau_mesh_interp = np.asarray(tau_mesh_interp, dtype=float)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "tau_mesh_interp must be a number, a list of numbers, or a 1D numpy array of numbers."
                ) from exc
        else:
            raise TypeError(
                "tau_mesh_interp must be a number, a list of numbers, or a 1D numpy array of numbers."
            )

        if tau_mesh_interp.ndim != 1:
            raise ValueError(
                "tau_mesh_interp must be one-dimensional, got shape {}.".format(tau_mesh_interp.shape)
            )

        if rel_notation:
            # convert to [0, beta] notation for sparse_ir library
            tau_mesh_interp = 0.5 * (tau_mesh_interp + 1.0) * self.beta

        Tlt = self.Tlt_ff if stats == 'f' else self.Tlt_bb
        nt = self.nt_f if stats == 'f' else self.nt_b
        Ttl_interp = self._bases.basis_f.u(tau_mesh_interp).T if stats == 'f' else self._bases.basis_b.u(tau_mesh_interp).T

        if ph_sym:
            nt_half = nt // 2 + nt % 2
            Tlt_pos = np.zeros((Tlt.shape[0], nt_half), dtype=Tlt.dtype)
            for l in range(Tlt.shape[0]):
                for it in range(nt_half):
                    imt = nt - it - 1
                    Tlt_pos[l, it] = Tlt[l, it] if it == imt else Tlt[l, it] + Tlt[l, imt]
            
        Ttt = np.dot(Ttl_interp, Tlt) if not ph_sym else np.dot(Ttl_interp, Tlt_pos)
        if Ot.shape[0] != Ttt.shape[1]:
            raise ValueError(
                "t_interpolate: Number of tau points are inconsistent: {} and {}".format(Ot.shape[0], Ttt.shape[1]))

        Ot_shape = Ot.shape
        Ot = Ot.reshape(Ot.shape[0], -1)
        Ot_interp = np.dot(Ttt, Ot)

        Ot = Ot.reshape(Ot_shape)
        Ot_interp = Ot_interp.reshape((np.shape(tau_mesh_interp)[0],) + Ot_shape[1:])
        return Ot_interp

    def check_leakage(self, Ot, stats: str, name: str="", w_input: bool=False):
        stats = _normalize_stats(stats)
        if w_input:
            Ot_ = self.w_to_tau(Ot, stats)
            self.check_leakage(Ot_, stats, name, w_input=False)
            return

        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        nts = self.nt_f if stats == 'f' else self.nt_b
        Tlt = self.Tlt_ff if stats == 'f' else self.Tlt_bb
        if nts != Ot.shape[0]:
            raise ValueError("Inconsistency between nts = {} and Ot.shape[0] = {}".format(nts, Ot.shape[0]))

        O_l0_i = np.einsum('t,ti->i', Tlt[0], Ot.reshape(nts, -1))
        coeff_first = np.max(np.abs(O_l0_i))

        O_lm2_t = np.einsum('lt,ti->li', Tlt[-2:], Ot.reshape(nts, -1))
        coeff_last = np.max(np.abs(O_lm2_t))

        leakage = coeff_last / coeff_first
        print("IAFT leakage of {}: {}".format(name, leakage))
        if leakage >= 1e-5:
            print("[WARNING] check_leakage: coeff_last/coeff_first = {} >= 1e-5; "
                  "coeff_last = {}, coeff_first = {}".format(leakage, coeff_last, coeff_first))
        sys.stdout.flush()

    def check_leakage_phsym(self, Ot, stats: str, name: str="", w_input: bool=False):
        stats = _normalize_stats(stats)
        if stats != 'b':
            raise ValueError("FT w/ particle-hole symmetry only support bosonic correlation functions")

        if w_input:
            Ot_ = self.w_to_tau_phsym(Ot, stats)
            self.check_leakage_phsym(Ot_, stats, name, w_input=False)
            return

        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))

        nts = self.nt_b
        nt_half = self.nt_b // 2 if self.nt_b % 2 == 0 else self.nt_b // 2 + 1
        Tlt = self.Tlt_bb
        if nt_half != Ot.shape[0]:
            raise ValueError("Inconsistency between nts_half = {} and Ot.shape[0] = {}".format(nt_half, Ot.shape[0]))

        Tl0_t_pos = np.zeros(nt_half, dtype=complex)
        for it in range(nt_half):
            imt = nts - it - 1
            Tl0_t_pos[it] = Tlt[0, it] if it == imt else Tlt[0, it] + Tlt[0, imt]
        O_l0_i = np.einsum('t,ti->i', Tl0_t_pos, Ot.reshape(nt_half, -1))
        coeff_first = np.max(np.abs(O_l0_i))

        Tlm2_t_pos = np.zeros((2, nt_half), dtype=complex)
        nl = Tlt.shape[0]
        for it in range(nt_half):
            imt = nts - it - 1
            Tlm2_t_pos[0, it] = Tlt[nl - 2, it] if it == imt else Tlt[nl - 2, it] + Tlt[nl - 2, imt]
            Tlm2_t_pos[1, it] = Tlt[nl - 1, it] if it == imt else Tlt[nl - 1, it] + Tlt[nl - 1, imt]
        O_lm2_t = np.einsum('lt,ti->li', Tlm2_t_pos, Ot.reshape(nt_half, -1))
        coeff_last = np.max(np.abs(O_lm2_t))

        leakage = coeff_last / coeff_first
        print("IAFT leakage of {}: {}".format(name, leakage))
        if leakage >= 1e-5:
            print("[WARNING] check_leakage_phsym: coeff_last/coeff_first = {} >= 1e-5; "
                  "coeff_last = {}, coeff_first = {}".format(leakage, coeff_last, coeff_first))
        sys.stdout.flush()
