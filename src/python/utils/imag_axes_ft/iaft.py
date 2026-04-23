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

import numpy as np

from .iaft_sparse_ir import _IAFTIRAdapter

try:
    from coqui._lib.iaft_module import IAFT as IAFTCpp
    from coqui._lib.iaft_module import build_g_tau_ref, build_g_iw_ref
    from coqui._lib.iaft_module import string_to_basis_enum
    _IAFT_CPP_IMPORT_ERROR = None
except ImportError as exc:
    IAFTCpp = None
    build_g_tau_ref = None
    build_g_iw_ref = None
    string_to_basis_enum = None
    _IAFT_CPP_IMPORT_ERROR = exc

"""
Kernel for Fourier transform on the imaginary axis, supporting both IR and DLR basis. 
"""


def _validate_accuracy_args(basis: str, prec, eps):
    if basis not in ("dlr", "ir"):
        raise ValueError("basis must be 'dlr' or 'ir', got '{}'".format(basis))

    if prec is not None:
        _validate_prec_string(prec)

    if eps is not None:
        if isinstance(eps, bool) or not isinstance(eps, (int, float, np.integer, np.floating)):
            raise TypeError("'eps' must be a floating-point value when provided.")
        if float(eps) <= 0.0:
            raise ValueError("'eps' must be positive.")

    if basis == "ir":
        if eps is not None:
            raise ValueError("IR basis accepts only 'prec'; 'eps' is not supported.")
        if prec is None:
            raise ValueError("IR basis requires 'prec'.")
    else:
        if prec is None and eps is None:
            raise ValueError("DLR basis requires at least one of 'prec' or 'eps'.")

        if prec == "custom" and eps is None:
            raise ValueError("For DLR, prec='custom' requires 'eps'.")


def _validate_prec_string(precision):
    if not isinstance(precision, str):
        raise TypeError("Precision must be a string.")
    if precision not in ("high", "medium", "low", "custom"):
        raise ValueError("Unknown precision value: {}. ".format(precision))

def _normalize_stats(stats_input: str) -> str:
    """
    Normalize stats input to C++ string form.
    Accepts both short forms ('f', 'b') and long forms ('fermion', 'boson').

    :param stats_input: str
    Statistics string ('f', 'fermion', 'b', or 'boson')
    :return: str
    Normalized C++ form ('fermion' or 'boson')
    :raises ValueError: if stats_input is not a recognized statistics format
    """
    if stats_input == 'f' or stats_input == 'fermion':
        return 'fermion'
    elif stats_input == 'b' or stats_input == 'boson':
        return 'boson'
    else:
        raise ValueError(
            "Unknown statistics '{}'. Acceptable options are 'f'/'fermion' for fermions "
            "and 'b'/'boson' for bosons.".format(stats_input))


class _IAFTCppAdapter(object):
    """
    Internal implementation class based on the C++ imag_axes_ft::IAFT class.
    Supports both DLR and IR basis. 

    Accuracy policy used here:
    - `prec` is a string; `eps` is a positive float.
    - If only one of `prec`/`eps` is provided, use that one.
    - If both are provided:
        - `prec == "custom"` -> build with `eps`
        - otherwise -> build with `prec`
    """
    def __init__(self, beta: float, wmax: float, *, prec=None, eps=None, verbose: bool=True, basis: str="dlr"):
        """
        :param beta: float
            Inverse temperature (a.u.)
        :param wmax: float
            Frequency cutoff (a.u.)
        :param prec: str or None
            Precision label for basis
        :param eps: float or None
            Explicit DLR accuracy target
        :param verbose: bool
            Print metadata (currently unused; kept for API compatibility)
        :param basis: str
            C++ backend basis, one of {"dlr", "ir"}
        """

        if IAFTCpp is None or string_to_basis_enum is None:
            raise ImportError(
                "DLR backend requires IAFTCpp binding from CoQuí C++ library. "
                "Please rebuild CoQuí with 'COQUI_UPDATE_PYTHON_BINDINGS=ON'. "
            ) from _IAFT_CPP_IMPORT_ERROR

        self.basis = basis
        self.beta = beta
        self.lmbda = wmax * beta
        self.wmax = wmax

        # Create C++ IAFT instance with selected C++ basis backend
        build_with_eps = (eps is not None) and (prec is None or prec == "custom")
        if build_with_eps:
            self._iaft_cpp = IAFTCpp(
                self.beta, self.wmax, string_to_basis_enum(self.basis), 
                float(eps), verbose
            )
        else:
            _validate_prec_string(prec)
            self._iaft_cpp = IAFTCpp(
                self.beta, self.wmax, string_to_basis_enum(self.basis), 
                prec, verbose
            )

        self.prec = self._iaft_cpp.prec()
        self.eps = self._iaft_cpp.eps()

        # Cache mesh properties from C++ backend
        self._tau_mesh_f = self._iaft_cpp.tau_mesh_f()
        self._tau_mesh_b = self._iaft_cpp.tau_mesh_b()
        self._wn_mesh_f = self._iaft_cpp.wn_mesh_f()
        self._wn_mesh_b = self._iaft_cpp.wn_mesh_b()
        self.nt_f, self.nw_f = self._tau_mesh_f.shape[0], self._wn_mesh_f.shape[0]
        self.nt_b, self.nw_b = self._tau_mesh_b.shape[0], self._wn_mesh_b.shape[0]

    # Expose C++ backend matrices as new numpy arrays
    @property
    def Ttw_ff(self):
        return np.array(self._iaft_cpp.Ttw_ff())

    @property
    def Twt_ff(self):
        return np.array(self._iaft_cpp.Twt_ff())
    
    @property
    def Ttw_bb(self):
        return np.array(self._iaft_cpp.Ttw_bb())

    @property
    def Twt_bb(self):
        return np.array(self._iaft_cpp.Twt_bb())
    
    @property
    def Ttt_bf(self):
        return np.array(self._iaft_cpp.Ttt_bf())

    @property
    def Ttt_fb(self):
        return np.array(self._iaft_cpp.Ttt_fb())
    
    @property
    def T_beta_t_ff(self):
        return np.array(self._iaft_cpp.T_beta_t_ff())

    @property
    def T_zero_t_ff(self):
        return np.array(self._iaft_cpp.T_zero_t_ff())
    
    @property
    def Tct_ff(self):
        return np.array(self._iaft_cpp.Tct_ff())

    @property
    def Tct_bb(self):
        return np.array(self._iaft_cpp.Tct_bb())

    def save(self, h5_grp):
        if 'imaginary_fourier_transform' in h5_grp:
            return
        h5_grp.create_group('imaginary_fourier_transform')
        h5_grp['imaginary_fourier_transform']['beta'] = self.beta
        h5_grp['imaginary_fourier_transform']['wmax'] = self.wmax
        h5_grp['imaginary_fourier_transform']['prec'] = self.prec
        h5_grp['imaginary_fourier_transform']['eps'] = self.eps
        h5_grp['imaginary_fourier_transform']['basis'] = self.basis

    def __str__(self):
        self._iaft_cpp.metadata_log()
        return ""

    def __eq__(self, other):
        if not isinstance(other, _IAFTCppAdapter):
            return NotImplemented
        return (
                self.beta == other.beta and
                self.lmbda == other.lmbda and
                self.prec == other.prec and
                self.eps == other.eps and
                self.basis == other.basis
        )

    def tau_mesh(self, stats: str):
        """
        Return imaginary-time sampling points, following the convention t = [-1, 1]. 
        
        :param stats: str
            statistics: 'f'/'fermion' for fermions and 'b'/'boson' for bosons
        :return: numpy.ndarray(dim=1)
            imaginary-time sampling points
        """
        stats = _normalize_stats(stats)
        return np.array(self._tau_mesh_f, dtype=float) if stats=="fermion" else np.array(self._tau_mesh_b, dtype=float)

    def wn_mesh(self, stats: str, ir_notation: bool=True, *, positive_only=False):
        """
        Return Matsubara frequency indices.
        :param stats: str
            statistics: 'f'/'fermion' for fermions and 'b'/'boson' for bosons
        :param ir_notation: bool
            Whether wn_mesh is in CoQui's notation where iwn = n*pi/beta regardless of statistics.
        :return: numpy.ndarray(dim=1)
            Matsubara frequency indices
        """
        stats = _normalize_stats(stats)
        wn_mesh = np.array(self._wn_mesh_f, dtype=int) if stats=="fermion" else np.array(self._wn_mesh_b, dtype=int)
        if not ir_notation:
            wn_mesh = (wn_mesh-1)//2 if stats=="fermion" else wn_mesh//2
        if positive_only:
            nw_half = wn_mesh.shape[0]//2
            return wn_mesh[nw_half:]
        else:
            return wn_mesh

    def tau_to_w(self, Ot, stats: str):
        """
        Fourier transform from imaginary-time axis to Matsubara-frequency axis
        :param Ot: numpy.ndarray
            imaginary-time object with dimensions (nts, ...)
        :param stats: str
            statistics: 'f'/'fermion' for fermions and 'b'/'boson' for bosons
        :return: numpy.ndarray
            Matsubara-frequency object with dimensions (nw, ...)
        """
        stats = _normalize_stats(stats)
        Ot_shape = Ot.shape
        Ot_2d = Ot.reshape(Ot.shape[0], -1)
        Ow_2d = self._iaft_cpp.tau_to_w_2d(Ot_2d, stats)
        Ow = Ow_2d.reshape((Ow_2d.shape[0],) + Ot_shape[1:])
        return Ow

    def w_to_tau(self, Ow, stats: str):
        """
        Fourier transform from Matsubara-frequency axis to imaginary-time axis.
        :param Ow: numpy.ndarray
            Matsubara-frequency object with dimensions (nw, ...)
        :param stats: str
            statistics: 'f'/'fermion' for fermions and 'b'/'boson' for bosons
        :return: numpy.ndarray
            Imaginary-time object with dimensions (nt, ...)
        """
        stats = _normalize_stats(stats)
        Ow_shape = Ow.shape
        Ow_2d = Ow.reshape(Ow.shape[0], -1)
        Ot_2d = self._iaft_cpp.w_to_tau_2d(Ow_2d, stats)
        Ot = Ot_2d.reshape((Ot_2d.shape[0],) + Ow_shape[1:])
        return Ot

    def tau_to_w_phsym(self, Ot, stats: str):
        """
        Fourier transform from imaginary-time axis to Matsubara-frequency axis w/ particle-hole symmetry
        :param Ot: numpy.ndarray
            imaginary-time object with dimensions (nts, ...)
        :param stats: str
            statistics: 'f'/'fermion' for fermions and 'b'/'boson' for bosons
        :return: numpy.ndarray
            Matsubara-frequency object with dimensions (nw, ...)
        """
        stats = _normalize_stats(stats)
        if stats != 'boson':
            raise ValueError("FT w/ particle-hole symmetry only support bosonic correlation functions")
        Ot_shape = Ot.shape
        Ot_2d = Ot.reshape(Ot.shape[0], -1)
        Ow_2d = self._iaft_cpp.tau_to_w_phsym_2d(Ot_2d)
        Ow = Ow_2d.reshape((Ow_2d.shape[0],) + Ot_shape[1:])
        return Ow

    def w_to_tau_phsym(self, Ow, stats: str):
        """
        Fourier transform from Matsubara-frequency axis to imaginary-time axis w/ particle-hole symmetry.
        :param Ow: numpy.ndarray
            Matsubara-frequency object with dimensions (nw, ...)
        :param stats: str
            statistics: 'f'/'fermion' for fermions and 'b'/'boson' for bosons
        :return: numpy.ndarray
            Imaginary-time object with dimensions (nt, ...)
        """
        stats = _normalize_stats(stats)
        if stats != 'boson':
            raise ValueError("FT w/ particle-hole symmetry only support bosonic correlation functions")
        Ow_shape = Ow.shape
        Ow_2d = Ow.reshape(Ow.shape[0], -1)
        Ot_2d = self._iaft_cpp.w_to_tau_phsym_2d(Ow_2d)
        Ot = Ot_2d.reshape((Ot_2d.shape[0],) + Ow_shape[1:])
        return Ot

    def w_interpolate(self, Ow, target, stats: str, ir_notation: bool=True, ph_sym: bool=False):
        if isinstance(target, _IAFTCppAdapter):
            wn_mesh = target.wn_mesh(stats, ir_notation=ir_notation, positive_only=ph_sym)
            return self._w_interpolate(Ow, wn_mesh, stats, ph_sym=ph_sym, ir_notation=ir_notation)
        else:
            return self._w_interpolate(Ow, target, stats, ph_sym=ph_sym, ir_notation=ir_notation)

    def w_interpolate_phsym(self, Ow, target, stats: str, ir_notation: bool=True):
        return self.w_interpolate(Ow, target, stats, ir_notation=ir_notation, ph_sym=True)

    def _w_interpolate(self, Ow, wn_mesh_interp, stats: str, *, ph_sym: bool=False, ir_notation: bool=True):
        if isinstance(wn_mesh_interp, int):
            wn_mesh_interp = np.array([wn_mesh_interp], dtype=int)

        stats = _normalize_stats(stats)

        # convert to ir notation if needed
        if not ir_notation:
            wn_mesh_interp = (2*wn_mesh_interp+1) if stats=="fermion" else (2*wn_mesh_interp)

        T_ww = self._iaft_cpp.construct_w_interpolate_matrix(wn_mesh_interp, stats, ph_sym=ph_sym)
        if Ow.shape[0] != T_ww.shape[1]:
            raise ValueError(
                "w_interpolate: Number of w points are inconsistent: {} and {}".format(
                Ow.shape[0], T_ww.shape[1]))

        Ow_shape = Ow.shape
        Ow_2d = Ow.reshape(Ow.shape[0], -1)
        Ow_interp = np.dot(T_ww, Ow_2d)
        Ow_interp = Ow_interp.reshape((Ow_interp.shape[0],) + Ow_shape[1:])
        return Ow_interp

    def tau_interpolate(self, Ot, target, stats: str, ph_sym: bool=False):
        if isinstance(target, _IAFTCppAdapter):
            tau_mesh = target._tau_mesh_f if _normalize_stats(stats)=='fermion' else target._tau_mesh_b
            if ph_sym:
                nt_half = tau_mesh.shape[0] // 2 + tau_mesh.shape[0] % 2
                tau_mesh = tau_mesh[:nt_half]
            return self._tau_interpolate(Ot, tau_mesh, ph_sym)
        else:
            return self._tau_interpolate(Ot, target, ph_sym)

    def tau_interpolate_phsym(self, Ot, target, stats: str):
        return self.tau_interpolate(Ot, target, stats, ph_sym=True)

    def _tau_interpolate(self, Ot, tau_mesh_interp, ph_sym: bool=False):
        if isinstance(tau_mesh_interp, float):
            tau_mesh_interp = np.array([tau_mesh_interp], dtype=float)

        T_tt = self._iaft_cpp.construct_tau_interpolate_matrix(tau_mesh_interp, ph_sym=ph_sym)
        if Ot.shape[0] != T_tt.shape[1]:
            raise ValueError(
                "tau_interpolate: Number of tau points are inconsistent: {} and {}".format(Ot.shape[0], T_tt.shape[1]))
    
        Ot_shape = Ot.shape
        Ot_2d = Ot.reshape(Ot.shape[0], -1)
        Ot_interp = np.dot(T_tt, Ot_2d)

        Ot_interp = Ot_interp.reshape((Ot_interp.shape[0],) + Ot_shape[1:])
        return Ot_interp

    def check_leakage(self, Ot, stats: str, name: str="", w_input: bool=False):
        """
        Check decay of the C++-backend coefficients to assess basis quality.
        Intentionally no operation until C++ leakage diagnostics in DLR are implemented.
        """
        return None

    def check_leakage_phsym(self, Ot, stats: str, name: str="", w_input: bool=False):
        """
        Check decay of the C++-backend coefficients w/ particle-hole symmetry.
        Intentionally no operation until C++ leakage diagnostics in DLR are implemented.
        """
        return None


class IAFT(object):
    """
    Fourier transform driver on the imaginary axis.

    This class wraps two backends:
    - basis="ir": legacy sparse_ir path
    - basis="dlr": C++ IAFT backend via `_IAFTCppAdapter`

    For `prec`/`eps` behavior, follow `_IAFTCppAdapter` policy.
    """
    def __init__(self, beta: float, wmax: float, *, prec=None, eps=None, 
                 verbose: bool=True, basis: str="ir"):
        """
        Initialize IAFT with specified basis backend.
        
        :param beta: float
            Inverse temperature (a.u.)
        :param wmax: float
            Frequency cutoff (a.u.)
        :param prec: str or None
            Precision selector for basis
        :param eps: float or None
            Explicit DLR accuracy target
        :param verbose: bool
            Print metadata on initialization
        :param basis: str
            Backend basis: "ir" (sparse_ir, default) or "dlr" (C++ IAFTCpp)
        :raises ValueError: if basis is not "ir" or "dlr"
        :raises ImportError: if required backend library is not available
        """
        _validate_accuracy_args(basis, prec, eps)

        # Check backend availability and create adapter
        if basis == "ir":
            try:
                import sparse_ir
            except ImportError:
                raise ImportError(
                    "IR backend requires sparse_ir library. "
                    "Please install with: pip install sparse-ir[xprec]==1.1.7"
                )
            self._impl = _IAFTIRAdapter(beta, wmax, prec, verbose)
        else:  # basis == "dlr"
            self._impl = _IAFTCppAdapter(beta, wmax, prec=prec, eps=eps, verbose=verbose, basis="dlr")

        self.basis = basis

    # Delegate properties to implementation
    @property
    def beta(self):
        return self._impl.beta

    @property
    def wmax(self):
        return self._impl.wmax

    @property
    def lmbda(self):
        return self._impl.lmbda

    @property
    def prec(self):
        return self._impl.prec

    @property
    def eps(self):
        return self._impl.eps

    @property
    def statisics(self):
        return self._impl.statisics

    @property
    def nt_f(self):
        return self._impl.nt_f

    @property
    def nt_b(self):
        return self._impl.nt_b

    @property
    def nw_f(self):
        return self._impl.nw_f

    @property
    def nw_b(self):
        return self._impl.nw_b

    @property
    def Ttw_ff(self):
        return self._impl.Ttw_ff

    @property
    def Twt_ff(self):
        return self._impl.Twt_ff

    @property
    def Ttw_bb(self):
        return self._impl.Ttw_bb

    @property
    def Twt_bb(self):
        return self._impl.Twt_bb

    @classmethod
    def from_coqui_chkpt(cls, chkpt_h5, verbose: bool=True):
        """
        Load IAFT from checkpoint file, auto-detecting backend from checkpoint.
        Defaults to IR basis if not specified in checkpoint (backward compatibility).
        
        :param chkpt_h5: str
            Path to HDF5 checkpoint file
        :param verbose: bool
            Print metadata on initialization
        :return: IAFT
            Initialized with parameters from checkpoint
        """
        from h5 import HDFArchive
        with HDFArchive(chkpt_h5, 'r') as ar:
            iaft_grp = ar['imaginary_fourier_transform']
            beta = iaft_grp['beta']
            prec = iaft_grp['prec'] if 'prec' in iaft_grp else None
            eps = iaft_grp['eps'] if 'eps' in iaft_grp else None
            if 'wmax' in iaft_grp:
                wmax = iaft_grp['wmax']
            else:
                ir_lambda = iaft_grp['lambda']
                wmax = ir_lambda / beta
            # Prefer basis metadata; fall back to source for backward compatibility.
            basis = iaft_grp['basis'] if 'basis' in iaft_grp else (iaft_grp['source'] if 'source' in iaft_grp else "ir")

        if basis == "ir":
            return cls(beta, wmax, prec=prec, verbose=verbose, basis=basis)
        elif basis == "dlr":
            return cls(beta, wmax, prec=prec, eps=eps, verbose=verbose, basis=basis)
        else:
            raise ValueError("iaft.py::IAFT.from_coqui_chkpt: unknown basis '{}' in checkpoint".format(basis))

    def save(self, h5_grp):
        """
        Save IAFT parameters to HDF5 checkpoint group.
        Stores only (beta, wmax, prec) for format compatibility with C++ checkpoints.
        
        :param h5_grp: h5 group
            Target HDF5 group
        """
        self._impl.save(h5_grp)

    def __str__(self):
        return self._impl.__str__()

    def __eq__(self, other):
        if not isinstance(other, IAFT):
            return NotImplemented
        return self._impl == other._impl

    # Delegate all transform methods
    def wn_mesh(self, stats: str, ir_notation: bool=True, *, positive_only=False):
        return self._impl.wn_mesh(stats, ir_notation, positive_only=positive_only)
    
    def tau_mesh(self, stats: str):
        return self._impl.tau_mesh(stats)

    def tau_to_w(self, Ot, stats: str, ph_sym: bool=False):
        return self._impl.tau_to_w(Ot, stats) if not ph_sym else self._impl.tau_to_w_phsym(Ot, stats)

    def w_to_tau(self, Ow, stats: str, ph_sym: bool=False):
        return self._impl.w_to_tau(Ow, stats) if not ph_sym else self._impl.w_to_tau_phsym(Ow, stats)

    def tau_to_w_phsym(self, Ot, stats: str):
        return self._impl.tau_to_w_phsym(Ot, stats)

    def w_to_tau_phsym(self, Ow, stats: str):
        return self._impl.w_to_tau_phsym(Ow, stats)

    def w_interpolate(self, Ow, target, stats: str, ir_notation: bool=True, ph_sym: bool=False):
        if isinstance(target, IAFT):
            return self._impl.w_interpolate(Ow, target._impl, stats, ir_notation, ph_sym)
        else:
            return self._impl.w_interpolate(Ow, target, stats, ir_notation, ph_sym)

    def w_interpolate_phsym(self, Ow, target, stats: str, ir_notation: bool=True):
        return self.w_interpolate(Ow, target, stats, ir_notation, ph_sym=True)

    def tau_interpolate(self, Ot, target, stats: str, ph_sym: bool=False):
        if isinstance(target, IAFT):
            return self._impl.tau_interpolate(Ot, target._impl, stats, ph_sym)
        else:
            return self._impl.tau_interpolate(Ot, target, stats, ph_sym)

    def tau_interpolate_phsym(self, Ot, target, stats: str):
        return self.tau_interpolate(Ot, target, stats, ph_sym=True)

    def check_leakage(self, Ot, stats: str, name: str="", w_input: bool=False, ):
        return self._impl.check_leakage(Ot, stats, name, w_input)

    def check_leakage_phsym(self, Ot, stats: str, name: str="", w_input: bool=False):
        return self._impl.check_leakage_phsym(Ot, stats, name, w_input)

    def build_g_tau_ref(self, norb: int, stats: str, ph_sym: bool = False) -> np.ndarray:
        """
        Build a reference Green's function on the imaginary time mesh with a known analytical form.

        The analytical form is the same as used internally for testing IAFT accuracy.
        The tau convention is [-1, 1] mapping to physical imaginary time [0, beta].

        :param norb: number of orbitals (output shape will be (nt, norb, norb)).
        :param stats: statistics, 'f'/'fermion' for fermions or 'b'/'boson' for bosons.
        :param ph_sym: if True, only the first half of tau points is constructed (particle-hole symmetry).
        :return: numpy array of shape (nt, norb, norb).
        """
        if build_g_tau_ref is None:
            raise ImportError(
                "build_g_tau_ref requires IAFTCpp binding from CoQuí C++ library."
            ) from _IAFT_CPP_IMPORT_ERROR
        return build_g_tau_ref(self.tau_mesh(stats), self.beta, norb, ph_sym)


    def build_g_iw_ref(self, norb: int, stats: str, ph_sym: bool = False) -> np.ndarray:
        """
        Build a reference Green's function on the Matsubara frequency mesh with a known analytical form.

        The analytical form is the same as used internally for testing IAFT accuracy.
        The Matsubara index convention is n = (2k+1) for fermions and n = 2k for bosons.

        :param norb: number of orbitals (output shape will be (nw, norb, norb)).
        :param stats: statistics, 'f'/'fermion' for fermions or 'b'/'boson' for bosons.
        :param ph_sym: if True, only the positive Matsubara frequencies are constructed (particle-hole symmetry).
        :return: numpy array of shape (nw, norb, norb).
        """
        stats = _normalize_stats(stats)
        if build_g_iw_ref is None:
            raise ImportError(
                "build_g_iw_ref requires IAFTCpp binding from CoQuí C++ library."
            ) from _IAFT_CPP_IMPORT_ERROR
        return build_g_iw_ref(self.wn_mesh(stats), self.beta, stats, norb, ph_sym)


if __name__ == '__main__':
    # Initialize IAFT object for given inverse temperature, lambda and precision
    ft = IAFT(1000.0, 10.0, prec="low")

    print(ft.wn_mesh('f', True))

    Gt = np.zeros((ft.nt_f, 2, 2, 2))
    Gw = ft.tau_to_w(Gt, 'f')
    print(Gw.shape)

    # Interpolate to arbitrary tau point
    tau_interp = np.array([0.0, ft.beta])
    Gt_interp = ft.tau_interpolate(Gt, tau_interp, 'f')
    print(Gt_interp.shape)

    # wn in spare_ir notation
    w_interp = np.array([-1,1,3,5], dtype=int)
    Gw_interp = ft.w_interpolate(Gw, w_interp, 'f', True)
    print(Gw_interp.shape)

    # wn in physical notation
    w_interp = np.array([-1,0,1,2,3,4], dtype=int)
    Gw_interp = ft.w_interpolate(Gw, w_interp, 'f', False)
    print(Gw_interp.shape)

    Gt2 = ft.w_to_tau(Gw, 'f')
    print(Gt2.shape)
