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

import json
import numpy as np
from coqui import IAFT

from coqui._lib import post_proc_module as pproc_mod


def ac(mf, params):
    pproc_mod.ac(mf, json.dumps(params))


def band_interpolation(mf, params):
    pproc_mod.band_interpolation(mf, json.dumps(params))


def spectral_interpolation(mf, params):
    pproc_mod.spectral_interpolation(mf, json.dumps(params))


def local_dos(mf, params):
    pproc_mod.local_dos(mf, json.dumps(params))


def unfold_bz(mf, params):
    pproc_mod.unfold_bz(mf, json.dumps(params))


def dump_vxc(mf, params):
    pproc_mod.dump_vxc(mf, json.dumps(params))


def dump_hartree(mf, params):
    pproc_mod.dump_hartree(mf, json.dumps(params))


def pade(g_iw, iaft, stats, wmin, wmax, Nw, *, Nfit=-1, eta=0.0, ph_sym=False):
  """
  Perform Pade analytical continuation using CoQui backend.
  
  Parameters:
    g_iw (np.ndarray): Input array on Matsubara frequency axis.
                       Can be arbitrary shape; first dimension is treated as Matsubara axis.
    iaft (IAFT): Fourier transform kernel on imaginary axis 
    stats (str): 'fermion' or 'boson' 
    wmin (float): Minimum real frequency for output.
    wmax (float): Maximum real frequency for output.
    Nw (int): Number of real frequency points.
    eta (float): Broadening parameter (imaginary offset). Default: 0.0
    Nfit (int): Number of Pade fitting points. Default: -1 (use all)
  
  Returns:
    g_w (np.ndarray): Complex array on real frequency axis with shape [Nw] + g_iw.shape[1:].
    w_mesh (np.ndarray): Real frequency mesh (1D complex array).
  """
  
  g_iw = np.asarray(g_iw, dtype=np.complex128)
  if isinstance(iaft, IAFT):
      iw_mesh = 1j*iaft.wn_mesh(stats=stats, positive_only=ph_sym) * np.pi / iaft.beta
  else:
      iw_mesh = np.asarray(iaft)
  
  target_shape = g_iw.shape[1:]
  niw = g_iw.shape[0]
  
  # Validate iw_mesh length
  if len(iw_mesh) != niw:
    raise ValueError(f"iw_mesh length ({len(iw_mesh)}) must match first dimension of g_iw ({niw})")
  
  # Reshape to 2D for backend: [niw, dim1]
  g_iw_2D = g_iw.reshape(niw, -1)
  
  # Call C++ backend
  g_w_2D, w_mesh = pproc_mod.pade(g_iw_2D, iw_mesh, wmin, wmax, Nw, Nfit, eta, ph_sym)
  
  # Reshape output back to original shape with first dimension = Nw
  g_w = g_w_2D.reshape((Nw,) + target_shape)
  
  return g_w, w_mesh


def aaa_adapol(g_iw, iaft, stats, wmin, wmax, Nw, *, Nfit=40, eta=None, ph_sym=False):
    try:
        from adapol import anacont as adapol_anacont
    except ImportError:
        raise ImportError("aaa_adapol requires the adapol package (https://github.com/flatironinstitute/adapol). \n"
                          "Ensure that it is installed. ")

    if stats in ['fermion', 'f']:
        statistic = "Fermion"
    elif stats in ['boson', 'b']:
        statistic = "Boson"
    else:
        raise ValueError("Invalid statistic. Use 'fermion' or 'boson'.")

    g_iw = np.asarray(g_iw)
    target_shape = g_iw.shape[1:]
    if g_iw.ndim  == 1:
        g_iw = g_iw[:, None, None]
    elif g_iw.ndim == 2:
        g_iw = np.array([ np.diag(x) for x in g_iw])
    elif g_iw.ndim > 3:
        raise ValueError("Input g_iw can only have 1, 2, or 3 dimensions.")

    if isinstance(iaft, IAFT):
        iw_mesh = 1j*iaft.wn_mesh(stats=stats, positive_only=ph_sym) * np.pi / iaft.beta
    else:
        iw_mesh = np.asarray(iaft)
    
    gw_func, error, poles, weights = adapol_anacont(
        g_iw, iw_mesh, 
        Np = Nfit, statistics=statistic
    )
    
    if eta is None:
        eta = np.pi / iaft.beta if isinstance(iaft, IAFT) else 0.001
    else: 
        eta = float(eta)
    w_mesh = np.linspace(wmin, wmax, Nw) + 1j * eta
    g_w = gw_func(w_mesh).reshape((Nw,) + target_shape)

    return g_w, w_mesh


def minipole(g_iw, iaft, stats, wmin, wmax, Nw, *, tol=1e-4, eta=None, ph_sym=False):
    try:
        from mini_pole import MiniPole
    except ImportError:
        raise ImportError("minipole function requires the mini_pole package (https://github.com/Green-Phys/MiniPole/tree/main). \n"
                          "Ensure that it is installed. ")

    if not isinstance(iaft, IAFT):
        raise ValueError("iaft must be an instance of IAFT for minipole fitting. ")
    
    g_iw = np.asarray(g_iw, dtype=np.complex128)
    ndim = g_iw.ndim
    if g_iw.ndim == 1 or g_iw.ndim == 2:
        alg_type = "scalar"
        g_iw = g_iw.reshape(g_iw.shape[0], -1)  # Ensure 2D shape for scalar case
    elif g_iw.ndim == 3:
        alg_type = "matrix"
        assert g_iw.shape[1] == g_iw.shape[2], "For matrix case, g_iw must have shape [niw, norb, norb]."
    else:
        raise ValueError("Input g_iw can only have 1, 2, or 3 dimensions.")
    
    # minipole requires positive uniformlly spaced frequencies  
    nw_max = iaft.wn_mesh(stats=stats)[-1]
    offset = 1 if stats in ['fermion', 'f'] else 0
    iw_mesh_uniform = np.arange(offset, nw_max, 2)
    g_iw_pos = iaft.w_interpolate(g_iw, iw_mesh_uniform, stats=stats, ph_sym=ph_sym)

    eta = np.pi / iaft.beta if eta is None else float(eta)
    w_mesh = np.linspace(wmin, wmax, Nw) + 1j * eta

    #tools for calculating the Green's function from the corresponding poles
    def cal_G_scalar(z, Al, xl):
        G_z = 0.0
        for i in range(xl.size):
            G_z += Al[i] / (z - xl[i])
        return G_z

    def cal_G_vector(z, Al, xl):
        G_z = 0.0
        for i in range(xl.size):
            G_z += Al[[i]] / (z.reshape(-1, 1) - xl[i])
        return G_z

    def _minipole_fit_scalar(g_iw, iw_mesh, w_mesh, tol):
        mp = MiniPole(g_iw, iw_mesh*np.pi/iaft.beta, err=tol)
        g_w = cal_G_scalar(w_mesh, mp.pole_weight, mp.pole_location)
        return g_w
    
    def _minipole_fit_matrix(g_iw, iw_mesh, w_mesh, tol):
        mp = MiniPole(g_iw, iw_mesh*np.pi/iaft.beta, err=tol)
        n_orb = g_iw.shape[1]
        g_w = cal_G_vector(w_mesh, mp.pole_weight.reshape(-1, n_orb ** 2), mp.pole_location).reshape(-1, n_orb, n_orb)
        return g_w

    if alg_type == "scalar":
        g_w = np.zeros((Nw, g_iw_pos.shape[1]), dtype=np.complex128)
        for i in range(g_iw_pos.shape[1]):
            g_w[:, i] = _minipole_fit_scalar(g_iw_pos[:, i], iw_mesh_uniform, w_mesh, tol)
    else:
        g_w = _minipole_fit_matrix(g_iw_pos, iw_mesh_uniform, w_mesh, tol)

    if ndim == 1:
        g_w = g_w.reshape(-1)

    return g_w, w_mesh
