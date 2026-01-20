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

"""
Utility functions for bath fitting 
"""
import triqs.utility.mpi as mpi
import numpy as np


def bath_fitting(A_wsab, iw_mesh, statistics, Np=5,
                 *, name="", iw_mesh_out=None):
    mpi.report(f"Causal projection for {name} with nbath/orbital = {Np} and statistics = {statistics}")
    try:
        from adapol import hybfit as adapol_hybfit
    except ImportError:
        raise ImportError("bath fitting requires the adapol package (https://github.com/flatironinstitute/adapol). \n"
                          "Please ensure that it is installed. ")
    statistics = 'fermion' if statistics == 'f' else statistics
    statistics = 'boson' if statistics == 'b' else statistics
    if statistics not in ("fermion", "boson"):
        raise ValueError(f"Invalid statistics: {statistics!r}. Use 'fermion' or 'boson'.")
    if A_wsab.ndim < 3:
        raise ValueError("A_wsab must be at least 3D: (Nw, norb, norb) or (Nw, nspin, norb, norb).")
    if A_wsab.shape[-1] != A_wsab.shape[-2]:
        raise ValueError("The last two dimensions of A_wsab must be equal (square matrices).")
    if A_wsab.shape[0] != iw_mesh.shape[0]:
        raise ValueError("Mismatch: A_wsab.shape[0] (Nw) != iw_mesh.shape[0].")

    if iw_mesh_out is None:
        iw_mesh_out = iw_mesh

    original_shape = None
    if len(A_wsab.shape) != 4:
        original_shape = A_wsab.shape
        A_wsab = A_wsab.reshape(A_wsab.shape[0], -1, A_wsab.shape[-2], A_wsab.shape[-1])

    nspin, error = A_wsab.shape[1], -1
    A_out = np.zeros((iw_mesh_out.shape[0],) + A_wsab.shape[1:], dtype=A_wsab.dtype)
    for s in np.arange(nspin):
        _, __, fit_error, func = adapol_hybfit(
            A_wsab[:, s], iw_mesh, Np=Np, solver='sdp', verbose=False, statistics=statistics
        )
        A_out[:, s] = func(iw_mesh_out)
        error = max(error, abs(fit_error))
    mpi.report(f"Causal projection error =  {error}\n")

    if original_shape is not None:
        A_wsab = A_wsab.reshape(original_shape)
        A_out = A_out.reshape((iw_mesh_out.shape[0],) + original_shape[1:])

    return A_out


def causal_projection_boson(A_iw, ir_kernel, causal_params, ph_symmetry, target_name=""):
    """
    Perform causal projection for bosonic Green's functions.

    This function fits the input bosonic Green's function `A_iw` to a causal
    representation using bath fitting. It supports optional particle-hole
    symmetry enforcement and allows for customization of the fitting process
    through `causal_params`.

    Parameters:
        A_iw (numpy.ndarray): Input bosonic Green's function data.
        ir_kernel: Kernel object providing the Matsubara frequency mesh and
                   other transformations.
        causal_params (dict): Parameters for the causal projection, including:
                              - 'nbath_per_orbital': Number of bath orbitals.
                              - 'exclude_w0' (optional): Whether to exclude
                                the zero-frequency component.
        ph_symmetry (bool): If True, enforces particle-hole symmetry by
                            setting the imaginary part to zero.
        target_name (str, optional): Name of the target for reporting purposes.

    Returns:
        numpy.ndarray: The fitted bosonic Green's function in causal form.
    """
    if causal_params is None:
        return A_iw

    iw_mesh_b = ir_kernel.wn_mesh('b', positive_only=ph_symmetry) * np.pi / ir_kernel.beta
    exclude_w0 = causal_params.get('exclude_w0', False)
    zero_index = np.where(iw_mesh_b == 0.0)[0][0]
    iw_inputs = np.delete(iw_mesh_b, zero_index) if exclude_w0 else iw_mesh_b
    A_iw_input = np.delete(A_iw, zero_index, axis=0) if exclude_w0 else A_iw

    A_iw_fit = bath_fitting(
        A_iw_input, 1j*iw_inputs,
        statistics="boson", name=target_name,
        Np=causal_params["nbath_per_orbital"],
        iw_mesh_out=1j*iw_mesh_b
    )
    if ph_symmetry is True:
        A_iw_fit[:].imag = 0.0

    return A_iw_fit


def fit_impurity_results_boson(imp_res, ir_kernel, causal_params):
    if causal_params is None or causal_params.get("target", "both")=="local":
        return

    fit_res = {}
    if mpi.is_master_node():
        fit_res = {
            "Pi_iw_data": causal_projection_boson(
                imp_res["Pi_iw_data"][0], ir_kernel, causal_params, ph_symmetry=True,
                target_name="impurity polarizability"
            ),
            "W_iw_data": causal_projection_boson(
                imp_res["W_iw_data"][0], ir_kernel, causal_params, ph_symmetry=True,
                target_name="impurity screened interaction"
            )
        }
    fit_res = mpi.bcast(fit_res)
    imp_res["Pi_iw_data"][0] = fit_res["Pi_iw_data"]
    imp_res["W_iw_data"][0] = fit_res["W_iw_data"]


def fit_local_results_boson(local_res, ir_kernel, causal_params):
    if causal_params is None or causal_params.get("target", "both")=="impurity":
        return

    wloc_t_fit = None
    if mpi.is_master_node():
        wloc_iw = ir_kernel.tau_to_w_phsym(local_res["Wloc_t"], 'b')
        nbnd = wloc_iw.shape[-1]
        wloc_iw_fit = causal_projection_boson(
            wloc_iw.reshape(-1, nbnd*nbnd, nbnd*nbnd), ir_kernel, causal_params,
            ph_symmetry=True, target_name="local screened interaction"
        )
        wloc_t_fit = ir_kernel.w_to_tau_phsym(
            wloc_iw_fit.reshape(-1, nbnd, nbnd, nbnd, nbnd), 'b'
        )

    wloc_t_fit = mpi.bcast(wloc_t_fit)
    local_res["Wloc_t"] = wloc_t_fit


def fit_u_weiss(u_weiss_iw, ir_kernel, causal_params):
    if causal_params is None or causal_params.get("target", "both")=="impurity":
        return u_weiss_iw

    u_iw_fit = None
    if mpi.is_master_node():
        nbnd = u_weiss_iw.shape[-1]
        u_iw_fit = causal_projection_boson(
            u_weiss_iw.reshape(-1, nbnd*nbnd, nbnd*nbnd), ir_kernel, causal_params,
            ph_symmetry=True, target_name="bosonic Weiss field"
        )
        u_iw_fit = u_iw_fit.reshape(-1, nbnd, nbnd, nbnd, nbnd)

    u_iw_fit = mpi.bcast(u_iw_fit)
    return u_iw_fit
