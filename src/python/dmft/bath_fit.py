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
    mpi.report(f"Causal projection for {name} with nbath/orbital = {Np} and statistics={statistics}")
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
    mpi.report(f"Causal projection error =  {error}")

    if original_shape is not None:
        A_wsab = A_wsab.reshape(original_shape)
        A_out = A_out.reshape((iw_mesh_out.shape[0],) + original_shape[1:])

    return A_out

def causal_projection_boson_impl(pi_iw_data, w_iw_data, ir_kernel, causal_params):
    if causal_params is None:
        return {}

    iw_mesh_b = ir_kernel.wn_mesh('b', positive_only=True) * np.pi / ir_kernel.beta
    exclude_w0 = causal_params.get('exclude_w0', False)
    iw_inputs = iw_mesh_b[1:] if exclude_w0 else iw_mesh_b
    pi_iw_input = pi_iw_data[0][1:] if exclude_w0 else pi_iw_data[0]
    w_iw_input = w_iw_data[0][1:] if exclude_w0 else w_iw_data[0]

    pi_iw_fit = bath_fitting(
        pi_iw_input, 1j*iw_inputs,
        statistics="boson", name="impurity polarizability",
        Np=causal_params["nbath_per_orbital"],
        iw_mesh_out=1j*iw_mesh_b
    )
    pi_iw_fit.imag = 0.0

    w_iw_fit = bath_fitting(
        w_iw_input, 1j*iw_inputs,
        statistics="boson", name="impurity screened interaction",
        Np=causal_params["nbath_per_orbital"],
        iw_mesh_out=1j*iw_mesh_b
    )
    w_iw_fit.imag = 0.0

    return {"Pi_iw_data": [pi_iw_fit], "W_iw_data": [w_iw_fit]}


def causal_projection_boson(pi_iw_data, w_iw_data, ir_kernel, causal_params):
    fit_res = {}
    if mpi.is_master_node():
        fit_res = causal_projection_boson_impl(pi_iw_data, w_iw_data, ir_kernel, causal_params)
    fit_res = mpi.bcast(fit_res)
    return fit_res
