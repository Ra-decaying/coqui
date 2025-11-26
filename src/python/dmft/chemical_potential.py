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
Utility functions for impurity chemical potential
"""
import triqs.utility.mpi as mpi
import numpy as np

from triqs.gf import inverse, iOmega_n#, Gf, make_gf_dlr, BlockGf, MeshImFreq, MeshDLRImFreq
import coqui.dmft as coqui_dmft


def compute_nelec(h0, delta_iw, sigma_infty, sigma_iw, mu):
    g_iw = sigma_iw.copy()
    for blk_idx, (blk_name, g) in enumerate(g_iw):
        g << inverse(iOmega_n + mu - h0[blk_idx] - sigma_infty[blk_idx] - delta_iw[blk_name] - sigma_iw[blk_name])

    nelec = 0.0
    for blk_name, occ in g_iw.density().items():
        nelec += np.diag(occ).sum()
    if nelec.imag >= 1e-6:
        mpi.report(f"WARNING: nelec.imag = {nelec.imag} >= 1e-6")

    return nelec.real


def compute_nelec_from_solver(gf_struct, h0_sab, delta_iw, u_weiss_iw, h_int, mu, **solver_params):
    # update h0 = h0 - mu_imp
    h0_mat_shifted = np.array([ h0_mat - np.eye(h0_mat.shape[0])*mu for h0_mat in h0_sab ])
    h0_op = coqui_dmft.h0_operator(h0_mat_shifted, gf_struct, diagonal=True, force_real=True)

    density_res = coqui_dmft.ctseg.solve_density_dynamic_u(delta_iw, h0_op, u_weiss_iw, h_int, **solver_params)
    imp_density = 0.0
    for blk_name, occ in density_res['orbital_occupations'].items():
        imp_density += occ.sum()

    return imp_density


def compute_mu_impurity(nelec_target, compute_nelec_fcn, tolerance=1e-2, mu0=0):

    mu, delta = mu0, 0.2
    nelec = compute_nelec_fcn(mu=mu0)

    mpi.report(f"Initial impurity chemical potential (mu_imp) = {mu0}, nelec = {nelec}")

    if abs(nelec - nelec_target) < tolerance:
        mpi.report(f"Impurity chemical potential (mu_imp) found = {mu}")
        mpi.report(f"Number of electron per impurity = {nelec}\n")
        return mu, nelec

    if nelec >= nelec_target:
        mu1, mu2 = mu0-delta, mu0
        nelec1 = compute_nelec_fcn(mu=mu1)
        while nelec1 > nelec_target:
            mu1 -= delta
            nelec1 = compute_nelec_fcn(mu=mu1)
        mpi.report(f"mu = {mu1}, nelec = {nelec1}")
    else:
        mu1, mu2 = mu0, mu0+delta
        nelec2 = compute_nelec_fcn(mu=mu2)
        while nelec2 < nelec_target:
            mu2 += delta
            nelec2 = compute_nelec_fcn(mu=mu2)
        mpi.report(f"mu = {mu2}, nelec = {nelec2}")
    mu_mid = (mu1 + mu2) * 0.5
    nelec = compute_nelec_fcn(mu=mu_mid)
    mpi.report(f"mu = {mu_mid}, nelec = {nelec}")
    while abs(nelec - nelec_target) >= tolerance:
        if nelec >= nelec_target:
            mu2 = mu_mid
        else:
            mu1 = mu_mid
        mu_mid = (mu1 + mu2) * 0.5
        nelec = compute_nelec_fcn(mu=mu_mid)
        mpi.report(f"mu = {mu_mid}, nelec = {nelec}")

    mu = mu_mid
    mpi.report(f"Impurity chemical potential (mu_imp) found = {mu}")
    mpi.report(f"Number of electron per impurity = {nelec}\n")

    return mu, nelec
