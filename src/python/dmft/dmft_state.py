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

import os
import numpy as np

import triqs.utility.mpi as mpi

from coqui.utils.imag_axes_ft import IAFT
import coqui.embed.triqs_interface.utils as edmft_utils
import coqui.embed.triqs_interface.io as edmft_io

"""
Data structure for DMFT state 
"""

def _mix_into(curr, prev, mixing_factor=0.7):
    """In-place: handles ndarray OR list-of-ndarrays; falls back to object algebra."""
    # list of arrays
    if isinstance(curr, list) and isinstance(prev, list):
        if len(curr) != len(prev):
            raise ValueError(f"List length mismatch: {len(curr)} vs {len(prev)}")
        for i, (c, p) in enumerate(zip(curr, prev)):
            if isinstance(c, np.ndarray) and isinstance(p, np.ndarray):
                if c.shape != p.shape:
                    raise ValueError(f"Shape mismatch at list[{i}]: {c.shape} vs {p.shape}")
                c *= mixing_factor
                c += (1 - mixing_factor) * p
            else:
                # fall back to reassignment (TRIQS objects or scalars)
                curr[i] = mixing_factor * c + (1 - mixing_factor) * p
        return

    # plain ndarray
    if isinstance(curr, np.ndarray) and isinstance(prev, np.ndarray):
        if curr.shape != prev.shape:
            raise ValueError(f"Shape mismatch: {curr.shape} vs {prev.shape}")
        curr *= mixing_factor
        curr += (1 - mixing_factor) * prev
        return

    # fallback: rely on object algebra (e.g., TRIQS Gf/BlockGf support)
    curr *= mixing_factor
    curr += (1 - mixing_factor) * prev
    return


class DMFTState(object):
    """
    """
    def __init__(self, embedding_1e, embedding_2e,
                 iw_mesh_f, iw_mesh_b, ir_crystal,
                 spin_average=False, verbal=False):
        self.iteration = 0
        self.spin_average = spin_average
        self.embedding = {'1e': embedding_1e, '2e': embedding_2e}
        self.ir_kernel = ir_crystal

        self.local_sigma_w = None
        self.local_sigma_infty = None
        self.local_pi_w = None

        self.solver_inputs = [ {
            'gf_struct': self.embedding['1e'].imp_block_shape[imp],
            'Gloc_t': None,
            'Wloc_t': None,
            'Vloc': None,
            'u_weiss_iw': None,
            'g_weiss_iw': None,
            'delta_iw': None,
            'h0': None
        } for imp in range(self.embedding['1e'].n_impurities) ]

        self.solver_results = [ {
            'gf_struct': self.embedding['1e'].imp_block_shape[imp],
            'symm_blks': None,
            'iw_mesh_f': iw_mesh_f,
            'iw_mesh_b': iw_mesh_b,
            'Sigma_infty': None,
            'Sigma_iw': None,      # triqs BlockGf version
            'Sigma_iw_data': None, # numpy array on IR mesh
            'Pi_iw': None,         # triqs Gf version
            'Pi_iw_data': None,    # numpy array on IR mesh
            'Sigma_infty_dc': None,
            'Sigma_iw_dc_data': None,
            'Pi_iw_dc_data': None
        } for imp in range(self.embedding['1e'].n_impurities) ]

        #if verbal:
        #    mpi.report(self)
        #    mpi.barrier()

    def __str__(self):
        return ("DMFT State                        \n" \
                "----------------------------------\n" )

    def __eq__(self, other):
        if not isinstance(other, DMFTState):
            return NotImplemented

        return (
                self.embedding['1e'] == other.embedding['1e'] and
                self.embedding['2e'] == other.embedding['2e']
        )

    @classmethod
    def make_dmft_state(cls, coqui_h5, embedding_1e, embedding_2e,
                        wmax_imp=None, prec_imp=None, spin_average=False,
                        verbal=False):
        from triqs.gf import MeshDLRImFreq
        ir_kernel = IAFT.from_coqui_chkpt(coqui_h5, verbose=verbal)
        # Compatible TRIQS meshes: one for fermion and one for boson, where all triqs Gfs live.
        if wmax_imp is None:
            wmax_imp = ir_kernel.wmax
        if prec_imp is None:
            prec_imp = ir_kernel.prec
        iw_mesh_f = MeshDLRImFreq(
            beta=ir_kernel.beta, statistic='Fermion', w_max=wmax_imp, eps=prec_imp, symmetrize=True
        )
        iw_mesh_b = MeshDLRImFreq(
            beta=ir_kernel.beta, statistic='Boson', w_max=wmax_imp, eps=prec_imp, symmetrize=True
        )
        return cls(embedding_1e, embedding_2e, iw_mesh_f, iw_mesh_b,
                   ir_kernel, spin_average, verbal)


    def load(self, solver_chkpt):
        if not os.path.isfile(solver_chkpt):
            mpi.report("No solver checkpoint file found. Will skip loading impurity results.\n")
            return

        # TODO for "each" impurity, load the previous results if existing, otherwise initialize to empty
        self.iteration = edmft_io.read_impurity_chkpt(self.solver_results, solver_chkpt) + 1
        for res in self.solver_results:
            assert res['Sigma_iw'].mesh == res['iw_mesh_f'], (
                "Incompatible fermionic Matsubara mesh in loaded impurity results."
            )
            gf_struct = [(bl, gf.target_shape[0]) for (bl, gf) in res['Sigma_iw']]
            assert gf_struct == res['gf_struct'], (
                "Incompatible gf_struct in loaded impurity results."
            )
            assert res['Pi_iw'].mesh == res['iw_mesh_b'], (
                "Incompatible bosonic Matsubara mesh in loaded impurity results."
            )

        self.local_sigma_w, self.local_sigma_infty, self.local_pi_w = edmft_utils.embedding(
            self.embedding['1e'], self.embedding['2e'], self.solver_results,
            self.ir_kernel, self.spin_average
        )
        if mpi.is_master_node():
            self.ir_kernel.check_leakage(self.local_sigma_w["imp"], stats='f', name='Sigma_imp', w_input=True)
            self.ir_kernel.check_leakage(self.local_sigma_w["dc"], stats='f', name='Sigma_dc', w_input=True)
            self.ir_kernel.check_leakage_phsym(self.local_pi_w["imp"], stats='b', name='Pi_imp', w_input=True)
            self.ir_kernel.check_leakage_phsym(self.local_pi_w["dc"], stats='b', name='Pi_dc', w_input=True)
            mpi.report("")
        mpi.barrier()


    def save(self, solver_chkpt):
        if mpi.is_master_node():
            edmft_io.save_impurities(solver_chkpt, self.solver_results, self.solver_inputs, self.iteration)
        mpi.barrier()


    def embed_impurity_results(self):
        local_sigma_w, local_sigma_infty, local_pi_w = (
            edmft_utils.embedding(
                self.embedding['1e'], self.embedding['2e'],
                self.solver_results,
                self.ir_kernel,
                self.spin_average
            )
        )
        # check convergence:
        if self.local_sigma_w and self.local_sigma_infty and self.local_pi_w:
            max_diff_sigma_w = np.max(
                np.abs(local_sigma_w["imp"] - local_sigma_w["dc"]
                   - self.local_sigma_w["imp"] + self.local_sigma_w["dc"])
            )
            max_diff_sigma_infty = np.max(
                np.abs(local_sigma_infty["imp"] - local_sigma_infty["dc"]
                   - self.local_sigma_infty["imp"] + self.local_sigma_infty["dc"])
            )
            max_diff_pi_w = np.max(
                np.abs(local_pi_w["imp"] - local_pi_w["dc"]
                       - self.local_pi_w["imp"] + self.local_pi_w["dc"])
            )
            mpi.report(f"Max difference in embedded impurity results: \n"
                       f"|Delta Sigma_w|     = {max_diff_sigma_w}, \n"
                       f"|Delta Sigma_infty| = {max_diff_sigma_infty}, \n"
                       f"|Delta Pi_w|        = {max_diff_pi_w}\n")

        self.local_sigma_w     = local_sigma_w
        self.local_sigma_infty = local_sigma_infty
        self.local_pi_w        = local_pi_w

        if mpi.is_master_node():
            self.ir_kernel.check_leakage(self.local_sigma_w["imp"], stats='f', name='Sigma_imp', w_input=True)
            self.ir_kernel.check_leakage(self.local_sigma_w["dc"], stats='f', name='Sigma_dc', w_input=True)
            self.ir_kernel.check_leakage_phsym(self.local_pi_w["imp"], stats='b', name='Pi_imp', w_input=True)
            self.ir_kernel.check_leakage_phsym(self.local_pi_w["dc"], stats='b', name='Pi_dc', w_input=True)
            mpi.report("")
        mpi.barrier()


    def damp_impurity_results(self, solver_chkpt, mixing=0.7):
        if self.iteration <= 0:
            # no iterative solve on the first iteration
            return

        solver_results_prev = [ {} for _ in range(self.embedding['1e'].n_impurities) ]
        iteration_prev = edmft_io.read_impurity_chkpt(solver_results_prev, solver_chkpt, self.iteration-1)
        assert iteration_prev == self.iteration-1, (
            "Oh oh, something went wrong when reading previous iteration impurity results."
        )
        keys_to_damp = ['Sigma_infty', 'Sigma_infty_dc',
                        'Sigma_iw', 'Sigma_iw_dc_data',
                        'Pi_iw', 'Pi_iw_dc_data']
        for imp_idx, (res, res_prev) in enumerate(zip(self.solver_results, solver_results_prev)):
            for key in keys_to_damp:
                if key not in res or key not in res_prev:
                    raise KeyError(f"Missing key '{key}' for impurity {imp_idx} during damping.")
                _mix_into(res[key], res_prev[key], mixing)
