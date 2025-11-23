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
from h5 import HDFArchive

from coqui.utils.imag_axes_ft import IAFT
import coqui.dmft.utils as dmft_utils
import coqui.dmft.io as dmft_io

"""
Data structure for DMFT state 
"""

# TODO use coqui MPIHandler? or triqs mpi wrapper?

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
                 spin_average=False, screen_type='gw_edmft',
                 verbal=False):
        self.iteration = 0
        self.spin_average = spin_average
        self.embedding = {'1e': embedding_1e, '2e': embedding_2e}
        self.ir_kernel = ir_crystal

        self.local_sigma_w = None
        self.local_sigma_infty = None
        self.local_pi_w = None
        assert screen_type in {"rpa", "gw_edmft", "gw_edmft_density"}, (
            f"DMFTState: incompatible screen_type = {screen_type} detected. "
            f"Available choices are \"rpa\", \"gw_edmft\" and \"gw_edmft_density\"."
        )

        self.solver_inputs = [ {
            'gf_struct': self.embedding['1e'].imp_block_shape[imp],
            'Gloc_t': None,
            'Wloc_t': None,
            'Vloc': None,
            'u_weiss_iw': None,
            'g_weiss_iw': None,
            'delta_iw': None,
            'h0': None,
            'screen_type': screen_type
        } for imp in range(self.embedding['1e'].n_impurities) ]

        self.solver_results = [ {
            'gf_struct': self.embedding['1e'].imp_block_shape[imp],
            'symm_blks': None,
            'iw_mesh_f': iw_mesh_f,
            'iw_mesh_b': iw_mesh_b,
            'mu_imp': 0.0,
            'Sigma_infty': None,
            'Sigma_iw': None,      # triqs BlockGf version
            'Sigma_iw_data': None, # numpy array on IR mesh
            'Pi_iw': None,         # triqs Gf version
            'Pi_iw_data': None,    # numpy array on IR mesh
            'W_iw': None,          # triqs Gf version
            'W_iw_data': None,     # numpy array on IR mesh
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
                        screen_type='gw_edmft', verbal=False):
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
                   ir_kernel, spin_average, screen_type, verbal)


    def load(self, solver_chkpt):
        if not os.path.isfile(solver_chkpt):
            mpi.report("No solver checkpoint file found. Will skip loading impurity results.\n")
            return

        # TODO for "each" impurity, load the previous results if existing, otherwise initialize to empty
        self.iteration = dmft_io.update_impurity_results_from_chkpt(self.solver_results, solver_chkpt) + 1

        # check dimensions of loaded impurity results
        nw_b_half = self.ir_kernel.nw_b//2 if self.ir_kernel.nw_b%2==0 else self.ir_kernel.nw_b//2 + 1
        for imp_idx, res in enumerate(self.solver_results):
            for blk_idx, (blk_name, blk_dim) in enumerate(res['gf_struct']):
                assert (res['Sigma_infty'][blk_idx].shape[0] == blk_dim and
                        res['Sigma_infty'][blk_idx].shape[1] == blk_dim), (
                    "Incompatible block dimension for the loaded impurity Sigma"
                )
                assert res['Sigma_iw_data'][blk_idx].shape[0] == self.ir_kernel.nw_f, (
                    "Incompatible fermionic Matsubara mesh for the loaded impurity Sigma"
                )
                assert (res['Sigma_iw_data'][blk_idx].shape[1] == blk_dim and
                        res['Sigma_iw_data'][blk_idx].shape[2] == blk_dim), (
                    "Incompatible block dimension for the loaded impurity Sigma"
                )
            gf_struct_2e = self.embedding['2e'].imp_block_shape[imp_idx]
            for blk_idx, (blk_name, blk_dim) in enumerate(gf_struct_2e):
                assert res['Pi_iw_data'][blk_idx].shape[0] == nw_b_half, (
                    "Incompatible bosonic Matsubara mesh for the loaded impurity Pi"
                )
                assert (res['Pi_iw_data'][blk_idx].shape[1] == blk_dim and
                        res['Pi_iw_data'][blk_idx].shape[2] == blk_dim), (
                    "Incompatible block dimension for the loaded impurity Pi"
                )

        self.local_sigma_w, self.local_sigma_infty, self.local_pi_w = dmft_utils.embed_impurities(
            self.embedding['1e'], self.embedding['2e'], self.solver_results, self.spin_average
        )
        if mpi.is_master_node():
            self.ir_kernel.check_leakage(self.local_sigma_w["imp"], stats='f', name='Sigma_imp', w_input=True)
            self.ir_kernel.check_leakage(self.local_sigma_w["dc"], stats='f', name='Sigma_dc', w_input=True)
            self.ir_kernel.check_leakage_phsym(self.local_pi_w["imp"], stats='b', name='Pi_imp', w_input=True)
            self.ir_kernel.check_leakage_phsym(self.local_pi_w["dc"], stats='b', name='Pi_dc', w_input=True)
            mpi.report("")
        mpi.barrier()


    def save_impurity_inputs(self, solver_chkpt, impurity_index):
        if mpi.is_master_node():
            with HDFArchive(solver_chkpt, 'a') as ar:
                if "dmft" not in ar.keys():
                    ar.create_group("dmft")
                dmft_io.save_impurities(
                    ar["dmft"], solver_inputs = self.solver_inputs,
                    impurity_index = impurity_index, iteration = self.iteration
                )
        mpi.barrier()


    def save_impurity_results(self, solver_chkpt, impurity_index):
        if mpi.is_master_node():
            with HDFArchive(solver_chkpt, 'a') as ar:
                if "dmft" not in ar.keys():
                    ar.create_group("dmft")
                dmft_io.save_impurities(
                    ar["dmft"], solver_results = self.solver_results,
                    impurity_index = impurity_index, iteration = self.iteration
                )
        mpi.barrier()

    def save(self, solver_chkpt):
        if mpi.is_master_node():
            with HDFArchive(solver_chkpt, 'a') as ar:
                if "dmft" not in ar.keys():
                    ar.create_group("dmft")
                dmft_io.save_impurities(
                    ar["dmft"], solver_results = self.solver_results,
                    solver_inputs = self.solver_inputs, iteration = self.iteration
                )
        mpi.barrier()


    def embed_impurity_results(self):
        # pi impurities are embedded even if screen_type == "rpa".
        # This is okay since coqui.run_gw() and coqui.downfold_wloc()
        # automatically omit pi_imp if screen_type="rpa"
        local_sigma_w, local_sigma_infty, local_pi_w = dmft_utils.embed_impurities(
            self.embedding['1e'], self.embedding['2e'], self.solver_results, self.spin_average
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


    def damp_impurity_results(self, solver_chkpt, mixing=0.7, *, impurity_indices=None):
        if self.iteration <= 0: # no damping in the first iteration
            return

        if impurity_indices is not None:
            assert isinstance(impurity_indices, list), "impurity_indices should be a list of integers."
        else:
            impurity_indices = np.arange(len(self.solver_results))

        mpi.report(f"Damping impurity results for impurities {impurity_indices}\n")
        solver_results_prev = dmft_io.read_impurity_chkpt(
            solver_chkpt, self.iteration-1, read="results", impurity_indices=impurity_indices
        )

        keys_to_damp = ['Sigma_infty', 'Sigma_infty_dc',
                        'Sigma_iw_data', 'Sigma_iw_dc_data',
                        'Pi_iw_data', 'Pi_iw_dc_data']
        for idx, imp_idx in enumerate(impurity_indices):
            res = self.solver_results[imp_idx]
            res_prev = solver_results_prev[idx]
            for key in keys_to_damp:
                if key not in res or key not in res_prev:
                    raise KeyError(f"Missing key '{key}' for impurity {imp_idx} during damping.")
                _mix_into(res[key], res_prev[key], mixing)
