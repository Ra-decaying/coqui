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

import triqs.utility.mpi as mpi
from h5 import HDFArchive
import numpy as np

from triqs.gf import Gf, make_gf_dlr, BlockGf, Block2Gf, MeshImFreq, MeshDLRImFreq


def print_title_box(name, box_width=19):
    top_left = '╔'
    top_right = '╗'
    bottom_left = '╚'
    bottom_right = '╝'
    horizontal = '═'
    vertical = '║'

    title = f"{vertical}{name.center(box_width - 2)}{vertical}"
    top_border = f"{top_left}{horizontal * (box_width - 2)}{top_right}"
    bottom_border = f"{bottom_left}{horizontal * (box_width - 2)}{bottom_right}"

    mpi.report("\n"+top_border)
    mpi.report(title)
    mpi.report(bottom_border+"\n")


def print_degenerate_blks(deg_blks, gf_struct):
    mpi.report(f"Degenerate blocks for impurity")
    mpi.report("-------------------------------")
    for i, blks in enumerate(deg_blks):
        subset = [gf_struct[i] for i in blks]
        mpi.report(f"Shell {i}: {subset}\n")


def save_impurities(h5_ar, *, solver_results=None, solver_inputs=None, impurity_index=-1, iteration=-1):
    if solver_results is None and solver_inputs is None:
        raise ValueError("Either solver_results or solver_inputs must be provided.")

    if iteration == -1:
        iteration = 1 if "final_iteration" not in h5_ar.keys() else h5_ar["final_iteration"] + 1

    if solver_results and solver_inputs:
        assert len(solver_results) == len(solver_inputs), "solver_results and solver_inputs must have the same length."
        num_impurities = len(solver_results)
    elif solver_results:
        num_impurities = len(solver_results)
    else:
        num_impurities = len(solver_inputs)

    if impurity_index == -1:
        impurity_list = np.arange(num_impurities)
    else:
        assert impurity_index < num_impurities, f"impurity_index {impurity_index} is out of range [0, {num_impurities})."
        impurity_list = [impurity_index]

    if f"iter{iteration}" not in h5_ar.keys():
        h5_ar.create_group(f"iter{iteration}")
    iter_grp = h5_ar[f"iter{iteration}"]

    for imp_i in impurity_list:
        if f"impurity_{imp_i}" not in iter_grp.keys():
            iter_grp.create_group(f"impurity_{imp_i}")
        imp_grp = iter_grp[f"impurity_{imp_i}"]
        if solver_results is not None:
            _write_impurity_results(imp_grp, solver_results[imp_i])
        if solver_inputs is not None:
            _write_impurity_inputs(imp_grp, solver_inputs[imp_i])

    h5_ar["final_iteration"] = iteration


def _write_impurity_results(h5_grp, impurity_results):
    if "results" not in h5_grp.keys():
        h5_grp.create_group("results")
    res_grp = h5_grp["results"]
    res_grp['gf_struct'] = impurity_results['gf_struct']
    res_grp['G_iw'] = impurity_results['G_iw']
    res_grp['Sigma_infty'] = impurity_results['Sigma_infty']
    res_grp['Sigma_iw'] = impurity_results['Sigma_iw']
    res_grp['Pi_iw'] = impurity_results['Pi_iw']
    res_grp['W_iw'] = impurity_results['W_iw']
    res_grp['orbital_occupations'] = impurity_results['orbital_occupations']
    res_grp['average_order'] = impurity_results['average_order']
    res_grp['average_sign'] = impurity_results['average_sign']

    # dc corrections if present
    res_grp['Sigma_infty_dc'] = impurity_results.get('Sigma_infty_dc', None)
    res_grp['Sigma_iw_dc_data'] = impurity_results.get('Sigma_iw_dc_data', None)
    res_grp['Pi_iw_dc_data'] = impurity_results.get('Pi_iw_dc_data', None)


def _write_impurity_inputs(h5_grp, impurity_inputs):
    if "inputs" not in h5_grp.keys():
        h5_grp.create_group("inputs")
    inp_grp = h5_grp["inputs"]
    inp_grp['Gloc_t'] = impurity_inputs['Gloc_t']
    inp_grp['Wloc_t'] = impurity_inputs['Wloc_t']
    inp_grp['Vloc'] = impurity_inputs['Vloc']
    inp_grp['u_weiss_iw'] = impurity_inputs['u_weiss_iw']
    inp_grp['g_weiss_iw'] = impurity_inputs['g_weiss_iw']
    inp_grp['delta_iw'] = impurity_inputs['delta_iw']
    inp_grp['h0'] = impurity_inputs['h0']


def read_impurity_chkpt(h5_filename, iteration=-1, *, only_results=False, impurity_indices=None):
    mpi.report(f"Reading impurity checkpoint file: {h5_filename}")
    solver_results, solver_inputs = [], []
    res_tmp, inp_tmp = {}, {}
    num_impurities = -1

    if mpi.is_master_node():
        with HDFArchive(h5_filename, 'r') as ar:
            if iteration == -1:
                iteration = ar["final_iteration"]

            iter_grp = ar[f"iter{iteration}"]
            num_impurities = len([k for k in iter_grp.keys() if k.startswith("impurity_")])
            mpi.bcast(num_impurities)

            # Determine which impurities to read
            if impurity_indices is None:
                impurity_indices = np.arange(num_impurities)
            else:
                assert isinstance(impurity_indices, list), "impurity_indices must be a list of integers."
            mpi.report(f"impurity list = {impurity_indices}\n")

            for i in impurity_indices:
                res_grp = iter_grp[f"impurity_{i}/results"]
                res_tmp = read_all_keys(res_grp)
                mpi.bcast(res_tmp)
                solver_results.append(res_tmp)

                if not only_results:
                    inp_grp = iter_grp[f"impurity_{i}/inputs"]
                    inp_tmp = read_all_keys(inp_grp)
                    mpi.bcast(inp_tmp)
                    solver_inputs.append(inp_tmp)
    else:
        num_impurities = mpi.bcast(num_impurities)

        # Determine which impurities to read
        if impurity_indices is None:
            impurity_indices = np.arange(num_impurities)
        else:
            assert isinstance(impurity_indices, list), "impurity_indices must be a list of integers."

        for _ in impurity_indices:
            res_tmp = mpi.bcast(res_tmp)
            solver_results.append(res_tmp)
            if not only_results:
                inp_tmp = mpi.bcast(inp_tmp)
                solver_inputs.append(inp_tmp)

    mpi.barrier()

    if only_results:
        return solver_results
    return solver_results, solver_inputs


def update_impurity_results_from_chkpt(solver_results, h5_filename, iteration=-1):
    mpi.report(f"Updating the impurity results from checkpoint file: {h5_filename}\n")
    res_tmp = {}
    if mpi.is_master_node():
        with HDFArchive(h5_filename, 'r') as ar:
            if iteration == -1:
                iteration = ar['final_iteration']
            for imp_index, res in enumerate(solver_results):
                # TODO check if impurity results exist, if not skip it
                imp_grp = ar[f'iter{iteration}/impurity_{imp_index}/results']
                res_tmp.update(read_all_keys(imp_grp))
                res_tmp = mpi.bcast(res_tmp)
                res.update(res_tmp)
    else:
        for imp_index, res in enumerate(solver_results):
            res_tmp = mpi.bcast(res_tmp)
            res.update(res_tmp)

    iteration = mpi.bcast(iteration)

    return iteration


def read_all_keys(h5_grp):
    output = {}
    for key in h5_grp.keys():
        output[key] = h5_grp[key]
    return output
