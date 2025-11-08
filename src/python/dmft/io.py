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


def save_impurities(h5_filename, solver_results, solver_inputs=None, iteration=-1):
    with HDFArchive(h5_filename, 'a') as ar:
        if iteration == -1:
            iteration = 1 if "final_iteration" not in ar.keys() else ar["final_iteration"] + 1

        ar.create_group(f"iter{iteration}")
        iter_grp = ar[f"iter{iteration}"]

        # Handle single or multiple results
        if isinstance(solver_results, (list, tuple)):
            for i, res in enumerate(solver_results):
                iter_grp.create_group(f"impurity_{i}")
                imp_grp = iter_grp[f"impurity_{i}"]
                _write_impurity_results(imp_grp, res)
                if solver_inputs is not None:
                    _write_impurity_inputs(imp_grp, solver_inputs[i])
        else:
            # Single result only
            iter_grp.create_group(f"impurity_0")
            imp_grp = iter_grp["impurity_0"]
            _write_impurity_results(imp_grp, solver_results)

        ar["final_iteration"] = iteration


def _write_impurity_results(h5_grp, impurity_results):
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
    h5_grp.create_group("inputs")
    inp_grp = h5_grp["inputs"]
    inp_grp['Gloc_t'] = impurity_inputs['Gloc_t']
    inp_grp['Wloc_t'] = impurity_inputs['Wloc_t']
    inp_grp['Vloc'] = impurity_inputs['Vloc']
    inp_grp['u_weiss_iw'] = impurity_inputs['u_weiss_iw']
    inp_grp['g_weiss_iw'] = impurity_inputs['g_weiss_iw']
    inp_grp['delta_iw'] = impurity_inputs['delta_iw']
    inp_grp['h0'] = impurity_inputs['h0']


def read_impurity_chkpt(solver_results, h5_filename, iteration=-1):
    mpi.report(f"Reading impurity results from checkpoint file: {h5_filename}\n")
    res_tmp = {}
    if mpi.is_master_node():
        with HDFArchive(h5_filename, 'r') as ar:
            if iteration == -1:
                iteration = ar['final_iteration']
            for imp_index, res in enumerate(solver_results):
                # TODO check if impurity results exist, if not skip it
                imp_grp = ar[f'iter{iteration}/impurity_{imp_index}/results']
                res_tmp.update(_read_impurity_results(imp_grp))
                res_tmp = mpi.bcast(res_tmp)
                res.update(res_tmp)
    else:
        for imp_index, res in enumerate(solver_results):
            res_tmp = mpi.bcast(res_tmp)
            res.update(res_tmp)

    iteration = mpi.bcast(iteration)

    return iteration


def _read_impurity_results(h5_grp):
    res = {}
    res['G_iw'] = h5_grp['G_iw']
    res['Sigma_infty'] = h5_grp['Sigma_infty']
    res['Sigma_iw'] = h5_grp['Sigma_iw']
    res['Pi_iw'] = h5_grp['Pi_iw']
    res['W_iw'] = h5_grp['W_iw']
    res['Sigma_infty_dc'] = h5_grp['Sigma_infty_dc']
    res['Sigma_iw_dc_data'] = h5_grp['Sigma_iw_dc_data']
    res['Pi_iw_dc_data'] = h5_grp['Pi_iw_dc_data']

    return res
