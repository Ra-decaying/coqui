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
from h5 import HDFArchive

import triqs_modest as modest

import coqui
import coqui.dmft as coqui_dmft

Hartree_eV = 27.211386245988

def gw_edmft_loop(mf, thc, solver_chkpt_h5,
                  proj_info, embedding_1e, embedding_2e,
                  gw_params, gloc_params, wloc_params, impurity_params, embed_params,
                  inner_num_iter, outer_num_iter, restart=False):
    """
    GW+EDMFT self-consistency loop
    :return:
    """

    coqui_mpi = mf.mpi()

    coqui_chkpt_h5 = gw_params['output']+".mbpt.h5"
    iterative_params = impurity_params.pop('iter_alg', None)
    assert iterative_params['alg'] == 'damping', (
        "Only \'damping\' iterative algorithm is supported in GW+EDMFT at the moment."
    )
    if isinstance(impurity_params['solver'], dict):
        impurity_params['solver'] = [impurity_params['solver']]

    # DMFT state container
    dmft_state = coqui_dmft.DMFTState.make_dmft_state(
        coqui_chkpt_h5, embedding_1e, embedding_2e,
        wmax_imp=impurity_params.pop('dlr_wmax', None),
        prec_imp=impurity_params.pop('dlr_eps', None),
        spin_average=mf.nspin()==1,
        verbal=coqui_mpi.root()
    )
    if restart:
        dmft_state.load(solver_chkpt_h5)

    for iteration in range(outer_num_iter):
        with HDFArchive(coqui_chkpt_h5, 'r') as ar:
            input_type = "embed" if "embed" in ar.keys() else "scf"
            input_iter = ar[f"{input_type}/final_iter"]

        # GW
        gw_params["input_type"] = input_type
        gw_params["input_iter"] = input_iter
        coqui.run_gw(gw_params, h_int = thc, projector_info = proj_info,
                     local_polarizabilities = dmft_state.local_pi_w)
        coqui_mpi.barrier()

        # Set the Green's function for the non-local RPA polarizability
        with HDFArchive(coqui_chkpt_h5, 'r') as ar:
            mbpt_final_iter = ar["scf/final_iter"]
            pi_rpa_input = ar[f"scf/iter{mbpt_final_iter}/input_grp"]
            pi_rpa_input_iter = ar[f"scf/iter{mbpt_final_iter}/input_iter"]
        wloc_params["input_type"] = pi_rpa_input
        wloc_params["input_iter"] = pi_rpa_input_iter
        coqui_mpi.barrier()

        # inner EDMFT loop
        edmft_loop(
            mf, thc, proj_info, dmft_state, solver_chkpt_h5,
            gloc_params, wloc_params, impurity_params['solver'], embed_params,
            iterative_params, inner_num_iter
        )



def edmft_loop(mf, thc, proj_info, dmft_state, solver_chkpt_h5,
               gloc_params, wloc_params, solver_params_list, embed_params,
               iterative_params, num_iter):

    coqui_mpi = mf.mpi()
    coqui_chkpt_h5 = gloc_params['outdir']+"/"+gloc_params['prefix']+".mbpt.h5"

    for iteration in range(dmft_state.iteration, dmft_state.iteration+num_iter):
        with HDFArchive(coqui_chkpt_h5, 'r') as ar:
            input_type = "embed" if "embed" in ar.keys() else "scf"
            input_iter = ar[f"{input_type}/final_iter"]

        # downfold for W_loc
        # input_type and input_iter should be fixed during the inner loop
        Vloc, Wloc_t = coqui.downfold_local_coulomb(
            thc, wloc_params,
            projector_info=proj_info,
            local_polarizabilities=dmft_state.local_pi_w
        )

        # downfold for G_loc
        gloc_params["input_type"] = input_type
        gloc_params["input_iter"] = input_iter
        Gloc_t = coqui.downfold_local_gf(mf, gloc_params, projector_info=proj_info)
        if mf.nspin() == 1:
            Gloc_t = np.repeat(Gloc_t, repeats=2, axis=1) # duplicate the spin channel

        if coqui_mpi.root():
            dmft_state.ir_kernel.check_leakage(Gloc_t, stats='f', name='Gloc')
            dmft_state.ir_kernel.check_leakage_phsym(Wloc_t, stats='b', name='Wloc')

        # Extract local Green's function and screened interactions for each impurity
        Gimp_C    = dmft_state.embedding['1e'].extract_wij(Gloc_t)   # block matrix
        Vimp_C    = dmft_state.embedding['2e'].extract_ijkl(Vloc)    # (norb, norb, norb, norb)
        Wimp_C    = dmft_state.embedding['2e'].extract_wijkl(Wloc_t) # (nts, norb, norb, norb, norb)

        for imp_index, (G_t, W_t, V) in enumerate(zip(Gimp_C, Wimp_C, Vimp_C)):
            coqui_dmft.print_title_box(f"IMPURITY {imp_index}")

            solver_params = solver_params_list[imp_index]
            Res = dmft_state.solver_results[imp_index]
            Input  = dmft_state.solver_inputs[imp_index]
            Input['Gloc_t'], Input['Wloc_t'], Input['Vloc'] = G_t, W_t, V

            # Fermionic and bosonic Weiss fields
            Input['g_weiss_iw'], Input['u_weiss_iw'] = (
                coqui_dmft.compute_weiss_fields_w(
                    ir_kernel = dmft_state.ir_kernel,
                    local_gf = {
                        "Gloc_t": coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Res['gf_struct']),
                        "Wloc_t": Input['Wloc_t'],
                        "Vloc": Input['Vloc']
                    },
                    impurity_selfenergies = {
                        "Vhf_imp": coqui_dmft.blk_arr_to_arr(Res['Sigma_infty'], Res['gf_struct']),
                        "Sigma_imp_w": coqui_dmft.blk_arr_to_arr(Res['Sigma_iw_data'], Res['gf_struct']),
                        "Pi_imp_w": Res['Pi_iw_data'][0] if Res['Pi_iw_data'] else None
                    },
                    density_only=True
                )
            )
            Ub, Ubp, Jb_spin, Jb_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc'])
            U, Up, J_spin, J_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc']+Input['u_weiss_iw'][0])
            if coqui_mpi.root():
                print("Hubbard-Kanamori interaction at bare and zero frequency")
                print("--------------------------------------------------------")
                print(f"  intra-orbital                  = {Ub*Hartree_eV:.4f}, {U*Hartree_eV} eV")
                print(f"  inter-orbital                  = {Ubp*Hartree_eV:.4f}, {Up*Hartree_eV} eV")
                print(f"  Hund's coupling (spin-flip)    = {Jb_spin*Hartree_eV:.4f}, {J_spin*Hartree_eV:.4f} eV")
                print(f"  Hund's coupling (pair-hopping) = {Jb_pair*Hartree_eV:.4f}, {J_pair*Hartree_eV:.4f} eV\n")

            # h0: (nspin, norb, norb), delta_iw: (niw, nspin, norb, norb)
            Input['h0'], Input['delta_iw'] = coqui_dmft.extract_h0_and_delta(
                Input['g_weiss_iw'], dmft_state.ir_kernel
            )

            dmft_state.save_impurity_inputs(solver_chkpt_h5, imp_index)

            # GW double counting contributions
            Res.update(
                coqui_dmft.solve_gw_dc(
                    coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Res['gf_struct']),
                    Input['Vloc'], Input['Wloc_t'], Input['u_weiss_iw'],
                    dmft_state.ir_kernel, density_only=True,
                    gf_struct=Res['gf_struct']
                )
            )

            # Convert CoQuí outputs to TRIQS containers
            h0, delta_iw, h_int, u_weiss_iw = coqui_dmft.to_triqs_containers(
                Input['h0'], Input['delta_iw'], Input['Vloc'], Input['u_weiss_iw'],
                dmft_state.ir_kernel, gf_struct = Res['gf_struct'],
                triqs_iw_mesh = {"fermion": Res['iw_mesh_f'], "boson": Res['iw_mesh_b']},
                density_hamiltonian = True, real_hamiltonian = True
            )

            # Analyze block symmetry
            if solver_params['degenerate_blk'] is None:
                if coqui_mpi.root():
                    print("Analyzing block symmetries via the hybridization function...\n")
                solver_params['degenerate_blk'] = modest.analyze_degenerate_blocks(
                    delta_iw, threshold=solver_params['degenerate_blk_thresh']
                )
            else:
                solver_params['degenerate_blk'] = [np.array(x) for x in solver_params["degenerate_blk"]]
            coqui_dmft.print_degenerate_blks(solver_params['degenerate_blk'], Res['gf_struct'])
            delta_iw = modest.symmetrize(delta_iw, solver_params['degenerate_blk'])

            # Call impurity solver, and store sigma_imp, vhf_imp, and pi_imp in "Res"
            solver_kwargs = solver_params.copy()
            solver_kwargs.pop('degenerate_blk_thresh', None)
            Res.update(
                coqui_dmft.ctseg.solve_dynamic_imp(delta_iw, h0, u_weiss_iw, h_int, **solver_kwargs)
            )

            dmft_state.damp_impurity_results(
                solver_chkpt_h5, mixing = iterative_params.get('mixing', 0.7), impurity_indices=[imp_index]
            )

            # save solver results for current impuprity
            dmft_state.save_impurity_results(solver_chkpt_h5, imp_index)

        # Embed impurity results
        dmft_state.embed_impurity_results()

        # Upfolding
        coqui.dmft_embed(
            mf, embed_params,
            projector_info = proj_info,
            local_hf_potentials = dmft_state.local_sigma_infty,
            local_sigma_dynamic = dmft_state.local_sigma_w
        )
        dmft_state.iteration += 1
        coqui_mpi.barrier()

