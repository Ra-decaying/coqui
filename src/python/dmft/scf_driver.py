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
from copy import deepcopy
from functools import partial
import numpy as np
from h5 import HDFArchive

import triqs_modest as modest

import coqui
from coqui.utils.imag_axes_ft import IAFT
import coqui.dmft as coqui_dmft

Hartree_eV = 27.211386245988


def gw_edmft_loop(mf, thc, proj_info, embedding_1e, 
                  niter, gw_iter_per_loop=1, edmft_iter_per_loop=1, 
                  inner_loop_alg=1, **gw_edmft_params):
    """
    GW+EDMFT self-consistency loop
    :return:
    """
    coqui_mpi = mf.mpi()
    if coqui_mpi.root():
        # http://patorjk.com/software/taag/#p=display&f=Calvin+S&t=COQUI+GW%2BEDMFT&x=none&v=4&h=4&w=80&we=false
        print("╔═╗╔═╗╔═╗ ╦ ╦╦  ╔═╗┬ ┬╔═╗┌┬┐┌┬┐┌─┐┌┬┐\n"
              "║  ║ ║║═╬╗║ ║║  ║ ╦│││║╣  │││││├┤  │ \n"
              "╚═╝╚═╝╚═╝╚╚═╝╩  ╚═╝└┴┘╚═╝─┴┘┴ ┴└   ┴ \n")
        print(f"  Total GW+EDMFT cycles (niter)       = {niter}")
        print(f"  GW iterations per GW+EDMFT cycle    = {gw_iter_per_loop}")
        print(f"  EDMFT iterations per GW+EDMFT cycle = {edmft_iter_per_loop}")
        print(f"    - Fix Gloc and Wloc during EDMFT iterations = {inner_loop_alg==2}\n")

    embedding_2e = embedding_1e.make_2particle.make_spinless
    if coqui_mpi.root():
        print(embedding_1e.description(True))
        print(embedding_2e.description(True))

    # deep copy to avoid any changes in gw_edmft_params affecting outside the function.
    gw_edmft_params = deepcopy(gw_edmft_params)
    try:
        gw_params        = gw_edmft_params.pop('gw', None)
        wloc_params      = gw_edmft_params.pop('wloc')
        gloc_params      = gw_edmft_params.pop('gloc')
        embed_params     = gw_edmft_params.pop('dmft_embed')
        impurity_params  = gw_edmft_params.pop('impurity')
        iterative_params = impurity_params.pop('iter_alg', None)
    except KeyError as e:
        raise KeyError(f"gw_edmft_loop: Missing required gw_edmft_params key: {e.args[0]}")

    if isinstance(impurity_params['solver'], dict):
        impurity_params['solver'] = [impurity_params['solver']]
    assert isinstance(impurity_params['solver'], list), (
        "impurity parameters should be a list of params - one for each impurity"
    )
    for solver_params in impurity_params['solver']:
        solver_params['n_cycles'] = int(solver_params['n_cycles']/coqui_mpi.comm_size())
        mu_params = solver_params.get('chemical_potential')
        if mu_params and 'n_cycles' in mu_params.keys():
            mu_params['n_cycles'] = int(mu_params['n_cycles']/coqui_mpi.comm_size())

    if iterative_params is not None:
        assert iterative_params['alg'] == 'damping', (
            "Only \'damping\' iterative algorithm is supported in GW+EDMFT at the moment."
        )

    if gw_params:
        assert gw_params['screen_type'] == wloc_params['screen_type']

    wloc_params['output_in_tau'] = True

    coqui_chkpt_h5 = embed_params['outdir']+"/"+embed_params['prefix']+".mbpt.h5"
    solver_chkpt_h5 = impurity_params.pop('chkpt_h5', coqui_chkpt_h5)

    # DMFT state container
    dmft_state = coqui_dmft.DMFTState.make_dmft_state(
        coqui_chkpt_h5, embedding_1e, embedding_2e,
        wmax_imp=impurity_params.pop('dlr_wmax', None),
        prec_imp=impurity_params.pop('dlr_eps', None),
        spin_average=mf.nspin()==1,
        screen_type=wloc_params['screen_type'],
        verbal=coqui_mpi.root()
    )
    if impurity_params.pop('restart', False):
        dmft_state.load(solver_chkpt_h5)


    for iteration in range(niter):

        if gw_params is not None and gw_iter_per_loop >= 1:
            # update GW solution with fixed impurity self-energies and polarizabilities
            _gw_loop(
                mf, thc, proj_info, 
                dmft_state, coqui_chkpt_h5, 
                gw_params, embed_params, gw_iter_per_loop
            )

        if edmft_iter_per_loop >= 1:
            # Set the Green's function for the non-local RPA polarizability
            with HDFArchive(coqui_chkpt_h5, 'r') as ar:
                mbpt_final_iter = ar["scf/final_iter"]
                try:
                    gf_for_wloc_source = ar[f"scf/iter{mbpt_final_iter}/greens_func_source"]
                    gf_for_wloc_iteration = ar[f"scf/iter{mbpt_final_iter}/greens_func_iteration"]
                except KeyError:
                    gf_for_wloc_source = ar[f"scf/iter{mbpt_final_iter}/input_grp"]
                    gf_for_wloc_iteration = ar[f"scf/iter{mbpt_final_iter}/input_iter"]
            wloc_params["greens_func_source"] = gf_for_wloc_source
            wloc_params["greens_func_iteration"] = gf_for_wloc_iteration
            coqui_mpi.barrier()

            # inner EDMFT loop
            edmft_alg = {1: _edmft_loop, 2: _edmft_loop_fixed_gloc_and_wloc}
            try:
                edmft_impl = edmft_alg[inner_loop_alg]
            except KeyError:
                raise ValueError(f"Unknown inner_loop_alg={inner_loop_alg!r} (expected 1, or 2)")
        
            edmft_impl(
                mf, thc, proj_info, dmft_state, solver_chkpt_h5, coqui_chkpt_h5, 
                gloc_params, wloc_params, impurity_params['solver'], embed_params, 
                iterative_params, edmft_iter_per_loop
            )


def _gw_loop(mf, thc, proj_info, 
             dmft_state, coqui_chkpt_h5, 
             gw_params, embed_params, gw_iter_per_loop):
    if gw_iter_per_loop < 1:
        return

    coqui_mpi = mf.mpi()
    for gw_iteration in range(gw_iter_per_loop):
        with HDFArchive(coqui_chkpt_h5, 'r') as ar:
            greens_func_source = "embed" if "embed" in ar.keys() else "scf"
            greens_func_iteration = ar[f"{greens_func_source}/final_iter"]
        coqui_mpi.barrier()

        # GW if gw_params presents
        gw_params["greens_func_source"] = greens_func_source
        gw_params["greens_func_iteration"] = greens_func_iteration
        coqui.run_gw(
            gw_params, h_int = thc, projector_info = proj_info,
            local_polarizabilities = dmft_state.local_pi_w
        )
        coqui_mpi.barrier()

        # Don't upfold the results if gw_iter_per_loop==1. 
        # Not sure if this is the best choice, but it allows us to skip one upfolding in the common case of 
        # doing just one GW iteration per GW+EDMFT loop, which can save some time.
        if gw_iter_per_loop > 1: 
            # Updates GW+EDMFT solution with the latest GW results while keeping the impurity solutions fixed.
            # Upfolding
            coqui.dmft_embed(
                mf, embed_params,
                projector_info = proj_info,
                local_hf_potentials = dmft_state.local_sigma_infty,
                local_sigma_dynamic = dmft_state.local_sigma_w
            )
            coqui_mpi.barrier()


def _edmft_loop(mf, thc, proj_info, dmft_state, solver_chkpt_h5, coqui_chkpt_h5, 
               gloc_params, wloc_params, solver_params_list, embed_params,
               iterative_params, num_iter):

    coqui_mpi = mf.mpi()

    for iteration in range(num_iter):
        with HDFArchive(coqui_chkpt_h5, 'r') as ar:
            greens_func_source = "embed" if "embed" in ar.keys() else "scf"
            greens_func_iteration = ar[f"{greens_func_source}/final_iter"]

        # downfold for W_loc
        # greens_func_source and greens_func_iteration should be fixed during the inner loop
        Vloc, Wloc_t = coqui.downfold_coulomb(
            thc, wloc_params,
            projector_info=proj_info,
            local_polarizabilities=dmft_state.local_pi_w
        )

        # downfold for G_loc
        gloc_params["greens_func_source"] = greens_func_source
        gloc_params["greens_func_iteration"] = greens_func_iteration
        Gloc_t = coqui.downfold_local_gf(mf, gloc_params, projector_info=proj_info)

        if coqui_mpi.root():
            dmft_state.ir_kernel.check_leakage(Gloc_t, stats='f', name='Gloc in the full MLWF space')
            dmft_state.ir_kernel.check_leakage_phsym(Wloc_t, stats='b', name='Wloc in the full MLWF space')

        # Convert spin axis → list of length nspin
        Vloc, Wloc_t = [Vloc], [Wloc_t]
        Gloc_t = [Gloc_t[:, s] for s in range(Gloc_t.shape[1])]
        if mf.nspin() == 1:
            Gloc_t = [Gloc_t[0], Gloc_t[0].copy()]
        
        # Extract local Green's function and screened interactions for each impurity
        Gloc_C    = dmft_state.embedding['1e'].extract(Gloc_t)   # block matrix
        Vloc_C    = dmft_state.embedding['2e'].extract(Vloc)
        Wloc_C    = dmft_state.embedding['2e'].extract(Wloc_t)
        Vloc_C    = [ V[0] for V in Vloc_C ]      # spinless
        Wloc_C    = [ W_t[0] for W_t in Wloc_C ]  # spinless

        for imp_index, (G_t, W_t, V) in enumerate(zip(Gloc_C, Wloc_C, Vloc_C)):
            coqui_dmft.print_title_box(f"IMPURITY {imp_index}")
            
            solver_params = solver_params_list[imp_index]
            Res, Input = dmft_state.solver_results[imp_index], dmft_state.solver_inputs[imp_index]
            Input['Gloc_t'] = G_t
            Input['Wloc_t'] = coqui_dmft.chemistry_to_product_basis(W_t)
            Input['Vloc'] = coqui_dmft.chemistry_to_product_basis(V)

            coqui_dmft.fit_local_results_boson(
                Input, dmft_state.ir_kernel, solver_params.get("causal_projection")
            )

            # Fermionic and bosonic Weiss fields
            Input['g_weiss_iw'], Input['u_weiss_iw'] = _compute_weiss_fields(
                coqui_mpi, Res, Input, solver_params, dmft_state.ir_kernel
            )
            Input['u_weiss_iw'] = coqui_dmft.fit_u_weiss(
                Input['u_weiss_iw'], dmft_state.ir_kernel, solver_params.get("causal_projection")
            )

            # h0: (nspin, norb, norb), delta_iw: (niw, nspin, norb, norb)
            Input['h0'], Input['delta_iw'] = coqui_dmft.extract_h0_and_delta(
                Input['g_weiss_iw'], dmft_state.ir_kernel
            )

            Ub, Ubp, Jb_spin, Jb_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc'])
            U, Up, J_spin, J_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc']+Input['u_weiss_iw'][0])
            dm = -dmft_state.ir_kernel.tau_interpolate(
                coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Input["gf_struct"]),
                [dmft_state.ir_kernel.beta], 'f')[0]
            Input['density'] = (np.diag(dm[0]).sum() + np.diag(dm[1]).sum()).real
            if coqui_mpi.root():
                print("Hubbard-Kanamori interaction at bare and zero frequency")
                print("--------------------------------------------------------")
                print(f"  intra-orbital                  = {Ub*Hartree_eV:.4f}, {U*Hartree_eV:.4f} eV")
                print(f"  inter-orbital                  = {Ubp*Hartree_eV:.4f}, {Up*Hartree_eV:.4f} eV")
                print(f"  Hund's coupling (spin-flip)    = {Jb_spin*Hartree_eV:.4f}, {J_spin*Hartree_eV:.4f} eV")
                print(f"  Hund's coupling (pair-hopping) = {Jb_pair*Hartree_eV:.4f}, {J_pair*Hartree_eV:.4f} eV\n")

                print("Local densities ")
                print("-------------------")
                print(f"Total: {Input['density']:.4f}")
                print(f"Spin up: {np.diag(dm[0]).real}")
                print(f"Spin down: {np.diag(dm[1]).real}\n")

            dmft_state.save_impurity_inputs(solver_chkpt_h5, imp_index)

            # Convert CoQuí outputs to TRIQS containers
            h0, delta_iw, h_int, u_weiss_iw = coqui_dmft.to_triqs_containers(
                Input['h0'], Input['delta_iw'], Input['Vloc'], Input['u_weiss_iw'],
                dmft_state.ir_kernel, gf_struct = Res['gf_struct'],
                triqs_iw_mesh = {"fermion": Res['iw_mesh_f'], "boson": Res['iw_mesh_b']},
                density_hamiltonian = True, real_hamiltonian = True,
                screen_j_in_u_dd = solver_params.get('screen_j', False)
            )

            # Analyze block symmetry
            if solver_params.get('degenerate_blk'):
                solver_params['degenerate_blk'] = [np.array(x) for x in solver_params["degenerate_blk"]]
            elif solver_params.get('degenerate_blk_thresh'):
                if coqui_mpi.root():
                    print("Analyzing block symmetries via the hybridization function...\n")
                solver_params['degenerate_blk'] = modest.analyze_degenerate_blocks(
                    delta_iw, threshold=solver_params['degenerate_blk_thresh']
                )
            if solver_params.get('degenerate_blk'):
                coqui_dmft.print_degenerate_blks(solver_params['degenerate_blk'], Res['gf_struct'])
                delta_iw   = modest.symmetrize(delta_iw, solver_params['degenerate_blk'])
                h0         = coqui_dmft.symmetrize_h0_op(h0, solver_params['degenerate_blk'], Res['gf_struct'])
                u_weiss_iw = coqui_dmft.symmetrize_blk2_gf(u_weiss_iw, solver_params['degenerate_blk'], Res['gf_struct'])

            # Call impurity solver, and store sigma_imp, vhf_imp, and pi_imp in "Res"
            Res.update(
                _solver_inner_loop(coqui_mpi, h0, delta_iw, u_weiss_iw, h_int, Input['density'], **solver_params)
            )
            # convert from triqs Gf to numpy arrays and ir mesh
            Res.update(coqui_dmft.imp_results_to_raw_data(
                Res['G_iw'], Res['Sigma_iw'], Res['W_iw'], Res['Pi_iw'], dmft_state.ir_kernel)
            )
            # Causal projection
            coqui_dmft.fit_impurity_results_boson(
                Res, dmft_state.ir_kernel, solver_params.get("causal_projection"))

            # TODO Option to use Gimp and Wimp
            # GW double counting contributions
            Res.update(
                coqui_dmft.solve_gw_dc(
                    coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Res['gf_struct']),
                    Input['Vloc'], Input['Wloc_t'], Input['u_weiss_iw'],
                    dmft_state.ir_kernel, density_only=True,
                    gf_struct=Res['gf_struct']
                )
            )

            # mixing impurity and dc solutions to facilitate convergence
            dmft_state.damp_impurity_results(
                solver_chkpt_h5, mixing=iterative_params.get('mixing', 0.7), impurity_indices=[imp_index], 
                mix_in_first_iter=iterative_params.get('mix_in_first_iter', False)
            )

            # save solver results for current impurity
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


def _edmft_loop_fixed_gloc_and_wloc(
        mf, thc, proj_info, dmft_state, solver_chkpt_h5, coqui_chkpt_h5, 
        gloc_params, wloc_params, solver_params_list, embed_params,
        iterative_params, num_iter):
    coqui_mpi = mf.mpi()
    with HDFArchive(coqui_chkpt_h5, 'r') as ar:
        greens_func_source = "embed" if "embed" in ar.keys() else "scf"
        greens_func_iteration = ar[f"{greens_func_source}/final_iter"]

    # downfold for W_loc
    # greens_func_source and greens_func_iteration should be fixed during the inner loop
    Vloc, Wloc_t = coqui.downfold_coulomb(
        thc, wloc_params,
        projector_info=proj_info,
        local_polarizabilities=dmft_state.local_pi_w
    )

    # downfold for G_loc
    gloc_params["greens_func_source"] = greens_func_source
    gloc_params["greens_func_iteration"] = greens_func_iteration
    Gloc_t = coqui.downfold_local_gf(mf, gloc_params, projector_info=proj_info)

    if coqui_mpi.root():
        dmft_state.ir_kernel.check_leakage(Gloc_t, stats='f', name='Gloc in the full MLWF space')
        dmft_state.ir_kernel.check_leakage_phsym(Wloc_t, stats='b', name='Wloc in the full MLWF space')

    # Convert spin axis → list of length nspin
    Gloc_t = [Gloc_t[:, s] for s in range(Gloc_t.shape[1])]
    if mf.nspin() == 1:
        Gloc_t = [Gloc_t[0], Gloc_t[0].copy()]

    # Extract local Green's function and screened interactions for each impurity
    Gloc_C    = dmft_state.embedding['1e'].extract(Gloc_t)   # block matrix
    Vloc_C    = dmft_state.embedding['2e'].extract([Vloc])    # (norb, norb, norb, norb)
    Wloc_C    = dmft_state.embedding['2e'].extract([Wloc_t]) # (nts, norb, norb, norb, norb)
    Vloc_C    = [ V[0] for V in Vloc_C ]      # spinless
    Wloc_C    = [ W_t[0] for W_t in Wloc_C ]  # spinless

    for iteration in range(num_iter):
        for imp_index, (G_t, W_t, V) in enumerate(zip(Gloc_C, Wloc_C, Vloc_C)):
            coqui_dmft.print_title_box(f"IMPURITY {imp_index}")

            solver_params = solver_params_list[imp_index]
            Res, Input = dmft_state.solver_results[imp_index], dmft_state.solver_inputs[imp_index]
            Input['Gloc_t'] = G_t
            Input['Wloc_t'] = coqui_dmft.chemistry_to_product_basis(W_t)
            Input['Vloc'] = coqui_dmft.chemistry_to_product_basis(V)

            coqui_dmft.fit_local_results_boson(
                Input, dmft_state.ir_kernel, solver_params.get("causal_projection"))

            # Fermionic and bosonic Weiss fields
            Input['g_weiss_iw'], Input['u_weiss_iw'] = _compute_weiss_fields(
                coqui_mpi, Res, Input, solver_params, dmft_state.ir_kernel
            )
            Input['u_weiss_iw'] = coqui_dmft.fit_u_weiss(
                Input['u_weiss_iw'], dmft_state.ir_kernel, solver_params.get("causal_projection")
            )

            # h0: (nspin, norb, norb), delta_iw: (niw, nspin, norb, norb)
            Input['h0'], Input['delta_iw'] = coqui_dmft.extract_h0_and_delta(
                Input['g_weiss_iw'], dmft_state.ir_kernel
            )

            Ub, Ubp, Jb_spin, Jb_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc'])
            U, Up, J_spin, J_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc']+Input['u_weiss_iw'][0])
            dm = -dmft_state.ir_kernel.tau_interpolate(
                coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Input["gf_struct"]),
                [dmft_state.ir_kernel.beta], 'f')[0]
            Input['density'] = (np.diag(dm[0]).sum() + np.diag(dm[1]).sum()).real
            if coqui_mpi.root():
                print("Hubbard-Kanamori interaction at bare and zero frequency")
                print("--------------------------------------------------------")
                print(f"  intra-orbital                  = {Ub*Hartree_eV:.4f}, {U*Hartree_eV:.4f} eV")
                print(f"  inter-orbital                  = {Ubp*Hartree_eV:.4f}, {Up*Hartree_eV:.4f} eV")
                print(f"  Hund's coupling (spin-flip)    = {Jb_spin*Hartree_eV:.4f}, {J_spin*Hartree_eV:.4f} eV")
                print(f"  Hund's coupling (pair-hopping) = {Jb_pair*Hartree_eV:.4f}, {J_pair*Hartree_eV:.4f} eV\n")

                print("Local densities ")
                print("-------------------")
                print(f"Total: {Input['density']:.4f}")
                print(f"Spin up: {np.diag(dm[0]).real}")
                print(f"Spin down: {np.diag(dm[1]).real}\n")

            dmft_state.save_impurity_inputs(solver_chkpt_h5, imp_index)

            # Convert CoQuí outputs to TRIQS containers
            h0, delta_iw, h_int, u_weiss_iw = coqui_dmft.to_triqs_containers(
                Input['h0'], Input['delta_iw'], Input['Vloc'], Input['u_weiss_iw'],
                dmft_state.ir_kernel, gf_struct = Res['gf_struct'],
                triqs_iw_mesh = {"fermion": Res['iw_mesh_f'], "boson": Res['iw_mesh_b']},
                density_hamiltonian = True, real_hamiltonian = True,
                screen_j_in_u_dd = solver_params.get('screen_j', False)
            )

            # Analyze block symmetry
            if solver_params.get('degenerate_blk') is None:
                if coqui_mpi.root():
                    print("Analyzing block symmetries via the hybridization function...\n")
                solver_params['degenerate_blk'] = modest.analyze_degenerate_blocks(
                    delta_iw, threshold=solver_params['degenerate_blk_thresh']
                )
            else:
                solver_params['degenerate_blk'] = [np.array(x) for x in solver_params["degenerate_blk"]]
            coqui_dmft.print_degenerate_blks(solver_params['degenerate_blk'], Res['gf_struct'])
            delta_iw   = modest.symmetrize(delta_iw, solver_params['degenerate_blk'])
            h0         = coqui_dmft.symmetrize_h0_op(h0, solver_params['degenerate_blk'], Res['gf_struct'])
            u_weiss_iw = coqui_dmft.symmetrize_blk2_gf(u_weiss_iw, solver_params['degenerate_blk'], Res['gf_struct'])

            # Call impurity solver, and store sigma_imp, vhf_imp, and pi_imp in "Res"
            Res.update(
                _solver_inner_loop(coqui_mpi, h0, delta_iw, u_weiss_iw, h_int, Input['density'], **solver_params)
            )
            # convert from triqs Gf to numpy arrays and ir mesh
            Res.update(coqui_dmft.imp_results_to_raw_data(
                Res['G_iw'], Res['Sigma_iw'], Res['W_iw'], Res['Pi_iw'], dmft_state.ir_kernel)
            )
            # Causal projection
            coqui_dmft.fit_impurity_results_boson(
                Res, dmft_state.ir_kernel, solver_params.get("causal_projection"))

            # GW double counting contributions
            Res.update(
                coqui_dmft.solve_gw_dc(
                    coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Res['gf_struct']),
                    Input['Vloc'], Input['Wloc_t'], Input['u_weiss_iw'],
                    dmft_state.ir_kernel, density_only=True,
                    gf_struct=Res['gf_struct']
                )
            )

            # mixing impurity and dc solutions to facilitate convergence
            dmft_state.damp_impurity_results(
                solver_chkpt_h5, mixing = iterative_params.get('mixing', 0.7), impurity_indices=[imp_index], 
                mix_in_first_iter=iterative_params.get('mix_in_first_iter', False)
            )

            # save solver results for current impurity
            dmft_state.save_impurity_results(solver_chkpt_h5, imp_index)

        # Embed impurity results
        dmft_state.embed_impurity_results()

        dmft_state.iteration += 1
        coqui_mpi.barrier()

    # Upfolding
    coqui.dmft_embed(
        mf, embed_params,
        projector_info = proj_info,
        local_hf_potentials = dmft_state.local_sigma_infty,
        local_sigma_dynamic = dmft_state.local_sigma_w
    )
    coqui_mpi.barrier()


def _compute_weiss_fields(coqui_mpi, imp_results, imp_inputs, solver_params, ir_kernel):

    gloc_t_mat = coqui_dmft.blk_arr_to_arr(imp_inputs['Gloc_t'], imp_inputs['gf_struct'])

    if imp_results['Sigma_iw_data'] is not None:
        if solver_params.get('set_sigma_infty_to_dc', False):
            vhf_imp =  imp_results['Sigma_infty_dc']
        else:
            vhf_imp =  imp_results['Sigma_infty']

        if imp_inputs['screen_type'] == 'rpa':
            if coqui_mpi.root():
                print("screen_type = \"rpa\" -> "
                      "Set impurity polarizability to RPA for bosonic Weiss field.\n")
            # eval Pi_dc using the current Gloc
            pi_imp = ir_kernel.tau_to_w_phsym(
                coqui_dmft.eval_pi_rpa(gloc_t_mat, density_only=True), stats='b'
            )
        else:
            pi_imp = imp_results['Pi_iw_data'][0] if imp_results['Pi_iw_data'] else None

        return (
            coqui_dmft.compute_weiss_fields_w(
                ir_kernel = ir_kernel,
                local_gf = {
                    "Gloc_t": gloc_t_mat,
                    "Wloc_t": imp_inputs['Wloc_t'],
                    "Vloc": imp_inputs['Vloc']
                },
                impurity_selfenergies = {
                    "Vhf_imp": coqui_dmft.blk_arr_to_arr(vhf_imp, imp_results['gf_struct']),
                    "Sigma_imp_w": coqui_dmft.blk_arr_to_arr(imp_results['Sigma_iw_data'], imp_results['gf_struct']),
                    "Pi_imp_w": pi_imp
                },
                density_only=True
            )
        )
    else:
        return (
            coqui_dmft.init_weiss_fields_w(
                ir_kernel = ir_kernel,
                local_gf = {
                    "Gloc_t": gloc_t_mat,
                    "Wloc_t": imp_inputs['Wloc_t'],
                    "Vloc": imp_inputs['Vloc']
                },
                init_imp_results = solver_params.get('init_imp_results', 'dc'),
                density_only=True
            )
        )


def _solver_inner_loop(coqui_mpi, h0, delta_iw, u_weiss_iw, h_int,
                       target_density, **solver_params):

    solver_params.pop('degenerate_blk_thresh', None)
    solver_params.pop('set_sigma_infty_to_dc', None)
    solver_params.pop('init_imp_results', None)
    solver_params.pop("causal_projection", None)
    solver_params.pop("screen_j", None)
    mu_params = solver_params.pop('chemical_potential', None)

    if mu_params is not None:
        dens_solver_params = solver_params.copy()
        mu_tol = mu_params.get('tolerance', 0.1)
        if not mu_params.get('verbosity', False):
            dens_solver_params['verbosity'] = 0
        if mu_params.get('n_warmup_cycles'):
            dens_solver_params['n_warmup_cycles'] = mu_params.get('n_warmup_cycles')
        if mu_params.get('length_cycle'):
            dens_solver_params['length_cycle'] = mu_params.get('length_cycle')
        if mu_params.get('n_cycles'):
            dens_solver_params['n_cycles'] = mu_params.get('n_cycles')

        gf_struct = [(bl, gf.target_shape[0]) for (bl, gf) in delta_iw]
        h0_sab = coqui_dmft.h0_operator_to_array(h0, gf_struct)
        compute_nelec_fcn = partial(
            coqui_dmft.compute_nelec_from_solver,
            gf_struct=gf_struct, h0_sab=h0_sab,
            delta_iw=delta_iw, u_weiss_iw=u_weiss_iw, h_int=h_int,
            **dens_solver_params
        )
        mu_imp, imp_density = coqui_dmft.compute_mu_impurity(
            target_density, compute_nelec_fcn,
            tolerance=mu_tol, mu0=0.0 # always start from mu=0 s.t. mu_imp falls back to 0 at convergence
        )
        # update h0 = h0 - mu_imp
        h0_mat_shifted = np.array([ h0_mat - np.eye(h0_mat.shape[0])*mu_imp for h0_mat in h0_sab ])
        h0 = coqui_dmft.h0_operator(h0_mat_shifted, gf_struct, force_real=True)
    else:
        mu_imp = 0.0

    solver_results = coqui_dmft.ctseg.solve_dynamic_imp(delta_iw, h0, u_weiss_iw, h_int, **solver_params)
    solver_results['mu_imp'] = mu_imp
    # impurity total density
    imp_density = 0.0
    for blk_name, occ in solver_results['orbital_occupations'].items():
        imp_density += occ.sum()

    if coqui_mpi.root():
        print(f"Total impurity densities = {imp_density}")
        print(f"Convergence of impurity density: {imp_density - target_density}\n")

    return solver_results


def solve_impurities_from_chkpt(coqui_mpi, dmft_iteration=-1, imp_indices=None, **params):

    imp_params = params.pop('impurity')
    solver_params_list = imp_params['solver']
    for solver_params in solver_params_list:
        solver_params['n_cycles'] = int(solver_params['n_cycles']/coqui_mpi.comm_size())
        mu_params = solver_params.get('chemical_potential')
        if mu_params and 'n_cycles' in mu_params.keys():
            mu_params['n_cycles'] = int(mu_params['n_cycles']/coqui_mpi.comm_size())

    ir_kernel = IAFT.from_coqui_chkpt(imp_params['chkpt_h5'], verbose=coqui_mpi.root())
    wmax_imp, prec_imp = imp_params.pop('dlr_wmax', ir_kernel.wmax), imp_params.pop('dlr_eps', ir_kernel.prec)

    solver_inputs = coqui_dmft.read_impurity_chkpt(
        imp_params['chkpt_h5'], dmft_iteration, read="inputs", impurity_indices=imp_indices
    )
    solver_results = []
    for imp_index, inputs in enumerate(solver_inputs):
        coqui_dmft.print_title_box(f"IMPURITY {imp_index}")
        solver_params = solver_params_list[imp_index]
        Input = solver_inputs[imp_index]

        Input['u_weiss_iw'] = coqui_dmft.fit_u_weiss(
            Input['u_weiss_iw'], ir_kernel, solver_params.get("causal_projection")
        )

        Ub, Ubp, Jb_spin, Jb_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc'])
        U, Up, J_spin, J_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc']+Input['u_weiss_iw'][0])
        dm = -ir_kernel.tau_interpolate(
            coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Input["gf_struct"]),
            [ir_kernel.beta], 'f')[0]
        Input['density'] = (np.diag(dm[0]).sum() + np.diag(dm[1]).sum()).real
        if coqui_mpi.root():
            print("Hubbard-Kanamori interaction at bare and zero frequency")
            print("--------------------------------------------------------")
            print(f"  intra-orbital                  = {Ub*Hartree_eV:.4f}, {U*Hartree_eV:.4f} eV")
            print(f"  inter-orbital                  = {Ubp*Hartree_eV:.4f}, {Up*Hartree_eV:.4f} eV")
            print(f"  Hund's coupling (spin-flip)    = {Jb_spin*Hartree_eV:.4f}, {J_spin*Hartree_eV:.4f} eV")
            print(f"  Hund's coupling (pair-hopping) = {Jb_pair*Hartree_eV:.4f}, {J_pair*Hartree_eV:.4f} eV\n")

            print("Local densities ")
            print("-------------------")
            print(f"Total: {Input['density']:.4f}")
            print(f"Spin up: {np.diag(dm[0]).real}")
            print(f"Spin down: {np.diag(dm[1]).real}\n")

        if coqui_mpi.root():
            ir_kernel.check_leakage(Input['delta_iw'], 'f', 'delta', w_input=True)
            ir_kernel.check_leakage_phsym(Input['u_weiss_iw'], 'b', 'u_weiss', w_input=True)

        # Convert CoQuí outputs to TRIQS containers
        h0, delta_iw, h_int, u_weiss_iw = coqui_dmft.to_triqs_containers(
            Input['h0'], Input['delta_iw'], Input['Vloc'], Input['u_weiss_iw'],
            ir_kernel, gf_struct = Input['gf_struct'],
            triqs_iw_mesh = {"dlr_wmax": wmax_imp, "dlr_eps": prec_imp},
            density_hamiltonian = True, real_hamiltonian = True,
            screen_j_in_u_dd = solver_params.get('screen_j', False)
        )

        # Analyze block symmetry
        if solver_params.get('degenerate_blk'):
            solver_params['degenerate_blk'] = [np.array(x) for x in solver_params["degenerate_blk"]]
        elif solver_params.get('degenerate_blk_thresh'):
            if coqui_mpi.root():
                print("Analyzing block symmetries via the hybridization function...\n")
            solver_params['degenerate_blk'] = modest.analyze_degenerate_blocks(
                delta_iw, threshold=solver_params['degenerate_blk_thresh']
            )
        if solver_params.get('degenerate_blk'):
            coqui_dmft.print_degenerate_blks(solver_params['degenerate_blk'], Input['gf_struct'])
            delta_iw   = modest.symmetrize(delta_iw, solver_params['degenerate_blk'])
            h0         = coqui_dmft.symmetrize_h0_op(h0, solver_params['degenerate_blk'], Input['gf_struct'])
            u_weiss_iw = coqui_dmft.symmetrize_blk2_gf(u_weiss_iw, solver_params['degenerate_blk'], Input['gf_struct'])

        # Call impurity solver, and store sigma_imp, vhf_imp, and pi_imp in "Res"
        Res = _solver_inner_loop(coqui_mpi, h0, delta_iw, u_weiss_iw, h_int, Input['density'], **solver_params)

        # convert from triqs Gf to numpy arrays and ir mesh
        Res.update(coqui_dmft.imp_results_to_raw_data(
            Res['G_iw'], Res['Sigma_iw'], Res['W_iw'], Res['Pi_iw'], ir_kernel)
        )

        # Causal projection
        coqui_dmft.fit_impurity_results_boson(
            Res, ir_kernel, solver_params.get("causal_projection"))

        solver_results.append(Res)

    return solver_results
