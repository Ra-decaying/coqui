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
Utility functions for TRIQS interface to EDMFT.

This module provides helper routines that facilitate the conversion and
manipulation of data between CoQuí and the TRIQS ecosystem in the context
of EDMFT, including:

- Constructing TRIQS Green’s functions (Gf, BlockGf, Block2Gf) from raw data
  such as Weiss fields, hybridization functions, and interaction tensors.
- Handling mappings between orbital subspaces, solver structures, and block
  Green’s functions.
- Extracting and reducing interaction tensors (e.g. to Hubbard–Kanamori form).
- Enforcing density–density or real-Hamiltonian approximations as required by
  TRIQS solvers (e.g. CT-SEG).
- Miscellaneous utilities for mesh handling, and TRIQS container construction.

Dependencies
------------
This module relies on the TRIQS ecosystem, in particular:
- `triqs.gf` for Green’s function containers and operations
- `triqs.operators` for operator algebra
- `triqs.utility.mpi` for parallelism
"""
import triqs.utility.mpi as mpi
from h5 import HDFArchive
import numpy as np
import itertools

from triqs.gf import Gf, make_gf_dlr, BlockGf, Block2Gf, MeshImFreq, MeshDLRImFreq
from triqs.operators import c_dag, c, Operator, util


def make_h5_sumk_format(mlwf_h5, orb_list=None):
    l, SO = 2, 0
    n_inequiv_shells = 1
    with HDFArchive(mlwf_h5, 'a') as ar:
        dft_inp = ar["dft_input"]

        # modify proj_mat and band_window
        if orb_list is not None:
            proj_mat = dft_inp["proj_mat"]
            dft_inp["proj_mat"] = proj_mat[:, :, :, orb_list]

            shell = dft_inp["corr_shells"]
            shell[0]["dim"] = len(orb_list)
            dft_inp["corr_shells"] = shell

            shell = dft_inp["shells"]
            shell[0]["dim"] = len(orb_list)
            dft_inp["shells"] = shell

        # dummy values that are unused but required by SumkDFT
        kpts_w90 = dft_inp['kpts']
        dft_inp["n_k"] = kpts_w90.shape[0]
        dft_inp["k_dep_projection"] = 0
        dft_inp["n_shells"] = n_inequiv_shells
        dft_inp["n_corr_shells"] = n_inequiv_shells
        dft_inp["n_inequiv_shells"] = n_inequiv_shells
        dft_inp["corr_to_inequiv"] = [0]
        dft_inp["inequiv_to_corr"] = [0]

        dft_inp["n_reps"] = [1 for i in range(n_inequiv_shells)]
        dft_inp["dim_reps"] = [0 for i in range(n_inequiv_shells)]

        ll = 2 * l + 1
        lmax = ll * (SO + 1)
        dft_inp["T"] = [np.zeros([lmax, lmax], dtype=complex) for i in range(n_inequiv_shells)]


def combine_impurities(C_ksIai, imp_dims):
    nkpts, nspins, nImps, nImpOrbs, Norbs = C_ksIai.shape
    C_ksIai_new = np.zeros((nkpts, nspins, 1, np.sum(imp_dims), Norbs), dtype=C_ksIai.dtype)
    offset = 0
    for I, dim in enumerate(imp_dims):
        C_ksIai_new[:, :, 0, offset:offset+dim] = C_ksIai[:, :, I, :dim]
        offset += dim

    return C_ksIai_new


def read_proj_info(wannier_h5):
  with HDFArchive(wannier_h5, 'r') as ar:
    C_ksIai = ar['dft_input/proj_mat']
    band_window = ar['dft_misc_input/band_window']
    kpts_w90 = ar['dft_input/kpts']

    nImps = band_window.shape[0]
    imp_dims = []
    for I in range(nImps):
        imp_dims.append(ar[f'dft_input/shells/{I}/dim'])

    if nImps > 1:
        # combine multiple impurities into one
        C_ksIai = combine_impurities(C_ksIai, imp_dims)
        band_window = band_window[:1]

  return {'proj_mat': C_ksIai, 'band_window': band_window, 'kpts_w90': kpts_w90}


def set_n_iw(ir_kernel):
    iw_idx_f = ir_kernel.wn_mesh('f', False)
    iw_idx_b = ir_kernel.wn_mesh('b', False)
    max_idx = max(abs(iw_idx_f[0]), abs(iw_idx_f[-1]), abs(iw_idx_b[0]), abs(iw_idx_b[-1]))
    return int(max_idx + 1)


def get_c_to_solver_mapping(solver_struct):
    # Ex: 
    # C space = [ ('up', nbnd), ('down', nbnd) ]
    # solver  = [ ('up_0', a), ('up_1', b), ('down_0', a), ('down_1', b) ]
    #             where a + b = nbnd
    c_to_solver = {}
    o1_up, o1_dn = 0, 0
    for blk_name, blk_dim in solver_struct:
        if blk_name[:2] == "up":
            for i in range(blk_dim):
                c_to_solver[("up", i+o1_up)] = (blk_name, i)
            o1_up += blk_dim
        else:
            for i in range(blk_dim):
                c_to_solver[("down", i+o1_dn)] = (blk_name, i)
            o1_dn += blk_dim

    return c_to_solver


def Vijkl_in_triqs_notation(V_ijkl):
    # switch inner two indices to match triqs notation
    V = np.zeros(V_ijkl.shape, dtype=V_ijkl.dtype)
    nbnd = V.shape[0]
    for or1, or2, or3, or4 in itertools.product(range(nbnd), repeat=4):
        V[or1, or2, or3, or4] = V_ijkl[or1, or3, or2, or4]
    return V


def h_int_density_density(V_abcd, gf_struct, force_real=True):    
    c_to_solver = get_c_to_solver_mapping(gf_struct)
    V, Vp = util.reduce_4index_to_2index(
        Vijkl_in_triqs_notation(V_abcd.real if force_real else V_abcd)
    )
    h_int = util.h_int_density(['up', 'down'], V.shape[0], U=V, Uprime=Vp, 
                               map_operator_structure=c_to_solver)
    return h_int


def h_int_slater(V_abcd, gf_struct, force_real=True):
    c_to_solver = get_c_to_solver_mapping(gf_struct)
    return util.h_int_slater(['up', 'down'], V_abcd.shape[-1], 
                              Vijkl_in_triqs_notation(V_abcd.real if force_real else V_abcd), 
                              map_operator_structure=c_to_solver)
    

def h0_operator(h0_sab, gf_struct, *, diagonal=True, force_real=True):
    assert len(h0_sab.shape) == 3, "incorrect h0_sab.shape"
    H0 = Operator()
    o1 = [0, 0]
    for blk_name, blk_dim in gf_struct:
        s = 0 if blk_name[:2] == "up" else 1
        for i in range(blk_dim):
            if force_real:
                H0 += h0_sab[s, o1[s]+i, o1[s]+i].real * c_dag(blk_name, i) * c(blk_name, i)
            else:
                H0 += h0_sab[s, o1[s]+i, o1[s]+i] * c_dag(blk_name, i) * c(blk_name, i)
            o1[s] += blk_dim

    return H0


def product_basis_to_density_density(A_abcd):
    if A_abcd.ndim < 4:
        raise ValueError("A must have at least 4 dimensions.")
    n1, n2, n3, n4 = A_abcd.shape[-4:]
    if not (n1 == n2 == n3 == n4):
        raise ValueError("The last four axes must have equal length (nbnd).")

    lead_shape = A_abcd.shape[:-4]
    out_shape = lead_shape + (n1, n1)
    A_ab = np.zeros(out_shape, dtype=A_abcd.dtype)

    for a, b in itertools.product(range(n1), repeat=2):
        if lead_shape:  # with leading dims, e.g. (nw, norb, norb, norb, norb)
            A_ab[..., a, b] = A_abcd[..., a, a, b, b]
        else:           # no leading dims, just (norb, norb, norb, norb)
            A_ab[a, b] = A_abcd[a, a, b, b]

    return A_ab


def density_density_to_product_basis(A_ab):
    if A_ab.ndim < 2:
        raise ValueError("A_ab must have at least 2 dimensions (..., norb, norb).")

    n1, n2 = A_ab.shape[-2:]
    if n1 != n2:
        raise ValueError("Last two dimensions of A_ab must be equal (norb, norb).")

    lead_shape = A_ab.shape[:-2]
    out_shape = lead_shape + (n1, n1, n1, n1)
    A_abcd = np.zeros(out_shape, dtype=A_ab.dtype)

    # Iterate over orbital indices
    for a, b in itertools.product(range(n1), repeat=2):
        if lead_shape:  # with leading dims, e.g. (nw, norb, norb)
            A_abcd[..., a, a, b, b] = A_ab[..., a, b]
        else:           # no leading dims, just (norb, norb)
            A_abcd[a, a, b, b] = A_ab[a, b]

    return A_abcd


def to_block_gf(giw_ir, ir_kernel, gf_struct, mesh_iw):

    assert mesh_iw.statistic == 'Fermion', "Only mesh_iw.statistic == Fermion is supported."
    assert len(giw_ir.shape) == 4, "giw_ir needs to have dimensions (nw, nspins, nbnd, nbnd)"
    assert giw_ir.shape[2] == giw_ir.shape[3], "giw_ir needs to have dimensions (nw, nspins, nbnd, nbnd)"
    assert giw_ir.shape[0] == ir_kernel.nw_f, "giw_ir.shape[0] != ir_kernel.nw_f"
    
    # giw_ir = (nw, nspin, nbnd, nbnd)
    nbnd = giw_ir.shape[-1]

    blk_gf = BlockGf(mesh=mesh_iw, gf_struct=gf_struct)
    mesh_iw_idx = np.array([iwn.index for iwn in mesh_iw])

    # E.g. block structure = [ ("up_0", 1), ("down_0", 1), ("up_1", 2), ("down_1", 2) ]
    offset = [0, 0]
    for blk_name, blk_dim in gf_struct:
        s = 0 if blk_name[:2] == "up" else 1
        blk_gf[blk_name].data[:] = ir_kernel.w_interpolate(
            giw_ir[:, s, offset[s]:offset[s]+blk_dim, offset[s]:offset[s]+blk_dim],
            mesh_iw_idx,
            stats="f",
            ir_notation=False
        )
        offset[s] += blk_dim
        assert offset[s] <= nbnd, f"Spin {s} block exceeds band range"

    return blk_gf


def to_block2_gf(Diw_ir, ir_kernel, gf_struct, mesh_iw):

    assert mesh_iw.statistic == 'Boson', "Only mesh_iw.statistic == Boson is supported."
    assert len(Diw_ir.shape) == 3, "Diw_ir needs to have dimensions (nw, nbnd, nbnd)"
    assert Diw_ir.shape[1] == Diw_ir.shape[2], "Diw_ir needs to have dimensions (nw, nbnd, nbnd)"
    nw_half = ir_kernel.nw_b//2 if ir_kernel.nw_b%2==0 else ir_kernel.nw_b//2 + 1
    assert Diw_ir.shape[0] == nw_half, "Diw_ir.shape[0] != nw_b_half"
    
    # Diw_ir = (nw, nbnd, nbnd)
    nbnd = Diw_ir.shape[-1]

    mesh_iw_idx = np.array([iwn.index for iwn in mesh_iw])
    Diw_data = ir_kernel.w_interpolate_phsym(Diw_ir, mesh_iw_idx, stats="b", ir_notation=False)

    gf_array = []
    o1 = [0, 0]
    for name1, dim1 in gf_struct:
        s1 = 0 if name1[:2] == "up" else 1

        gf_list = []
        o2 = [0, 0]
        for name2, dim2 in gf_struct:
            s2 = 0 if name2[:2] == "up" else 1

            gf = Gf(mesh = mesh_iw, target_shape = (dim1, dim2))
            gf.data[:] = Diw_data[:, o1[s1]:o1[s1]+dim1, o2[s2]:o2[s2]+dim2]

            o2[s2] += dim2
            assert o2[s2] <= nbnd, f"Spin {s2} block exceeds band range"

            gf_list.append(gf)

        o1[s1] += dim1
        assert o1[s1] <= nbnd, f"Spin {s1} block exceeds band range"

        gf_array.append(gf_list)

    names = [name for name, _ in gf_struct]
    return Block2Gf(names, names, gf_array)


def to_triqs_containers(h0, delta_iw, Vimp, u_weiss_iw, ir_kernel, 
                        gf_struct, triqs_iw_mesh, 
                        density_hamiltonian, real_hamiltonian=True):
    """
    Convert raw CoQui outputs (NumPy arrays) into TRIQS containers 
    (e.g. `triqs.operators.many_body_operator`, `BlockGf`, and `Block2Gf`).

    This function provides a bridge between CoQui’s raw many-body data and 
    TRIQS’ Green’s function representation. It constructs the one-particle and 
    two-particle objects required for TRIQS-based impurity solvers or DMFT-like 
    workflows.

    Parameters
    ----------
    h0 : np.ndarray
        One-particle Hamiltonian matrix in the impurity subspace.
    delta_iw : np.ndarray
        Hybridization function Δ(iωₙ) on the Matsubara frequency axis, in raw array form.
    Vimp : np.ndarray
        Bare Coulomb interaction tensor in the impurity basis.
    u_weiss_iw : np.ndarray
        Dynamical screened interaction U(iωₙ) from cRPA or EDMFT preprocessing, 
        given as raw arrays on the Matsubara axis.
    ir_kernel : object
        Imaginary-time/frequency transform kernel (e.g. IR/IAFT object) used for 
        Fourier transforms between τ and iωₙ.
    gf_struct : dict
        Block structure of Green’s functions, mapping orbital/spin indices to block labels.
    triqs_iw_mesh : dict
        Dictionary of TRIQS mesh objects:
        - `"fermion"` : Matsubara mesh for fermionic objects (Δ, G).
        - `"boson"`   : Matsubara mesh for bosonic objects (U).
    density_hamiltonian : bool
        Whether to use the density-density approximation for the Coulomb interaction. 
        This will also enforce the non-interacting Hamiltonian to density operator only. 
        Currently must be `True`. Non-density-density interactions are not implemented.
    real_hamiltonian : bool, optional
        If `True`, enforce the bare Hamiltonian (h0, Vimp) to be real-valued. 
        Default is `True`.

    Returns
    -------
    h0 : triqs.operators.many_body_operator
        One-particle Hamiltonian in TRIQS operator form.
    delta_iw : triqs.gf.BlockGf
        Hybridization function Δ(iωₙ) as a TRIQS block Green’s function.
    h_int : triqs.operators.many_body_operator
        Local interaction Hamiltonian in density-density approximation.
    u_weiss_iw : triqs.gf.Block2Gf
        Dynamical screened interaction U(iωₙ) as a TRIQS two-particle block Green’s function.

    Notes
    -----
    - This conversion assumes the density-density approximation for Coulomb 
      interactions; a full four-index mapping is not yet implemented.
    """

    assert density_hamiltonian==True, (
        "Convertion to non-density-density Hamiltonian is not implemented yet."
    )

    if real_hamiltonian:
        # FT to tau space and enforce to real values
        delta_tau = ir_kernel.w_to_tau(delta_iw, 'f')
        delta_tau.imag = 0.0
        delta_iw = ir_kernel.tau_to_w(delta_tau, 'f')

        u_weiss_tau = ir_kernel.w_to_tau_phsym(u_weiss_iw, 'b')
        u_weiss_tau.imag = 0.0
        u_weiss_iw = ir_kernel.tau_to_w_phsym(u_weiss_tau, 'b')

    # one-particle
    h0 = h0_operator(h0, gf_struct, diagonal=density_hamiltonian, 
                     force_real=real_hamiltonian)
    delta_iw = to_block_gf(delta_iw, ir_kernel, 
                           gf_struct, triqs_iw_mesh["fermion"])
    
    # two-particle
    u_weiss_iw = to_block2_gf(
        product_basis_to_density_density(u_weiss_iw),
        ir_kernel, gf_struct, triqs_iw_mesh["boson"]
    )
    h_int = h_int_density_density(
        Vimp, gf_struct, 
        force_real=real_hamiltonian
    )

    return h0, delta_iw, h_int, u_weiss_iw


def Gf_dlr_from_ir(Giw_ir, ir_kernel, mesh_dlr_iw, dim=2):
    
    nbnd = Giw_ir.shape[-1]
    stats = 'f' if mesh_dlr_iw.statistic == 'Fermion' else 'b'

    if stats == 'b':
        Giw_ir_pos = Giw_ir.copy()
        Giw_ir = np.zeros([Giw_ir_pos.shape[0] * 2 - 1] + [nbnd] * dim, dtype=complex)
        Giw_ir[: Giw_ir_pos.shape[0]] = Giw_ir_pos[::-1]
        Giw_ir[Giw_ir_pos.shape[0] :] = Giw_ir_pos[1:]
    
    Gf_dlr_iw = Gf(mesh=mesh_dlr_iw, target_shape=[nbnd] * dim)

    # prepare idx array for spare ir
    if stats == 'f':
        mesh_dlr_iw_idx = np.array([iwn.index for iwn in mesh_dlr_iw])
    else:
        mesh_dlr_iw_idx = np.array([iwn.index for iwn in mesh_dlr_iw])

    Gf_dlr_iw.data[:] = ir_kernel.w_interpolate(Giw_ir, mesh_dlr_iw_idx, stats=stats, ir_notation=False)

    return make_gf_dlr(Gf_dlr_iw)


def estimate_zero_moment(Aw, iw_mesh):
    """
    Estimate the zeroth moment (high-frequency constant term) of A(iω). 

    Assumes that at large Matsubara frequencies the function behaves as

        A(iω) ≈ t + B / (iω) + O(1/ω²),

    where `t` is the zeroth moment (constant term). The estimate uses the values
    of A(iω) at the last two frequencies in `iw_mesh` to extrapolate to the
    ω → ∞ limit.

    Parameters
    ----------
    Aw : array of complex, shape (nw, ...)
        Values of A(iω) on the Matsubara frequency mesh. Must be ordered such
        that the last entries correspond to the largest |ω|.
    iw_mesh : array of complex, shape (nw)
        Matsubara frequency mesh corresponding to `Aw`. The last two entries are
        used for the extrapolation.

    Returns
    -------
    t : complex
        Estimated zeroth moment (constant term) of A(iω) in the high-frequency
        expansion.

    Notes
    -----
    - Accuracy depends on how far into the asymptotic regime the last two
      frequencies are. If the frequencies are not sufficiently large, the
      estimate may be biased.
    """
    iw_m1 = iw_mesh[-1]
    iw_m2 = iw_mesh[-2]
    t = Aw[-1].real - (Aw[-1] - Aw[-2]).real * iw_m2 ** 2 / (
           iw_m2 ** 2 - iw_m1 ** 2)
    t = t.astype(complex)

    return t


def extract_h0_and_delta(g_weiss_wsab, ir_kernel, high_freq_multiplier=10):
    """
    Estimate the static one-body term h₀ (as t_sIab) and the hybridization function Δ(iω)
    from a Weiss Green's function G₀(iω) sampled on a fermionic Matsubara mesh.

    The method:
      1) Interpolate G₀(iω) to a few very large Matsubara frequencies (scaled by
         `high_freq_multiplier`) to probe the asymptotic regime.
      2) Construct W(iω) = iω·I - [G₀(iω)]⁻¹ and estimate its zeroth moment
         t_sIab = lim_{|ω|→∞} W(iω) via `estimate_zero_moment`.
      3) Build Δ(iω) from the Dyson-like relation:
            Δ(iω) = iω·I - t_sIab - [G₀(iω)]⁻¹.

    Parameters
    ----------
    g_weiss_wsab : ndarray, complex, shape (nw, nspin, nbnd, nbnd)
        Weiss Green's function G₀(iωₙ) on the fermionic Matsubara mesh returned by `ir_kernel`.
        The leading dimension is frequency index; s,a,b are spin and orbital indices.
    ir_kernel : IAFT object
    high_freq_multiplier : float, default 10
        Multiplier applied to the last few (three) IR fermionic frequencies (in IR notation)
        before converting to physical Matsubara frequencies, to push evaluation deep into
        the asymptotic region for a more stable moment estimate.

    Returns
    -------
    t_sIab_estimate : ndarray, complex, shape (nspin, nbnd, nbnd)
        Estimate of the static one-body term (zeroth moment) per spin block.
    delta_estimate : ndarray, complex, shape (nw, nspin, nbnd, nbnd)
        Estimated hybridization function Δ(iωₙ) on the original fermionic mesh.

    Notes
    -----
    - Accuracy of `t_sIab_estimate` depends on how large the interpolated frequencies are.
    """
    nspin = g_weiss_wsab.shape[1]
    
    # 1) Interpolate G0 to very high fermionic frequencies to improve the accuracy of high-frequency fitting
    iwn_interp = ir_kernel.wn_mesh('f', ir_notation=False)[-3:] * high_freq_multiplier
    g_weiss_interp = ir_kernel.w_interpolate(g_weiss_wsab, iwn_interp, 'f', ir_notation=False)
    iwn_interp = (2*iwn_interp.astype(float) + 1) * np.pi / ir_kernel.beta
    weiss_tmp = np.zeros(g_weiss_interp.shape, dtype=complex)
    for n, g in enumerate(g_weiss_interp):
        for s in range(nspin):
            weiss_tmp[n, s] = 1j * iwn_interp[n] - np.linalg.inv(g[s])

    # 2) Fitting the zeroth moment as the non-interacting Hamiltonian
    t_sIab_estimate = estimate_zero_moment(weiss_tmp, iwn_interp)

    # 3) Construct Δ(iω) = iω·I - t_sIab - [G0(iω)]^{-1} on the original mesh
    iwn_mesh_imp = ir_kernel.wn_mesh('f') * np.pi / ir_kernel.beta
    delta_estimate = np.zeros(g_weiss_wsab.shape, dtype=complex)
    nbnd = t_sIab_estimate.shape[-1]
    for n in range(delta_estimate.shape[0]):
        for s in range(nspin):
            g_weiss_inv = np.linalg.inv(g_weiss_wsab[n, s])
            delta_estimate[n, s] = 1j * iwn_mesh_imp[n] * np.eye(nbnd) - t_sIab_estimate[s] - g_weiss_inv

    # 4) checking the leakage of the resulting Δ(iω)
    if mpi.is_master_node():
        ir_kernel.check_leakage(delta_estimate, 'f', 'delta', w_input=True)
    mpi.report("")
    mpi.barrier()

    return t_sIab_estimate, delta_estimate


def compute_weiss_fields_w(*, ir_kernel, local_gf, impurity_selfenergies=None, density_only=False):
    # checking inputs 
    missing = {"Gloc_t", "Wloc_t", "Vloc"} - local_gf.keys()
    if missing:
        raise ValueError(f"Missing keys in local_gf: {missing}")

    if impurity_selfenergies is not None:
        missing = {"Vhf_imp", "Sigma_imp_w", "Pi_imp_w"} - impurity_selfenergies.keys()
        if missing:
            raise ValueError(f"Missing keys in impurity_selfenergies: {missing}")
    else:
        impurity_selfenergies = {"Vhf_imp": None, "Sigma_imp_w": None, "Pi_imp_w": None}

    # bosonic first 
    if impurity_selfenergies["Pi_imp_w"] is not None:
        mpi.report("Evaluate the bosonic Weiss field using the provided impurity polarizability.")
        Pi_imp_w = impurity_selfenergies["Pi_imp_w"]
    else:
        mpi.report("Evaluate the bosonic Weiss field at the RPA level.")
        Pi_imp_w = ir_kernel.tau_to_w_phsym(eval_pi_rpa(local_gf["Gloc_t"], density_only=density_only), stats='b')

    # check if Pi_imp_w contains only density-density
    if len(Pi_imp_w.shape) == 3:
        Pi_imp_w_pb = density_density_to_product_basis(Pi_imp_w)
    else:
        Pi_imp_w_pb = Pi_imp_w
        
    u_weiss_w = compute_weiss_boson_w(
        local_gf["Vloc"],
        ir_kernel.tau_to_w_phsym(local_gf["Wloc_t"], stats='b'),
        Pi_imp_w_pb
    )
    if mpi.is_master_node():
        ir_kernel.check_leakage_phsym(u_weiss_w, 'b', 'u_weiss', w_input=True)
    mpi.report("")
    mpi.barrier()

    # fermionic 
    if impurity_selfenergies["Vhf_imp"] is not None and impurity_selfenergies["Sigma_imp_w"] is not None:
        mpi.report("Evaluate the fermionic Weiss field using the provided impurity self-energy.")
        Vhf_imp = impurity_selfenergies["Vhf_imp"]
        Sigma_imp_w = impurity_selfenergies["Sigma_imp_w"]
    else:
        mpi.report("Evaluate the fermionic Weiss field using the local GW self-energy.")
        Vhf_imp = eval_hf_dc(
            -ir_kernel.tau_interpolate(local_gf["Gloc_t"], [ir_kernel.beta], stats='f')[0], 
            local_gf["Vloc"], 
            u_weiss_w[0]+local_gf["Vloc"]
        )
        Sigma_imp_w = ir_kernel.tau_to_w(eval_gw_dc_t(local_gf["Gloc_t"], local_gf["Wloc_t"]), stats='f')
                   
    g_weiss_w = compute_weiss_fermion_w(
        ir_kernel.tau_to_w(local_gf["Gloc_t"], stats='f'),
        Vhf_imp, 
        Sigma_imp_w
    )

    if mpi.is_master_node():
        ir_kernel.check_leakage(g_weiss_w, 'f', 'g_weiss', w_input=True)
    mpi.report("")
    mpi.barrier()

    return g_weiss_w, u_weiss_w


def compute_weiss_fermion_w(G_wsab, vhf_sab, sigma_wsab):
    nw, nspin, nbnd = G_wsab.shape[:3]
    g_wsab = np.zeros(G_wsab.shape, dtype=G_wsab.dtype)

    #  g(w) = [G(w)^{-1} + Sigma_imp]^{-1]
    for ws in range(nw*nspin):
        w = ws // nspin
        s = ws % nspin
        X_inv = np.linalg.solve(G_wsab[w, s], np.eye(nbnd)) + vhf_sab[s] + sigma_wsab[w, s]
        g_wsab[w, s] = np.linalg.solve(X_inv, np.eye(nbnd))

    return g_wsab


def compute_weiss_boson_w(V_abcd, W_wabcd, Pi_wabcd):
    nbnd = V_abcd.shape[0]
    nbnd2 = nbnd*nbnd
    Wfull_pb = (W_wabcd + V_abcd).reshape(-1,nbnd2,nbnd2)
    Pi_pb = Pi_wabcd.reshape(-1,nbnd2,nbnd2)

    # U(w) = W(w)[I + Pi(w)*W(w)]^{-1}
    U_pb = np.zeros(Wfull_pb.shape, dtype=Wfull_pb.dtype)
    for n, W in enumerate(Wfull_pb):
        X = np.eye(nbnd2) + Pi_pb[n] @ W
        X_inv = np.linalg.solve(X, np.eye(nbnd2))
        U_pb[n] = W @ X_inv

    return U_pb.reshape(W_wabcd.shape) - V_abcd


def eval_pi_rpa(G_tsab, *, density_only=False):
    nts, nspin, nbnd = G_tsab.shape[:3]
    nts_half = nts//2 if nts%2==0 else nts//2 + 1
    spin_factor = -2.0 if nspin == 1 else -1.0
    if not density_only:
        pi_t = np.zeros((nts_half, nbnd, nbnd, nbnd, nbnd), dtype=complex)
        for t in range(nts_half):
            mt = nts-t-1
            pi_t[t] += spin_factor * np.einsum(
                'sbd,sca->abcd',
                G_tsab[t], G_tsab[mt]
            )
    else:
        pi_t = np.zeros((nts_half, nbnd, nbnd), dtype=complex)
        for t in range(nts_half):
            mt = nts-t-1
            for s in range(nspin):
                for a, b in itertools.product(range(nbnd), repeat=2):
                    pi_t[t, a, b] += spin_factor * G_tsab[t, s, a, b] * G_tsab[mt, s, b, a]
    return pi_t


def eval_hf_dc(Dm_sab, V_abcd, U0_abcd):
    """
    Evaluate the Hartree and exchange contributions to the density matrix.

    Parameters:
    - Dm_sab: Density matrix in spin and band indices.
    - V_abcd: Bare interaction on a product basis.
    - U0_abcd: Static screened interaction on a product basis.

    Returns:
    - Hartree-Fock potential for an impurity with dynamic interactions.
    """

    Vhf_sab = np.zeros(Dm_sab.shape, dtype=Dm_sab.dtype)

    # Hartree contribution
    spin_factor = 2.0 if Dm_sab.shape[0] == 1 else 1.0
    for s in range(Dm_sab.shape[0]):
        Vhf_sab += spin_factor * np.einsum('dc,bacd->ab', Dm_sab[s], U0_abcd)

    # Exchange contribution
    Vhf_sab -= np.einsum('sab,aibj->sij', Dm_sab, V_abcd)

    return Vhf_sab


def eval_gw_dc_t(G_tsab, W_tabcd):
    """
    Evaluate the GW self-energy contribution to the impurity Green's function.

    Parameters:
    - G_tsab: Impurity Green's function in time, spin, and band indices.
    - W_tabcd: Dynamic interaction on a product basis.

    Returns:
    - GW self-energy contribution to the impurity Green's function.
    """

    nts, nts_half = G_tsab.shape[0], W_tabcd.shape[0]
    sigma_tsab = np.zeros(G_tsab.shape, dtype=G_tsab.dtype)

    for t in range(nts_half):
        mt = nts-t-1
        sigma_tsab[t] = -1 * np.einsum('sab,aibj->sij', G_tsab[t], W_tabcd[t])
        if mt != t:
            sigma_tsab[mt] = -1 * np.einsum('sab,aibj->sij', G_tsab[mt], W_tabcd[t])

    return sigma_tsab


def solve_gw_dc(G_t, V, W_t, u_weiss_iw, ir_kernel, density_only=True,
                *, gf_struct=None):
    dm = -ir_kernel.tau_interpolate(G_t, [ir_kernel.beta], stats='f')[0]
    vhf_dc = eval_hf_dc(dm, V, u_weiss_iw[0]+V)
    
    sigma_dc_iw = ir_kernel.tau_to_w(eval_gw_dc_t(G_t, W_t), stats='f')

    # (niw, norb, norb) if density_only else (niw, norb, norb, norb, norb)
    pi_dc_iw = ir_kernel.tau_to_w_phsym(eval_pi_rpa(G_t, density_only=density_only), stats='b')

    if gf_struct is not None:
        # convert to block matrix format
        vhf_dc = arr_to_blk_arr(vhf_dc, gf_struct)
        sigma_dc_iw = arr_to_blk_arr(sigma_dc_iw, gf_struct)
        pi_dc_iw = [pi_dc_iw]

    return {
        "Sigma_infty_dc": vhf_dc,
        "Sigma_iw_dc_data": sigma_dc_iw,
        "Pi_iw_dc_data": pi_dc_iw
    }


def imp_results_to_raw_data(sigma_iw, pi_iw, ir_kernel=None):
    if isinstance(sigma_iw.mesh, MeshDLRImFreq) and isinstance(pi_iw.mesh, MeshDLRImFreq):
        return _dlr_imp_results_to_raw_data(sigma_iw, pi_iw, ir_kernel)
    elif isinstance(sigma_iw.mesh, MeshImFreq) and isinstance(pi_iw.mesh, MeshImFreq):
        return _full_mesh_imp_results_to_raw_data(sigma_iw, pi_iw, ir_kernel)
    else:
        raise ValueError("Incompatible mesh types for sigma_iw and pi_iw.")


def _dlr_imp_results_to_raw_data(sigma_iw, pi_iw, ir_kernel=None):
    """
    Convert impurity self-energy and polarization Green's functions from TRIQS objects
    to raw NumPy array representations, optionally projected onto an intermediate
    representation (IR) Matsubara frequency mesh.

    Parameters
    ----------
    sigma_iw : BlockGf on MeshDLRImFreq mesh
    pi_iw : Gf on MeshDLRImFreq mesh
    ir_kernel : optional
        IR kernel object providing methods to obtain the IR Matsubara frequency meshes.
        If provided, both `sigma_iw` and `pi_iw` will be interpolated on these IR grids.
        If None, the data on the original full Matsubara mesh are returned.

    Returns
    -------
    dict
        A dictionary containing:
        - ``"Sigma_iw_data"`` : list of ndarray
            Block array representation of the impurity self-energy.
        - ``"Pi_iw_data"`` : list of ndarray
            Block array representation of the impurity polarizability.
    """
    if ir_kernel is None:
        # solver_res.Sigma_iw (TRIQS BlockGf) -> solver_res.Sigma_iw_data (block array)
        sigma_iw_data = blk_gf_to_blk_arr(sigma_iw)
        # solver_res.pi_iw (TRIQS Gf) -> solver_res.pi_iw_data (block array)
        pi_iw_data = [pi_iw.data[:]]

        return {"Sigma_iw_data": sigma_iw_data, "Pi_iw_data": pi_iw_data}

    # converter Sigma and Pi to IR Matsubara mesh
    Sigma_dlr = make_gf_dlr(sigma_iw)
    ir_idx_f = ir_kernel.wn_mesh(stats='f', ir_notation=False)
    nw_f_half = ir_kernel.nw_f // 2
    iw_mesh_uniform_f = MeshImFreq(
        beta=sigma_iw.mesh.beta,
        statistic=sigma_iw.mesh.statistic,
        n_iw=ir_idx_f[-1]+10
    )

    sigma_iw_data = []
    for blk_name, sigma in Sigma_dlr:
        sigma_dyn_ir = np.zeros((ir_kernel.nw_f,) + sigma.data[:].shape[1:], dtype=complex)
        for idx in range(nw_f_half):
            iw_pos = nw_f_half + idx
            iw_neg = nw_f_half - idx - 1
            sigma_dyn_ir[iw_pos] = sigma(iw_mesh_uniform_f(ir_idx_f[iw_pos]))
            sigma_dyn_ir[iw_neg] = sigma(iw_mesh_uniform_f(ir_idx_f[iw_pos])).conj()
        sigma_iw_data.append(sigma_dyn_ir)

    Pi_dlr = make_gf_dlr(pi_iw)
    ir_idx_b = ir_kernel.wn_mesh(stats='b', ir_notation=False, positive_only=True)
    nw_b_pos = len(ir_idx_b)
    iw_mesh_uniform_b = MeshImFreq(
        beta=Pi_dlr.mesh.beta,
        statistic=Pi_dlr.mesh.statistic,
        n_iw=ir_idx_b[-1]+10
    )
    pi_iw_ir = np.zeros((nw_b_pos,) + pi_iw.data[:].shape[1:], dtype=complex)
    for i, idx in enumerate(ir_idx_b):
        pi_iw_ir[i] = Pi_dlr(iw_mesh_uniform_b(idx))
    pi_iw_data = [pi_iw_ir]

    return {"Sigma_iw_data": sigma_iw_data, "Pi_iw_data": pi_iw_data}


def _full_mesh_imp_results_to_raw_data(sigma_iw, pi_iw, ir_kernel=None):
    """
    Convert impurity self-energy and polarization Green's functions from TRIQS objects
    to raw NumPy array representations, optionally projected onto an intermediate
    representation (IR) Matsubara frequency mesh.

    Parameters
    ----------
    sigma_iw : BlockGf on MeshImFreq mesh
    pi_iw : Gf on MeshImFreq mesh
    ir_kernel : optional
        IR kernel object providing methods to obtain the IR Matsubara frequency meshes.
        If provided, both `sigma_iw` and `pi_iw` will be interpolated on these IR grids.
        If None, the data on the original full Matsubara mesh are returned.

    Returns
    -------
    dict
        A dictionary containing:
        - ``"Sigma_iw_data"`` : list of ndarray
            Block array representation of the impurity self-energy.
        - ``"Pi_iw_data"`` : list of ndarray
            Block array representation of the impurity polarizability.
    """
    # solver_res.Sigma_iw (TRIQS BlockGf) -> solver_res.Sigma_iw_data (block array)
    sigma_iw_data = blk_gf_to_blk_arr(sigma_iw)
    # solver_res.Pi_iw (TRIQS Gf) -> solver_res.Pi_iw_data (block array)
    pi_iw_data = [pi_iw.data[:]]

    if ir_kernel is None:
        return {"Sigma_iw_data": sigma_iw_data, "Pi_iw_data": pi_iw_data}

    # converter Sigma and Pi to IR Matsubara mesh
    ir_idx_f = ir_kernel.wn_mesh(stats='f', ir_notation=False)
    nw_f = len(ir_idx_f)
    nw_f_half = nw_f // 2
    for i, sigma_dyn in enumerate(sigma_iw_data):
        sigma_dyn_ir = np.zeros((nw_f,) + sigma_dyn.shape[1:], dtype=complex)
        for idx in range(nw_f_half):
            iw_pos = nw_f_half + idx
            iw_neg = nw_f_half - idx - 1
            data_idx = sigma_iw.mesh.to_data_index(ir_idx_f[iw_pos])
            sigma_dyn_ir[iw_pos] = sigma_dyn[data_idx]
            sigma_dyn_ir[iw_neg] = sigma_dyn[data_idx].conj()
        sigma_iw_data[i] = sigma_dyn_ir

    # interpolate solver_res.Pi_iw to ir grid
    ir_idx_b = ir_kernel.wn_mesh(stats='b', ir_notation=False, positive_only=True)
    nw_b_pos = len(ir_idx_b)
    pi_iw_ir = np.zeros((nw_b_pos,) + pi_iw_data[0].shape[1:], dtype=complex)
    for idx in range(nw_b_pos):
        data_idx = pi_iw.mesh.to_data_index(ir_idx_b[idx])
        pi_iw_ir[idx] = pi_iw_data[0][data_idx]
    pi_iw_data[0] = pi_iw_ir

    return {"Sigma_iw_data": sigma_iw_data, "Pi_iw_data": pi_iw_data}


def block_gf_to_gf(block_gf, gf_struct):
    
    n_orb = sum(dim for name, dim in gf_struct if name[:2]=="up")
    gf_sab = Gf(mesh = block_gf.mesh, target_shape=(2, n_orb, n_orb))
    gf_sab.data[:] = 0.0
    
    offsets = [0, 0]
    for blk_name, dim in gf_struct:
        s = 0 if blk_name[:2] == "up" else 1
        gf_sab.data[:, s, offsets[s]:offsets[s]+dim, offsets[s]:offsets[s]+dim] = block_gf[blk_name].data[:]
        offsets[s] += dim
    
    return gf_sab


def blk_gf_to_blk_arr(block_gf):

    blk_array = []
    for blk_name, gf in block_gf:
        blk_array.append(gf.data[:])

    return blk_array


def blk_arr_to_arr(blk_array, gf_struct):
    """
    Convert a list of spin/orbital block arrays into a single dense array.

    Parameters
    ----------
    blk_array : list of ndarray
        List of block data arrays. Each block corresponds to one entry in `gf_struct`,
        and all blocks must have the same leading dimensions (e.g. frequency, k-points).
        The last two dimensions of each block are the orbital indices for that block.
    gf_struct : list of (str, int)
        Structure definition for the blocks. Each element is a tuple `(block_name, dim)`,
        where `block_name` is a string like `"up_0"`, `"down_1"`, and `dim` is the orbital
        dimension of that block.

    Returns
    -------
    array : ndarray
        Combined dense array with shape
        ``arr_shape[:-2] + (nspin, n_orb, n_orb)``,
        where `arr_shape` is the shape of the first element of `blk_array`.
        The extra dimension `nspin` is the number of spin blocks (e.g. 2 for "up"/"down"),
        and `n_orb` is the total number of orbitals per spin.
        Each block from `blk_array` is placed into the correct slice of the full array
        according to `gf_struct`.

    Notes
    -----
    - Spin blocks are identified by the prefix of `block_name` before the underscore
      (e.g. `"up"` in `"up_0"`).
    - Orbital sub-blocks are placed contiguously along the orbital axes, with offsets
      determined by the cumulative dimensions of the blocks.
    - Assumes each spin has the same total orbital count; raises if not.
    """

    if blk_array is None:
        return None

    assert len(blk_array) == len(gf_struct),  (
        f"Inconistent number of blocks between block_array ({len(blk_array)}) "
        f"and gf_struct ({len(gf_struct)})."
    )

    spin_blk = []
    for name, _ in gf_struct:
        spin = name.split('_', 1)[0]
        if spin not in spin_blk:
            spin_blk.append(spin)
    spin_to_idx = {s: i for i, s in enumerate(spin_blk)}
    nspin = len(spin_blk)

    # Compute per-spin orbital totals and validate they match
    per_spin_counts = {s: 0 for s in spin_blk}
    for name, dim in gf_struct:
        s = name.split('_', 1)[0]
        per_spin_counts[s] += dim
    counts = set(per_spin_counts.values())
    assert len(counts) == 1, (
        "Per-spin orbital totals must match across spins; got "
        f"{per_spin_counts}"
    )
    n_orb = counts.pop()

    # Infer leading shape/dtype from first block
    arr_shape = blk_array[0].shape
    leading_shape = arr_shape[:-2]

    # Allocate dense array
    array = np.zeros(leading_shape+(nspin, n_orb, n_orb), dtype=blk_array[0].dtype)
    
    # Fill with block data
    offsets = [0] * nspin
    for (blk_name, dim), blk_data in zip(gf_struct, blk_array):
        s_name = blk_name.split('_', 1)[0]
        s = spin_to_idx[s_name]

        assert blk_data.shape[:-2] == leading_shape, (
            f"Leading dims mismatch for block {blk_name}: "
            f"{blk_data.shape[:-2]} vs {leading_shape}"
        )

        start, stop = offsets[s], offsets[s]+dim
        array[..., s, start:stop, start:stop] = blk_data
        offsets[s] += dim

    return array


def arr_to_blk_arr(array, gf_struct):
    """
    Convert a dense array into a list of spin/orbital block arrays per `gf_struct`.

    Parameters
    ----------
    array : ndarray
        Dense data with shape: leading_shape + (nspin, n_orb, n_orb).
    gf_struct : list[tuple[str, int]]
        List of (block_name, dim), e.g. [("up_0", d0), ("up_1", d1), ("down_0", d2), ...].
        The spin label is the prefix before the first underscore.

    Returns
    -------
    blk_array : list of ndarray
        Blocks in the same order as `gf_struct`. Each has shape: leading_shape + (dim, dim).

    Raises
    ------
    AssertionError
        If shapes are inconsistent with `gf_struct` or per-spin orbital totals
        do not match the dense array’s orbital size.
    """
    assert array.ndim >= 3, "array must have at least 3 dimensions (…, nspin, n_orb, n_orb)"
    leading_shape = array.shape[:-3]
    nspin, n_orb_0, n_orb_1 = array.shape[-3:]
    assert n_orb_0 == n_orb_1, f"Last two dims must be square; got {(n_orb_0, n_orb_1)}"

    # Preserve spin order as first seen in gf_struct
    spin_blk = []
    for name, _ in gf_struct:
        s = name.split('_', 1)[0]
        if s not in spin_blk:
            spin_blk.append(s)
    assert len(spin_blk) == nspin, (
        f"Spin count mismatch: array has {nspin}, gf_struct implies {len(spin_blk)}"
    )
    spin_to_idx = {s: i for i, s in enumerate(spin_blk)}

    # Check per-spin total orbitals vs array
    per_spin_counts = {s: 0 for s in spin_blk}
    for name, dim in gf_struct:
        s = name.split('_', 1)[0]
        per_spin_counts[s] += dim
    totals = set(per_spin_counts.values())
    assert len(totals) == 1 and totals.pop() == n_orb_0, (
        "Per-spin orbital totals must match the dense array's orbital size; "
        f"got per-spin {per_spin_counts}, array n_orb={n_orb_0}"
    )

    # Extract blocks
    offsets = [0] * nspin
    blk_array = []
    for blk_name, dim in gf_struct:
        s_name = blk_name.split('_', 1)[0]
        s = spin_to_idx[s_name]

        start, stop = offsets[s], offsets[s]+dim
        assert stop <= n_orb_0, (
            f"Block {blk_name} (dim={dim}) exceeds spin-orbital range "
            f"[{start}:{stop}) with n_orb={n_orb_0}"
        )

        blk_array.append(array[..., s, start:stop, start:stop].copy())
        offsets[s] += dim

    for s_idx, off in enumerate(offsets):
        assert off == n_orb_0, (
            f"Unused orbital slots remain for spin index {s_idx}: "
            f"filled {off} / {n_orb_0}"
        )

    return blk_array


def blk_gf_to_arr(block_gf, gf_struct):
    blk_array = blk_gf_to_blk_arr(block_gf)
    return blk_arr_to_arr(blk_array, gf_struct)


def compute_rot_matrix(embedding, A_ij):
    """
    Compute the rotation matrix U for degenerate blocks of A_ij.

    The degeneracy is provided by the embedding object,
    and the rotation is computed by diagonalizing blocks of A_ij

    :param embedding:
    :param A_ij:
    :return:
    """

    n_orb = A_ij.shape[0]
    U = np.eye(n_orb, dtype=float)

    return U


def embedding(embeding_1e, embeding_2e, solver_results, ir_kernel=None, spin_average=False):

    # convert from TRIQS Gfs to numpy arrays and optionally on the IR basis
    for Res in solver_results:
        Res.update( imp_results_to_raw_data(Res['Sigma_iw'], Res['Pi_iw'], ir_kernel) )

    # A list of 3D arrays (w, i, j) with the length of the list = number of spins
    Sigma_imp_embed = embeding_1e.embed_wij([ Res['Sigma_iw_data'] for Res in solver_results ])
    Vhf_imp_embed   = embeding_1e.embed_ij([ Res['Sigma_infty'] for Res in solver_results ])
    Pi_imp_embed    = embeding_2e.embed_wij([ Res['Pi_iw_data'] for Res in solver_results ])

    # The same applied to the DC terms
    Sigma_dc_embed = embeding_1e.embed_wij([ Res['Sigma_iw_dc_data'] for Res in solver_results ])
    Vhf_dc_embed   = embeding_1e.embed_ij([ Res['Sigma_infty_dc'] for Res in solver_results ])
    Pi_dc_embed    = embeding_2e.embed_wij([ Res['Pi_iw_dc_data'] for Res in solver_results ])

    #combine spins to a single array and add auxiliary impurity index
    Sigma_imp_embed = np.stack(Sigma_imp_embed, axis=1)[:,:,None]
    Sigma_dc_embed  = np.stack(Sigma_dc_embed, axis=1)[:,:,None]
    Vhf_imp_embed   = np.stack(Vhf_imp_embed, axis=0)[:,None]
    Vhf_dc_embed    = np.stack(Vhf_dc_embed, axis=0)[:,None]
    Pi_imp_embed    = Pi_imp_embed[0]
    Pi_dc_embed     = Pi_dc_embed[0]

    if spin_average:
        # Average over spin
        Sigma_imp_embed = np.sum(Sigma_imp_embed, axis=1)[:, None] * 0.5
        Sigma_dc_embed = np.sum(Sigma_dc_embed, axis=1)[:, None] * 0.5
        Vhf_imp_embed = np.sum(Vhf_imp_embed, axis=0)[None, :] * 0.5
        Vhf_dc_embed = np.sum(Vhf_dc_embed, axis=0)[None, :] * 0.5

    local_sigma_w = {'imp': Sigma_imp_embed, 'dc': Sigma_dc_embed}
    local_hf      = {'imp': Vhf_imp_embed, 'dc': Vhf_dc_embed}
    local_pi_w    = {
        'imp': density_density_to_product_basis(Pi_imp_embed),
        'dc': density_density_to_product_basis(Pi_dc_embed)
    }

    return local_sigma_w, local_hf, local_pi_w


def hubbard_kanamori_coulomb(V_abcd):
    n_orb = V_abcd.shape[0]
    U, Up, J_pair, J_spin = 0.0, 0.0, 0.0, 0.0

    # intra-orbital U
    for i in range(n_orb):
        U += V_abcd[i, i, i, i]
    U /= n_orb

    if n_orb == 1:
        return U, Up, J_pair, J_spin

    # inter-orbital U
    for i in range(n_orb):
        for j in range(i+1, n_orb):
            if i != j:
                Up += V_abcd[i, i, j, j]
    Up /= (n_orb * (n_orb - 1)) / 2

    # Hund's coupling J: pair-hopping
    for i in range(n_orb):
        for j in range(i+1, n_orb):
            if i != j:
                J_pair += V_abcd[i, j, i, j]
    J_pair /= (n_orb * (n_orb - 1)) / 2

    # Hund's coupling J: spin-flip
    for i in range(n_orb):
        for j in range(i+1, n_orb):
            if i != j:
                J_spin += V_abcd[i, j, j, i]
    J_spin /= (n_orb * (n_orb - 1)) / 2

    if U.imag > 1e-8:
        mpi.report("Warning: complex value encountered in intra-orbital U.imag = {U.imag}.")
    if Up.imag > 1e-8:
        mpi.report("Warning: complex value encountered in inter-orbital U.imag = {Up.imag}.")
    if J_pair.imag > 1e-8:
        mpi.report("Warning: complex value encountered in pair-hopping J.imag = {J_pair.imag}.")
    if J_spin.imag > 1e-8:
        mpi.report("Warning: complex value encountered in spin-flip J.imag = {J_spin.imag}.")

    return U.real, Up.real, J_pair.real, J_spin.real
