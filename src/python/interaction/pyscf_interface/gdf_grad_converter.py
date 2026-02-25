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

# TODO: merge gdf_grad_converter into gdf_converter

import os

import numpy as np
import h5._h5py as h5

from pyscf import lib
from pyscf.pbc import gto, df
import pyscf.df as mol_df

def mol_gdf_grad_dump_to_h5(gdf, auxbasis_response=True, mo=False, outdir=None):

    if not isinstance(gdf, mol_df.DF):
        raise ValueError('mol_gdf_dump_to_h5: DF obejct has to be a DF instance')
    if outdir is None:
        outdir = './'
    if mo:
        raise ValueError('mol_gdf_dump_to_h5: MO is not supported yet!')
    if gdf.auxmol is None:
        gdf.build()

    print('# Dump GDF-ERI gradients from pyscf')

    mol   = gdf.mol
    natoms = mol.natm
    nkpts, nqpts = 1, 1
    nbnd  = mol.nao_nr()
    naux  = gdf.auxmol.nao_nr()
    kpts  = np.zeros((1,3), dtype=float)
    qpts  = np.zeros((1,3), dtype=float)
    qk_to_kmq = np.zeros((1,1), dtype=int)

    if not os.path.exists(outdir):
        os.system('mkdir ' + outdir)
    filename = outdir + '/chol_info.h5'
    f = h5.File(filename, 'w')
    g = h5.Group(f).create_group('Interaction')
    h5.h5_write(g, 'Np', np.int32(naux))
    h5.h5_write(g, 'tol', -1.0)
    h5.h5_write(g, 'nkpts', np.int32(nkpts))
    h5.h5_write(g, 'nbnd', np.int32(nbnd))
    h5.h5_write(g, 'kpts', kpts)
    h5.h5_write(g, 'qpts', qpts)
    h5.h5_write(g, 'qk_to_kmq', qk_to_kmq.astype(np.int32))
    del f

    filename = outdir + '/Vq0.h5'
    f = h5.File(filename, 'w')
    g = h5.Group(f).create_group("Interaction")
    h5.h5_write(g, 'natoms', np.int32(natoms))
    h5.h5_write(g, 'Np', np.int32(naux))
    h5.h5_write(g, 'nbnd', np.int32(nbnd))

    # imformation of atoms and AO slices

    slice = gdf.mol.aoslice_by_atom()
    mol_ao_slice_by_atom = np.zeros((natoms, 2), dtype=int)
    for a in range(natoms):
        mol_ao_slice_by_atom[a, :] = slice[a, 2:4]
    slice = gdf.auxmol.aoslice_by_atom()
    auxmol_ao_slice_by_atom = np.zeros((natoms, 2), dtype=int)
    for a in range(natoms):
        auxmol_ao_slice_by_atom[a, :] = slice[a, 2:4]
    print()
    print('mol_ao_slice_by_atom')
    print(mol_ao_slice_by_atom)
    print('auxmol_ao_slice_by_atom')
    print(auxmol_ao_slice_by_atom)
    print()
    h5.h5_write(g, 'mol_ao_slice_by_atom', mol_ao_slice_by_atom)
    h5.h5_write(g, 'auxmol_ao_slice_by_atom', auxmol_ao_slice_by_atom)


    # integral derivatives

    V_a3ijQ_dijQ = np.zeros((natoms, 3, nbnd, nbnd, naux), dtype=complex)

    # ( d/dX i, j | Q )
    int3c_e1 = mol_df.incore.aux_e2(gdf.mol, gdf.auxmol, intor='int3c2e_ip1', aosym='s1', comp=3)
    print('shape of int3c_e1 = ', int3c_e1.shape)
    V_3ijQ_di = -int3c_e1.astype(complex)
    del int3c_e1

    for a in range(natoms):
        for d in range(3):
            for Q in range(naux):
                V_a3ijQ_dijQ[a, d, mol_ao_slice_by_atom[a][0]:mol_ao_slice_by_atom[a][1], :, Q] \
                    += V_3ijQ_di[d, mol_ao_slice_by_atom[a][0]:mol_ao_slice_by_atom[a][1], :, Q]
                V_a3ijQ_dijQ[a, d, :, mol_ao_slice_by_atom[a][0]:mol_ao_slice_by_atom[a][1], Q] \
                    += V_3ijQ_di[d, mol_ao_slice_by_atom[a][0]:mol_ao_slice_by_atom[a][1], :, Q].T
    del V_3ijQ_di

    if auxbasis_response:

        # ( i, j | d/dX Q )
        int3c_e2 = mol_df.incore.aux_e2(gdf.mol, gdf.auxmol, intor='int3c2e_ip2', aosym='s1', comp=3)
        print('shape of int3c_e2 = ', int3c_e2.shape)
        V_3ijQ_dQ = -int3c_e2.astype(complex)
        del int3c_e2

        for a in range(natoms):
            for d in range(3):
                V_a3ijQ_dijQ[a, d, :, :, auxmol_ao_slice_by_atom[a][0]:auxmol_ao_slice_by_atom[a][1]] \
                    += V_3ijQ_dQ[d, :, :, auxmol_ao_slice_by_atom[a][0]:auxmol_ao_slice_by_atom[a][1]]
        del V_3ijQ_dQ

    # ( i, j | Q )
    int3c = mol_df.incore.aux_e2(gdf.mol, gdf.auxmol, intor='int3c2e', aosym='s1', comp=1)
    print('shape of int3c = ', int3c.shape)
    V_ijQ = int3c.astype(complex)
    del int3c

    # ( P | Q ) and ( P | Q )^{-1}

    int2c = gdf.auxmol.intor('int2c2e', aosym='s1', comp=1)
    print('shape of int2c = ', int2c.shape)
    int2c_inv = np.linalg.inv(int2c)
    eig, U = np.linalg.eigh(int2c)
    int2c_inv_sqrt = np.zeros((naux, naux))
    for Q in range(naux):
        int2c_inv_sqrt[Q, Q] = 1 / np.sqrt(eig[Q])
    int2c_inv_sqrt = U @ int2c_inv_sqrt @ U.T.conj()
    V_PQ = int2c.astype(complex)
    V_PQ_inv = int2c_inv.astype(complex)
    V_PQ_inv_sqrt = int2c_inv_sqrt.astype(complex)
    del int2c
    del int2c_inv
    del int2c_inv_sqrt

    V_ijQ = V_ijQ.reshape((nbnd*nbnd, naux)) @ V_PQ_inv_sqrt

    V_tilde_a3ijQ = np.zeros((natoms, 3, nbnd*nbnd, naux), dtype=complex)

    for a in range(natoms):
        for d in range(3):
            V_tilde_a3ijQ[a, d] += V_a3ijQ_dijQ[a, d].reshape(nbnd*nbnd, naux) @ V_PQ_inv_sqrt
    del V_a3ijQ_dijQ

    if auxbasis_response:

        V_a3PQ_dP = np.zeros((natoms, 3, naux, naux), dtype=complex)

        # ( d/dX P | Q )
        int2c_e1 = gdf.auxmol.intor('int2c2e_ip1', aosym='s1', comp=3)
        print('shape of int2c_e1 = ', int2c_e1.shape)
        V_3PQ_dP = -int2c_e1.astype(complex)
        del int2c_e1

        for a in range(natoms):
            for d in range(3):
                V_a3PQ_dP[a, d, auxmol_ao_slice_by_atom[a][0]:auxmol_ao_slice_by_atom[a][1], :] \
                    += V_3PQ_dP[d, auxmol_ao_slice_by_atom[a][0]:auxmol_ao_slice_by_atom[a][1], :]
        del V_3PQ_dP

        for a in range(natoms):
            for d in range(3):
                V_tilde_a3ijQ[a, d] -= V_ijQ @ V_PQ_inv_sqrt @ V_a3PQ_dP[a, d] @ V_PQ_inv_sqrt
        del V_a3PQ_dP

    V_Qskij = np.zeros((naux, 1, 1, nbnd, nbnd), dtype=complex)
    V_ijQ = V_ijQ.reshape((nbnd, nbnd, naux))
    for Q in range(naux):
        for i in range(nbnd):
            for j in range(nbnd):
                V_Qskij[Q, 0, 0, i, j] = V_ijQ[i, j, Q]

    V_tilde_a3Qskij = np.zeros((natoms, 3, naux, 1, 1, nbnd, nbnd), dtype=complex)
    V_tilde_a3ijQ = V_tilde_a3ijQ.reshape((natoms, 3, nbnd, nbnd, naux))
    for a in range(natoms):
        for d in range(3):
            for Q in range(naux):
                for i in range(nbnd):
                    for j in range(nbnd):
                        V_tilde_a3Qskij[a, d, Q, 0, 0, i, j] = V_tilde_a3ijQ[a, d, i, j, Q]

    h5.h5_write(g, 'Vq0', V_Qskij)
    h5.h5_write(g, 'dVq0', V_tilde_a3Qskij)

    del V_PQ
    del V_PQ_inv
    del V_PQ_inv_sqrt
    del V_ijQ
    del V_Qskij
    del V_tilde_a3ijQ
    del V_tilde_a3Qskij

    del slice
    del mol_ao_slice_by_atom
    del auxmol_ao_slice_by_atom

    del f
