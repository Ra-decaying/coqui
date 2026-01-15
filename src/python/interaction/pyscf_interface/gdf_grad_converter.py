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
import h5._h5py as h5

from pyscf import lib
from pyscf.pbc import gto, df
import pyscf.df as mol_df

def mol_gdf_grad_dump_to_h5(gdf, mo=False, outdir=None):

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
    filename = outdir + '/chol_grad_info.h5'
    f = h5.File(filename, 'w')
    g = h5.Group(f).create_group('Interaction_Gradient')
    h5.h5_write(g, 'natoms', np.int32(natoms))
    h5.h5_write(g, 'Np', np.int32(naux))
    h5.h5_write(g, 'tol', -1.0)
    h5.h5_write(g, 'nkpts', np.int32(nkpts))
    h5.h5_write(g, 'nbnd', np.int32(nbnd))
    h5.h5_write(g, 'kpts', kpts)
    h5.h5_write(g, 'qpts', qpts)
    h5.h5_write(g, 'qk_to_kmq', qk_to_kmq.astype(np.int32))
    del f

    # integral derivatives

    filename = outdir + 'Vq0_grad.h5'
    f = h5.File(filename, 'w')
    g = h5.Group(f).create_group("Interaction_Gradient")
    h5.h5_write(g, 'natoms', np.int32(natoms))
    h5.h5_write(g, 'Np', np.int32(naux))
    h5.h5_write(g, 'nbnd', np.int32(nbnd))

    # ( P | d/dX i, j )
    V_3Qskij = np.zeros((3, naux, 1, nkpts, nbnd, nbnd), dtype=complex)
    int3c_e1 = mol_df.incore.aux_e2(gdf.mol, gdf.auxmol, intor='int3c2e_ip1', aosym='s1', comp=3)
    print('shape of int3c_e1 = ', int3c_e1.shape)
    for d in range(3):
        for Q in range(naux):
            for k in range(nkpts):
                for i in range(nbnd):
                    for j in range(nbnd):
                        V_3Qskij[d, Q, 0, k, i, j] = int3c_e1[d, i, j, Q]
    h5.h5_write(g, 'Vq0_3Qskij_di', V_3Qskij)
    del V_3Qskij
    del int3c_e1

    # (d/dX Q | i, j )
    V_3Qskij = np.zeros((3, naux, 1, nkpts, nbnd, nbnd), dtype=complex)
    int3c_e2 = mol_df.incore.aux_e2(gdf.mol, gdf.auxmol, intor='int3c2e_ip2', aosym='s1', comp=3)
    print('shape of int3c_e2 = ', int3c_e2.shape)
    for d in range(3):
        for Q in range(naux):
            for k in range(nkpts):
                for i in range(nbnd):
                    for j in range(nbnd):
                        V_3Qskij[d, Q, 0, k, i, j] = int3c_e2[d, i, j, Q]
    h5.h5_write(g, 'Vq0_3Qskij_dQ', V_3Qskij)
    del V_3Qskij
    del int3c_e2

    # ( Q | i, j )
    V_Qskij = np.zeros((naux, 1, nkpts, nbnd, nbnd), dtype=complex)
    int3c = mol_df.incore.aux_e2(gdf.mol, gdf.auxmol, 'int3c2e', aosym='s1', comp=1)
    print('shape of int3c = ', int3c.shape)
    for d in range(3):
        for Q in range(naux):
            for k in range(nkpts):
                for i in range(nbnd):
                    for j in range(nbnd):
                            V_Qskij[Q, 0, k, i, j] = int3c[i, j, Q]
    h5.h5_write(g, 'Vq0_Qskij', V_Qskij)
    del V_Qskij
    del int3c

    # (d/dX P|Q)
    V_3PQsk = np.zeros((3, naux, naux, 1, nkpts), dtype=complex)
    int2c_e1 = gdf.auxmol.intor('int2c2e_ip1', aosym='s1', comp=3)
    print('shape of int2c_e1 = ', int2c_e1.shape)
    for d in range(3):
        for P in range(naux):
            for Q in range(naux):
                for k in range(nkpts):
                    V_3PQsk[d, P, Q, 0, k] = int2c_e1[d, P, Q]
    h5.h5_write(g, 'Vq0_3PQsk_dP', V_3PQsk)
    del V_3PQsk
    del int2c_e1

    # ( P | Q ) and ( P | Q )^{-1}
    V_PQsk = np.zeros((naux, naux, 1, nkpts), dtype=complex)
    V_PQsk_inv = np.zeros((naux, naux, 1, nkpts), dtype=complex)
    int2c = gdf.auxmol.intor('int2c2e', aosym='s1', comp=1)
    int2c_inv = np.linalg.inv(int2c)
    print('shape of int2c = ', int2c.shape)
    for P in range(naux):
        for Q in range(naux):
            for k in range(nkpts):
                V_PQsk[P, Q, 0, k] = int2c[P, Q]
                V_PQsk_inv[P, Q, 0, k] = int2c_inv[P, Q]
    h5.h5_write(g, 'Vq0_PQsk', V_PQsk)
    h5.h5_write(g, 'Vq0_PQsk_inv', V_PQsk_inv)
    del V_PQsk
    del V_PQsk_inv
    del int2c
    del int2c_inv

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
    del slice
    del mol_ao_slice_by_atom
    del auxmol_ao_slice_by_atom

    del f
