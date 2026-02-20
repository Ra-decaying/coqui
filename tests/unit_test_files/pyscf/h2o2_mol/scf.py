

import numpy as np
from pyscf import gto, df, scf, dft
import h5._h5py as h5
import converter
import gdf_converter
import gdf_grad_converter

mol = gto.M(atom='H -1 -1 0; O 0 0 0; O 1 0 0; H 1 0 1', basis='cc-pvdz', verbose=7)

mydf = df.DF(mol, auxbasis='cc-pvdz-jkfit')
mydf._cderi_to_save = "cderi.h5"
mydf.build()
gdf_grad_converter.mol_gdf_grad_dump_to_h5(mydf, mo=False, outdir='gdf_eri_grad')

mf = scf.RHF(mol).density_fit()
mf.with_df = mydf
mf.kernel()
mf.analyze()
converter.mol_dump_to_h5(mf, becke_grid_level=5, mo=False, outdir='./', prefix='pyscf')

mf_grad = mf.Gradients()
mf_grad.auxbasis_response = True
grad_total = mf_grad.kernel()

file = h5.File('./pyscf.h5', 'a')
group = h5.Group(file).open_group('SCF')
h5.h5_write(group, 'grad_total', grad_total)

