import numpy as np
import tomllib

import triqs.utility.mpi as mpi
import triqs_modest as modest

import coqui
import coqui.dmft as coqui_dmft

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"
wan_h5 = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/mlwf/svo.mlwf.h5"

coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=2)

outer_niter, inner_niter = 1, 1

# Define correlated subspace via ModEST
obe = modest.make_one_body_elements_gw(wan_h5)
mpi.report("-"*20+"One-Particle Embedding"+"-"*20)
E1 = modest.make_embedding(obe.C_space)
mpi.report(E1.description(True))
mpi.report("-"*20+"Two-Particle Embedding"+"-"*20)
E2 = modest.make_embedding(obe.C_space, use_atom_decomp=True).make_spinless
mpi.report(E2.description(True))

# Update the rotation matrix
proj_info = coqui_dmft.get_proj_info(obe.P)

# read input parameters
with open("gw_edmft_params.toml", "rb") as f:
    coqui_params = tomllib.load(f)

mf = coqui.make_mf(coqui_mpi, coqui_params['mean_field']['qe'], mf_type='qe')
thc = coqui.make_thc_coulomb(mf=mf, params=coqui_params['interaction']['thc'])

# GW+EDMFT driver
coqui_dmft.gw_edmft_loop(
    mf, thc, proj_info, E1, E2, 
    inner_niter, outer_niter, **coqui_params
)
