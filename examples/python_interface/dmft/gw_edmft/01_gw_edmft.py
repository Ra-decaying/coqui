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

outer_niter, inner_niter = 10, 1

# Embedding class from ModEST
mpi.report("-"*20+"One-Particle Embedding"+"-"*20)
P, E1 = modest.make_embedding_from_h5(wan_h5)
E1 = E1.split_block(0, 0, [1, 1, 1])
mpi.report(E1)

mpi.report("-"*20+"Two-Particle Embedding"+"-"*20)
_, E2 = modest.make_embedding_from_h5(wan_h5)
E2 = E2.make_spinless
mpi.report(E2)

# Update the rotation matrix
proj_info = {}
if mpi.is_master_node():
    proj_info = coqui_dmft.read_proj_info(wan_h5)
proj_info = mpi.bcast(proj_info)

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
