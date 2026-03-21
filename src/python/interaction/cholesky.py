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

import json
import os
from coqui._lib.eri_module import CholCoulomb


def make_chol_coulomb(mf, params):
  # Create output directory if it does not exist
  path = os.path.abspath(os.path.expanduser(params.get("path", "./")))
  if not os.path.exists(path):
    try:
      os.makedirs(path)
    except Exception as e:
      raise RuntimeError(f"make_chol_coulomb: Failed to create directory '{path}': {e}") from e
  elif not os.path.isdir(path):
    raise RuntimeError(f"make_chol_coulomb: params['path'] exists but is not a directory: {path}")

  return CholCoulomb(mf, json.dumps(params))

