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


def plot_edmft_convergence(dmft_chkpt, *, impurity_index=0, check_w=True,
                           figure_name="edmft_convergence"):
  """Plot EDMFT convergence diagnostics for a single impurity.

  Reads the convergence history from the DMFT checkpoint file and produces a
  multi-panel figure.  The following quantities are plotted when available:

  - ``U(w=0)``       – orbital-averaged static local interaction at w=0 (should be constant)
  - ``|Gloc - Gimp|``  – fermionic self-consistency residual (should → 0)
  - ``|dg_weiss|``   – change in fermionic Weiss field between iterations (should → 0)
  - ``|du_weiss|``   – change in bosonic Weiss field between iterations (should → 0)
  - ``|Wloc - Wimp|``  – bosonic self-consistency residual (plotted only when
                          ``check_w=True`` and at least one stored value is valid)
  - ``mu_imp``       – impurity chemical potential correction (plotted only when
                          at least one stored value is non-zero)
  - Orbital occupations per spin block (should be constant)

  All convergence data are read from
  ``dmft/iter{i}/impurity_{impurity_index}/results/convergence`` (a length-6
  numpy array ``[U_w0, A_w0, diff_g, diff_g_weiss, diff_u_weiss, diff_w]``),
  ``dmft/iter{i}/impurity_{impurity_index}/results/mu_imp``, and
  ``dmft/iter{i}/impurity_{impurity_index}/results/density``.

  Parameters
  ----------
  dmft_chkpt : str
    Path to the DMFT HDF5 checkpoint file.
  impurity_index : int, optional
    Index of the impurity to plot. Default is 0.
  check_w : bool, optional
    Whether to include the ``|Wloc - Wimp|`` panel. Even when ``True``, the
    panel is suppressed if all stored ``diff_w`` values equal ``-1.0``.
    Default is ``True``.
  figure_name : str, optional
    Output filename for the figure.  The string ``_imp{impurity_index}`` is
    always appended before the extension so that different impurities do not
    overwrite each other.  Default is ``"edmft_convergence"``.

  Returns
  -------
  str
    The actual filename the figure was saved to.
  """
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError as e:
    raise ImportError(
      "plot_edmft_convergence requires matplotlib. "
      "Install it with: pip install matplotlib"
    ) from e

  out_file = f"{figure_name}_imp{impurity_index}.png"

  # ---- Read checkpoint -------------------------------------------------------
  iters        = []
  U_w0_list    = []
  diff_g_list       = []
  diff_g_weiss_list = []
  diff_u_weiss_list = []
  diff_w_list       = []
  mu_imp_list       = []
  density_list      = []

  with HDFArchive(dmft_chkpt, 'r') as ar:
    if "dmft" not in ar.keys():
      raise KeyError(f"No 'dmft' group found in {dmft_chkpt}.")
    dmft_grp = ar["dmft"]

    # Collect all iter{i} groups in order
    iter_keys = sorted(
      [k for k in dmft_grp.keys() if k.startswith("iter")],
      key=lambda k: int(k[4:])
    )
    if not iter_keys:
      raise KeyError(f"No 'iter*' groups found under 'dmft' in {dmft_chkpt}.")

    imp_key = f"impurity_{impurity_index}"
    for ik in iter_keys:
      iter_grp = dmft_grp[ik]
      if imp_key not in iter_grp.keys():
        continue
      res_grp = iter_grp[f"{imp_key}/results"]

      if "convergence" not in res_grp.keys():
        continue

      conv = np.asarray(res_grp["convergence"])  # [U_w0, A_w0, diff_g, diff_g_weiss, diff_u_weiss, diff_w]
      iters.append(int(ik[4:]))
      U_w0_list.append(conv[0])
      diff_g_list.append(conv[2])
      diff_g_weiss_list.append(conv[3])
      diff_u_weiss_list.append(conv[4])
      diff_w_list.append(conv[5])

      # mu_imp (optional; default 0.0 when absent)
      mu_val = float(res_grp["mu_imp"]) if "mu_imp" in res_grp.keys() else 0.0
      mu_imp_list.append(mu_val)

      # density (optional; -1.0 sentinel when absent)
      density_val = float(res_grp["density"]) if "density" in res_grp.keys() else -1.0
      density_list.append(density_val)

  if not iters:
    raise RuntimeError(
      f"No convergence data found for impurity {impurity_index} in {dmft_chkpt}. "
      "Ensure the DMFT loop has completed at least one iteration."
    )

  iters = np.array(iters)
  diff_w_arr = np.array(diff_w_list)

  # ---- Decide which optional panels to include --------------------------------
  show_w    = check_w and np.any(diff_w_arr > -0.5)   # at least one valid diff_w
  show_mu   = np.any(np.abs(mu_imp_list) > 1e-14)

  # ---- Build panel list -------------------------------------------------------
  # Fixed panels (always shown): U_w0/A_w0, diff_g, diff_g_weiss, diff_u_weiss, imp_density
  n_panels = 5
  if show_w:
    n_panels += 1
  if show_mu:
    n_panels += 1

  fontsize, markersize = 18, 10
  fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.2 * n_panels), sharex=True)
  if n_panels == 1:
    axes = [axes]

  panel = 0

  # Panel 1: U(w=0) and A(w=0)
  ax = axes[panel]; panel += 1
  ax.plot(iters, U_w0_list, 'o-', markersize=markersize, color='C0')
  ax.set_ylabel(r"$u(\omega=0)$ (a.u.)", fontsize=fontsize )
  ax.grid(True, alpha=0.3, linestyle='--')

  # Panel 2: |Gloc - Gimp|
  ax = axes[panel]; panel += 1
  ax.semilogy(iters, diff_g_list, 'o-', markersize=markersize, color='C1')
  ax.set_ylabel(r"$|G_{\rm loc} - G_{\rm imp}|$", fontsize=fontsize)
  ax.grid(True, alpha=0.3, linestyle='--')

  # Panel 3: |dg_weiss|
  ax = axes[panel]; panel += 1
  valid = np.array(diff_g_weiss_list) > -0.5
  if np.any(valid):
    ax.semilogy(iters[valid], np.array(diff_g_weiss_list)[valid], 'o-', markersize=markersize, color='C2')
  ax.set_ylabel(r"$|\Delta g|$", fontsize=fontsize)
  ax.grid(True, alpha=0.3, linestyle='--')

  # Panel 4: |du_weiss|
  ax = axes[panel]; panel += 1
  valid = np.array(diff_u_weiss_list) > -0.5
  if np.any(valid):
    ax.semilogy(iters[valid], np.array(diff_u_weiss_list)[valid], 'o-', markersize=markersize, color='C3')
  ax.set_ylabel(r"$|\Delta u|$", fontsize=fontsize)
  ax.grid(True, alpha=0.3, linestyle='--')

  # Panel 5 (optional): mu_imp
  if show_mu:
    ax = axes[panel]; panel += 1
    ax.plot(iters, mu_imp_list, 'o-', markersize=markersize, color='C4')
    ax.axhline(0.0, color='k', linewidth=0.8, linestyle='--')
    ax.set_ylabel(r"$\mu_{\rm imp}$ (a.u.)", fontsize=fontsize)
    ax.grid(True, alpha=0.3, linestyle='--')

  # Panel 6: impurity total density
  ax = axes[panel]; panel += 1
  density_arr = np.array(density_list)
  valid = density_arr > -0.5
  if np.any(valid):
    ax.plot(iters[valid], density_arr[valid], 'o-', markersize=markersize, color='C5')
  ax.set_ylabel(r"$n_{\rm imp}$", fontsize=fontsize)
  ax.grid(True, alpha=0.3, linestyle='--')

  # Panel 7 (optional): |Wloc - Wimp|
  if show_w:
    ax = axes[panel]; panel += 1
    valid = diff_w_arr > -0.5
    ax.semilogy(iters[valid], diff_w_arr[valid], 'o-', markersize=markersize, color='C6')
    ax.set_ylabel(r"$|W_{\rm loc} - W_{\rm imp}|$", fontsize=fontsize)
    ax.grid(True, alpha=0.3, linestyle='--')

  ax.set_xticks(iters)
  ax.set_xlabel("DMFT iteration", fontsize=fontsize)

  for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)

  fig.tight_layout()
  fig.savefig(out_file, dpi=120, bbox_inches='tight')
  plt.close(fig)
  return out_file