import l3d
import sumpy.toys as t
import numpy as np
import numpy.linalg as la
import pandas as pd
import multiprocessing
import collections
import itertools
import logging
import re


logger = logging.getLogger(__name__)


MAX_WORKERS = 1 + multiprocessing.cpu_count()

SCALING = 1 / (4 * np.pi)


# {{{ sampling

def sphere_sample(npoints_approx, r=1):
    """Generate points regularly distributed on a sphere.

    Based on: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf

    Returns an array of shape (npoints, 3). npoints does not generally equally
    npoints_approx.
    """

    points = []

    count = 0
    a = 4 * np.pi / npoints_approx
    d = a ** 0.5
    M_theta = int(np.ceil(np.pi / d))
    d_theta = np.pi / M_theta
    d_phi = a / d_theta

    for m in range(M_theta):
        theta = np.pi * (m + 0.5) / M_theta
        M_phi = int(np.ceil((2 * np.pi * np.sin(theta) / d_phi)))
        for n in range(M_phi):
            phi = 2 * np.pi * n / M_phi
            points.append([r * np.sin(theta) * np.cos(phi),
                           r * np.sin(theta) * np.sin(phi),
                           r * np.cos(theta)])
            count += 1

    # Add poles.
    for i in range(3):
        for sign in [-1, +1]:
            pole = np.zeros(3)
            pole[i] = r * sign
            points.append(pole)

    return np.array(points)


def visualize_sphere_sample(npoints=72):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = sphere_sample(npoints).T
    ax.scatter(points[0,:], points[1,:], points[2,:])

    plt.show()


def ball_sample(nshells, npoints_approx_per_shell, r=1):
    """Generate points sampled along a number of shells of a ball."""
    points = [(0, 0, 0)]

    sphere = sphere_sample(npoints_approx_per_shell)

    for i in range(1, 1 + nshells):
        points.extend(list(sphere * r * (i / nshells)))

    return np.array(points)

# }}}


Result = collections.namedtuple("Result", "R, r, rho, ratio, p, q, error")
ExperimentParam = collections.namedtuple("ExperimentParam", "R, r, rho, ratio, p, q")


BIG_R_VALS = (0.1, 1, 10)
RHO_VALS = BIG_R_VALS
R_OVER_RHO_VALS = (1/4, 1/2, 3/4)
Q_VALS = (3, 5, 10, 15, 20)
P_VALS = Q_VALS


SPHERE_SAMPLE = sphere_sample(32)
BALL_SAMPLE = ball_sample(4, 7)


# {{{ m2p

def run_m2p_experiment():
    p = multiprocessing.Pool(MAX_WORKERS)

    experiments = [ExperimentParam(R, rho * r_over_rho, rho, r_over_rho, p, q)
        for R, rho, r_over_rho, p, q
        in itertools.product(BIG_R_VALS, RHO_VALS, R_OVER_RHO_VALS, P_VALS, Q_VALS)]

    results = set()

    n = len(experiments)

    for i, result in enumerate(p.imap_unordered(m2p_experiment, experiments)):
        results.add(result)
        logger.info(f"{i+1}/{n} finished")

    p.close()
    p.join()

    df = pd.DataFrame.from_records(list(results), columns=Result._fields)
    return df


def m2p_experiment(param):
    from sumpy.kernel import LaplaceKernel
    ctx = l3d.InterpretedToyContext(
        LaplaceKernel(3), l3d.L3DMultipoleExpansion, l3d.L3DLocalExpansion)

    assert param.r / param.rho < 1

    mpole_center_dist = param.R + param.rho

    sources = param.r * SPHERE_SAMPLE
    sources[:, 2] += mpole_center_dist

    mpole_center = np.array([0., 0., mpole_center_dist])
    local_centers = BALL_SAMPLE * param.R

    error = 0

    fmm_order = param.p
    qbx_order = param.q

    for src in sources:
        src = src.reshape(3, -1)

        src_pot = t.PointSources(ctx, src, weights=[1])
        mpole_pot = t.multipole_expand(src_pot, mpole_center, order=fmm_order)

        for local_center in local_centers:
            targets = (
                (param.R - la.norm(local_center)) * SPHERE_SAMPLE + local_center)
            assert (la.norm(targets, axis=1) <= param.R * (1 + 1e-5)).all()

            if qbx_order == np.inf:
                actual = src_pot.eval(targets.T) - mpole_pot.eval(targets.T)
            else:
                qbx_src_pot = t.local_expand(src_pot, local_center,
                                             order=qbx_order)
                qbx_mpole_pot = t.local_expand(mpole_pot, local_center,
                                               order=qbx_order)
                diff = qbx_src_pot.with_coeffs(
                    qbx_src_pot.coeffs - qbx_mpole_pot.coeffs)
                actual = diff.eval(targets.T)

            error = max(error, np.max(np.abs(actual)))

    return Result(param.R, param.r, param.rho, param.ratio, param.p, param.q, error)

# }}}


# {{ l2p

def run_l2p_experiment():
    p = multiprocessing.Pool(MAX_WORKERS)

    experiments = [ExperimentParam(None, rho * r_over_rho, rho, r_over_rho, p, q)
        for rho, r_over_rho, p, q
        in itertools.product(RHO_VALS, R_OVER_RHO_VALS, P_VALS, Q_VALS)]

    results = set()

    n = len(experiments)

    for i, result in enumerate(p.imap_unordered(l2p_experiment, experiments)):
        results.add(result)
        logger.info(f"{i+1}/{n} finished")

    p.close()
    p.join()

    df = pd.DataFrame.from_records(list(results), columns=Result._fields)
    # R is unused
    del df["R"]

    return df


def l2p_experiment(param):
    from sumpy.kernel import LaplaceKernel
    ctx = l3d.InterpretedToyContext(
        LaplaceKernel(3), l3d.L3DMultipoleExpansion, l3d.L3DLocalExpansion)

    assert param.r / param.rho < 1

    origin = np.zeros(3)

    source_loc = np.array([0., 0., param.rho])
    local_centers = BALL_SAMPLE * param.r

    fmm_order = param.p
    qbx_order = param.q

    src_pot = t.PointSources(ctx, source_loc.reshape(3, -1), weights=[1])
    local_pot = t.local_expand(src_pot, origin, order=fmm_order)

    error = 0

    for ctr in local_centers:
        targets = (param.r - la.norm(ctr)) * SPHERE_SAMPLE + ctr
        assert (la.norm(targets, axis=1) <= param.r * (1 + 1e-5)).all()

        if qbx_order == np.inf:
            actual = src_pot.eval(targets.T) - local_pot.eval(targets.T)
        else:
            qbx_src_pot = t.local_expand(src_pot, ctr, order=qbx_order)
            qbx_local_pot = t.local_expand(local_pot, ctr, order=qbx_order)
            diff = qbx_src_pot.with_coeffs(
                qbx_src_pot.coeffs - qbx_local_pot.coeffs)
            actual = diff.eval(targets.T)

        error = max(error, np.max(np.abs(actual)))

    return Result(param.R, param.r, param.rho, param.ratio, param.p, param.q, error)

# }}}


# {{{ m2l

def run_m2l_experiment():
    p = multiprocessing.Pool(MAX_WORKERS)

    experiments = [ExperimentParam(R, rho * r_over_rho, rho, r_over_rho, p, q)
        for R, rho, r_over_rho, p, q
        in itertools.product(BIG_R_VALS, RHO_VALS, R_OVER_RHO_VALS, P_VALS, Q_VALS)]

    results = set()

    n = len(experiments)

    for i, result in enumerate(p.imap_unordered(m2l_experiment, experiments)):
        results.add(result)
        logger.info(f"{i+1}/{n} finished")

    p.close()
    p.join()

    df = pd.DataFrame.from_records(list(results), columns=Result._fields)

    return df


def m2l_experiment(param):
    from sumpy.kernel import LaplaceKernel
    ctx = l3d.InterpretedToyContext(
        LaplaceKernel(3), l3d.L3DMultipoleExpansion, l3d.L3DLocalExpansion)

    assert param.r / param.rho < 1

    mpole_center_dist = param.R + param.rho

    sources = param.r * SPHERE_SAMPLE
    sources[:, 2] += mpole_center_dist

    origin = np.zeros(3)
    mpole_center = np.array([0., 0., mpole_center_dist])
    local_centers = BALL_SAMPLE * param.R

    error = 0

    fmm_order = param.p
    qbx_order = param.q

    for src in sources:
        src = src.reshape(3, -1)

        src_pot = t.PointSources(ctx, src, weights=[1])
        mpole_pot = t.multipole_expand(src_pot, mpole_center, order=fmm_order)
        local_pot = t.local_expand(mpole_pot, origin, order=fmm_order)

        for local_center in local_centers:
            targets = (
                (param.R - la.norm(local_center)) * SPHERE_SAMPLE + local_center)
            assert (la.norm(targets, axis=1) <= param.R * (1 + 1e-5)).all()

            if qbx_order == np.inf:
                actual = src_pot.eval(targets.T) - local_pot.eval(targets.T)
            else:
                qbx_src_pot = t.local_expand(src_pot, local_center,
                                             order=qbx_order)
                qbx_local_pot = t.local_expand(local_pot, local_center,
                                               order=qbx_order)
                diff = qbx_src_pot.with_coeffs(
                    qbx_src_pot.coeffs - qbx_local_pot.coeffs)
                actual = diff.eval(targets.T)

            error = max(error, np.max(np.abs(actual)))

    return Result(param.R, param.r, param.rho, param.ratio, param.p, param.q, error)

# }}}


# {{{ data processing

HEADER = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{siunitx}
\usepackage[margin=1in]{geometry}

\title{Raw Experimental Accuracy Data for 3D FMM Translations}

\newcommand{\cc}[1]{\multicolumn{1}{c}{#1}}

\sisetup{
  table-number-alignment = center,
  table-format = 1.3e-2,
  table-sign-exponent,
  round-mode = figures,
  round-precision = 4,
}
"""


def get_table(df, caption):
    lines = [r"\begin{table}"]
    lines.append(r"\small")
    lines.append(r"\centering")
    lines.append(tabulate(df))
    lines.append(rf"\caption{{{caption}}}")
    lines.append(r"\end{table}")
    return lines


def generate_report():
    report = [HEADER]
    report.append(r"\begin{document}")
    report.append(r"\maketitle")
    report.append(r"\listoftables")

    # Generate raw output tables.

    m2p_results = pd.read_pickle(M2P_FILE)
    l2p_results = pd.read_pickle(L2P_FILE)
    m2l_results = pd.read_pickle(M2L_FILE)

    for R in BIG_R_VALS:
        df = m2p_results.query(f"R == {R}")
        caption = f"M2P results, $R = {R}$"
        report.extend(get_table(df, caption))

    caption = "L2P results"
    report.extend(get_table(l2p_results, caption))

    for R in BIG_R_VALS:
        df = m2p_results.query(f"R == {R}")
        caption = f"M2L results, $R = {R}$"
        report.extend(get_table(df, caption))

    # Generate estimates of overshoot factors.

    report.append(r"\begin{table}")
    report.append(r"\centering")
    report.append(r"\begin{tabular}{cr}")
    report.append(r"\toprule")
    report.append(r"\cc{Translation} & \cc{Factor} \\")
    report.append(r"\midrule")

    r, rho, p = (m2p_results[col] for col in ("r", "rho", "p"))
    m2p_results["bound"] = SCALING / (rho - r) * (r / rho) ** (p + 1)
    c_m2p = (m2p_results["error"] / m2p_results["bound"]).max()
    report.append(rf"M2P & {c_m2p} \\")

    r, rho, p = (l2p_results[col] for col in ("r", "rho", "p"))
    l2p_results["bound"] = SCALING /(rho - r) * (r / rho) ** (p + 1)
    c_l2p = (l2p_results["error"] / l2p_results["bound"]).max()
    report.append(rf"L2P & {c_l2p} \\")

    R, r, rho, p = (m2l_results[col] for col in ("R", "r", "rho", "p"))
    m2l_results["bound"] = SCALING / (rho - r) * (
        (R / (R + rho - r)) ** (p + 1) + (r/rho) ** (p + 1))
    c_m2l = (m2l_results["error"] / m2l_results["bound"]).max()
    report.append(rf"M2L & {c_m2l} \\")

    report.append(r"\bottomrule")
    report.append(r"\end{tabular}")
    report.append(r"\caption{Factors by which a translation operator "
                  r"exceeds the point FMM error estimate}")
    report.append(r"\end{table}")

    report.append(r"\end{document}")

    return "\n".join(report)


def tabulate(df):
    rows = []

    # Header material
    rows.append(r"\begin{tabular}{rrrSSSSS}")
    rows.append(r"\toprule")
    indices = ("ratio", "p", "rho")
    tex_indices = dict(ratio=r"$r/\rho$", p="$p$", rho=r"$\rho$")
    column_names = [" & ".join(rf"\cc{{{tex_indices[index]}}}" for
                               index in indices)]
    for q in Q_VALS:
        column_names.append(rf"\cc{{$q={q}$}}")
    rows.append(" & ".join(column_names) + r"\\")

    # Body material
    tab = df.pivot_table(values="error", index=indices, columns="q")

    def rho_format(rho):
        if rho.is_integer():
            rho = int(rho)
        return str(rho)

    tab = tab.rename(rho_format, axis="index", level=indices.index("rho"))
    body = tab.to_latex(float_format="%e", multirow=True).split("\n")
    del body[:body.index(r"\midrule")]

    # Hackery to get ride of certain formatting
    # - replace cline with cmidrule
    # - get rid of repeated instances of cline
    # - remote multirow statements
    for line in body:
        if line.startswith(r"\cline"):
            if line.startswith(r"\cline{1"):
                rows.append(line.replace("cline", "cmidrule"))
            else:
                if not rows[-1].startswith("\cmidrule"):
                    rows[-1] += "[1ex]"
        else:
            line = re.sub(r"\\multirow\{\d+\}\{\*\}\{([0-9.]*)\}", r"\1", line)
            rows.append(line)

    # Generated body includes footer material
    return "\n".join(rows)

# }}}


M2P_FILE = "m2p-results.df"
L2P_FILE = "l2p-results.df"
M2L_FILE = "m2l-results.df"


def main():
    if 0:
        visualize_sphere_sample()
        return

    import os
    os.nice(19)

    logging.basicConfig(level=logging.INFO)

    logger.info("sphere sample has %d points" % SPHERE_SAMPLE.shape[0])
    logger.info("ball sample has %d points" % BALL_SAMPLE.shape[0])

    if not os.path.exists(M2P_FILE):
        df = run_m2p_experiment()
        df.to_pickle(M2P_FILE)

    if not os.path.exists(L2P_FILE):
        df = run_l2p_experiment()
        df.to_pickle(L2P_FILE)

    if not os.path.exists(M2L_FILE):
        df = run_m2l_experiment()
        df.to_pickle(M2L_FILE)

    print(generate_report())


if __name__ == "__main__":
    main()
