import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import brentq
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def sinkhorn_logdomain(alpha, beta, C, nu_vals, eps, num_iters=100000, tol=1e-3):
    log_alpha, log_beta = np.log(alpha), np.log(beta)
    log_nu = np.log(nu_vals)
    phi = np.zeros(len(alpha))
    psi = np.zeros(len(beta))

    for _ in range(num_iters):
        phi_prev, psi_prev = phi.copy(), psi.copy()

        phi = -logsumexp(-C / eps + psi[None, :] + log_nu[None, :], axis=1)
        psi = log_beta - log_nu - logsumexp(
            -C.T / eps + phi[None, :] + log_alpha[None, :], axis=1
        )

        if np.linalg.norm(phi - phi_prev) < tol and np.linalg.norm(psi - psi_prev) < tol:
            break

    return eps * phi, eps * psi


def entropic_ot_dual_value(u, v, alpha, beta, nu_vals, C, eps):
    U = u[:, None]
    V = v[None, :]
    exponent = (U + V - C) / eps
    exp_term = np.exp(exponent)
    interaction = np.sum(exp_term * (alpha[:, None] * nu_vals[None, :]))
    return np.dot(u, alpha) + np.dot(v, beta) - eps * interaction + eps


def sinkhorn_discrepancy_refnu(X, Y, alpha, beta, eps):
    C = cdist(X, Y, "sqeuclidean")

    d = Y.shape[1]
    nu_pdf = multivariate_normal(mean=np.zeros(d), cov=np.eye(d))
    nu_vals = nu_pdf.pdf(Y)
    nu_vals /= np.sum(nu_vals)

    u, v = sinkhorn_logdomain(alpha, beta, C, nu_vals, eps)
    return entropic_ot_dual_value(u, v, alpha, beta, nu_vals, C, eps)


def find_sinkhorn_step(center, direction, rho, sinkhorn_cost_fn, eps, alpha, beta, t_max=2.0, tol=1e-4, max_iter=100):
    def f_t(t):
        Y = center + t * direction
        val = sinkhorn_cost_fn(center, Y, alpha, beta, eps)
        if np.isnan(val) or np.isinf(val):
            return np.inf
        return val - rho

    f_hi = f_t(t_max)
    iters = 0
    while f_hi < 0 and iters < max_iter:
        t_max *= 2.0
        f_hi = f_t(t_max)
        iters += 1

    if f_hi < 0:
        raise RuntimeError("Could not bracket target rho within max_iter expansion.")

    t_star = brentq(f_t, 0.0, t_max, rtol=tol)
    Y_star = center + t_star * direction
    cost_val = sinkhorn_cost_fn(center, Y_star, alpha, beta, eps)
    return Y_star, t_star, cost_val


def build_center_particles(nominal_path, n_particles):
    mat = loadmat(nominal_path)
    X_center_small = mat["X"].T  # MATLAB (dN, n) -> rows are samples
    base_n, d = X_center_small.shape

    k = n_particles // base_n
    if k * base_n != n_particles:
        raise ValueError(
            f"n_particles={n_particles} must be a multiple of base samples={base_n}."
        )

    X_center = np.repeat(X_center_small, k, axis=0)
    return X_center, d


def generate_one(nominal_path, n_particles, rho, eps, seed):
    X_center, d = build_center_particles(nominal_path, n_particles)
    alpha = np.ones(n_particles) / n_particles
    beta = np.ones(n_particles) / n_particles

    rng = np.random.default_rng(seed)
    direction = rng.standard_normal((n_particles, d))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    Y_pert, t_star, cost = find_sinkhorn_step(
        X_center, direction, rho, sinkhorn_discrepancy_refnu, eps, alpha, beta
    )
    return Y_pert.T, t_star, cost


def parse_floats(csv_text):
    return [float(x.strip()) for x in csv_text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate Sinkhorn-ball adversarial test sets without touching existing scripts."
    )
    parser.add_argument("--nominal", type=str, default="nominal.mat")
    parser.add_argument("--out-dir", type=str, default="adversarial_sets")
    parser.add_argument("--n-particles", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--eps-list", type=str, default="1e-5,1.2e-5,1.5e-5,2e-5")
    parser.add_argument("--rho-scales", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    args = parser.parse_args()

    eps_list = parse_floats(args.eps_list)
    rho_scales = parse_floats(args.rho_scales)
    seeds = [int(float(x.strip())) for x in args.seeds.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for eps in eps_list:
        for scale in rho_scales:
            rho_target = args.rho * scale
            for seed in seeds:
                X_test, t_star, realized_cost = generate_one(
                    nominal_path=args.nominal,
                    n_particles=args.n_particles,
                    rho=rho_target,
                    eps=eps,
                    seed=seed,
                )

                name = f"tester_eps{eps:.6g}_rhos{scale:.3f}_seed{seed}.mat"
                out_path = out_dir / name
                savemat(out_path, {"X_test": X_test})

                manifest_rows.append((str(out_path), eps, scale, rho_target, seed, t_star, realized_cost))
                print(
                    f"Saved {out_path.name} | eps={eps:.3e} rho={rho_target:.3e} seed={seed} "
                    f"t*={t_star:.4f} cost={realized_cost:.4f}"
                )

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("file,eps,rho_scale,rho_target,seed,t_star,sinkhorn_cost\n")
        for row in manifest_rows:
            f.write(
                f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]}\n"
            )

    print(f"\nManifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
