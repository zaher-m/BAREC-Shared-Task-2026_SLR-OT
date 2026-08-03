"""exp018 step 10: estimate the peer's error variance on blind without any gold labels.

Everything so far brackets the blend over an unknown c = var(peer error) / var(base error),
because the peer's accuracy on blind cannot be measured directly. But c might be recoverable
from the disagreement structure alone.

For predictors of the same target, pred_j = t + e_j, so

    D_jk = E[(pred_j - pred_k)^2] = E[(e_j - e_k)^2]

which needs no labels. Assume a one-factor error model, e_j = f_j g + u_j, i.e. one shared
error direction, which is plausible for 48 encoders trained on the same corpus plus one
outsider. Then

    D_jk = psi_j + psi_k + (f_j - f_k)^2 ,        psi_j = var(u_j),  v_j = psi_j + f_j^2

With 48 members that is 1,128 equations for ~96 parameters. What is not identified: D only
depends on differences of f, so the common level of f is invisible. An error component shared
equally by every predictor cancels out of every pairwise distance. That is the limit of
label-free estimation here, and it is why one anchor is needed. The anchor is the ensemble's
ensemble's error variance on blind, which follows from its measured 85.50 QWK.

The peer's loading f_p is identified relative to the members', because D(peer, member) is much larger
than D(member, member), which is the 0.76 vs 0.95 correlation gap.

Validated on public test first: run it there using predictions only, then compare the recovered
v_j against the true v_j from gold. If it cannot recover those, it cannot be trusted on blind.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from slra_ot.paths import REGENERATED  # noqa: E402

# re-runs write here, never over the committed record
OUT_DIR = os.path.join(str(REGENERATED), os.path.basename(HERE))
os.makedirs(OUT_DIR, exist_ok=True)

from slra_ot import combiners as C  # noqa: E402
from slra_ot.members import aligned, blind_matrix, pinned_members  # noqa: E402

AD = ("_ad", "_ad2", "_ad3", "_ad4")


def pairwise_D(U):
    """D[j,k] = mean((U[:,j] - U[:,k])^2). No labels involved."""
    n, m = U.shape
    q = (U * U).mean(0)
    G = U.T @ U / n
    return q[:, None] + q[None, :] - 2 * G


def fit_factor(D, iters=4000, lr=0.05, seed=0):
    """Least-squares fit of D_jk = psi_j + psi_k + (f_j - f_k)^2 on the off-diagonal.

    psi is clipped at 0 and f is centred each step to pin down the unidentified shift. Plain
    gradient descent is fine, the problem is small (m ~ 50) and smooth.
    """
    m = D.shape[0]
    rng = np.random.default_rng(seed)
    off = ~np.eye(m, dtype=bool)
    psi = np.full(m, D[off].mean() / 2)
    f = rng.normal(0, 0.1, m)
    for it in range(iters):
        Dh = psi[:, None] + psi[None, :] + (f[:, None] - f[None, :]) ** 2
        R = (Dh - D) * off
        gpsi = 2 * R.sum(1)
        diff = f[:, None] - f[None, :]
        gf = 2 * (R * 2 * diff).sum(1) * 2
        psi = np.maximum(psi - lr * gpsi / m / m * m, 1e-6)
        f = f - lr * gf / m / m * m
        f = f - f.mean()
    Dh = psi[:, None] + psi[None, :] + (f[:, None] - f[None, :]) ** 2
    rms = float(np.sqrt(((Dh - D)[off] ** 2).mean()))
    return psi, f, rms


def solve_shift(psi, f, w, v_target):
    """Choose the common level of f so the ensemble's variance comes out at v_target."""
    Fbar = float(w @ f)
    rem = v_target - float((w ** 2) @ psi)
    if rem <= 0:
        return None, rem
    return float(np.sqrt(rem) - Fbar), rem


def main():
    members = [m for m in pinned_members(HERE) if not m["tag"].endswith(AD)]
    tags = [m["tag"] for m in members]
    ids, Xt, yt = aligned(members, "test")

    # put every member on the label scale with its own linear fit against test gold
    ab = np.array([np.polyfit(Xt[:, j], yt, 1) for j in range(len(tags))])
    Ut = Xt * ab[:, 0] + ab[:, 1]
    v_true = np.array([np.var(yt - Ut[:, j]) for j in range(len(tags))])

    # ---------- VALIDATION on the public test split ---------------------------
    print("validation: recover v_j on public test from predictions only")
    D = pairwise_D(Ut)
    psi, f, rms = fit_factor(D)
    print(f"  one-factor fit to {len(tags)}x{len(tags)} disagreement matrix: "
          f"rms residual {rms:.4f} against mean D {D[~np.eye(len(tags),dtype=bool)].mean():.4f}")

    W = C.GreedyMultiset(rounds=30).fit(Xt, yt).weights_
    # the ensemble on the label scale, and the error variance that can be measured
    s = Xt @ W
    a_o, b_o = np.polyfit(s, yt, 1)
    v_o_true = float(np.var(yt - (a_o * s + b_o)))
    # weights expressed against the label-scale member columns
    wl = W * ab[:, 0]
    wl = wl / wl.sum()
    sh, rem = solve_shift(psi, f, wl, v_o_true)
    print(f"  anchor: true ensemble v_o {v_o_true:.4f}  ->  common f level {sh:.4f}")
    v_hat = psi + (f + sh) ** 2
    use = W > 1e-9
    err = (v_hat - v_true) / v_true
    print(f"  recovery of v_j over all {len(tags)} members: "
          f"median |rel err| {100*np.median(np.abs(err)):.1f}%  "
          f"mean {100*np.mean(err):+.1f}%")
    print(f"  over the {use.sum()} weighted members:            "
          f"median |rel err| {100*np.median(np.abs(err[use])):.1f}%  "
          f"mean {100*np.mean(err[use]):+.1f}%")
    print(f"  {'member':34s} {'v_true':>8s} {'v_hat':>8s} {'rel':>7s}")
    order = np.argsort(-W)
    for j in order[:6]:
        print(f"  {tags[j]:34s} {v_true[j]:8.4f} {v_hat[j]:8.4f} {100*err[j]:+6.1f}%")
    ok = np.median(np.abs(err[use])) < 0.15

    # ---------- APPLY on blind ------------------------------------------------
    print(f"\nAPPLY on blind")
    peer = np.load(os.path.join(HERE, "peer_blind_nogold.npy"))
    cov = ~np.isnan(peer)
    Xb = blind_matrix(tags)
    Ub = (Xb * ab[:, 0] + ab[:, 1])[cov]
    # the peer is already on the label scale (it is a level), de-meaned onto the base
    pv = peer[cov] - peer[cov].mean() + Ub.mean()
    Uall = np.column_stack([Ub, pv])
    print(f"  {cov.sum()} covered blind rows, {Uall.shape[1]} columns "
          f"({len(tags)} members + peer)")

    Db = pairwise_D(Uall)
    psib, fb, rmsb = fit_factor(Db)
    m = Uall.shape[1]
    offb = ~np.eye(m, dtype=bool)
    print(f"  one-factor fit: rms residual {rmsb:.4f} against mean D {Db[offb].mean():.4f}")
    print(f"  mean D(member,member) {Db[:len(tags),:len(tags)][~np.eye(len(tags),dtype=bool)].mean():.4f}"
          f"   mean D(member,peer) {Db[:len(tags),-1].mean():.4f}")

    # anchor on the ensemble's blind error variance, implied by its measured 85.50 QWK.
    # (1 - QWK) moves with error variance at comparable marginals, so scale the measured
    # test value by the ratio of (1 - QWK).
    QWK_TEST, QWK_BLIND = 86.94, 85.50
    v_o_blind = v_o_true * (1 - QWK_BLIND / 100) / (1 - QWK_TEST / 100)
    print(f"  anchor: v_o on blind = {v_o_true:.4f} x "
          f"{(1-QWK_BLIND/100)/(1-QWK_TEST/100):.4f} = {v_o_blind:.4f}")
    wl2 = np.concatenate([wl, [0.0]])
    shb, remb = solve_shift(psib, fb, wl2, v_o_blind)
    if shb is None:
        raise SystemExit(f"anchor infeasible (residual {remb:.4f})")
    v_hat_b = psib + (fb + shb) ** 2
    v_peer = float(v_hat_b[-1])
    c_hat = v_peer / v_o_blind
    peer_qwk = 100 * (1 - (1 - QWK_BLIND / 100) * c_hat)
    print(f"\n  estimated peer error variance on blind  v_p = {v_peer:.4f}")
    print(f"  the ensemble                            v_o = {v_o_blind:.4f}")
    print(f"  ==> c = {c_hat:.3f}   (peer alone would score about {peer_qwk:.1f} QWK)")

    # what that c implies for the blend
    Dpo = float(Db[:len(tags), -1] @ wl)          # weighted member-to-peer distance
    # use the ensemble-vs-peer distance directly
    uo = Ub @ wl
    Dop = float(np.mean((uo - pv) ** 2))
    rho = (v_o_blind + v_peer - Dop) / (2 * np.sqrt(v_o_blind * v_peer))
    print(f"  D(ensemble, peer) = {Dop:.4f}  ->  error correlation rho = {rho:.3f}")
    w_star = (v_o_blind - rho * np.sqrt(v_o_blind * v_peer)) / (v_o_blind + v_peer - 2 * rho * np.sqrt(v_o_blind * v_peer))
    print(f"  variance-optimal peer weight w* = {w_star:.3f}")
    print(f"\n{'w':>6s} {'var ratio':>10s} {'blind QWK':>10s}")
    covf = float(cov.mean())
    best = None
    for wv in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, float(np.clip(w_star,0,1))]:
        vr = ((1 - wv) ** 2 * v_o_blind + wv ** 2 * v_peer
              + 2 * wv * (1 - wv) * rho * np.sqrt(v_o_blind * v_peer)) / v_o_blind
        tot = covf * vr + (1 - covf)
        q = 100 * (1 - (1 - QWK_BLIND / 100) * tot)
        if best is None or q > best[1]:
            best = (wv, q)
        print(f"{wv:6.3f} {vr:10.4f} {q:10.2f}")
    print(f"\nbest w {best[0]:.3f} -> {best[1]:.2f}  (+{best[1]-QWK_BLIND:.2f} over 85.50, "
          f"before channel A)")
    print(f"validation verdict: {'procedure recovers v_j, estimate is usable' if ok else 'poor recovery, treat c as unreliable and keep the bracket'}")

    json.dump({"validation": {"rms": rms, "median_rel_err_weighted": float(np.median(np.abs(err[use]))),
                              "usable": bool(ok)},
               "blind": {"rms": rmsb, "v_peer": v_peer, "v_o": v_o_blind, "c": c_hat,
                         "implied_peer_qwk": peer_qwk, "rho": float(rho),
                         "w_star": float(w_star), "coverage": covf,
                         "best_w": best[0], "projected_qwk": best[1]}},
              open(os.path.join(OUT_DIR, "s10_estimate_c.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
