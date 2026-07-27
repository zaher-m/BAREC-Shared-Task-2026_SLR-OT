"""BAREC metrics: QWK plus the secondary numbers the leaderboard reports.

Same definitions as the shared task's eval.py, so local scores match the leaderboard.
"""
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error

# level collapse maps, copied from the official eval.py
BAREC_7 = {1:1,2:1,3:1,4:1,5:2,6:2,7:2,8:3,9:3,10:4,11:4,12:5,13:5,14:6,15:6,16:7,17:7,18:7,19:7}
BAREC_5 = {1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:2,9:2,10:2,11:2,12:3,13:3,14:4,15:4,16:5,17:5,18:5,19:5}
BAREC_3 = {1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:2,13:2,14:3,15:3,16:3,17:3,18:3,19:3}

N_CLASSES = 19

# quadratic weights over levels 1..19, precomputed for fast_qwk
_W = (np.subtract.outer(np.arange(N_CLASSES), np.arange(N_CLASSES)) ** 2) / (N_CLASSES - 1) ** 2


def qwk(y_true, y_pred):
    return cohen_kappa_score(list(map(int, y_true)), list(map(int, y_pred)), weights="quadratic")


def fast_qwk(y_true0, y_pred0):
    """Same as qwk() but much faster. Labels must be 0-based (0..18)."""
    n = len(y_true0)
    O = (np.bincount(y_true0 * N_CLASSES + y_pred0, minlength=N_CLASSES ** 2)
         .reshape(N_CLASSES, N_CLASSES))
    ht = np.bincount(y_true0, minlength=N_CLASSES).astype(float)
    hp = np.bincount(y_pred0, minlength=N_CLASSES).astype(float)
    E = np.outer(ht, hp) / n
    num = (_W * O).sum()
    den = (_W * E).sum()
    return 1.0 - num / den if den > 0 else 0.0


def full_report(y_true, y_pred):
    y_true = [int(x) for x in y_true]
    y_pred = [int(x) for x in y_pred]
    return {
        "QWK": cohen_kappa_score(y_true, y_pred, weights="quadratic") * 100,
        "Acc19": accuracy_score(y_true, y_pred) * 100,
        "Acc7": accuracy_score([BAREC_7[l] for l in y_true], [BAREC_7[p] for p in y_pred]) * 100,
        "Acc5": accuracy_score([BAREC_5[l] for l in y_true], [BAREC_5[p] for p in y_pred]) * 100,
        "Acc3": accuracy_score([BAREC_3[l] for l in y_true], [BAREC_3[p] for p in y_pred]) * 100,
        "Adj+-1": float(np.mean([abs(p - l) <= 1 for p, l in zip(y_pred, y_true)])) * 100,
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def print_report(name, y_true, y_pred):
    r = full_report(y_true, y_pred)
    print(f"[{name}] QWK={r['QWK']:.4f} Acc19={r['Acc19']:.2f} Acc7={r['Acc7']:.2f} "
          f"Acc5={r['Acc5']:.2f} Acc3={r['Acc3']:.2f} Adj+-1={r['Adj+-1']:.2f} MAE={r['MAE']:.4f}")
    return r
