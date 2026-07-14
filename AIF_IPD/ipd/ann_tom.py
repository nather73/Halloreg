"""
ipd.ann_tom
===========

**학습된 인공신경망(ANN)이 베이지안 Theory-of-Mind 을 재현하는가?**

[동기 — 기존 한계]
HalloReg 의 ToM 은 입자필터(particle filter) 기반 베이지안 역추론이다(inversion.py).
이는 규범적(normative)이나 (i) 신경 구현의 그럴듯함(plausibility)과 (ii) 학습을 통한
창발 여부는 다루지 못한다. 본 모듈은 **입자필터 사후를 교사(teacher)로 하는 지식
증류(distillation)** 로, 두 신경망 구조가 베이지안 ToM 의 두 핵심 기능
    (a) 잠재 특성 θ=(α,ρ,β,λ_j) 추론
    (b) **특성별 신뢰도(reliability) 배정**
을 재현할 수 있는지 검증한다.

[구조 — 두 계열]
1. **Schwarcz-style RNN (GRU)** — Schwarcz et al. (2025):
   RL 로 훈련된 순환망이 베이즈-최적 믿음 갱신을 순환 동역학에 내재화함을 보였다.
   여기서는 관측 스트림을 GRU 로 처리해 매 라운드 θ 사후평균 + 특성별 로그정밀도를
   출력한다(단일 스트림에서 잠재상태·맥락 동시 추론).

2. **Kim-style GCN+RNN (관계형 + 스포트라이트 주의)** — Kim et al. (2026):
   affordance-가중 간선의 그래프합성곱(GCN)으로 관계 구조를 인코딩하고, **간선 엔트로피
   = 스포트라이트 주의**로 특성별 신뢰도/주의를 배정(dACC 대응). 여기서는 특성 노드가
   시간 맥락(GRU)에 대해 주의(attention)하고, 그 주의분포의 엔트로피(낮을수록 집중=고
   신뢰)를 **특성별 reliability** 로 산출한다. 특성 노드 간 1-hop 메시지 전달(학습된
   인접행렬=관계구조)로 상호 의존(예: β 불확실 → α 식별 저하)을 포착한다.

[검증]
  · 교사(입자필터 사후) 대비 특성 회복 RMSE/R² + 참(true) θ 대비 회복.
  · 보정(calibration): 예측 정밀도 ↔ 실제 오차(신뢰도가 실제 정확도를 반영하는가).
  · A-B-A 일반화: 상대 θ 가 A→B→A 로 전환할 때 특성 추론이 복구되는가(§ABA 연계).

JAX/Equinox 로 구현(사용자 스택). CPU 로 수 분 내 완주.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from AIF_IPD.core.constants import COOP, DEFECT, joint_index
from AIF_IPD.ipd.tom.inversion import OpponentInversion, ObservationContext, THETA_AXES

# θ 사전 범위(생성용) — 식별 가능한 넓은 범위
THETA_RANGES = {
    "alpha": (-1.5, 1.5), "rho": (-0.2, 1.6),
    "beta": (0.6, 6.0), "lambda_j": (0.0, 1.0),
}
_AX = list(THETA_AXES)


def _logistic(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


# ------------------------------------------------------------- 매개적 상대
@dataclass
class ParametricOpponent:
    """
    입자필터가 가정하는 것과 **동일한** 행동 생성모형으로 행동하는 상대(참 θ 보유):
        P(a=C|h,θ) = σ(β·(α + ρ·f + (5λ_j − p − 1)))
    f: 내 직전 행동 호혜신호(+1 협력/−1 배신/0), p: 내 협력율에 대한 상대 믿음(추적).
    이로써 교사(입자필터)의 회복 가능성과 ANN 의 재현을 공정히 평가할 수 있다.
    """
    alpha: float
    rho: float
    beta: float
    lambda_j: float
    seed: int = 0
    my_last: Optional[int] = None
    my_coop_sum: float = 0.0
    my_n: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def _f(self) -> float:
        if self.my_last is None:
            return 0.0
        return 1.0 - 2.0 * self.my_last

    @property
    def my_coop_rate(self) -> float:
        return self.my_coop_sum / self.my_n if self.my_n else 0.5

    def act(self) -> int:
        logit = self.beta * (self.alpha + self.rho * self._f()
                             + (5.0 * self.lambda_j - self.my_coop_rate - 1.0))
        pC = float(_logistic(np.array(logit)))
        return COOP if self.rng.random() < pC else DEFECT

    def observe(self, my_action: int):
        self.my_last = int(my_action)
        self.my_coop_sum += 1.0 if my_action == COOP else 0.0
        self.my_n += 1


@dataclass
class ProbingFocal:
    """
    식별성(identifiability)을 위해 탐색적으로 행동하는 focal: 관대한 TFT + ε-탐색.
    ρ(호혜성) 식별에는 내 행동의 변동이 필요하므로 무작위 탐색을 섞는다.
    """
    eps: float = 0.25
    seed: int = 0
    opp_last: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def act(self) -> int:
        if self.rng.random() < self.eps:
            return COOP if self.rng.random() < 0.5 else DEFECT
        if self.opp_last is None:
            return COOP
        return COOP if self.opp_last == COOP or self.rng.random() < 0.2 else DEFECT

    def observe(self, opp_action: int):
        self.opp_last = int(opp_action)


# ------------------------------------------------------- 라운드 특성 인코딩
N_FEAT = 8


def _encode_round(my_last: Optional[int], opp_action: int, my_action: int,
                  t: int, T: int) -> np.ndarray:
    """ANN 입력 특성(라운드별): 최소 마르코프 관측 + 호혜신호 + 시점."""
    st = joint_index(my_action, opp_action)
    onehot = np.zeros(4); onehot[st] = 1.0
    f = 0.0 if my_last is None else (1.0 - 2.0 * my_last)
    return np.array([float(opp_action), 0.0 if my_last is None else float(my_last),
                     f, *onehot, (t + 1) / T], dtype=np.float32)


# ---------------------------------------------------------- 데이터 생성
def _theta_scaler(theta_true: np.ndarray):
    """참 θ 분포로 표준화 스케일러(축별 mean/std)."""
    mu = theta_true.mean(axis=0); sd = theta_true.std(axis=0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def generate_dataset(n_dyads: int, n_rounds: int, seed: int,
                     n_particles: int = 300,
                     theta_schedule: Optional[Callable[[int, int, np.random.Generator], np.ndarray]] = None
                     ) -> Dict:
    """
    n_dyads 개 다이애드 생성. 반환:
      X        (n_dyads, T, N_FEAT)   관측 스트림
      Y_mean   (n_dyads, T, 4)        교사(입자필터) θ 사후평균
      Y_lprec  (n_dyads, T, 4)        교사 특성별 로그정밀도(=−log std) — reliability 목표
      theta    (n_dyads, T, 4)        참 θ(정적이면 브로드캐스트; A-B-A 면 시변)
    theta_schedule(t,T,rng) 를 주면 라운드별 참 θ(비정상 A-B-A) 생성.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_dyads, n_rounds, N_FEAT), np.float32)
    Ym = np.zeros((n_dyads, n_rounds, 4), np.float32)
    Yp = np.zeros((n_dyads, n_rounds, 4), np.float32)
    TH = np.zeros((n_dyads, n_rounds, 4), np.float32)

    for d in range(n_dyads):
        if theta_schedule is None:
            th0 = np.array([rng.uniform(*THETA_RANGES[a]) for a in _AX], np.float32)
            theta_of_t = lambda t: th0
        else:
            theta_of_t = lambda t: theta_schedule(t, n_rounds, rng)

        opp = None
        focal = ProbingFocal(eps=0.25, seed=int(rng.integers(1 << 30)))
        teacher = OpponentInversion(n_particles=n_particles,
                                    seed=int(rng.integers(1 << 30)))
        my_last = None
        for t in range(n_rounds):
            th = theta_of_t(t)
            # 상대 상태(호혜/협력율)는 유지하되 θ 만 스케줄에 따라 교체
            if opp is None:
                opp = ParametricOpponent(*[float(v) for v in th],
                                         seed=int(rng.integers(1 << 30)))
            else:
                opp.alpha, opp.rho, opp.beta, opp.lambda_j = [float(v) for v in th]

            my_a = focal.act()
            opp_a = opp.act()
            focal.observe(opp_a); opp.observe(my_a)

            X[d, t] = _encode_round(my_last, opp_a, my_a, t, n_rounds)
            ctx = ObservationContext(my_last_action=my_last, their_last_action=opp_a,
                                     joint_outcome=joint_index(my_a, opp_a),
                                     round_number=t)
            # 교사에 내 협력율 반영(상대 공감 항 계산에 필요)
            teacher.my_cooperation_rate = opp.my_coop_rate
            teacher.update(opp_a, ctx)
            m = teacher.posterior_means(); s = teacher.posterior_stds()
            Ym[d, t] = [m[a] for a in _AX]
            Yp[d, t] = [-np.log(max(s[a], 1e-3)) for a in _AX]
            TH[d, t] = th
            my_last = my_a
    return {"X": X, "Y_mean": Ym, "Y_lprec": Yp, "theta": TH,
            "n_rounds": n_rounds}


# ================================================================== 모델
class SchwarczGRU(eqx.Module):
    """관측 스트림 → GRU → 매 스텝 (θ 사후평균 4, 로그정밀도 4)."""
    gru: eqx.nn.GRUCell
    head_mean: eqx.nn.MLP
    head_lprec: eqx.nn.MLP
    h0: jax.Array
    hidden: int = eqx.field(static=True)

    def __init__(self, key, n_feat=N_FEAT, hidden=48):
        k1, k2, k3 = jax.random.split(key, 3)
        self.gru = eqx.nn.GRUCell(n_feat, hidden, key=k1)
        self.head_mean = eqx.nn.MLP(hidden, 4, width_size=hidden, depth=1,
                                    activation=jax.nn.tanh, key=k2)
        self.head_lprec = eqx.nn.MLP(hidden, 4, width_size=hidden, depth=1,
                                     activation=jax.nn.tanh, key=k3)
        self.h0 = jnp.zeros(hidden)
        self.hidden = hidden

    def __call__(self, x_seq):
        def step(h, x):
            h = self.gru(x, h)
            return h, (self.head_mean(h), self.head_lprec(h))
        _, (means, lprecs) = jax.lax.scan(step, self.h0, x_seq)
        return means, lprecs, None


class KimGCNRNN(eqx.Module):
    """
    관계형(GCN) + 순환(RNN) + 스포트라이트 주의.
    GRU 로 시간 맥락 h_t 산출 → 4개 특성 질의(query)가 h_t 로부터 유도된 key/value 에
    주의 → 주의분포 엔트로피 = 특성별 reliability(간선 엔트로피). 특성 노드 간 학습된
    인접행렬로 1-hop 메시지 전달(관계 구조). 노드별 readout → (mean, lprec).
    """
    gru: eqx.nn.GRUCell
    q: jax.Array                     # (4, dq) 특성 질의
    key_proj: eqx.nn.Linear
    val_proj: eqx.nn.Linear
    adj: jax.Array                   # (4,4) 학습 인접(관계구조)
    msg: eqx.nn.Linear
    read_mean: eqx.nn.MLP
    read_lprec: eqx.nn.MLP
    h0: jax.Array
    n_key: int = eqx.field(static=True)
    dq: int = eqx.field(static=True)

    def __init__(self, key, n_feat=N_FEAT, hidden=48, dq=24, n_key=6):
        ks = jax.random.split(key, 8)
        self.gru = eqx.nn.GRUCell(n_feat, hidden, key=ks[0])
        self.q = 0.1 * jax.random.normal(ks[1], (4, dq))
        self.key_proj = eqx.nn.Linear(hidden, n_key * dq, key=ks[2])
        self.val_proj = eqx.nn.Linear(hidden, n_key * dq, key=ks[3])
        self.adj = 0.1 * jax.random.normal(ks[4], (4, 4))
        self.msg = eqx.nn.Linear(dq, dq, key=ks[5])
        self.read_mean = eqx.nn.MLP(dq, 1, width_size=dq, depth=1,
                                    activation=jax.nn.tanh, key=ks[6])
        self.read_lprec = eqx.nn.MLP(dq, 1, width_size=dq, depth=1,
                                     activation=jax.nn.tanh, key=ks[7])
        self.h0 = jnp.zeros(hidden)
        self.n_key = n_key
        self.dq = dq

    def _readout(self, h):
        # h_t → n_key 개 key/value 노드 (affordance 증거)
        K = self.key_proj(h).reshape(self.n_key, self.dq)
        V = self.val_proj(h).reshape(self.n_key, self.dq)
        # 특성 질의 × key → 주의 로짓 (4, n_key)
        logits = (self.q @ K.T) / jnp.sqrt(self.dq)
        attn = jax.nn.softmax(logits, axis=-1)                  # (4, n_key)
        # 간선 엔트로피 → reliability(낮은 엔트로피=집중=고신뢰): rel=1−H/logK
        ent = -jnp.sum(attn * jnp.log(attn + 1e-9), axis=-1)
        rel = 1.0 - ent / jnp.log(self.n_key)                   # (4,)
        node = attn @ V                                          # (4, dq) 특성 노드값
        # 특성 노드 간 1-hop 메시지 전달(학습 인접 = 관계 구조)
        A = jax.nn.softmax(self.adj, axis=-1)
        node = node + jax.nn.tanh(A @ jax.vmap(self.msg)(node))
        mean = jax.vmap(self.read_mean)(node).squeeze(-1)       # (4,)
        lprec = jax.vmap(self.read_lprec)(node).squeeze(-1)     # (4,)
        return mean, lprec, rel

    def __call__(self, x_seq):
        def step(h, x):
            h = self.gru(x, h)
            mean, lprec, rel = self._readout(h)
            return h, (mean, lprec, rel)
        _, (means, lprecs, rels) = jax.lax.scan(step, self.h0, x_seq)
        return means, lprecs, rels


# ================================================================== 학습
def _standardize_targets(Y_mean, scaler):
    mu, sd = scaler
    return (Y_mean - mu) / sd


def _destandardize(pred, scaler):
    mu, sd = scaler
    return pred * sd + mu


def train_model(model, data, scaler, key, epochs=45, lr=3e-3, batch=32,
                warm=10, verbose=True, tag=""):
    """
    교사 사후평균(표준화) 회귀 + 정밀도 헤드의 가우시안 NLL(불확실성 보정) 학습.
    라운드 warm 이전은 사후가 미성숙하므로 손실에서 가중 축소.
    """
    X = jnp.asarray(data["X"])
    Ym = jnp.asarray(_standardize_targets(data["Y_mean"], scaler))
    T = X.shape[1]
    w_t = jnp.asarray((np.arange(T) >= warm).astype(np.float32))  # (T,)

    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    def seq_loss(model, x, ym):
        means, lprecs, _ = model(x)                              # (T,4)
        lprecs = jnp.clip(lprecs, -3.0, 3.0)
        # 가우시안 NLL(표준화 공간): 0.5*(exp(2lp)*(y-μ)² ... ) — 간이형
        prec = jnp.exp(lprecs)
        nll = 0.5 * (prec ** 2 * (ym - means) ** 2) - lprecs
        return jnp.mean(w_t[:, None] * nll)

    @eqx.filter_jit
    def batch_loss(model, xb, yb):
        return jnp.mean(jax.vmap(lambda x, y: seq_loss(model, x, y))(xb, yb))

    @eqx.filter_jit
    def step(model, opt_state, xb, yb):
        loss, grads = eqx.filter_value_and_grad(batch_loss)(model, xb, yb)
        updates, opt_state = opt.update(grads, opt_state,
                                        eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    n = X.shape[0]
    hist = []
    for ep in range(epochs):
        key, sk = jax.random.split(key)
        perm = np.asarray(jax.random.permutation(sk, n))
        ep_loss = 0.0; nb = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            model, opt_state, loss = step(model, opt_state, X[idx], Ym[idx])
            ep_loss += float(loss); nb += 1
        hist.append(ep_loss / max(nb, 1))
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  [{tag}] epoch {ep:3d}  loss={hist[-1]:.4f}")
    return model, hist


# ================================================================== 평가
def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def predict(model, X):
    """(n, T, ·) 예측 (표준화 공간) — means, lprecs, rels."""
    def one(x):
        m, lp, rel = model(x)
        return m, lp, (rel if rel is not None else jnp.zeros_like(m))
    means, lprecs, rels = jax.vmap(one)(jnp.asarray(X))
    return np.asarray(means), np.asarray(lprecs), np.asarray(rels)


def evaluate(model, data, scaler, eval_from: int = 15) -> Dict:
    """
    교사·참 θ 대비 특성 회복 RMSE/R²(표준화 후반 라운드 평균) + 보정.
    eval_from: 사후 성숙 이후 라운드만 평가.
    """
    means_s, lprecs, rels = predict(model, data["X"])
    means = _destandardize(means_s, scaler)
    T = data["X"].shape[1]
    sl = slice(eval_from, T)
    teach = data["Y_mean"][:, sl]           # (n, T', 4)
    true = data["theta"][:, sl]
    pred = means[:, sl]
    out = {"per_trait": {}, "eval_from": eval_from}
    for j, ax in enumerate(_AX):
        p = pred[:, :, j].ravel()
        t_teach = teach[:, :, j].ravel(); t_true = true[:, :, j].ravel()
        out["per_trait"][ax] = {
            "rmse_teacher": float(np.sqrt(np.mean((p - t_teach) ** 2))),
            "r2_teacher": _r2(t_teach, p),
            "rmse_true": float(np.sqrt(np.mean((p - t_true) ** 2))),
            "r2_true": _r2(t_true, p),
        }
    # 보정: 예측 정밀도(로그) ↔ 실제 |오차| (상관이 음수여야 = 고정밀→저오차)
    lp = lprecs[:, sl].reshape(-1, 4)
    err = np.abs(_destandardize(means_s, scaler)[:, sl] - true).reshape(-1, 4)
    calib = {}
    for j, ax in enumerate(_AX):
        if lp[:, j].std() > 1e-6:
            calib[ax] = float(np.corrcoef(lp[:, j], err[:, j])[0, 1])
        else:
            calib[ax] = float("nan")
    out["calibration_lprec_vs_abserr"] = calib
    out["mean_r2_teacher"] = float(np.mean([out["per_trait"][a]["r2_teacher"] for a in _AX]))
    out["mean_r2_true"] = float(np.mean([out["per_trait"][a]["r2_true"] for a in _AX]))
    # 식별가능 합성: 행동모형에서 α 와 5λ_j 는 가법 축퇴 → α+5λ_j 만 식별가능.
    # 개별 참값 회복이 낮아도 이 합성은 잘 회복되어야 한다(생성모형 성질의 정직한 반영).
    ia = _AX.index("alpha"); il = _AX.index("lambda_j")
    comp_pred = (pred[:, :, ia] + 5.0 * pred[:, :, il]).ravel()
    comp_teach = (teach[:, :, ia] + 5.0 * teach[:, :, il]).ravel()
    comp_true = (true[:, :, ia] + 5.0 * true[:, :, il]).ravel()
    out["identifiable_composite"] = {
        "name": "alpha+5*lambda_j",
        "r2_teacher": _r2(comp_teach, comp_pred),
        "r2_true": _r2(comp_true, comp_pred),
        "rmse_true": float(np.sqrt(np.mean((comp_pred - comp_true) ** 2)))}
    return out


# ------------------------------------------------------ A-B-A 일반화
def make_aba_theta_schedule(theta_A: np.ndarray, theta_B: np.ndarray, T: int):
    """A→B→A 참 θ 스케줄(세 등분)."""
    third = max(T // 3, 1)

    def sched(t, TT, rng):
        return theta_A if (t < third or t >= 2 * third) else theta_B
    return sched


def aba_generalization(models: Dict[str, object], scaler, seed: int = 7,
                       n_dyads: int = 40, n_rounds: int = 90) -> Dict:
    """
    A(협력적: 高α,高ρ,高λ)→B(착취적: 低α,低ρ,低λ)→A 로 상대 θ 가 전환할 때, ANN 의
    특성 추론이 교사와 함께 복구되는지. phase 별 회복오차·궤적 반환.
    """
    theta_A = np.array([1.0, 1.2, 3.0, 0.8], np.float32)   # 협력적
    theta_B = np.array([-1.0, 0.0, 3.0, 0.05], np.float32)  # 착취적
    sched = make_aba_theta_schedule(theta_A, theta_B, n_rounds)
    data = generate_dataset(n_dyads, n_rounds, seed, theta_schedule=sched)
    third = max(n_rounds // 3, 1)
    A1 = slice(0, third); A3 = slice(2 * third, n_rounds)
    res = {"theta_A": theta_A.tolist(), "theta_B": theta_B.tolist(),
           "third": third, "n_rounds": n_rounds, "models": {},
           "teacher_traj": data["Y_mean"].mean(axis=0).tolist(),
           "true_traj": data["theta"].mean(axis=0).tolist()}
    for name, model in models.items():
        means_s, _, _ = predict(model, data["X"])
        means = _destandardize(means_s, scaler)            # (n, T, 4)
        traj = means.mean(axis=0)                          # (T,4)
        # 복구오차: phase A3 특성 평균 − phase A1 특성 평균 의 |·| (축평균)
        a1 = means[:, A1].mean(axis=1); a3 = means[:, A3].mean(axis=1)
        rec_err = float(np.mean(np.abs(a3 - a1)))
        # 교사 대비 A3 정합(회복 후 교사와 얼마나 일치)
        teach_a3 = data["Y_mean"][:, A3].mean(axis=1)
        align = float(np.sqrt(np.mean((a3 - teach_a3) ** 2)))
        res["models"][name] = {"traj": traj.tolist(), "rec_err": rec_err,
                               "align_teacher_A3": align}
    return res
