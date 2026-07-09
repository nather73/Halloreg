"""
core.pymdp_backend
==================

**옵션(optional) pymdp 1.0.x (JAX 백엔드) 계산 경로.**

기본 계산은 `core.generative` 의 해석적 numpy EFE 를 사용한다(A=I₄·policy_len=1
하에서 pymdp EFE 와 소수점까지 동일함이 검증됨). 그러나 과제 요구사항은 "pymdp
1.0.x(JAX)를 반드시 사용할 수 있어야 한다"이므로, 본 모듈은 첨부 프로토타입
`empathic_ipd_pymdp.py` 가 확립한 패턴을 그대로 캡슐화하여 pymdp 를 *실제로* 사용하는
백엔드를 제공한다:

  * 생성 모형 A/B/C/D 를 선두 batch 차원(=1)의 jax 배열 list 로 구성.
  * pymdp `Agent(..., policy_len=1, action_selection="stochastic")`.
  * 조망수용(perspective-taking): equinox 불변 모듈에 `eqx.tree_at` 으로 C 를
    상대 보수로 교체한 pt_agent 를 만들고, B 는 라운드마다 ToM 예측 pc 로 교체.
  * `infer_policies(qs)` 가 반환하는 neg_efe 로 G_self / G_other 를 얻는다.

[중요] JAX 는 import 시점에 가속기(GPU/TPU) 를 탐색하며, 일부 환경에서는 이 탐색이
행(hang) 을 유발한다. 따라서 본 모듈은 **jax import 이전에** `JAX_PLATFORMS=cpu` 를
설정한다(Windows/AMD 환경 포함 CPU 전용 동작; 사용자 지정값은 존중).

사용:
    from HalloReg.core.pymdp_backend import PymdpEFE
    efe = PymdpEFE()
    g_self  = efe.G_self(pc=0.7)     # shape (2,) = [G_C, G_D]
    g_other = efe.G_other(pc=0.7)
    PymdpEFE.check_equivalence()     # numpy 경로와의 등가성 자동 검증
"""

from __future__ import annotations

import os

# --- jax import 이전 CPU 강제 (가속기 탐색 hang 방지) ---
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import numpy as np

from .constants import CC, CD, DC, DD, COOP, DEFECT, PAYOFF_SELF, PAYOFF_OTHER

_PYMDP_AVAILABLE = None
_IMPORT_ERROR = None


def pymdp_available() -> bool:
    """pymdp(JAX) 를 import 할 수 있는지 지연 확인."""
    global _PYMDP_AVAILABLE, _IMPORT_ERROR
    if _PYMDP_AVAILABLE is None:
        try:
            import jax.numpy as _  # noqa: F401
            import equinox as _e    # noqa: F401
            from pymdp.agent import Agent  # noqa: F401
            _PYMDP_AVAILABLE = True
        except Exception as exc:  # pragma: no cover
            _PYMDP_AVAILABLE = False
            _IMPORT_ERROR = exc
    return _PYMDP_AVAILABLE


def _B_from_pc(pc: float):
    """행동조건부 전이 B[0] : (batch=1, s', s, a)."""
    import jax.numpy as jnp
    pc = float(pc)
    pd = 1.0 - pc
    B0 = np.zeros((4, 4, 2))
    for s in range(4):
        B0[CC, s, COOP] = pc
        B0[CD, s, COOP] = pd
        B0[DC, s, DEFECT] = pc
        B0[DD, s, DEFECT] = pd
    return [jnp.asarray(B0)[None, ...]]


class PymdpEFE:
    """
    pymdp 1.0.x(JAX) 로 base EFE(G_self, G_other)를 계산하는 백엔드.

    첫 호출 시 JAX JIT 컴파일로 ~1초가 소요되고, 이후 호출은 ~0.25초/회다.
    상태추론(perception)이 필요하면 `infer_states` 헬퍼를 함께 제공한다.
    """

    def __init__(self, prior_opp_coop: float = 0.5):
        if not pymdp_available():
            raise ImportError(
                "pymdp(JAX) 를 불러올 수 없습니다. `pip install inferactively-pymdp` "
                f"후 재시도하세요. (원인: {_IMPORT_ERROR})"
            )
        import jax.numpy as jnp
        import equinox as eqx
        from pymdp.agent import Agent

        self._jnp = jnp
        self._eqx = eqx

        A = [jnp.eye(4)[None, ...]]
        B = _B_from_pc(prior_opp_coop)
        C_self = [jnp.asarray(PAYOFF_SELF)[None, ...]]
        C_other = [jnp.asarray(PAYOFF_OTHER)[None, ...]]
        D = [jnp.ones(4)[None, ...] / 4.0]

        self._agent = Agent(A=A, B=B, C=C_self, D=D, batch_size=1,
                            policy_len=1, gamma=1.0,
                            action_selection="stochastic")
        # 조망수용 에이전트: 선호만 상대 보수로 교체
        self._pt_agent = eqx.tree_at(lambda a: a.C, self._agent, C_other)
        self.D = self._agent.D

    # ---- base EFE ----
    def _neg_efe(self, agent, pc: float, qs=None) -> np.ndarray:
        eqx = self._eqx
        ag = eqx.tree_at(lambda a: a.B, agent, _B_from_pc(pc))
        if qs is None:
            qs = [d[:, None, :] for d in self._agent.D]
        _, neg_efe = ag.infer_policies(qs)
        return np.asarray(neg_efe)[0]  # (n_policies,) = [C, D]

    def G_self(self, pc: float, qs=None) -> np.ndarray:
        """내 관점 EFE (작을수록 선호). shape (2,)."""
        return -self._neg_efe(self._agent, pc, qs)

    def G_other(self, pc: float, qs=None) -> np.ndarray:
        """조망수용 EFE (선호=상대보수). shape (2,)."""
        return -self._neg_efe(self._pt_agent, pc, qs)

    # ---- perception (변분 상태추론) ----
    def infer_states(self, observed_state: int, empirical_prior=None):
        """joint outcome 관측으로 변분 상태추론. empirical_prior 미지정 시 D 사용."""
        jnp = self._jnp
        obs = [jnp.array([observed_state])]
        prior = empirical_prior if empirical_prior is not None else self._agent.D
        return self._agent.infer_states(obs, empirical_prior=prior)

    def update_empirical_prior(self, my_last_action: int, qs):
        """직전 내 행동으로 다음 라운드 사전 전파 (함수형 API)."""
        jnp = self._jnp
        act = jnp.array([[my_last_action]])
        return self._agent.update_empirical_prior(act, qs)

    # ---- 등가성 검증 ----
    @staticmethod
    def check_equivalence(pcs=(0.1, 0.5, 0.8), atol: float = 1e-4) -> bool:
        """
        pymdp EFE 와 numpy 해석적 EFE 의 등가성을 검증한다.
        불일치 시 AssertionError.
        """
        from . import generative as gen
        efe = PymdpEFE()
        for pc in pcs:
            gs_pymdp = efe.G_self(pc)
            go_pymdp = efe.G_other(pc)
            gs_np = gen.G_self(pc)
            go_np = gen.G_other_perspective(pc)
            assert np.allclose(gs_pymdp, gs_np, atol=atol), (
                f"G_self mismatch @pc={pc}: pymdp={gs_pymdp} numpy={gs_np}")
            assert np.allclose(go_pymdp, go_np, atol=atol), (
                f"G_other mismatch @pc={pc}: pymdp={go_pymdp} numpy={go_np}")
        return True


if __name__ == "__main__":
    # 등가성 자가검증
    if pymdp_available():
        ok = PymdpEFE.check_equivalence()
        print("pymdp <-> numpy EFE 등가성 검증:", "통과" if ok else "실패")
    else:
        print("pymdp(JAX) 미설치 — numpy 경로만 사용 가능:", _IMPORT_ERROR)
