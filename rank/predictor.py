"""
걸파 순위 예측 — 통합 코어 (신규).

백테스트(experiment/)에서 검증된 결과에 따라 예측 방식을 교체:
  기존: 직전 이벤트 1개에 GAM 외삽 + 잔차 ARIMA  (막판 급주를 과소예측)
  신규: 과거 다수 이벤트 기반 '티어별 그래디언트 부스팅'으로
        log(최종컷/현재컷) 배율을 회귀. 막판 급상승을 데이터에서 직접 학습.

검증 성능(LOO 백테스트, 진행도 85~95% 기준):
  naive(현재값) 막판오차 19.7% / GBM 2.7%, 편향 ~0.
  안전컷(α=0.95) 실제 커버리지 ≈ 89%.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from get_events import call_events
from bestdori.eventtracker import EventTracker

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "experiment", "cache")
GRID = np.linspace(0, 100, 201)          # 진행도 0..100 (0.5% 간격)
N = len(GRID)

# 분위수 설정
ALPHA_UB = 0.95   # 안전컷 (실제 커버리지 ≈ 89%)
ALPHA_LB = 0.10

# 학습 풀: 항상 같은 티어 + 같은 이벤트 타입(메들리/트라이/대반…)으로 좁힘.
# 타입 일치 표본이 MIN_TYPE_SAMPLES 미만이면 같은 티어 전체로 폴백.
# (백테스트: 티어+타입이 티어단독보다 약간 정확. 밴드 필터는 표본만 깎여 오히려 악화)
MIN_TYPE_SAMPLES = 15

GBM_KW = dict(n_estimators=120, max_depth=2, learning_rate=0.05,
              subsample=0.8, random_state=0)


def idx_at(p):
    return int(round(p / 100 * (N - 1)))


# ---------------------------------------------------------------
# 특징 추출 (백테스트와 동일 정의)
# ---------------------------------------------------------------
def features(curve, i_obs):
    v = curve[i_obs]
    if v <= 0:
        return None
    p = i_obs / (N - 1) * 100
    w10 = max(1, int(0.10 * (N - 1)))
    j = max(0, i_obs - w10)
    slope10 = (curve[i_obs] - curve[j]) / max(v, 1)
    w5 = max(1, int(0.05 * (N - 1)))
    j2 = max(0, i_obs - w5)
    slope5 = (curve[i_obs] - curve[j2]) / max(v, 1)
    i_half = min(idx_at(50), i_obs)
    half = curve[i_half] / v
    return [p, np.log(v), slope10, slope5, half]


# ---------------------------------------------------------------
# 과거 데이터 캐시 로드 / 정규화
#  (과거 곡선 생성은 experiment/fetch_data.py 의 normalized_curve 사용)
# ---------------------------------------------------------------
class SurgePredictor:
    def __init__(self):
        self.curves = {}      # (event,tier) -> np.array(N)
        self.meta = None
        self._type_of = {}    # event id -> eventType
        self._models = {}     # (tier, etype, i_obs, alpha) -> fitted model (요청 내 캐시)
        self.load_cache()

    # ---- 캐시 ----
    def load_cache(self):
        npz = os.path.join(CACHE_DIR, "curves.npz")
        pkl = os.path.join(CACHE_DIR, "meta.pkl")
        if not os.path.exists(npz):
            self.curves, self.meta = {}, call_events()
            return
        d = np.load(npz)
        self.curves = {}
        for e, t, c in zip(d["events"], d["tiers"], d["curves"]):
            self.curves[(int(e), int(t))] = c
        self.meta = pd.read_pickle(pkl) if os.path.exists(pkl) else call_events()
        self._build_type_lookup()

    def _build_type_lookup(self):
        self._type_of = {}
        try:
            for _, r in self.meta.iterrows():
                self._type_of[int(r["id"])] = r["eventType"]
        except Exception:
            self._type_of = {}
        # 캐시에 빠진 이벤트(예: 최신 진행이벤트)는 전체 목록에서 보충
        try:
            full = call_events()
            for _, r in full.iterrows():
                self._type_of.setdefault(int(r["id"]), r["eventType"])
        except Exception:
            pass

    def refresh_history(self, n_events=80, tiers=(100, 500, 1000, 2000)):
        """과거 이벤트 데이터를 새로 받아 캐시 갱신."""
        sys.path.insert(0, os.path.join(HERE, "experiment"))
        from fetch_data import build_curve_table
        curves, meta = build_curve_table(n_events=n_events, tiers=list(tiers))
        keys = list(curves.keys())
        arr = np.vstack([curves[k] for k in keys]) if keys else np.zeros((0, N))
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(os.path.join(CACHE_DIR, "curves.npz"),
                 grid=GRID,
                 events=np.array([k[0] for k in keys]),
                 tiers=np.array([k[1] for k in keys]),
                 curves=arr)
        meta.to_pickle(os.path.join(CACHE_DIR, "meta.pkl"))
        self.load_cache()
        return len(self.curves)

    # ---- 현재 이벤트 부분 곡선 ----
    def current_partial(self, event, tier, start, end, now=None):
        et = EventTracker(0, event)
        try:
            cutoffs = et.get_data(tier=tier).get("cutoffs", [])
        except Exception:
            cutoffs = []
        if not cutoffs:
            return None
        d = pd.DataFrame(cutoffs)
        d = d.sort_values("time")
        t = pd.to_datetime(d["time"], unit="ms") + pd.Timedelta(hours=9)
        ep = d["ep"].to_numpy(dtype=float)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        total = (end_ts - start_ts).total_seconds()
        if total <= 0:
            return None
        now = pd.Timestamp(now) if now is not None else pd.Timestamp.now()
        p_now = (now - start_ts).total_seconds() / total * 100.0
        p_now = float(np.clip(p_now, 0.5, 100))
        i_now = idx_at(p_now)
        # 진행도(%) = (관측시각 - 시작) / 전체기간  (단위 혼동 없이 초 단위로 계산)
        prog = ((t - start_ts).dt.total_seconds() / total * 100.0).to_numpy()
        prog = np.concatenate([[0.0], prog]); ep = np.concatenate([[0.0], ep])
        # 그리드 보간 (관측 구간까지만)
        full = np.interp(GRID, prog, ep, left=0, right=ep[-1])
        full = np.maximum.accumulate(full)
        partial = full[: i_now + 1].copy()
        return {
            "partial": partial, "i_now": i_now, "p_now": p_now,
            "v": float(partial[-1]),
            "obs_prog": prog.tolist(), "obs_ep": ep.tolist(),
        }

    def _training_rows(self, tier, etype):
        """학습 곡선 풀 구성. 같은 티어+같은 타입 우선, 표본 부족 시 티어 전체로 폴백."""
        same_type = [(e, c) for (e, t), c in self.curves.items()
                     if t == tier and (etype is None or self._type_of.get(e) == etype)]
        if len(same_type) >= MIN_TYPE_SAMPLES:
            pool = same_type
        else:
            pool = [(e, c) for (e, t), c in self.curves.items() if t == tier]
        return pool

    # ---- 모델 적합 ----
    def _fit(self, tier, etype, i_obs, alpha):
        key = (tier, etype, i_obs, alpha)
        if key in self._models:
            return self._models[key]
        pool = self._training_rows(tier, etype)
        X, y = [], []
        for e, c in pool:
            f = features(c, i_obs)
            v = c[i_obs]
            if f is None or v <= 0 or c[-1] <= 0:
                continue
            X.append(f); y.append(np.log(c[-1] / v))
        if len(X) < 8:
            self._models[key] = None
            return None
        X, y = np.array(X), np.array(y)
        if alpha is None:
            m = GradientBoostingRegressor(**GBM_KW)
        else:
            m = GradientBoostingRegressor(loss="quantile", alpha=alpha, **GBM_KW)
        m.fit(X, y)
        self._models[key] = m
        return m

    # ---- 예측 ----
    def forecast(self, event, tier, start, end, now=None):
        cur = self.current_partial(event, tier, start, end, now=now)
        if cur is None:
            return None
        partial, i_obs, v = cur["partial"], cur["i_now"], cur["v"]
        f0 = features(partial, i_obs)
        if f0 is None:
            return None
        etype = self._type_of.get(int(event))
        # 실제 사용된 학습 풀(타입 일치 여부) 기록
        same_type_n = sum(1 for (e, t) in self.curves
                          if t == tier and etype is not None and self._type_of.get(e) == etype)
        used_type = same_type_n >= MIN_TYPE_SAMPLES
        out = {}
        for name, alpha in [("mean", None), ("ub", ALPHA_UB), ("lb", ALPHA_LB)]:
            m = self._fit(tier, etype, i_obs, alpha)
            if m is None:
                out[name] = int(v)
            else:
                out[name] = int(v * np.exp(m.predict([f0])[0]))
        # 단조 보정: lb <= mean <= ub, 모두 현재값 이상
        out["mean"] = max(out["mean"], v)
        out["ub"] = max(out["ub"], out["mean"])
        out["lb"] = min(max(out["lb"], v), out["mean"])
        out["mean"], out["ub"], out["lb"] = int(out["mean"]), int(out["ub"]), int(out["lb"])
        out["p_now"] = cur["p_now"]
        out["current_value"] = int(v)
        out["i_now"] = i_obs
        out["n_train"] = len(self._training_rows(tier, etype))
        out["pool"] = f"{etype}+IN{tier}" if used_type else f"전체+IN{tier}"
        out["chart"] = self._chart_data(cur, out, tier, etype)
        return out

    def _chart_data(self, cur, out, tier, etype):
        """프런트(Chart.js)용 차트 데이터. 단위 명확, 다운샘플."""
        i_obs = cur["i_now"]
        v = cur["v"]
        # 관측 곡선: 그리드로 보간 후 다운샘플
        prog = np.array(cur["obs_prog"]); ep = np.array(cur["obs_ep"])
        full_obs = np.maximum.accumulate(np.interp(GRID, prog, ep, left=0, right=ep[-1]))
        obs_idx = list(range(0, i_obs + 1, max(1, (i_obs + 1) // 70)))
        if obs_idx[-1] != i_obs:
            obs_idx.append(i_obs)
        observed = [[round(GRID[i], 2), int(full_obs[i])] for i in obs_idx]

        # 예측 곡선: 같은 학습 풀의 평균 '형태'를 현재값 기준으로 투영
        pool = self._training_rows(tier, etype)
        shapes = [c / c[i_obs] for _, c in pool if c[i_obs] > 0]
        fut_idx = list(range(i_obs, len(GRID), 2))
        if fut_idx[-1] != len(GRID) - 1:
            fut_idx.append(len(GRID) - 1)
        if shapes:
            tmpl = np.mean(shapes, axis=0) * v          # 현재값에서 출발
            base_end = tmpl[-1] if tmpl[-1] > 0 else v
            def scaled(final):
                arr = tmpl * (final / base_end)
                arr[i_obs] = v                          # 현재 시점은 실측값에 고정
                return arr
            mean_c, lb_c, ub_c = scaled(out["mean"]), scaled(out["lb"]), scaled(out["ub"])
        else:
            mean_c = lb_c = ub_c = np.full(len(GRID), v)
        proj_mean = [[round(GRID[i], 2), int(mean_c[i])] for i in fut_idx]
        proj_lb = [[round(GRID[i], 2), int(lb_c[i])] for i in fut_idx]
        proj_ub = [[round(GRID[i], 2), int(ub_c[i])] for i in fut_idx]
        return {
            "observed": observed,
            "mean": proj_mean, "lb": proj_lb, "ub": proj_ub,
            "p_now": round(cur["p_now"], 2),
        }


# ---------------------------------------------------------------
# 진단(필요 불캔/시간/판수) — 기존 로직 정리 이식
# ---------------------------------------------------------------
def diagnose(forecast, current_score, end_time, score_per_game,
             minutes_per_game=3, natural_flames_per_day=35, current_cutline=None):
    end = pd.to_datetime(end_time)
    now = pd.Timestamp.now()
    remaining_days = max(0.0, (end - now).total_seconds() / 86400)

    target = forecast["ub"]  # 안전컷 기준
    togo = max(0, target - current_score)
    games = togo / score_per_game if score_per_game > 0 else 0
    minutes = games * minutes_per_game
    flames = games * 3
    natural = natural_flames_per_day * remaining_days
    flames_to_buy = max(0, flames - natural)

    max_games = max(0.0, (end - now).total_seconds() / (minutes_per_game * 60))
    max_final = current_score + max_games * score_per_game

    cur_togo = (current_cutline - current_score) if current_cutline else 0
    return {
        "remaining_days": remaining_days,
        "current_cutline": current_cutline,
        "current_togo": cur_togo,
        "target": target,
        "togo": int(togo),
        "games_needed": int(games),
        "minutes_needed": minutes,
        "flames_needed": int(flames),
        "natural_flames": int(natural),
        "flames_to_buy": int(flames_to_buy),
        "max_final_score": int(max_final),
    }


if __name__ == "__main__":
    sp = SurgePredictor()
    df = call_events()
    cur = df.iloc[-1]
    eid = int(cur["id"])
    print(f"현재 이벤트 {eid} ({cur['eventName']}) / {cur['Band']} / {cur['eventType']}")
    for tier in [100, 500, 1000]:
        r = sp.forecast(eid, tier, cur["startTime"], cur["endTime"])
        if r:
            print(f"  IN{tier}: 진행도 {r['p_now']:.1f}% 현재컷 {r['current_value']:,} "
                  f"-> 예측 {r['mean']:,} [{r['lb']:,} ~ 안전 {r['ub']:,}] "
                  f"(풀={r['pool']}, n={r['n_train']})")
