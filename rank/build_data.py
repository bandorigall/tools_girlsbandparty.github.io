"""
정적 사이트(github.io)용 data.json 생성기.
GitHub Actions가 1시간마다 실행 → 현재 이벤트를 실시간 fetch + 모델 학습/예측 →
data.json 으로 저장. 정적 index.html 이 이 파일을 읽어 표시한다.

학습용 과거 곡선은 동봉된 curves.npz 를 사용(현재 이벤트만 실시간 fetch).
"""
import os
import sys
import json
import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import predictor as P
P.CACHE_DIR = HERE                       # curves.npz 가 이 폴더에 있음
from predictor import SurgePredictor
from get_events import call_events

TIERS = [100, 500, 1000, 2000]
MARGIN_SHRINK = 0.5   # 과거 최소 로그간격의 절반만 강제 (표본 극값 위험 완화)


def _pav_dec(y):
    """등가중 비증가 isotonic 회귀 (PAV)."""
    blocks = []  # [합, 개수]
    for v in y:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            s, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
    out = []
    for s, n in blocks:
        out += [s / n] * n
    return np.array(out)


def _final_margins(eid, tiers):
    """과거 이벤트(현재 제외) 최종컷의 인접 티어 로그간격 최소값 × MARGIN_SHRINK."""
    d = np.load(os.path.join(HERE, "curves.npz"))
    fin = {(int(e), int(t)): c[-1] for e, t, c in zip(d["events"], d["tiers"], d["curves"])}
    evs = {e for e, _ in fin if e != eid and all((e, t) in fin for t in tiers)}
    return [MARGIN_SHRINK * max(0.0, min(np.log(fin[(e, a)] / fin[(e, b)]) for e in evs))
            for a, b in zip(tiers, tiers[1:])]


def reconcile(fcs, eid):
    """티어 순서(IN100≥IN500≥…) 보정. 로그척도 margin PAV를 mean/ub/lb 수준별로,
    이후 현재값 하한 clip, 마지막에 lb≤mean≤ub 정렬. (2026-10-23 전문가 3인 합의안)"""
    tiers = [t for t in TIERS if t in fcs]
    if len(tiers) < 2:
        return
    off = np.concatenate([[0.0], -np.cumsum(_final_margins(eid, tiers))])
    now = np.log([fcs[t]["current_value"] for t in tiers])
    floor = np.maximum.accumulate(now[::-1])[::-1]          # L_k = max_{j≥k} now_j
    for key in ("mean", "ub", "lb"):
        z = np.log([fcs[t][key] for t in tiers])
        z = np.maximum(_pav_dec(z - off) + off, floor)
        assert np.all(np.diff(z) <= 1e-9), f"{key} 티어 순서 위반"
        for t, v in zip(tiers, z):
            fcs[t][key] = int(round(np.exp(v)))
    for t in tiers:
        fcs[t]["lb"], fcs[t]["mean"], fcs[t]["ub"] = sorted((fcs[t]["lb"], fcs[t]["mean"], fcs[t]["ub"]))


def main():
    sp = SurgePredictor()
    df = call_events()
    cur = df.iloc[-1]
    eid = int(cur["id"])

    out = {
        "generated": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).replace(tzinfo=None).isoformat(timespec="seconds"),
        "event": {
            "id": eid,
            "name": str(cur["eventName"]),
            "band": str(cur["Band"]),
            "type": str(cur["eventType"]),
            "start": str(cur["startTime"]),
            "end": str(cur["endTime"]),
        },
        "progress": 0,
        "n_history": len(sp.curves),
        "tiers": {},
    }

    fcs = {}
    for tier in TIERS:
        try:
            fc = sp.forecast(eid, tier, cur["startTime"], cur["endTime"])
        except Exception as e:  # noqa
            print(f"  ! IN{tier} 예측 실패: {e}")
            fc = None
        if fc:
            fcs[tier] = fc
    reconcile(fcs, eid)

    for tier, fc in fcs.items():
        out["progress"] = round(fc["p_now"], 2)
        out["tiers"][str(tier)] = {
            "current_value": fc["current_value"],
            "mean": fc["mean"],
            "lb": fc["lb"],
            "ub": fc["ub"],
            "n_train": fc["n_train"],
            "pool": fc["pool"],
            "chart": fc["chart"],
        }
        print(f"  IN{tier}: now {fc['current_value']:,} -> mean {fc['mean']:,} "
              f"ub {fc['ub']:,} ({fc['pool']})")

    path = os.path.join(HERE, "data.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\n저장: {path} ({os.path.getsize(path)//1024} KB, 티어 {len(out['tiers'])}개)")


if __name__ == "__main__":
    main()
