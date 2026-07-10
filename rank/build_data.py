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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import predictor as P
P.CACHE_DIR = HERE                       # curves.npz 가 이 폴더에 있음
from predictor import SurgePredictor
from get_events import call_events

TIERS = [100, 500, 1000, 2000]


def main():
    sp = SurgePredictor()
    df = call_events()
    cur = df.iloc[-1]
    eid = int(cur["id"])

    out = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
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

    for tier in TIERS:
        try:
            fc = sp.forecast(eid, tier, cur["startTime"], cur["endTime"])
        except Exception as e:  # noqa
            print(f"  ! IN{tier} 예측 실패: {e}")
            fc = None
        if not fc:
            continue
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
