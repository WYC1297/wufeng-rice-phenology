"""
批次預先計算（展示模式與跨年比較用）

  python scripts/precompute.py --village 北勢里 --years 2019 2026

每個年份輸出至 cache/：
  {村里}_{年}_clear50.pkl.gz   影像時序（網頁選到此組合時直接載入，免重新讀取雲端影像）
  {村里}_{年}_summary.json     年度摘要：各期作稻作面積、物候日曆中位數、各段天數
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as P
import insights as I


def main():
    ap = argparse.ArgumentParser(description="批次預先計算村里年度成果")
    ap.add_argument("--village", nargs="+", default=["北勢里"])
    ap.add_argument("--years", type=int, nargs=2, default=[2019, 2025], metavar=("START", "END"))
    ap.add_argument("--clear-thr", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="忽略既有快取重新讀取影像")
    args = ap.parse_args()

    model = P.load_model()
    for village in args.village:
        for year in range(args.years[0], args.years[1] + 1):
            t0 = time.time()
            path = P.cache_path(village, year, args.clear_thr)
            try:
                if path.exists() and not args.force:
                    data = P.load_cache(path)
                else:
                    data = P.load_village_cube(village, year, args.clear_thr)
                    P.save_cache(data, path)
            except RuntimeError as e:
                print(f"[SKIP] {village} {year}：{e}")
                continue
            data["ndvi"], data["ndwi"], data["rvi"] = P.spectral_indices(data["cube"])
            clusters = I.cluster_seasons(data, args.k)
            ana = I.analyze(data, clusters, {s: c["suggested"] for s, c in clusters.items()}, model)
            summary = I.year_summary(data, ana)
            P.summary_path(village, year).write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
            areas = "、".join(f"{s} {v['稻作面積 (ha)']:.0f} ha" for s, v in summary["seasons"].items())
            print(f"[OK] {village} {year}（{time.time() - t0:.0f} s）：{areas}")


if __name__ == "__main__":
    main()
