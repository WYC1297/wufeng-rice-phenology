"""
輸出靜態成果圖（簡報／README 用）

  python scripts/export_figures.py --village 北勢里 --year 2025 --out docs --font NotoSansCJKtc-Regular.otf

輸出：
  stage_share.png   各階段比例折線圖
  stage_maps.png    各期物候階段圖
  demo_player.gif   左：預測分類圖、右：當期 Sentinel-2 真彩影像
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as P


def parse_args():
    ap = argparse.ArgumentParser(description="輸出霧峰區水稻物候 Demo 之靜態成果圖")
    ap.add_argument("--village", default="北勢里")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--clear-thr", type=float, default=0.5)
    ap.add_argument("--season-months", type=int, nargs=2, default=[1, 6], help="分群期作月份範圍")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="docs")
    ap.add_argument("--font", default="", help="中文字型檔路徑（.ttf/.otf）")
    ap.add_argument("--gif-ms", type=int, default=800, help="GIF 每格毫秒")
    return ap.parse_args()


def set_font(path):
    if path and Path(path).exists():
        fm.fontManager.addfont(path)
        plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def true_color(cube_t):
    rgb = np.stack([cube_t[2], cube_t[1], cube_t[0]], axis=-1)
    valid = ~np.isnan(rgb).any(-1)
    lo, hi = np.nanpercentile(rgb[valid], [2, 98]) if valid.any() else (0, 1)
    out = np.clip((np.nan_to_num(rgb) - lo) / (hi - lo + 1e-6), 0, 1)
    out[~valid] = (1.0, 0.25, 0.63)   # 雲遮
    return out


def main():
    args = parse_args()
    set_font(args.font)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = P.load_village_cube(args.village, args.year, args.clear_thr,
                            progress=lambda f, m: print(f"[{f:4.0%}] {m}", end="\r"))
    print()
    ndvi, ndwi, rvi = P.spectral_indices(d["cube"])
    m0, m1 = args.season_months
    idx = [i for i, s in enumerate(d["slot_dates"]) if m0 <= s.month <= m1]
    labels, _, _, suggested = P.cluster_village(ndvi, d["vmask"], idx, args.k)
    rr, cc = np.where(np.isin(labels, suggested))
    series = P.build_s2_month_series(ndvi, ndwi, rvi, rr, cc, d["slot_dates"])
    pred = P.predict_stages(P.load_model(), series)

    slot_dates, slot_acq = d["slot_dates"], d["slot_acq"]
    dates = slot_dates[P.SEQ_LEN:]
    interp = [slot_acq[j + P.SEQ_LEN] is None for j in range(len(dates))]
    share = np.stack([[(p == s).mean() * 100 for s in range(4)] for p in pred])
    rice_ha = len(rr) * P.GRID_RES ** 2 / 1e4
    title = f"{args.village} {args.year} 年"

    # 1) 比例折線圖
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for s in range(4):
        ax.plot(dates, share[:, s], "-", color=P.STAGE_COLORS[s], lw=2.5, label=P.STAGE_NAMES[s])
        for x, y, f in zip(dates, share[:, s], interp):
            ax.plot(x, y, "o", ms=6, mfc="white" if f else P.STAGE_COLORS[s],
                    mec=P.STAGE_COLORS[s], mew=1.5)
    ax.set_ylim(0, 102)
    ax.set_ylabel("判釋稻米像素中之比例 (%)")
    ax.grid(alpha=.3)
    ax.set_title(f"{title} 水稻物候階段比例（稻米 {rice_ha:.0f} ha；空心點＝插值期別）")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.12), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "stage_share.png", dpi=150)
    plt.close(fig)

    # 2) 各期階段圖
    cmap = ListedColormap(P.STAGE_COLORS)
    vm = d["vmask"]
    rows_any, cols_any = np.where(vm)
    crop = (slice(rows_any.min(), rows_any.max() + 1), slice(cols_any.min(), cols_any.max() + 1))
    stage_maps = []
    for p in pred:
        m = np.full(vm.shape, np.nan)
        m[rr, cc] = p
        stage_maps.append(m)

    n, cols = len(dates), 7
    nrows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(nrows, cols, figsize=(cols * 2.4, nrows * 2.6))
    for j, a in enumerate(axs.flat):
        a.axis("off")
        if j >= n:
            continue
        a.imshow(np.where(vm, 0.93, 1.0)[crop], cmap="gray", vmin=0, vmax=1)
        a.imshow(stage_maps[j][crop], cmap=cmap, vmin=-.5, vmax=3.5, interpolation="nearest")
        a.set_title(f"{dates[j]:%m/%d}" + ("（插值）" if interp[j] else ""), fontsize=10)
    fig.legend(handles=[Patch(color=c, label=l) for l, c in zip(P.STAGE_NAMES, P.STAGE_COLORS)],
               ncol=4, loc="lower center", frameon=False, fontsize=11)
    fig.suptitle(f"{title} 各期物候階段圖", fontsize=14)
    fig.tight_layout(rect=(0, .04, 1, .97))
    fig.savefig(out_dir / "stage_maps.png", dpi=130)
    plt.close(fig)

    # 3) 左右對照 GIF
    frames = []
    for j in range(n):
        slot = j + P.SEQ_LEN
        img_slot = slot if slot_acq[slot] is not None else min(
            (t for t, a in enumerate(slot_acq) if a is not None),
            key=lambda t: abs((slot_dates[t] - slot_dates[slot]).days))
        tc = true_color(d["cube"][img_slot])
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5.2))
        a1.imshow(tc * 0.55 + 0.45 * 0.5)                      # 淡化影像作為分類圖底圖
        a1.imshow(stage_maps[j], cmap=cmap, vmin=-.5, vmax=3.5, interpolation="nearest")
        a1.contour(vm, levels=[0.5], colors="#FFEB3B", linewidths=1.2)
        a1.set_title("預測分類圖")
        a2.imshow(tc)
        a2.contour(vm, levels=[0.5], colors="#FFEB3B", linewidths=1.2)
        a2.set_title(f"Sentinel-2 真彩影像（{slot_acq[img_slot]}）")
        for a in (a1, a2):
            a.axis("off")
        fig.suptitle(f"{title}　T{slot + 1}　{slot_dates[slot]}" + ("（插值）" if interp[j] else ""),
                     fontsize=14)
        fig.legend(handles=[Patch(color=c, label=l) for l, c in zip(P.STAGE_NAMES, P.STAGE_COLORS)]
                   + [Patch(color=(1.0, 0.25, 0.63), label="雲遮（已排除）")],
                   ncol=5, loc="lower center", frameon=False, fontsize=10)
        fig.tight_layout(rect=(0, .06, 1, .95))
        buf = io.BytesIO()
        fig.savefig(buf, dpi=90, format="png")
        plt.close(fig)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128))
    frames[0].save(out_dir / "demo_player.gif", save_all=True, append_images=frames[1:],
                   duration=args.gif_ms, loop=0, optimize=True)

    print(f"已輸出至 {out_dir.resolve()}：stage_share.png、stage_maps.png、demo_player.gif")


if __name__ == "__main__":
    main()
