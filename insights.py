"""
由逐期物候階段萃取可用於決策之資訊
====================================
  - 物候日曆：每像素之插秧（營養生長期開始）、生殖生長期開始、抽穗（成熟期開始）、收割日
  - 收割預報：以「截至某日」之資料預測收割日，並可與完整資料之判定結果比對（回測）
  - 巡查清單：生長進度落後／提前、生長勢偏弱、未進入成熟期之區塊
  - 年度摘要：各期作稻作面積與日曆中位數，供跨年比較

日期解析度受期別間隔限制（約 10–15 天），事件日期以「首次判定為該階段之期別日期」表示。
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import ndimage

import pipeline as P

# 期作切分：一期作 1/1–7/15、二期作 7/16–12/31（臺中地區一期作約 6–7 月收割、二期作 7–8 月插秧）
SEASONS = {
    "一期作": ((1, 1), (7, 15)),
    "二期作": ((7, 16), (12, 31)),
}
EVENTS = ["插秧", "生殖生長", "抽穗", "收割"]   # 依序對應階段 0、1、2、3 之首次出現
# 至下一事件之預設天數（無前一年資料時使用；一期作冬春低溫，營養生長期較長）
DEFAULT_DURATIONS = {
    "一期作": {"插秧": 80, "生殖生長": 35, "抽穗": 35},
    "二期作": {"插秧": 60, "生殖生長": 15, "抽穗": 30},
}
LATE_DAYS = 21              # 進度偏離村里中位數超過此天數視為異常
WEAK_Z = -2.5               # 季內 NDVI 峰值之穩健 z 分數門檻
MIN_PATCH_PX = 10           # 巡查區塊最小像素數（0.1 ha）


def season_mask(dates, season: str, year: int) -> np.ndarray:
    (m0, d0), (m1, d1) = SEASONS[season]
    lo, hi = date(year, m0, d0), date(year, m1, d1)
    return np.array([lo <= d <= hi for d in dates])


def _first_after(cond: np.ndarray, start: np.ndarray) -> np.ndarray:
    """cond (W, N)；回傳每欄自 start 起首次為 True 之索引，無則 -1（start = -1 亦回傳 -1）"""
    if cond.shape[0] == 0:
        return np.full(cond.shape[1], -1)
    idx = np.arange(cond.shape[0])[:, None]
    c = cond & (idx >= np.maximum(start, 0)[None, :])
    first = np.where(c.any(0), c.argmax(0), -1)
    return np.where(start >= 0, first, -1)


def phenology_calendar(pred: np.ndarray, pred_dates, season: str, year: int) -> dict:
    """
    pred (J, N)：逐期階段；回傳 {事件: (N,) 日期序數 (date.toordinal)，-1 = 未發生}
    及 censored：插秧落在季內第一期（實際可能更早）
    """
    w = np.where(season_mask(pred_dates, season, year))[0]
    p = pred[w]
    N = p.shape[1]
    zero = np.zeros(N, dtype=int)
    veg = _first_after(p == 0, zero)
    rep = _first_after(p == 1, np.where(veg >= 0, veg, 0))
    mat = _first_after(p == 2, np.maximum.reduce([rep, veg, zero]))
    har = _first_after(p == 3, mat)
    ords = np.array([pred_dates[i].toordinal() for i in w] or [-1])

    def to_ord(i):
        return np.where(i >= 0, ords[np.clip(i, 0, None)], -1)

    return {
        "插秧": to_ord(veg),
        "生殖生長": to_ord(rep),
        "抽穗": to_ord(mat),
        "收割": to_ord(har),
        "censored": veg == 0,
        "window": w,
    }


def stage_durations(cal: dict, mask: np.ndarray, season: str) -> dict:
    """由該期作已完成之轉換估計各段天數中位數（像素數不足 200 時採預設值）"""
    out = {}
    for a, b in zip(EVENTS[:-1], EVENTS[1:]):
        ok = mask & (cal[a] >= 0) & (cal[b] >= 0)
        if a == "插秧":
            ok &= ~cal["censored"]
        out[a] = float(np.median(cal[b][ok] - cal[a][ok])) if ok.sum() >= 200 \
            else float(DEFAULT_DURATIONS[season][a])
    return out


def forecast_harvest(cal: dict, durations: dict) -> np.ndarray:
    """
    以最近已發生之事件加上後續各段天數推估收割日（日期序數，-1 = 尚未插秧無法推估）。
    已觀測到收割者直接採用觀測值。
    """
    fc = np.full(cal["收割"].shape, -1, dtype=float)
    for ev in ["抽穗", "生殖生長", "插秧"]:
        remaining_ev = sum(durations[e] for e in EVENTS[EVENTS.index(ev):-1])
        has = (cal[ev] >= 0) & (fc < 0)
        fc[has] = cal[ev][has] + remaining_ev
    observed = cal["收割"] >= 0
    fc[observed] = cal["收割"][observed]
    return fc


def weekly_area(ordinals: np.ndarray, px_area_ha: float) -> pd.DataFrame:
    """日期序數 → 每週（週一起算）面積"""
    ok = ordinals >= 0
    if not ok.any():
        return pd.DataFrame(columns=["week", "ha"])
    days = pd.to_datetime([date.fromordinal(int(o)) for o in ordinals[ok]])
    week = days.to_period("W-SUN").start_time
    return (pd.Series(px_area_ha, index=week).groupby(level=0).sum()
            .rename("ha").rename_axis("week").reset_index())


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / (mad + 1e-6)


def anomaly_patches(cal: dict, season_peak_ndvi: np.ndarray, in_season: np.ndarray,
                    season_complete: bool, rows: np.ndarray, cols: np.ndarray, data: dict) -> tuple:
    """
    回傳 (DataFrame 巡查清單, (H, W) 類別圖 0 = 無異常, 1 落後, 2 提前, 3 偏弱, 4 未成熟)
    in_season：該期作判釋為稻米之像素（相對於 rows/cols）
    """
    H, W = data["vmask"].shape
    cls = np.zeros(N := len(rows), dtype=np.uint8)
    head = cal["抽穗"].astype(float)
    planted = in_season & ((cal["生殖生長"] >= 0) | (head >= 0))
    has_head = planted & (head >= 0)
    if has_head.sum() > 0:
        med = np.median(head[has_head])
        dev = np.where(has_head, head - med, np.nan)
        cls[has_head & (dev >= LATE_DAYS)] = 1
        cls[has_head & (dev <= -LATE_DAYS)] = 2
    else:
        dev = np.full(N, np.nan)
    if planted.sum() > 0:
        z = np.full(N, np.nan)
        z[planted] = robust_z(season_peak_ndvi[planted])
        cls[planted & (cls == 0) & (z < WEAK_Z)] = 3
    else:
        z = np.full(N, np.nan)
    if season_complete:
        cls[planted & (cal["生殖生長"] >= 0) & (head < 0)] = 4

    grid_cls = np.zeros((H, W), dtype=np.uint8)
    grid_cls[rows, cols] = cls
    dev_map = np.full((H, W), np.nan)
    dev_map[rows, cols] = dev
    peak_map = np.full((H, W), np.nan)
    peak_map[rows, cols] = season_peak_ndvi

    names = {1: "生長落後", 2: "生長提前", 3: "生長勢偏弱", 4: "未進入成熟期"}
    records = []
    final = np.zeros((H, W), dtype=np.uint8)
    for k, name in names.items():
        lab, n = ndimage.label(grid_cls == k, structure=np.ones((3, 3)))
        if n == 0:
            continue
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        for pid, size in enumerate(sizes, 1):
            if size < MIN_PATCH_PX:
                continue
            rr, cc = np.where(lab == pid)
            final[rr, cc] = k
            x, y = data["grid"].transform * (cc.mean() + 0.5, rr.mean() + 0.5)
            records.append({
                "類型": name,
                "面積 (ha)": round(size * P.GRID_RES ** 2 / 1e4, 2),
                "抽穗偏離 (天)": round(float(np.nanmean(dev_map[rr, cc])), 0) if k in (1, 2) else None,
                "NDVI 峰值": round(float(np.nanmean(peak_map[rr, cc])), 2),
                "_x": x, "_y": y,
            })
    df = pd.DataFrame(records)
    if len(df):
        from rasterio.warp import transform as warp_transform
        lon, lat = warp_transform(P.GRID_CRS, "EPSG:4326", df["_x"].tolist(), df["_y"].tolist())
        df["經度"] = np.round(lon, 5)
        df["緯度"] = np.round(lat, 5)
        df = (df.drop(columns=["_x", "_y"])
                .sort_values(["類型", "面積 (ha)"], ascending=[True, False])
                .reset_index(drop=True))
        df.insert(0, "編號", [f"P{i + 1:03d}" for i in range(len(df))])
    return df, final


def ordinal_to_str(o) -> str:
    return date.fromordinal(int(round(o))).isoformat() if o is not None and o >= 0 else "—"


def season_summary(cal: dict, in_season: np.ndarray) -> dict:
    """年度摘要：稻作面積與各事件日期中位數"""
    planted = in_season & ((cal["生殖生長"] >= 0) | (cal["抽穗"] >= 0))
    out = {"稻作面積 (ha)": round(planted.sum() * P.GRID_RES ** 2 / 1e4, 1)}
    for ev in EVENTS:
        ok = planted & (cal[ev] >= 0)
        if ev == "插秧":
            ok &= ~cal["censored"]
        out[f"{ev}中位數"] = ordinal_to_str(np.median(cal[ev][ok])) if ok.sum() >= 50 else "—"
    return out


# ============ 完整分析流程（網頁與批次腳本共用） ============

def this_year_slots(data):
    n_pre = data["n_pre"]
    return data["slot_dates"][n_pre:], n_pre


def cluster_seasons(data: dict, k: int) -> dict:
    """各期作分別以 NDVI 物候特徵做 K-means；建議群集不做退而求其次（可能該期作未種稻）"""
    ndvi = data["ndvi"]
    dates, n_pre = this_year_slots(data)
    out = {}
    for season in SEASONS:
        idx = [n_pre + i for i in np.where(season_mask(dates, season, data["year"]))[0]]
        if len(idx) < 3:
            continue
        labels, curves, counts, suggested = P.cluster_village(ndvi, data["vmask"], idx, k, fallback=False)
        out[season] = {"idx": idx, "labels": labels, "curves": curves,
                       "counts": counts, "suggested": suggested}
    return out


def predict_pixels(data: dict, rows, cols, model, upto: int = None):
    """對指定像素建立特徵並預測；upto = 僅使用 slot 0..upto（模擬「截至該日」之資料）"""
    end = len(data["slot_dates"]) if upto is None else upto + 1
    series = P.build_s2_month_series(data["ndvi"][:end], data["ndwi"][:end], data["rvi"][:end],
                                     rows, cols, data["slot_dates"][:end])
    return series, P.predict_stages(model, series)


def analyze(data: dict, clusters: dict, rice_ids: dict, model) -> dict:
    """
    rice_ids：{期作: [群集編號]}；回傳逐期階段、各期作物候日曆、稻米像素歸屬等
    """
    H, W = data["vmask"].shape
    season_rice = {}
    for season, c in clusters.items():
        season_rice[season] = np.isin(c["labels"], rice_ids.get(season, [])) & data["vmask"]
    union = np.zeros((H, W), bool)
    for m in season_rice.values():
        union |= m
    rows, cols = np.where(union)
    pred_dates, n_pre = this_year_slots(data)
    if len(rows) == 0:
        return {"rows": rows, "cols": cols, "pred": np.zeros((len(pred_dates), 0), np.uint8),
                "pred_dates": pred_dates, "season_rice": season_rice, "cal": {}, "in_season": {},
                "peak": {}, "complete": {}}
    series, pred = predict_pixels(data, rows, cols, model)
    ndvi_s = series[:, n_pre:, 0] * 2 - 1          # 還原正規化之 NDVI（SG 插值＋平滑後）
    cal, in_season, peak, complete = {}, {}, {}, {}
    for season in clusters:
        cal[season] = phenology_calendar(pred, pred_dates, season, data["year"])
        in_season[season] = season_rice[season][rows, cols]
        w = cal[season]["window"]
        peak[season] = ndvi_s[:, w].max(1) if len(w) else np.full(len(rows), np.nan)
        (_, _), (m1, d1) = SEASONS[season]
        complete[season] = pred_dates[-1] >= date(data["year"], m1, d1) - timedelta(days=30)
    return {"rows": rows, "cols": cols, "pred": pred, "pred_dates": pred_dates,
            "season_rice": season_rice, "cal": cal, "in_season": in_season,
            "peak": peak, "complete": complete}


def asof_forecast(data: dict, ana: dict, season: str, asof: date, model, prior: dict = None) -> dict:
    """
    回測：僅用 asof 當日以前之影像重新插值與預測，推估收割日，
    再與完整資料判定之收割日比較（僅評估 asof 時尚未收割之像素）
    prior：前一年同期作之各段天數（無則用預設值）
    """
    upto = max(i for i, d in enumerate(data["slot_dates"]) if d <= asof)
    n_pre = data["n_pre"]
    if upto < n_pre:
        raise ValueError("截至日期須在當年第 1 期之後")
    rows, cols = ana["rows"], ana["cols"]
    _, pred_a = predict_pixels(data, rows, cols, model, upto=upto)
    dates_a = data["slot_dates"][n_pre:upto + 1]
    cal_a = phenology_calendar(pred_a, dates_a, season, data["year"])
    durations = prior or DEFAULT_DURATIONS[season]
    source = f"{data['year'] - 1} 年同期作實際天數" if prior else "預設值（無前一年資料）"
    fc = forecast_harvest(cal_a, durations)
    obs = ana["cal"][season]["收割"]
    ins = ana["in_season"][season]
    pending = ins & (fc > asof.toordinal())               # 截至當日尚未收割、需要預報者
    evaluable = pending & (obs >= 0) & (fc >= 0)
    err = fc[evaluable] - obs[evaluable]
    return {
        "forecast": np.where(pending, fc, -1),
        "observed": np.where(pending, obs, -1),
        "durations": durations,
        "source": source,
        "n_pending": int(pending.sum()),
        "n_eval": int(evaluable.sum()),
        "mae": float(np.mean(np.abs(err))) if len(err) else None,
        "bias": float(np.mean(err)) if len(err) else None,
        "within15": float(np.mean(np.abs(err) <= 15)) if len(err) else None,
        "upto_date": data["slot_dates"][upto],
    }


def year_summary(data: dict, ana: dict) -> dict:
    """存成 JSON 之年度摘要（跨年比較與下一年預報之天數依據）"""
    out = {"village": data.get("village"), "year": data["year"], "seasons": {}}
    for season in ana["cal"]:
        cal, ins = ana["cal"][season], ana["in_season"][season]
        out["seasons"][season] = {
            **season_summary(cal, ins),
            "durations": stage_durations(cal, ins, season),
            "complete": bool(ana["complete"][season]),
        }
    return out
