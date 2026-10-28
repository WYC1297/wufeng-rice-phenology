"""
霧峰區水稻物候 Demo - 資料流程
================================
全部在記憶體內完成，不落地任何影像檔：

  1. 依村里界線查詢 Planetary Computer 之 Sentinel-2 L2A（STAC）
  2. 以 HTTP 視窗讀取（COG）僅取村里範圍，重投影至 EPSG:3826 / 10 m 網格
  3. SCL 逐像素排除雲／雲影／卷雲；整體晴空比例低於門檻之日期整期剔除
  4. 依訓練資料之 30 期日期網格（2023 S2_DATES 之月日）挑選每期最晴之影像
  5. 非監督 K-means（沿用 北勢_非監督非分類稻米.py 之 8 個 NDVI 物候特徵）
  6. 稻米像素 → SG 插值 + SG 平滑 + min-max 正規化 + 月份 sin/cos
     （與 prepare_dataset_s2_interpolation.py 一致）→ S2+Month LSTM (seq5)
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from scipy import interpolate, signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============ 參數設定區 ============
APP_DIR = Path(__file__).resolve().parent
VILLAGE_GEOJSON = APP_DIR / "wufeng_villages.geojson"
# 來源：phenology_stage_lstm/lstm_phenology_models_s2interp_sg_seq5_完整/model_S2plusMonth.pt
# 以 scripts/convert_model.py 轉為 NumPy 權重，推論不需安裝 torch
MODEL_PATH = APP_DIR / "models" / "model_S2plusMonth.npz"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
GRID_CRS = "EPSG:3826"   # 與原始裁切資料一致 (TWD97 / TM2)
GRID_RES = 10.0
MAX_SCENE_CLOUD = 95     # 整幅影像(100 km)雲量上限，僅作粗篩；實際以村里內 SCL 判斷
MAX_SLOT_DAYS = 8        # 影像日期距離期別名目日期之最大天數
GRID_BUFFER = 300        # 讀取範圍外擴(m)，供地圖顯示村里周邊之當期影像

# SCL：4 植生、5 裸露地、6 水體、7 未分類 視為有效；其餘(雲、雲影、卷雲、飽和、無資料)排除
SCL_VALID = (4, 5, 6, 7)

# 訓練資料之 30 期 S2 日期（取月日套用到所選年份）
S2_DATES_2023 = [
    "20230105", "20230115", "20230130", "20230209", "20230224", "20230306",
    "20230321", "20230331", "20230410", "20230425", "20230505", "20230520",
    "20230530", "20230614", "20230624", "20230704", "20230719", "20230729",
    "20230813", "20230823", "20230902", "20230922", "20231002", "20231012",
    "20231022", "20231101", "20231111", "20231126", "20231216", "20231226",
]

SEQ_LEN = 5
NORMALIZATION_PARAMS = {
    "ndvi": (-1.0, 1.0),
    "ndwi": (-1.0, 1.0),
    "rvi": (0.0, 10.0),
}

# 模型類別 0~3：淹水–分蘗、分蘗–抽穗開花、抽穗開花–採收、其他
STAGE_NAMES = ["營養生長期", "生殖生長期", "成熟期", "非稻作"]
STAGE_COLORS = ["#9BD770", "#1E7B34", "#E3B23C", "#A0A0A0"]


# ============ 村里與網格 ============

def load_villages() -> gpd.GeoDataFrame:
    return gpd.read_file(VILLAGE_GEOJSON)


@dataclass(frozen=True)
class Grid:
    transform: rasterio.Affine
    width: int
    height: int
    crs: str = GRID_CRS


def build_grid(geom_3826) -> Grid:
    minx, miny, maxx, maxy = geom_3826.buffer(GRID_BUFFER).bounds
    minx = np.floor(minx / GRID_RES) * GRID_RES
    maxy = np.ceil(maxy / GRID_RES) * GRID_RES
    width = int(np.ceil((maxx - minx) / GRID_RES))
    height = int(np.ceil((maxy - miny) / GRID_RES))
    return Grid(from_origin(minx, maxy, GRID_RES, GRID_RES), width, height)


def village_mask(geom_3826, grid: Grid) -> np.ndarray:
    """True = 村里範圍內"""
    return geometry_mask([geom_3826], out_shape=(grid.height, grid.width),
                         transform=grid.transform, invert=True, all_touched=True)


# ============ STAC 讀取 ============

def search_items(geom_4326, start: date, end: date):
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[COLLECTION],
        intersects=geom_4326.__geo_interface__,
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        query={"eo:cloud_cover": {"lt": MAX_SCENE_CLOUD}},
    )
    items = list(search.items())
    by_date = {}
    for it in items:
        by_date.setdefault(it.datetime.date(), []).append(it)
    return dict(sorted(by_date.items()))


def read_asset(href: str, grid: Grid, resampling=Resampling.nearest) -> np.ndarray:
    """以 WarpedVRT 僅讀取網格範圍並重投影，回傳 (H, W)，0 = 無資料"""
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=grid.crs, transform=grid.transform,
                       width=grid.width, height=grid.height,
                       resampling=resampling, src_nodata=0, nodata=0) as vrt:
            return vrt.read(1)


def mosaic_asset(items, key: str, grid: Grid, resampling=Resampling.nearest) -> np.ndarray:
    """同日多幅（不同 tile）以先到先填方式鑲嵌"""
    out = None
    for it in items:
        arr = read_asset(it.assets[key].href, grid, resampling)
        if out is None:
            out = arr
        else:
            fill = out == 0
            out[fill] = arr[fill]
    return out


def boa_offset(item) -> float:
    """處理基線 04.00 (2022-01-25) 起 L2A 數值加了 +1000 偏移量"""
    try:
        return 1000.0 if float(item.properties.get("s2:processing_baseline", "0")) >= 4.0 else 0.0
    except ValueError:
        return 0.0


def nominal_slot_dates(year: int):
    """
    前一年最後 SEQ_LEN 期 + 當年 30 期。
    前置期別作為 LSTM 輸入視窗，使當年第 1 期起即有預測（一期作插秧期才看得到）
    """
    this_year = [date(year, int(d[4:6]), int(d[6:8])) for d in S2_DATES_2023]
    prev_year = [date(year - 1, int(d[4:6]), int(d[6:8])) for d in S2_DATES_2023[-SEQ_LEN:]]
    return prev_year + this_year


def assign_slots(acq_df: pd.DataFrame, slot_dates, clear_thr: float):
    """
    每期挑選 ±MAX_SLOT_DAYS 內晴空比例最高（且達門檻）之影像日期；
    以晴空比例由高到低貪婪配對，每幅影像只配給一期，避免同一日期重複出現在時序中
    """
    usable = [(d, f) for d, f in zip(acq_df["date"], acq_df["clear_frac"]) if f >= clear_thr]
    pairs = sorted((-f, abs((d - sd).days), i, d)
                   for i, sd in enumerate(slot_dates) for d, f in usable
                   if abs((d - sd).days) <= MAX_SLOT_DAYS)
    chosen = [None] * len(slot_dates)
    used = set()
    for _, _, i, d in pairs:
        if chosen[i] is None and d not in used:
            chosen[i] = d
            used.add(d)
    return chosen


def load_village_cube(village: str, year: int, clear_thr: float, progress=None):
    """
    回傳 dict：
      cube       (T, 4, H, W) float32 反射率 [B02, B03, B04, B08]，雲遮像素 = NaN，無影像期別整期 NaN
                 （網格為村里外擴 GRID_BUFFER 之矩形，村里範圍見 vmask）
      slot_dates 名目期別日期（前 n_pre 期為前一年）、slot_acq 實際採用影像日期（None = 該期無可用影像）
      acq_df     所有搜尋到之影像日期與晴空比例
    """
    report = progress or (lambda frac, msg: None)
    villages = load_villages()
    row = villages[villages["VILLAGE"] == village].iloc[0]
    geom_4326 = row.geometry
    geom_3826 = villages[villages["VILLAGE"] == village].to_crs(GRID_CRS).geometry.iloc[0]
    grid = build_grid(geom_3826)
    vmask = village_mask(geom_3826, grid)
    n_village = int(vmask.sum())

    slot_dates = nominal_slot_dates(year)
    report(0.02, "查詢 Sentinel-2 影像清單…")
    by_date = search_items(geom_4326, slot_dates[0] - timedelta(days=MAX_SLOT_DAYS),
                           date(year, 12, 31))
    if not by_date:
        raise RuntimeError(f"{year} 年查無涵蓋 {village} 之 Sentinel-2 影像")

    # 1) 先只讀 SCL 計算村里內晴空比例
    dates = list(by_date)
    scl_cache = {}

    def _scl(d):
        return d, mosaic_asset(by_date[d], "SCL", grid, Resampling.nearest)

    with ThreadPoolExecutor(max_workers=12) as ex:
        for n, (d, scl) in enumerate(ex.map(_scl, dates), 1):
            scl_cache[d] = scl
            report(0.02 + 0.45 * n / len(dates), f"讀取雲遮圖層 SCL {n}/{len(dates)}（{d}）")

    records = []
    for d in dates:
        scl = scl_cache[d]
        valid = np.isin(scl, SCL_VALID) & vmask
        covered = (scl > 0) & vmask
        records.append({
            "date": d,
            "clear_frac": valid.sum() / n_village,
            "coverage": covered.sum() / n_village,
            "n_tiles": len(by_date[d]),
        })
    acq_df = pd.DataFrame(records)

    # 2) 期別配對
    slot_acq = assign_slots(acq_df, slot_dates, clear_thr)

    # 當年度尚未結束：截斷至最後一期有影像之期別
    last = max((i for i, a in enumerate(slot_acq) if a is not None), default=-1)
    if last < SEQ_LEN + 1:
        raise RuntimeError("可用期別太少（雲遮門檻可能太嚴格，或該年影像尚不足），無法進行 LSTM 預測")
    slot_dates = slot_dates[: last + 1]
    slot_acq = slot_acq[: last + 1]
    acq_df["used"] = acq_df["date"].isin([a for a in slot_acq if a is not None])

    # 3) 只對採用之日期讀取光學波段
    T = len(slot_dates)
    cube = np.full((T, 4, grid.height, grid.width), np.nan, dtype=np.float32)
    jobs = [(i, a) for i, a in enumerate(slot_acq) if a is not None]

    def _bands(job):
        i, d = job
        items = by_date[d]
        bands = []
        for key in ("B02", "B03", "B04", "B08"):
            out = None
            for it in items:
                arr = read_asset(it.assets[key].href, grid, Resampling.bilinear).astype(np.float32)
                arr = np.where(arr > 0, (arr - boa_offset(it)) / 10000.0, np.nan)
                out = arr if out is None else np.where(np.isnan(out), arr, out)
            bands.append(out)
        return i, np.stack(bands)

    with ThreadPoolExecutor(max_workers=8) as ex:
        for n, (i, arr) in enumerate(ex.map(_bands, jobs), 1):
            invalid = ~np.isin(scl_cache[slot_acq[i]], SCL_VALID)
            arr[:, invalid] = np.nan   # 村里外仍保留，僅供地圖顯示；分析一律以 vmask 限定
            cube[i] = arr
            report(0.47 + 0.5 * n / len(jobs), f"讀取光學波段 {n}/{len(jobs)}（{slot_acq[i]}）")

    report(1.0, "影像載入完成")
    return {
        "cube": cube,
        "slot_dates": slot_dates,
        "slot_acq": slot_acq,
        "acq_df": acq_df,
        "vmask": vmask,
        "grid": grid,
        "geom_4326": geom_4326,
        "village": village,
        "year": year,
        "n_pre": SEQ_LEN,      # slot_dates 前 n_pre 期屬前一年，僅作輸入視窗
    }


# ============ 本機快取（預先計算之展示資料） ============

CACHE_DIR = APP_DIR / "cache"


def cache_path(village: str, year: int, clear_thr: float) -> Path:
    return CACHE_DIR / f"{village}_{year}_clear{int(round(clear_thr * 100))}.pkl.gz"


def summary_path(village: str, year: int) -> Path:
    return CACHE_DIR / f"{village}_{year}_summary.json"


def save_cache(data: dict, path: Path):
    import gzip, pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = dict(data, cube=data["cube"].astype(np.float16))
    with gzip.open(path, "wb", compresslevel=3) as f:
        pickle.dump(slim, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache(path: Path) -> dict:
    import gzip, pickle
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    data["cube"] = data["cube"].astype(np.float32)
    return data


# ============ 指數計算 ============

def spectral_indices(cube: np.ndarray):
    """回傳 NDVI、NDWI、RVI，各為 (T, H, W)，與 extract_s2_features 公式一致"""
    green, red, nir = cube[:, 1], cube[:, 2], cube[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndwi = (green - nir) / (green + nir + 1e-8)
        rvi = nir / (red + 1e-8)
    return ndvi, ndwi, rvi


def _interp_matrix(t_valid: np.ndarray, n: int, extrapolate: bool) -> np.ndarray:
    """回傳 (n, len(t_valid)) 之線性插值矩陣，使 filled = values @ M.T"""
    t = np.arange(n)
    eye = np.eye(len(t_valid))
    if extrapolate:
        return np.stack([interpolate.interp1d(t_valid, e, kind="linear", fill_value="extrapolate",
                                              bounds_error=False)(t) for e in eye], axis=1)
    return np.stack([np.interp(t, t_valid, e) for e in eye], axis=1)


def _group_by_pattern(valid: np.ndarray):
    """依缺值樣態分組（雲遮具空間連續性，樣態數遠少於像素數），供向量化插值"""
    patterns, inverse = np.unique(valid, axis=0, return_inverse=True)
    return patterns, inverse.ravel()


def fill_linear(series: np.ndarray, min_valid: int = 3) -> np.ndarray:
    """(N, T) 線性內插補 NaN（兩端以最近值延伸），同 np.interp；有效點不足者回傳 NaN"""
    N, T = series.shape
    out = np.full_like(series, np.nan)
    valid = ~np.isnan(series)
    patterns, inverse = _group_by_pattern(valid)
    for p, pat in enumerate(patterns):
        rows = inverse == p
        if pat.sum() < min_valid:
            continue
        M = _interp_matrix(np.where(pat)[0], T, extrapolate=False)
        out[rows] = series[rows][:, pat] @ M.T
    return out


def sg_interpolation(series: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    """
    向量化版 sg_interpolation（prepare_dataset_s2_interpolation.py）：
    有效值先 SG 平滑 → 線性插值(可外插)補齊 → 依各像素有效值 1/99 百分位 clip
    """
    N, T = series.shape
    out = np.empty_like(series)
    valid = ~np.isnan(series)
    patterns, inverse = _group_by_pattern(valid)
    for p, pat in enumerate(patterns):
        rows = inverse == p
        x_valid = series[rows][:, pat]
        if pat.sum() < window:
            fill = np.nanmean(x_valid, axis=1) if pat.sum() > 0 else np.zeros(rows.sum())
            out[rows] = np.nan_to_num(fill)[:, None]
            continue
        smooth = signal.savgol_filter(x_valid, window, polyorder, axis=1)
        M = _interp_matrix(np.where(pat)[0], T, extrapolate=True)
        filled = smooth @ M.T
        lo = np.percentile(x_valid, 1, axis=1, keepdims=True)
        hi = np.percentile(x_valid, 99, axis=1, keepdims=True)
        out[rows] = np.clip(filled, lo, hi)
    return out


# ============ 非監督稻米判釋 ============

FEATURE_NAMES = ["min_ndvi", "max_ndvi", "amplitude", "mean_ndvi",
                 "std_ndvi", "rising_slope", "falling_slope", "max_t_ratio"]


def phenology_features(ndvi_filled: np.ndarray) -> np.ndarray:
    """與 北勢_非監督非分類稻米.py extract_phenology_features 相同之 8 個特徵（向量化）"""
    N, T = ndvi_filled.shape
    min_v, max_v = ndvi_filled.min(1), ndvi_filled.max(1)
    min_t, max_t = ndvi_filled.argmin(1), ndvi_filled.argmax(1)
    amplitude = max_v - min_v
    rising = np.where(max_t > min_t, amplitude / np.maximum(max_t - min_t, 1), 0.0)
    falling = np.where(T - 1 > max_t,
                       (ndvi_filled[:, -1] - max_v) / np.maximum(T - 1 - max_t, 1), 0.0)
    return np.stack([min_v, max_v, amplitude, ndvi_filled.mean(1), ndvi_filled.std(1),
                     rising, falling, max_t / T], axis=1).astype(np.float32)


def suggest_rice_clusters(mean_curves: np.ndarray, counts=None, fallback: bool = True):
    """
    依群集平均 NDVI 曲線建議水稻群集：「淹水低點 → 陡升 → 峰值」
      - 峰值 ≥ 0.55
      - 峰值之前的最低點到峰值上升 ≥ 0.3（收割後 NDVI 可能比淹水期更低，故不以全季最低點判斷）
      - 群集像素數 ≥ 總數 1%（排除雜訊小群集）
    """
    amp = np.nanmax(mean_curves, 1) - np.nanmin(mean_curves, 1)
    picks = []
    for k, curve in enumerate(mean_curves):
        t_peak = int(np.nanargmax(curve))
        before = curve[:t_peak]
        if t_peak == 0 or np.isnan(before).all():
            continue
        rise = curve[t_peak] - np.nanmin(before)
        big_enough = counts is None or counts[k] >= 0.01 * np.sum(counts)
        if curve[t_peak] >= 0.55 and rise >= 0.3 and big_enough:
            picks.append(k)
    if picks or not fallback:
        return picks
    return [int(np.argmax(amp))]


def cluster_village(ndvi: np.ndarray, vmask: np.ndarray, season_idx, n_clusters: int, seed: int = 42,
                    fallback: bool = True):
    """
    ndvi: (T, H, W)；season_idx: 用於分群之期別索引
    回傳 labels_map (H, W，-1 = 範圍外/無法計算)、各群集平均曲線、建議水稻群集
    """
    rows, cols = np.where(vmask)
    series = ndvi[season_idx][:, rows, cols].T  # (N, Ts)
    series = np.clip(series, -1, 1)
    filled = fill_linear(series)
    ok = ~np.isnan(filled).any(1)
    feats = phenology_features(filled[ok])

    scaler = StandardScaler()
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(scaler.fit_transform(feats))

    labels_map = np.full(vmask.shape, -1, dtype=np.int16)
    labels_map[rows[ok], cols[ok]] = labels

    raw = series[ok]
    mean_curves = np.stack([np.nanmean(raw[labels == k], axis=0) for k in range(n_clusters)])
    counts = np.bincount(labels, minlength=n_clusters)
    return labels_map, mean_curves, counts, suggest_rice_clusters(mean_curves, counts, fallback)


# ============ LSTM 物候預測 ============

class LSTMModel:
    """
    與 train_s2interp_all_methods.py 相同之架構（2 層 LSTM 128 → fc 32 → ReLU → fc 4），
    以 NumPy 實作推論；閘門順序同 PyTorch (i, f, g, o)，dropout 於推論時不作用
    """
    def __init__(self, weights: dict, num_layers: int = 2):
        w = {k: v.astype(np.float32) for k, v in weights.items()}
        # sigmoid(x) = 0.5·tanh(x/2) + 0.5：i、f、o 閘之權重先乘 0.5，四個閘只需一次 tanh
        hidden = w["lstm__weight_hh_l0"].shape[1]
        scale = np.full(4 * hidden, 0.5, np.float32)
        scale[2 * hidden:3 * hidden] = 1.0
        self.layers = []
        for layer in range(num_layers):
            self.layers.append((
                np.ascontiguousarray((w[f"lstm__weight_ih_l{layer}"] * scale[:, None]).T),
                np.ascontiguousarray((w[f"lstm__weight_hh_l{layer}"] * scale[:, None]).T),
                (w[f"lstm__bias_ih_l{layer}"] + w[f"lstm__bias_hh_l{layer}"]) * scale,
            ))
        self.w = w

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x (N, T, F) → logits (N, 4)"""
        h_seq = np.ascontiguousarray(x.transpose(1, 0, 2), dtype=np.float32)   # (T, N, F)
        for w_ih, w_hh, bias in self.layers:
            hidden = w_hh.shape[0]
            t_len, n, _ = h_seq.shape
            h = np.zeros((n, hidden), np.float32)
            c = np.zeros((n, hidden), np.float32)
            gates_x = h_seq @ w_ih + bias
            outs = np.empty((t_len, n, hidden), np.float32)
            for t in range(t_len):
                g = gates_x[t]
                g += h @ w_hh
                np.tanh(g, out=g)
                g *= 0.5
                g[:, 2 * hidden:3 * hidden] *= 2.0
                g[:, :2 * hidden] += 0.5
                g[:, 3 * hidden:] += 0.5
                i, f, gg, o = (g[:, k * hidden:(k + 1) * hidden] for k in range(4))
                c *= f
                c += i * gg
                h = outs[t]
                np.tanh(c, out=h)
                h *= o
            h_seq = outs
        z = np.maximum(h_seq[-1] @ self.w["fc1__weight"].T + self.w["fc1__bias"], 0)
        return z @ self.w["fc2__weight"].T + self.w["fc2__bias"]


def load_model(path: Path = MODEL_PATH) -> LSTMModel:
    with np.load(path) as f:
        return LSTMModel(dict(f))


def _normalize(x: np.ndarray, name: str) -> np.ndarray:
    lo, hi = NORMALIZATION_PARAMS[name]
    return (np.clip(x, lo, hi) - lo) / (hi - lo + 1e-8)


def build_s2_month_series(ndvi, ndwi, rvi, rice_rows, rice_cols, slot_dates) -> np.ndarray:
    """
    依 prepare_dataset_s2_interpolation.py 之步驟建立 (N, T, 5) 特徵：
    clip → SG 插值補缺 → SG(5,2) 平滑 → min-max 正規化，再接月份 sin/cos
    """
    feats = []
    for arr, name in ((ndvi, "ndvi"), (ndwi, "ndwi"), (rvi, "rvi")):
        lo, hi = NORMALIZATION_PARAMS[name]
        s = np.clip(arr[:, rice_rows, rice_cols].T, lo, hi)   # NaN 保留
        s = sg_interpolation(s)
        s = signal.savgol_filter(s, 5, 2, axis=1)
        feats.append(_normalize(s, name))
    months = np.array([d.month for d in slot_dates])
    angle = 2 * np.pi * months / 12
    n = len(rice_rows)
    feats.append(np.broadcast_to(np.sin(angle), (n, len(months))))
    feats.append(np.broadcast_to(np.cos(angle), (n, len(months))))
    return np.stack(feats, axis=2).astype(np.float32)


def predict_stages(model: LSTMModel, series: np.ndarray, batch_size: int = 16384) -> np.ndarray:
    """
    series (N, T, 5) → (T - SEQ_LEN, N)，第 j 列為期別 j + SEQ_LEN 之預測階段
    （訓練時以 [i, i+seq) 視窗預測 T{i+seq+1} 之標籤）
    """
    N, T, _ = series.shape
    out = np.empty((T - SEQ_LEN, N), dtype=np.uint8)
    for j, end in enumerate(range(SEQ_LEN, T)):
        window = series[:, end - SEQ_LEN:end]
        preds = []
        for b in range(0, N, batch_size):
            logits = model(window[b:b + batch_size])
            preds.append(logits.argmax(1).astype(np.uint8))
        out[j] = np.concatenate(preds) if preds else np.empty(0, dtype=np.uint8)
    return out
