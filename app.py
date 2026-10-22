"""
霧峰區水稻物候 Demo（Streamlit）

執行：
  ~/venvs/rice_demo/bin/streamlit run app.py
"""

import base64
import json
import io
from datetime import date

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from streamlit_folium import st_folium

import pipeline as P

st.set_page_config(page_title="霧峰區水稻物候 Demo", page_icon="🌾", layout="wide")

CLUSTER_COLORS = ["#E45756", "#4C78A8", "#F58518", "#72B7B2", "#54A24B",
                  "#B279A2", "#FF9DA6", "#9D755D"]
CLOUD_RGBA = (255, 64, 160, 150)
SEASONS = {"一期作（1–6 月）": (1, 6), "二期作（7–12 月）": (7, 12)}


# ============ 快取：只存在記憶體 ============

def get_cube(village, year, clear_thr):
    """影像資料存於本 session 記憶體（含進度條，不適用 st.cache_data 重播機制）"""
    store = st.session_state.setdefault("cubes", {})
    key = (village, year, clear_thr)
    if key not in store:
        bar = st.progress(0.0, text="準備中…")
        data = P.load_village_cube(village, year, clear_thr,
                                   progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
        bar.empty()
        ndvi, ndwi, rvi = P.spectral_indices(data["cube"])
        data.update(ndvi=ndvi, ndwi=ndwi, rvi=rvi)
        while len(store) >= 2:
            store.pop(next(iter(store)))
        store[key] = data
    return store[key]


@st.cache_data(show_spinner="K-means 分群中…", max_entries=8)
def cached_clusters(village, year, clear_thr, season, k, _data):
    m0, m1 = SEASONS[season]
    idx = [i for i, d in enumerate(_data["slot_dates"]) if m0 <= d.month <= m1]
    if len(idx) < 3:
        raise RuntimeError("所選期作內可用期別不足 3 期，無法分群")
    labels_map, curves, counts, suggested = P.cluster_village(_data["ndvi"], _data["vmask"], idx, k)
    return idx, labels_map, curves, counts, suggested


@st.cache_resource
def cached_model():
    return P.load_model()


@st.cache_data(show_spinner="LSTM 物候階段預測中…", max_entries=8)
def cached_stages(village, year, clear_thr, season, k, rice_ids, _data, _labels_map):
    rows, cols = np.where(np.isin(_labels_map, rice_ids))
    series = P.build_s2_month_series(_data["ndvi"], _data["ndwi"], _data["rvi"],
                                     rows, cols, _data["slot_dates"])
    preds = P.predict_stages(cached_model(), series)
    H, W = _labels_map.shape
    stage_maps = np.full((preds.shape[0], H, W), 255, dtype=np.uint8)
    stage_maps[:, rows, cols] = preds
    return stage_maps


# ============ 地圖輔助 ============

def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def to_overlay(rgba, grid, fmt="PNG"):
    """(H, W, 4) uint8 → 重投影至 WGS84 之 PNG／WEBP data URL 與 folium bounds"""
    H, W, _ = rgba.shape
    bounds = array_bounds(H, W, grid.transform)  # (west, south, east, north)
    dst_t, dw, dh = calculate_default_transform(grid.crs, "EPSG:4326", W, H,
                                                left=bounds[0], bottom=bounds[1],
                                                right=bounds[2], top=bounds[3])
    out = np.zeros((4, dh, dw), dtype=np.uint8)
    reproject(np.moveaxis(rgba, 2, 0), out, src_transform=grid.transform, src_crs=grid.crs,
              dst_transform=dst_t, dst_crs="EPSG:4326", resampling=Resampling.nearest)
    w, s, e, n = array_bounds(dh, dw, dst_t)
    buf = io.BytesIO()
    img = Image.fromarray(np.moveaxis(out, 0, 2), "RGBA")
    if fmt == "WEBP":
        img.save(buf, format="WEBP", quality=85)
    else:
        img.save(buf, format="PNG")
    url = f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()
    return url, [[s, w], [n, e]]


def show_map(layers, data, key, height=520):
    """layers：[(rgba, opacity, 名稱), ...]，由下而上疊加"""
    geom = data["geom_4326"]
    minx, miny, maxx, maxy = geom.bounds
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=14,
                   tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="衛星底圖").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)
    for rgba, opacity, name in layers:
        url, bounds = to_overlay(rgba, data["grid"])
        folium.raster_layers.ImageOverlay(url, bounds=bounds, opacity=opacity, name=name).add_to(m)
    folium.GeoJson(geom.__geo_interface__, name="村里界",
                   style_function=lambda _: {"color": "#FFEB3B", "weight": 2, "fillOpacity": 0}).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    st_folium(m, height=height, use_container_width=True, returned_objects=[], key=key)


def true_color_rgba(cube_t, vmask):
    """當期真彩影像（含村里周邊）；雲遮像素以桃紅色標示"""
    rgb = np.stack([cube_t[2], cube_t[1], cube_t[0]], axis=-1)
    valid = ~np.isnan(rgb).any(-1)
    out = np.zeros(rgb.shape[:2] + (4,), dtype=np.uint8)
    if valid.any():
        lo, hi = np.nanpercentile(rgb[valid & vmask] if (valid & vmask).any() else rgb[valid], [2, 98])
        out[..., :3] = (np.clip((np.nan_to_num(rgb) - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
        out[valid, 3] = 255
    out[~valid] = CLOUD_RGBA
    out[~valid & ~vmask, 3] = 90
    return out


def nearest_image_slot(i, slot_acq, slot_dates):
    """該期無影像時，回傳名目日期最接近之有影像期別"""
    avail = [t for t, a in enumerate(slot_acq) if a is not None]
    return min(avail, key=lambda t: abs((slot_dates[t] - slot_dates[i]).days))


@st.cache_data(show_spinner="產生各期圖層…", max_entries=8)
def build_player_frames(village, year, clear_thr, season, k, rice_ids, _data, _stage_maps):
    """預先產生每一預測期別之分類圖與當期影像（data URL），供前端播放器切換"""
    slot_dates, slot_acq, vmask = _data["slot_dates"], _data["slot_acq"], _data["vmask"]
    rice_mask = _stage_maps[0] != 255
    n_rice = max(int(rice_mask.sum()), 1)
    img_cache = {}
    frames, bounds = [], None
    for j, sm in enumerate(_stage_maps):
        slot = j + P.SEQ_LEN
        img_slot = slot if slot_acq[slot] is not None else nearest_image_slot(slot, slot_acq, slot_dates)
        if img_slot not in img_cache:
            img_cache[img_slot], _ = to_overlay(true_color_rgba(_data["cube"][img_slot], vmask),
                                                _data["grid"], fmt="WEBP")
        stage_url, bounds = to_overlay(categorical_rgba(sm, P.STAGE_COLORS, 255), _data["grid"])
        px = [int((sm[rice_mask] == s).sum()) for s in range(4)]
        frames.append({
            "label": f"T{slot + 1}",
            "date": str(slot_dates[slot]),
            "acq": str(slot_acq[img_slot]),
            "interp": slot_acq[slot] is None,
            "stage": stage_url,
            "image": img_cache[img_slot],
            "px": px,
            "ha": [round(p * P.GRID_RES ** 2 / 1e4, 1) for p in px],
            "pct": [round(p / n_rice * 100, 1) for p in px],
        })
    return {"frames": frames, "bounds": bounds}


def player_html(player, geom):
    """左：預測分類圖、右：當期 Sentinel-2 影像；兩圖連動，可拖曳期別或自動播放"""
    payload = json.dumps({
        "frames": player["frames"],
        "bounds": player["bounds"],
        "boundary": geom.__geo_interface__,
        "names": P.STAGE_NAMES,
        "colors": P.STAGE_COLORS,
    }, ensure_ascii=False)
    return """
<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  :root { --fg:#1f2328; --muted:#6b7280; --panel:#f6f7f9; --line:#d0d7de; --accent:#ff4b4b; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e6e6e6; --muted:#9aa0a6; --panel:#1a1c23; --line:#30343c; }
  }
  body { margin:0; font-family:"Source Sans Pro","Noto Sans TC",sans-serif; color:var(--fg); background:transparent; }
  .bar { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:10px; }
  button { background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:8px;
           padding:6px 12px; font-size:15px; cursor:pointer; }
  button.play { background:var(--accent); border-color:var(--accent); color:#fff; min-width:92px; }
  input[type=range] { accent-color:var(--accent); }
  #slider { flex:1; min-width:200px; }
  select { background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:4px; }
  .now { font-size:18px; font-weight:600; min-width:150px; }
  .tag { font-size:13px; color:var(--muted); }
  .maps { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .map { height:500px; border-radius:8px; border:1px solid var(--line); }
  .title { font-weight:600; margin:0 0 4px 2px; }
  .legend { display:flex; gap:14px; flex-wrap:wrap; margin-top:8px; font-size:14px; }
  .sw { display:inline-block; width:13px; height:13px; border-radius:3px; margin-right:5px; vertical-align:-2px; }
  table { border-collapse:collapse; margin-top:8px; font-size:14px; }
  td, th { padding:3px 12px; border-bottom:1px solid var(--line); text-align:right; }
  td:first-child, th:first-child { text-align:left; }
  .sub { color:var(--muted); font-size:13px; margin-top:4px; }
</style></head><body>
<div class="bar">
  <button class="play" id="play">▶ 播放</button>
  <button id="prev">⏮</button><button id="next">⏭</button>
  <span class="now" id="now"></span>
  <input type="range" id="slider" min="0" step="1">
  <label class="tag">速度 <select id="speed">
    <option value="1500">慢</option><option value="900" selected>中</option><option value="450">快</option>
  </select></label>
  <label class="tag">分類圖不透明度 <input type="range" id="opacity" min="0" max="1" step="0.05" value="0.9"></label>
</div>
<div class="maps">
  <div><div class="title">預測分類圖</div><div class="map" id="mapL"></div></div>
  <div><div class="title">原始影像（Sentinel-2 真彩）</div><div class="map" id="mapR"></div></div>
</div>
<div class="maps">
  <div>
    <div class="legend" id="legend"></div>
    <table id="stats"></table>
  </div>
  <div>
    <div class="legend"><span><span class="sw" style="background:rgb(255,64,160)"></span>雲／雲影（已排除）</span></div>
    <div class="sub" id="imginfo"></div>
  </div>
</div>
<script>
const D = __PAYLOAD__;
const F = D.frames;
const esri = () => L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  {attribution:"Esri World Imagery", maxZoom:19});
const mk = id => { const m = L.map(id, {zoomSnap:0.25}); esri().addTo(m); L.control.scale().addTo(m); return m; };
const mapL = mk("mapL"), mapR = mk("mapR");
const stageLayer = L.imageOverlay(F[0].stage, D.bounds, {opacity:0.9}).addTo(mapL);
const imgLayer = L.imageOverlay(F[0].image, D.bounds).addTo(mapR);
const edge = {color:"#FFEB3B", weight:2, fill:false};
const bl = L.geoJSON(D.boundary, {style:edge}).addTo(mapL);
L.geoJSON(D.boundary, {style:edge}).addTo(mapR);
mapL.fitBounds(bl.getBounds(), {padding:[10,10]});
mapR.fitBounds(bl.getBounds(), {padding:[10,10]});

// 兩圖連動
let syncing = false;
const sync = (a, b) => a.on("move", () => {
  if (syncing) return; syncing = true;
  b.setView(a.getCenter(), a.getZoom(), {animate:false}); syncing = false;
});
sync(mapL, mapR); sync(mapR, mapL);

// 預載影像避免播放時閃爍
F.forEach(f => { new Image().src = f.stage; new Image().src = f.image; });

document.getElementById("legend").innerHTML = D.names.map((n, i) =>
  `<span><span class="sw" style="background:${D.colors[i]}"></span>${n}</span>`).join("");

const slider = document.getElementById("slider");
slider.max = F.length - 1; slider.value = 0;
let idx = 0, timer = null;

function show(i) {
  idx = (i + F.length) % F.length;
  const f = F[idx];
  slider.value = idx;
  stageLayer.setUrl(f.stage);
  imgLayer.setUrl(f.image);
  document.getElementById("now").textContent = `${f.label}　${f.date}${f.interp ? "（插值）" : ""}`;
  document.getElementById("imginfo").textContent = f.interp
    ? `本期無可用影像（預測以插值序列完成），右圖改顯示最接近之 ${f.acq} 影像`
    : `右圖影像日期：${f.acq}`;
  document.getElementById("stats").innerHTML =
    "<tr><th>階段</th><th>像素</th><th>面積 (ha)</th><th>比例</th></tr>" +
    D.names.map((n, s) => `<tr><td><span class="sw" style="background:${D.colors[s]}"></span>${n}</td>` +
      `<td>${f.px[s].toLocaleString()}</td><td>${f.ha[s].toFixed(1)}</td><td>${f.pct[s].toFixed(1)}%</td></tr>`).join("");
}

function stop() { clearInterval(timer); timer = null; document.getElementById("play").textContent = "▶ 播放"; }
function play() {
  stop();
  document.getElementById("play").textContent = "⏸ 暫停";
  timer = setInterval(() => show(idx + 1), +document.getElementById("speed").value);
}
document.getElementById("play").onclick = () => timer ? stop() : play();
document.getElementById("prev").onclick = () => { stop(); show(idx - 1); };
document.getElementById("next").onclick = () => { stop(); show(idx + 1); };
document.getElementById("speed").onchange = () => { if (timer) play(); };
slider.oninput = () => { stop(); show(+slider.value); };
document.getElementById("opacity").oninput = e => stageLayer.setOpacity(+e.target.value);
show(0);
</script></body></html>
""".replace("__PAYLOAD__", payload)


def categorical_rgba(label_map, colors, nodata):
    out = np.zeros(label_map.shape + (4,), dtype=np.uint8)
    for k, c in enumerate(colors):
        out[label_map == k] = (*hex_rgb(c), 255)
    out[label_map == nodata, 3] = 0
    return out


def legend(items):
    html = " ".join(
        f"<span style='display:inline-flex;align-items:center;margin-right:14px'>"
        f"<span style='width:14px;height:14px;background:{c};border-radius:3px;margin-right:6px'></span>{n}</span>"
        for n, c in items)
    st.markdown(html, unsafe_allow_html=True)


# ============ 側欄 ============

villages = P.load_villages()
village_names = sorted(villages["VILLAGE"])
this_year = date.today().year

with st.sidebar:
    st.header("🌾 參數設定")
    village = st.selectbox("村里（臺中市霧峰區）", village_names, index=village_names.index("北勢里"))
    year = st.selectbox("年份", list(range(this_year, 2018, -1)), index=1)
    clear_thr = st.slider("單期最低晴空比例", 0.1, 0.95, 0.5, 0.05,
                          help="村里範圍內 SCL 判定為有效（非雲／雲影／卷雲）之像素比例，低於此值之日期整期剔除；"
                               "保留之日期仍逐像素排除雲遮，再以 SG 插值補齊")
    season = st.radio("非監督分群之期作", list(SEASONS))
    k = st.slider("K-means 群集數", 3, 8, 5)
    run = st.button("載入影像並分析", type="primary", use_container_width=True)
    st.caption("影像來源：Microsoft Planetary Computer Sentinel-2 L2A，以 HTTP 視窗讀取，不下載至本機。")

if run:
    st.session_state["params"] = (village, year, clear_thr, season, k)

st.title("霧峰區水稻物候階段 Demo")
st.caption("Sentinel-2 → 雲遮排除 → 非監督稻米判釋 → S2+Month LSTM（SG 插值、seq5）物候階段預測")

if "params" not in st.session_state:
    st.info("請於左側選擇村里與年份後按「載入影像並分析」。首次載入約需 1–3 分鐘（視村里面積與網路而定）。")
    show_map([], {"geom_4326": villages[villages.VILLAGE == village].geometry.iloc[0]}, key="preview")
    st.stop()

village, year, clear_thr, season, k = st.session_state["params"]

try:
    data = get_cube(village, year, clear_thr)
    season_idx, labels_map, curves, counts, suggested = cached_clusters(village, year, clear_thr, season, k, data)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

slot_dates, slot_acq, acq_df, vmask = data["slot_dates"], data["slot_acq"], data["acq_df"], data["vmask"]
n_used = sum(a is not None for a in slot_acq)

st.subheader(f"{village}・{year} 年")
c1, c2, c3, c4 = st.columns(4)
c1.metric("搜尋到之影像日期", len(acq_df))
c2.metric("達晴空門檻", int((acq_df.clear_frac >= clear_thr).sum()))
c3.metric("採用期別", f"{n_used} / {len(slot_dates)}")
c4.metric("村里面積", f"{vmask.sum() * P.GRID_RES ** 2 / 1e4:,.0f} ha")

tab1, tab2, tab3 = st.tabs(["① 影像篩選與雲遮", "② 非監督稻米判釋", "③ LSTM 物候階段"])

# ---------- ① 影像篩選 ----------
with tab1:
    fig = go.Figure()
    for used, name, color in ((False, "未採用", "#B0B0B0"), (True, "採用", "#2E7D32")):
        sub = acq_df[acq_df.used == used]
        fig.add_bar(x=sub.date, y=sub.clear_frac * 100, name=name, marker_color=color)
    fig.add_hline(y=clear_thr * 100, line_dash="dash", line_color="#D32F2F",
                  annotation_text=f"門檻 {clear_thr:.0%}")
    fig.update_layout(height=320, yaxis_title="村里內晴空比例 (%)", bargap=0.1,
                      margin=dict(t=20, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1, 2])
    with left:
        slot_tbl = pd.DataFrame({
            "期別": [f"T{i + 1}" for i in range(len(slot_dates))],
            "名目日期": slot_dates,
            "採用影像": [a if a is not None else "— 缺（插值補齊）" for a in slot_acq],
            "晴空比例": [f"{acq_df.set_index('date').clear_frac.get(a, np.nan):.0%}" if a else "" for a in slot_acq],
        })
        st.dataframe(slot_tbl, hide_index=True, height=480, use_container_width=True)
    with right:
        avail = [i for i, a in enumerate(slot_acq) if a is not None]
        pick = st.select_slider("檢視真彩影像", options=avail,
                                format_func=lambda i: f"T{i + 1}  {slot_acq[i]}")
        show_map([(true_color_rgba(data["cube"][pick], vmask), 1.0, f"Sentinel-2 {slot_acq[pick]}")],
                 data, key=f"tc_{pick}")
        legend([("雲／雲影／卷雲（已排除）", "rgb(255,64,160)")])

# ---------- ② 稻米判釋 ----------
with tab2:
    left, right = st.columns([1, 1])
    with left:
        x = [slot_dates[i] for i in season_idx]
        fig = go.Figure()
        for c in range(k):
            fig.add_scatter(x=x, y=curves[c], mode="lines+markers", connectgaps=True,
                            name=f"Cluster {c}（{counts[c]:,} px）",
                            line=dict(color=CLUSTER_COLORS[c], width=4 if c in suggested else 2))
        fig.update_layout(height=380, yaxis_title="平均 NDVI", title=f"各群集平均 NDVI 時序（{season}）",
                          margin=dict(t=50, b=10), legend=dict(font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("水稻典型曲線：淹水低點 → 陡升 → 高峰 → 緩降。系統依振幅、峰值與低點先於峰值自動建議，可自行調整。")
        rice_ids = st.multiselect("水稻群集", list(range(k)), default=suggested,
                                  format_func=lambda c: f"Cluster {c}", key=f"rice_{st.session_state['params']}")
        rice_px = int(np.isin(labels_map, rice_ids).sum())
        st.metric("稻米像素 / 面積", f"{rice_px:,} px", f"約 {rice_px * P.GRID_RES ** 2 / 1e4:,.1f} ha",
                  delta_color="off")
    with right:
        view = st.radio("圖層", ["稻米遮罩", "全部群集"], horizontal=True)
        if view == "稻米遮罩":
            rice_map = np.where(labels_map < 0, -1, np.isin(labels_map, rice_ids).astype(np.int16))
            rgba = categorical_rgba(rice_map, ["#BDBDBD", "#00E676"], -1)
            rgba[rice_map == 0, 3] = 90
            show_map([(rgba, 0.85, "稻米遮罩")], data, key=f"ricemap_{rice_ids}")
            legend([("水稻", "#00E676"), ("非水稻", "#BDBDBD")])
        else:
            show_map([(categorical_rgba(labels_map, CLUSTER_COLORS[:k], -1), 0.85, "K-means 群集")],
                     data, key="clusters")
            legend([(f"Cluster {c}", CLUSTER_COLORS[c]) for c in range(k)])

# ---------- ③ 物候階段 ----------
with tab3:
    if not rice_ids:
        st.warning("請先於「② 非監督稻米判釋」選擇至少一個水稻群集。")
        st.stop()
    stage_maps = cached_stages(village, year, clear_thr, season, k, tuple(sorted(rice_ids)), data, labels_map)
    pred_dates = slot_dates[P.SEQ_LEN:]
    rice_mask = stage_maps[0] != 255
    n_rice = int(rice_mask.sum())

    share = np.stack([[(sm[rice_mask] == s).mean() * 100 for s in range(4)] for sm in stage_maps])
    fig = go.Figure()
    interp = [slot_acq[j + P.SEQ_LEN] is None for j in range(len(pred_dates))]
    for s in range(4):
        fig.add_scatter(x=pred_dates, y=share[:, s], name=P.STAGE_NAMES[s], mode="lines+markers",
                        line=dict(color=P.STAGE_COLORS[s], width=3),
                        marker=dict(size=8, color=["white" if f else P.STAGE_COLORS[s] for f in interp],
                                    line=dict(color=P.STAGE_COLORS[s], width=2)),
                        hovertemplate="%{y:.1f}%")
    fig.update_layout(height=360, yaxis_title="判釋稻米像素中之比例 (%)", yaxis_range=[0, 102],
                      hovermode="x unified", margin=dict(t=20, b=10),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"以前 {P.SEQ_LEN} 期為輸入視窗預測下一期之階段，故自第 {P.SEQ_LEN + 1} 期起才有預測。"
               "空心點為該期無可用影像、以 SG 插值補齊後預測。")

    frames = build_player_frames(village, year, clear_thr, season, k, tuple(sorted(rice_ids)),
                                 data, stage_maps)
    st.iframe(player_html(frames, data["geom_4326"]), height=720)
    st.caption("模型：S2+Month LSTM（SG 插值、seq5），models/model_S2plusMonth.pt")
