"""
霧峰區水稻物候 Demo（Streamlit）

執行：
  ~/venvs/rice_demo/bin/streamlit run app.py
"""

import base64
import json
import io
from datetime import date, timedelta

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from streamlit_folium import st_folium

import insights as I
import pipeline as P

st.set_page_config(page_title="霧峰區水稻物候 Demo", page_icon="🌾", layout="wide")

CLUSTER_COLORS = ["#E45756", "#4C78A8", "#F58518", "#72B7B2", "#54A24B",
                  "#B279A2", "#FF9DA6", "#9D755D"]
CLOUD_RGBA = (255, 64, 160, 150)
PX_HA = P.GRID_RES ** 2 / 1e4
PATTERN_NAMES = ["僅一期作", "僅二期作", "兩期皆種"]
PATTERN_COLORS = ["#F2C14E", "#5AB4E5", "#1E7B34"]
ANOMALY_NAMES = {1: "生長落後", 2: "生長提前", 3: "生長勢偏弱", 4: "未進入成熟期"}
ANOMALY_COLORS = ["#E53935", "#8E24AA", "#FB8C00", "#3949AB"]


# ============ 快取：只存在記憶體 ============

def get_cube(village, year, clear_thr):
    """影像資料存於本 session 記憶體；有預先計算之快取時直接載入（展示模式）"""
    store = st.session_state.setdefault("cubes", {})
    key = (village, year, clear_thr)
    if key not in store:
        path = P.cache_path(village, year, clear_thr)
        if path.exists():
            with st.spinner("載入預先計算之影像時序…"):
                data = P.load_cache(path)
            data["source"] = "預先計算（展示模式）"
        else:
            bar = st.progress(0.0, text="準備中…")
            data = P.load_village_cube(village, year, clear_thr,
                                       progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
            bar.empty()
            data["source"] = "即時讀取 Planetary Computer"
        ndvi, ndwi, rvi = P.spectral_indices(data["cube"])
        data.update(ndvi=ndvi, ndwi=ndwi, rvi=rvi)
        while store:   # 只保留一組，降低雲端主機記憶體用量
            store.pop(next(iter(store)))
        store[key] = data
    return store[key]


@st.cache_data(show_spinner="K-means 分群中…", max_entries=8)
def cached_clusters(village, year, clear_thr, k, _data):
    return I.cluster_seasons(_data, k)


@st.cache_resource
def cached_model():
    return P.load_model()


@st.cache_data(show_spinner="LSTM 物候階段預測中…", max_entries=8)
def cached_analysis(village, year, clear_thr, k, rice_key, _data, _clusters):
    return I.analyze(_data, _clusters, dict(rice_key), cached_model())


def load_summary(village, year):
    path = P.summary_path(village, year)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner="以截至當日之資料重新預測…", max_entries=16)
def cached_asof(village, year, clear_thr, k, rice_key, season, asof, _data, _ana):
    prev = load_summary(village, year - 1)
    prior = prev["seasons"].get(season, {}).get("durations") if prev else None
    return I.asof_forecast(_data, _ana, season, asof, cached_model(), prior)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_anomalies(village, year, clear_thr, k, rice_key, season, _data, _ana):
    return I.anomaly_patches(_ana["cal"][season], _ana["peak"][season], _ana["in_season"][season],
                             _ana["complete"][season], _ana["rows"], _ana["cols"], _data)



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
def build_player_frames(village, year, clear_thr, k, rice_key, _data, _ana):
    """預先產生每一預測期別之分類圖與當期影像（data URL），供前端播放器切換"""
    slot_dates, slot_acq, vmask = _data["slot_dates"], _data["slot_acq"], _data["vmask"]
    rows, cols, pred, n_pre = _ana["rows"], _ana["cols"], _ana["pred"], _data["n_pre"]
    n_rice = max(len(rows), 1)
    H, W = vmask.shape
    img_cache = {}
    frames, bounds = [], None
    for j in range(pred.shape[0]):
        sm = np.full((H, W), 255, dtype=np.uint8)
        sm[rows, cols] = pred[j]
        slot = j + n_pre
        img_slot = slot if slot_acq[slot] is not None else nearest_image_slot(slot, slot_acq, slot_dates)
        if img_slot not in img_cache:
            img_cache[img_slot], _ = to_overlay(true_color_rgba(_data["cube"][img_slot], vmask),
                                                _data["grid"], fmt="WEBP")
        stage_url, bounds = to_overlay(categorical_rgba(sm, P.STAGE_COLORS, 255), _data["grid"])
        px = [int((pred[j] == s).sum()) for s in range(4)]
        frames.append({
            "label": f"T{j + 1}",
            "date": str(slot_dates[slot]),
            "acq": str(slot_acq[img_slot]),
            "interp": slot_acq[slot] is None,
            "stage": stage_url,
            "image": img_cache[img_slot],
            "px": px,
            "ha": [round(p * PX_HA, 1) for p in px],
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



def date_rgba(ord_map, cmap="Viridis"):
    """日期序數圖 → 依日期先後著色；回傳 (rgba, [(日期字串, 顏色), ...])"""
    from plotly.colors import sample_colorscale
    vals = np.unique(ord_map[ord_map >= 0]).astype(int)
    out = np.zeros(ord_map.shape + (4,), dtype=np.uint8)
    if len(vals) == 0:
        return out, []
    colors = sample_colorscale(cmap, np.linspace(0, 1, len(vals)) if len(vals) > 1 else [0.5])
    items = []
    for v, c in zip(vals, colors):
        rgb = tuple(int(float(x)) for x in c[c.index("(") + 1:-1].split(","))
        out[ord_map == v] = (*rgb, 255)
        items.append((date.fromordinal(v).strftime("%m/%d"), f"rgb{rgb}"))
    return out, items


def to_grid(values, rows, cols, shape, fill=-1, dtype=float):
    g = np.full(shape, fill, dtype=dtype)
    g[rows, cols] = values
    return g


def planted_mask(ana, season):
    cal = ana["cal"][season]
    return ana["in_season"][season] & ((cal["生殖生長"] >= 0) | (cal["抽穗"] >= 0))


def week_bars(fig, df, name, color, opacity=1.0):
    if len(df):
        fig.add_bar(x=df["week"], y=df["ha"], name=name, marker_color=color, opacity=opacity,
                    hovertemplate="%{x|%m/%d} 當週：%{y:.1f} ha<extra>" + name + "</extra>")


# ============ 側欄 ============

villages = P.load_villages()
village_names = sorted(villages["VILLAGE"])
this_year = date.today().year

with st.sidebar:
    st.header("🌾 參數設定")
    village = st.selectbox("村里（臺中市霧峰區）", village_names, index=village_names.index("北勢里"))
    year = st.selectbox("年份", list(range(this_year, 2018, -1)), index=1,
                        format_func=lambda y: f"{y}" + ("　⚡" if P.cache_path(village, y, 0.5).exists() else ""))
    clear_thr = st.slider("單期最低晴空比例", 0.1, 0.95, 0.5, 0.05,
                          help="村里範圍內 SCL 判定為有效（非雲／雲影／卷雲）之像素比例，低於此值之日期整期剔除；"
                               "保留之日期仍逐像素排除雲遮，再以 SG 插值補齊")
    k = st.slider("K-means 群集數", 3, 8, 5)
    run = st.button("載入影像並分析", type="primary", width="stretch")
    st.caption("⚡ 已預先計算（晴空門檻 0.5），可立即載入；其餘年份即時讀取 Planetary Computer 之 "
               "Sentinel-2 L2A（約 1–3 分鐘），影像不下載至本機。")

if run:
    st.session_state["params"] = (village, year, clear_thr, k)

st.title("霧峰區水稻物候監測")
st.caption("Sentinel-2 時序影像 → 雲遮排除 → 非監督稻米判釋 → LSTM 物候階段 → 收割預報與巡查清單")

if "params" not in st.session_state:
    st.info("請於左側選擇村里與年份後按「載入影像並分析」。")
    show_map([], {"geom_4326": villages[villages.VILLAGE == village].geometry.iloc[0]}, key="preview")
    st.stop()

params = st.session_state["params"]
village, year, clear_thr, k = params

try:
    data = get_cube(village, year, clear_thr)
    clusters = cached_clusters(village, year, clear_thr, k, data)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

rice_widget_key = lambda s: f"rice_{s}_{params}"
rice_ids = {s: st.session_state.get(rice_widget_key(s), c["suggested"]) for s, c in clusters.items()}
rice_key = tuple(sorted((s, tuple(sorted(v))) for s, v in rice_ids.items()))
ana = cached_analysis(village, year, clear_thr, k, rice_key, data, clusters)

slot_dates, slot_acq, acq_df, vmask = data["slot_dates"], data["slot_acq"], data["acq_df"], data["vmask"]
n_pre = data["n_pre"]
pred_dates = ana["pred_dates"]
rows, cols = ana["rows"], ana["cols"]
shape = vmask.shape
seasons = list(ana["cal"])

st.subheader(f"{village}・{year} 年")
st.caption(f"資料：{data['source']}；當年採用 {sum(a is not None for a in slot_acq[n_pre:])} / "
           f"{len(pred_dates)} 期影像，最新一期 {pred_dates[-1]}")

if len(rows) == 0:
    st.warning("目前沒有判釋為水稻的像素，請到「🔧 技術細節」選擇水稻群集。")

tabA, tabB, tabC, tabD, tabE = st.tabs(["📊 種植概況", "📅 收割時程與預報", "⚠️ 巡查清單",
                                        "🎞 物候動態", "🔧 技術細節"])

# ---------- 📊 種植概況 ----------
with tabA:
    planted = {s: planted_mask(ana, s) for s in seasons}
    cols_m = st.columns(len(seasons) + 2)
    for i, s in enumerate(seasons):
        cols_m[i].metric(f"{s}稻作面積", f"{planted[s].sum() * PX_HA:,.1f} ha")
    if len(seasons) == 2:
        both = planted[seasons[0]] & planted[seasons[1]]
        cols_m[-2].metric("兩期皆種", f"{both.sum() * PX_HA:,.1f} ha")
    cols_m[-1].metric("村里面積", f"{vmask.sum() * PX_HA:,.0f} ha")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**種植型態**")
        code = np.full(len(rows), -1, dtype=np.int16)
        if len(seasons) == 2:
            p1, p2 = planted[seasons[0]], planted[seasons[1]]
            code[p1 & ~p2], code[~p1 & p2], code[p1 & p2] = 0, 1, 2
        elif seasons:
            code[planted[seasons[0]]] = 0 if seasons[0] == "一期作" else 1
        pattern = to_grid(code, rows, cols, shape, -1, np.int16)
        show_map([(categorical_rgba(pattern, PATTERN_COLORS, -1), 0.85, "種植型態")], data,
                 key=f"pattern_{rice_key}")
        legend(list(zip(PATTERN_NAMES, PATTERN_COLORS)))
    with right:
        st.markdown("**歷年比較**")
        summaries = [s for y in range(2017, this_year + 1) if (s := load_summary(village, y))]
        if len(summaries) >= 2:
            fig = go.Figure()
            for s, color in zip(I.SEASONS, ["#F2C14E", "#5AB4E5"]):
                fig.add_bar(x=[x["year"] for x in summaries],
                            y=[x["seasons"].get(s, {}).get("稻作面積 (ha)") for x in summaries],
                            name=s, marker_color=color,
                            marker_line=dict(width=[3 if x["year"] == year else 0 for x in summaries],
                                             color="#FF4B4B"))
            fig.update_layout(barmode="group", height=300, yaxis_title="稻作面積 (ha)",
                              margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.15),
                              xaxis=dict(dtick=1))
            st.plotly_chart(fig, width="stretch")
            tbl = pd.DataFrame([{
                "年份": x["year"], "期作": s,
                "面積 (ha)": v["稻作面積 (ha)"],
                "插秧": v["插秧中位數"][5:], "抽穗": v["抽穗中位數"][5:], "收割": v["收割中位數"][5:],
            } for x in summaries for s, v in x["seasons"].items()])
            st.dataframe(tbl, hide_index=True, width="stretch", height=260)
            st.caption("日期為村里內稻作像素之中位數（月/日）。歷年資料由 scripts/precompute.py 以預設參數產生。")
        else:
            st.info("執行 `python scripts/precompute.py --village " + village +
                    " --years 2019 2025` 產生歷年資料後，此處會顯示跨年比較。")

# ---------- 📅 收割時程與預報 ----------
with tabB:
    if not seasons or len(rows) == 0:
        st.info("沒有可分析之期作。")
    else:
        season = st.radio("期作", seasons, horizontal=True, key="season_B")
        cal, ins = ana["cal"][season], ana["in_season"][season]
        plantedS = planted_mask(ana, season)
        summ = I.season_summary(cal, ins)
        live = None
        if not ana["complete"][season]:
            live = cached_asof(village, year, clear_thr, k, rice_key, season, pred_dates[-1], data, ana)
            harvest_all = np.where(plantedS & (cal["收割"] >= 0), cal["收割"], np.where(plantedS, live["forecast"], -1))
            ok = harvest_all >= 0
            summ["收割中位數"] = I.ordinal_to_str(np.median(harvest_all[ok])) if ok.any() else "—"
        c = st.columns(4)
        for col, ev in zip(c, I.EVENTS):
            label = f"{ev}（中位數{'，含預報' if live is not None and ev == '收割' else ''}）"
            col.metric(label, summ[f"{ev}中位數"])

        left, right = st.columns([1, 1])
        with left:
            ev = st.radio("日曆圖層", ["插秧", "抽穗", "收割"], horizontal=True, key="cal_layer")
            ords = np.where(plantedS, cal[ev], -1)
            if ev == "插秧":
                ords = np.where(cal["censored"], -1, ords)
            rgba, items = date_rgba(to_grid(ords, rows, cols, shape))
            show_map([(rgba, 0.9, f"{ev}日")], data, key=f"cal_{season}_{ev}_{rice_key}", height=460)
            legend(items)
            st.caption("日期為首次判定進入該階段之期別（解析度約 10–15 天）。"
                       + ("插秧發生於當年第 1 期之前者不計。" if ev == "插秧" else ""))
        with right:
            st.markdown("**每週收割面積**")
            fig = go.Figure()
            week_bars(fig, I.weekly_area(np.where(plantedS, cal["收割"], -1), PX_HA), "已判定收割", "#E3B23C")
            if live is not None:
                week_bars(fig, I.weekly_area(live["forecast"], PX_HA), "預報（尚未收割）", "#E53935", 0.75)
            fig.update_layout(barmode="stack", height=380, yaxis_title="面積 (ha)",
                              margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch")
            if live is not None:
                st.success(f"截至 {pred_dates[-1]}，尚有 {live['n_pending'] * PX_HA:,.1f} ha 未收割；"
                           "紅色為依目前物候階段推估之收割週。")
            st.caption("可用於收割機、烘乾廠排程，以及收割後稻草露天燃燒之稽查時段規劃。")

        st.divider()
        st.markdown("#### 預報可信度回測：假設今天是……")
        st.caption("只用所選日期以前的影像重新插值與預測，推估各田收割日，再與事後完整資料判定之收割日比較。"
                   "後續各階段天數採前一年同期作之實際天數（無資料時用預設值）。")
        if not ana["complete"][season]:
            st.info("本期作尚未結束，完成後即可回測。")
        else:
            (m0, d0), (m1, d1) = I.SEASONS[season]
            opts = [d for d in pred_dates if date(year, m0, d0) + timedelta(days=30) <= d < date(year, m1, d1)]
            harvest_med = summ["收割中位數"]
            target = (date.fromisoformat(harvest_med) - timedelta(days=45)) if harvest_med != "—" else opts[len(opts) // 2]
            default = min(opts, key=lambda d: abs((d - target).days)) if opts else None
            if not opts:
                st.info("可回測之期別不足。")
            else:
                asof = st.select_slider("截至日期", options=opts, value=default,
                                        format_func=lambda d: d.strftime("%Y-%m-%d"), key=f"asof_{season}")
                r = cached_asof(village, year, clear_thr, k, rice_key, season, asof, data, ana)
                c = st.columns(4)
                c[0].metric("需預報面積", f"{r['n_pending'] * PX_HA:,.1f} ha")
                c[1].metric("平均絕對誤差", f"{r['mae']:.1f} 天" if r["mae"] is not None else "—")
                c[2].metric("平均偏差", f"{r['bias']:+.1f} 天" if r["bias"] is not None else "—",
                            help="正值＝預報較實際晚")
                c[3].metric("誤差 ±15 天內", f"{r['within15']:.0%}" if r["within15"] is not None else "—")
                fig = go.Figure()
                week_bars(fig, I.weekly_area(r["forecast"], PX_HA), f"預報（{asof:%m/%d} 時）", "#E53935", 0.8)
                week_bars(fig, I.weekly_area(r["observed"], PX_HA), "事後判定", "#E3B23C", 0.8)
                fig.update_layout(barmode="group", height=320, yaxis_title="面積 (ha)",
                                  margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig, width="stretch")
                st.caption("採用天數：" + "、".join(f"{a}→下一階段 {v:.0f} 天" for a, v in r["durations"].items())
                           + f"（{r['source']}）。「事後判定」亦為本系統之判釋結果，非地面調查資料。")

# ---------- ⚠️ 巡查清單 ----------
with tabC:
    if not seasons or len(rows) == 0:
        st.info("沒有可分析之期作。")
    else:
        season = st.radio("期作", seasons, horizontal=True, key="season_C")
        df, amap = cached_anomalies(village, year, clear_thr, k, rice_key, season, data, ana)
        left, right = st.columns([3, 2])
        with left:
            show_map([(categorical_rgba(amap.astype(np.int16) - 1, ANOMALY_COLORS, -1), 0.95, "異常區塊")],
                     data, key=f"anom_{season}_{rice_key}")
            legend([(ANOMALY_NAMES[i + 1], c) for i, c in enumerate(ANOMALY_COLORS)])
        with right:
            if len(df):
                agg = df.groupby("類型")["面積 (ha)"].agg(["count", "sum"]).rename(
                    columns={"count": "區塊數", "sum": "面積 (ha)"}).round(2)
                st.dataframe(agg, width="stretch")
                st.dataframe(df, hide_index=True, width="stretch", height=300)
                st.download_button("下載巡查清單 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"巡查清單_{village}_{year}_{season}.csv", mime="text/csv")
            else:
                st.success("本期作未偵測到需巡查之區塊。")
        st.caption(f"判定標準：抽穗日較村里中位數晚／早 {I.LATE_DAYS} 天以上為生長落後／提前；"
                   f"季內 NDVI 峰值穩健 z 分數低於 {I.WEAK_Z} 為生長勢偏弱（可能缺水、病蟲害或災損）；"
                   f"期作結束仍未進入成熟期者為未進入成熟期。僅列出 ≥ {I.MIN_PATCH_PX * PX_HA:.1f} ha 之連續區塊。")

# ---------- 🎞 物候動態 ----------
with tabD:
    if len(rows) == 0:
        st.info("沒有判釋為水稻的像素。")
    else:
        pred = ana["pred"]
        share = np.stack([[(pred[j] == s).mean() * 100 for s in range(4)] for j in range(pred.shape[0])])
        interp = [slot_acq[j + n_pre] is None for j in range(len(pred_dates))]
        fig = go.Figure()
        for s in range(4):
            fig.add_scatter(x=pred_dates, y=share[:, s], name=P.STAGE_NAMES[s], mode="lines+markers",
                            line=dict(color=P.STAGE_COLORS[s], width=3),
                            marker=dict(size=8, color=["white" if f else P.STAGE_COLORS[s] for f in interp],
                                        line=dict(color=P.STAGE_COLORS[s], width=2)),
                            hovertemplate="%{y:.1f}%")
        fig.update_layout(height=340, yaxis_title="稻米像素中之比例 (%)", yaxis_range=[0, 102],
                          hovermode="x unified", margin=dict(t=20, b=10),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        st.caption("空心點為該期無可用影像、以 SG 插值補齊後預測。")
        frames = build_player_frames(village, year, clear_thr, k, rice_key, data, ana)
        st.iframe(player_html(frames, data["geom_4326"]), height=720)

# ---------- 🔧 技術細節 ----------
with tabE:
    st.markdown("#### 影像篩選與雲遮")
    fig = go.Figure()
    for used, name, color in ((False, "未採用", "#B0B0B0"), (True, "採用", "#2E7D32")):
        sub = acq_df[acq_df.used == used]
        fig.add_bar(x=sub.date, y=sub.clear_frac * 100, name=name, marker_color=color)
    fig.add_hline(y=clear_thr * 100, line_dash="dash", line_color="#D32F2F",
                  annotation_text=f"門檻 {clear_thr:.0%}")
    fig.update_layout(height=300, yaxis_title="村里內晴空比例 (%)", bargap=0.1,
                      margin=dict(t=20, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([1, 2])
    with left:
        slot_tbl = pd.DataFrame({
            "期別": [f"前年{i - n_pre}" if i < n_pre else f"T{i - n_pre + 1}" for i in range(len(slot_dates))],
            "名目日期": slot_dates,
            "採用影像": [str(a) if a is not None else "— 缺（插值補齊）" for a in slot_acq],
            "晴空比例": [f"{acq_df.set_index('date').clear_frac.get(a, np.nan):.0%}" if a else "" for a in slot_acq],
        })
        st.dataframe(slot_tbl, hide_index=True, height=480, width="stretch")
        st.caption(f"前 {n_pre} 期取自前一年年底，作為 LSTM 輸入視窗，使當年第 1 期即有預測。")
    with right:
        avail = [i for i, a in enumerate(slot_acq) if a is not None]
        pick = st.select_slider("檢視真彩影像", options=avail, value=avail[-1],
                                format_func=lambda i: f"{slot_acq[i]}")
        show_map([(true_color_rgba(data["cube"][pick], vmask), 1.0, f"Sentinel-2 {slot_acq[pick]}")],
                 data, key=f"tc_{pick}")
        legend([("雲／雲影／卷雲（已排除）", "rgb(255,64,160)")])

    st.markdown("#### 非監督稻米判釋（各期作分別分群）")
    cols_s = st.columns(max(len(clusters), 1))
    for col, (s, c) in zip(cols_s, clusters.items()):
        with col:
            x = [slot_dates[i] for i in c["idx"]]
            fig = go.Figure()
            for ci in range(k):
                fig.add_scatter(x=x, y=c["curves"][ci], mode="lines+markers", connectgaps=True,
                                name=f"Cluster {ci}（{c['counts'][ci]:,} px）",
                                line=dict(color=CLUSTER_COLORS[ci], width=4 if ci in rice_ids[s] else 2))
            fig.update_layout(height=330, yaxis_title="平均 NDVI", title=f"{s}：各群集平均 NDVI",
                              margin=dict(t=50, b=10), legend=dict(font=dict(size=10)))
            st.plotly_chart(fig, width="stretch")
            st.multiselect(f"{s}水稻群集", list(range(k)), default=c["suggested"],
                           format_func=lambda ci: f"Cluster {ci}", key=rice_widget_key(s))
    st.caption("水稻典型曲線：淹水低點 → 陡升 → 高峰 → 緩降。系統依振幅、峰值與低點先於峰值自動建議（粗線），可自行調整。")
    st.caption("模型：S2+Month LSTM（SG 插值、seq5），models/model_S2plusMonth.npz（NumPy 推論）")
