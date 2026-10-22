# 霧峰區水稻物候階段 Demo

以 Sentinel-2 時序影像，自動判釋臺中市霧峰區各村里的水稻田，並以 LSTM 預測每一期的水稻物候階段（營養生長期／生殖生長期／成熟期／非稻作）。

選擇村里與年份後，網頁會直接以 HTTP 讀取雲端影像中該村里的範圍，全程在記憶體內運算，**不需事先下載任何影像**。

![左：預測分類圖、右：當期 Sentinel-2 影像](docs/demo_player.gif)

## 流程

```
Planetary Computer STAC（Sentinel-2 L2A）
  │  COG 視窗讀取：只取村里範圍（外擴 300 m），重投影至 EPSG:3826 / 10 m
  ▼
雲遮排除
  │  SCL 逐像素排除雲／雲影／卷雲；村里內晴空比例低於門檻之日期整期剔除
  ▼
30 期時間網格
  │  每期於 ±8 天內挑選最晴之影像（每幅影像只用一次），無影像之期別留待插值
  ▼
非監督稻米判釋
  │  NDVI 物候特徵（最小值、最大值、振幅、上升／下降斜率、峰值時間…）＋ K-means
  │  依「淹水低點 → 陡升 → 峰值」曲線形狀自動建議水稻群集，可手動調整
  ▼
LSTM 物候階段預測
     NDVI / NDWI / RVI → SG 插值補缺 → SG 平滑 → min-max 正規化 ＋ 月份 sin/cos
     以前 5 期為輸入，預測下一期之階段
```

## 成果範例：北勢里 2025 年

一期作與二期作皆可清楚看到「營養生長 → 生殖生長 → 成熟 → 休耕」的循環。

![各階段比例](docs/stage_share.png)

![各期物候階段圖](docs/stage_maps.png)

## 網頁功能

| 頁籤 | 內容 |
|---|---|
| ① 影像篩選與雲遮 | 各日期之村里晴空比例、採用期別對照表、各期真彩影像（雲遮處以桃紅色標示） |
| ② 非監督稻米判釋 | 各群集平均 NDVI 曲線、自動建議／手動選擇水稻群集、稻米遮罩與面積 |
| ③ LSTM 物候階段 | 各階段比例折線圖；左右對照播放器（預測分類圖 ／ 當期影像），可拖曳期別或自動播放 |

## 模型

- 架構：2 層 LSTM（128 units）＋ 全連接層，4 類輸出
- 輸入：S2+Month 共 5 個特徵（NDVI、NDWI、RVI、Month_sin、Month_cos），序列長度 5
- 缺值補全：Savitzky–Golay 插值
- 訓練資料：臺中地區 4 個訓練樣區、2 個獨立測試樣區之逐像素物候標註（2023 年）

| 測試樣區 | Accuracy | F1 | Mean IoU |
|---|---|---|---|
| test_area1 | 0.898 | 0.898 | 0.806 |
| test_area2 | 0.861 | 0.860 | 0.739 |

完整指標見 [`models/evaluation_metrics.csv`](models/evaluation_metrics.csv)（含 S1、S1+S2 版本供比較；本 Demo 僅使用光學影像，故採 S2+Month 模型）。

## 執行

需要 Python 3.10 以上，以及可連外的網路（讀取 Planetary Computer 影像）。

```bash
python -m venv .venv
source .venv/bin/activate            # Windows：.venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
streamlit run app.py
```

瀏覽器開啟 http://localhost:8501，於左側選擇村里與年份後按「載入影像並分析」。首次載入約 1–3 分鐘（視村里面積與網路速度而定）。

輸出靜態成果圖（README 圖片即由此產生）：

```bash
python scripts/export_figures.py --village 北勢里 --year 2025 --out docs --font <中文字型檔.otf>
```

## 專案結構

```
app.py                      Streamlit 網頁
pipeline.py                 影像讀取、雲遮、分群、特徵與 LSTM 推論
scripts/export_figures.py   輸出成果圖與 GIF
models/                     模型權重與評估指標
wufeng_villages.geojson     霧峰區村里界
docs/                       README 圖片
```

## 限制

- 模型以研究區 2023 年資料訓練；Planetary Computer 之 L2A 產品與訓練所用影像之前處理不完全相同，其他年份與地區之結果僅供參考。
- 每年前 5 期作為輸入視窗，自第 6 期起才有預測。
- 長期連續雲遮（如梅雨季）之期別以插值補齊，預測不確定性較高。

## 資料來源

- Sentinel-2 L2A：Copernicus Sentinel data，經 [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) 取得
- 村里界：內政部國土測繪中心 村里界圖（政府資料開放授權條款）
- 底圖：Esri World Imagery、OpenStreetMap
