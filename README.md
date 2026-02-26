# Music Recommendation App
**核心概念**：結合音訊特徵分析與機器學習的推薦系統，能根據上傳音檔產出 5 首風格相似的歌曲並串接 Spotify 播放

## 核心功能 
* **音訊特徵提取**：利用 `Librosa` 庫分析音檔的 Tempo, MFCC, Loudness, Chroma 等關鍵音訊特徵。
* **推薦邏輯**：基於 `KNN (K-Nearest Neighbors)` 演算法，從 **20 萬筆** 歌曲資料庫中運算最相似曲目。
* **跨平台整合**：前端使用 `Dart` 開發，並串接 `Spotify API` 實現一鍵跳轉播放功能。
* **後端架構**：使用 `FastAPI` 處理前端請求，並透過 `MySQL` 進行大規模數據檢索。
