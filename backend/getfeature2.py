import os
import h5py
import numpy as np
import pandas as pd
import sqlite3

# 你的 subset 資料夾路徑
DATASET_PATH = "/Users/leeangel/Desktop/資料庫專題/MillionSongSubset"

# SQLite 資料庫位置
DB_PATH = "/Users/leeangel/Desktop/資料庫專題/features.db"

# 收集結果
rows = []

def get_feature_from_h5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            analysis = f['analysis']
            metadata = f['metadata']

            # 基本特徵
            tempo = analysis['songs']['tempo'][0]
            loudness = analysis['songs']['loudness'][0]
            key = analysis['songs']['key'][0]
            mode = analysis['songs']['mode'][0]

            # Timbre（類似 MFCC 概念）
            timbre = analysis['segments_timbre'][:]  # shape: (segments, 12)
            timbre_mean = np.mean(timbre, axis=0)

            # Pitches（類似 chroma）
            pitches = analysis['segments_pitches'][:]  # shape: (segments, 12)
            pitch_mean = np.mean(pitches, axis=0)

            # song ID
            song_id = metadata["songs"]["song_id"][0].decode()

            # title 與 artist_name
            title = metadata["songs"]["title"][0].decode()
            artist_name = metadata["songs"]["artist_name"][0].decode()

            # 打包成字典
            data = {
                "song_id": song_id,
                "title": title,
                "artist_name":artist_name,
                "tempo": tempo,
                "loudness": loudness,
                "key": key,
                "mode": mode,
            }

            # 加入 timbre 平均值
            for i in range(12):
                data[f"timbre_{i+1}"] = timbre_mean[i]

            # 加入 pitch 平均值
            for i in range(12):
                data[f"pitch_{i+1}"] = pitch_mean[i]

            return data

    except Exception as e:
        print("Error:", file_path, e)
        return None


print("開始提取特徵...")

count = 0

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".h5"):
            fpath = os.path.join(root, file)

            features = get_feature_from_h5(fpath)
            if features:
                rows.append(features)

            count += 1
            if count % 500 == 0:
                print(f"已處理 {count} 首歌")

print("特徵萃取完成，共", len(rows), "首歌")

# -----------------------------
# 存進 SQLite
# -----------------------------
df = pd.DataFrame(rows)

# 連線或建立 SQLite 資料庫
conn = sqlite3.connect(DB_PATH)
df.to_sql("songs_features", conn, if_exists="replace", index=False)
conn.close()

print(f"輸出至 SQLite 完成，資料庫路徑：{DB_PATH}")
