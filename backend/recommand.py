import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

# SQLite 資料庫路徑
DB_PATH = "features.db"

# 連線 SQLite 並讀取資料
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM songs_features", conn)
conn.close()
# 建立特徵矩陣
feature_cols = [c for c in df.columns if c not in ['song_id', 'title', 'artist_name']]
features = df[feature_cols].values

# song_id → index
song_id_to_index = {tid: idx for idx, tid in enumerate(df['song_id'])}

def recommend(song_id, top_k=5):
    if song_id not in song_id_to_index:
        print("Song ID 不存在！")
        return None

    idx = song_id_to_index[song_id]
    query_vector = features[idx].reshape(1, -1)

    # 計算餘弦相似度
    sim_scores = cosine_similarity(query_vector, features)[0]

    # 取 top_k 排除自己
    sim_indices = sim_scores.argsort()[::-1]
    sim_indices = [i for i in sim_indices if i != idx][:top_k]

    recommendations = df.iloc[sim_indices].copy()
    recommendations['similarity'] = sim_scores[sim_indices]

    return recommendations

# Demo
query_song_id = input("請輸入歌曲 song_id：")
rec = recommend(query_song_id, top_k=5)

if rec is not None:
    print("\n推薦歌曲清單：")
    for i, row in rec.iterrows():
        print(f"{i+1}. {row['song_id']} | {row.get('title', '未知名稱')} | {row.get('artist_name', '未知歌手')} | similarity: {row['similarity']:.3f}")
        print(f"   tempo: {row['tempo']}, loudness: {row['loudness']}, key: {row['key']}, mode: {row['mode']}")
        print("   timbre_mean:", [row[f'timbre_{j+1}'] for j in range(12)])
        print("   pitch_mean:", [row[f'pitch_{j+1}'] for j in range(12)])
        print("---------------------------------------------------")
