# 用 MySQL + CNN embedding 進行歌曲推薦
import pymysql
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

def recommend_demo():
    print("📂 連線 MySQL 資料庫...")

    conn = pymysql.connect(
        host="localhost",
        user="appuser",
        password="AppUser123",
        database="music_recommendation",
        charset="utf8mb4"
    )


    df = pd.read_sql(
        """
        SELECT song_id, title, artist_name, embedding
        FROM training_songs
        WHERE embedding IS NOT NULL
        """,
        conn
    )
    conn.close()

    if df.empty:
        print("❌ MySQL 裡沒有 embedding")
        return

    def to_array(blob):
        return np.frombuffer(blob, dtype=np.float32)

    matrix = np.stack(df["embedding"].apply(to_array).values)

    target_row = df.sample(1).iloc[0]
    target_idx = df.index.get_loc(target_row.name)

    print("-" * 50)
    print(f"🎵 目標歌曲: {target_row['title']} - {target_row['artist_name']}")
    print("-" * 50)

    sims = cosine_similarity(
        matrix[target_idx].reshape(1, -1),
        matrix
    )[0]

    top_indices = sims.argsort()[::-1][1:6]

    for i in top_indices:
        row = df.iloc[i]
        print(f"🔥 {row['title']} - {row['artist_name']} (相似度: {sims[i]:.4f})")

if __name__ == "__main__":
    recommend_demo()
