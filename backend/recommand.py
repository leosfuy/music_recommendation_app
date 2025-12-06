# 用bestcnn_model.pth歌曲推薦
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DB_FILE = "training_data.db"

def recommend_demo():
    print("📂 載入資料庫...")
    conn = sqlite3.connect(DB_FILE)
    
    # 1. 直接撈出全庫 Embedding
    df = pd.read_sql("SELECT song_id, title, artist_name, embedding FROM training_songs WHERE embedding IS NOT NULL", conn)
    conn.close()
    
    if df.empty:
        print("❌ 資料庫裡沒有 Embedding，請先執行 run_precompute.py")
        return

    # 轉成 Numpy Matrix
    def to_array(blob): return np.frombuffer(blob, dtype=np.float32)
    matrix = np.stack(df['embedding'].apply(to_array).values)
    
    # 2. 隨機選一首歌來測試 
    # 從 df 裡隨機抽一行，保證 ID 一定存在
    target_row = df.sample(1).iloc[0]
    target_idx = df.index.get_loc(target_row.name) # 取得它在 matrix 中的 index
    
    print("-" * 50)
    print(f"🎵 目標歌曲: {target_row['title']} - {target_row['artist_name']}")
    print("-" * 50)
    
    # 3. 計算相似度
    target_vec = matrix[target_idx].reshape(1, -1)
    sims = cosine_similarity(target_vec, matrix)[0]
    
    # 取前 5 名
    top_indices = sims.argsort()[::-1][1:6]
    
    for i in top_indices:
        rec_row = df.iloc[i]
        print(f"🔥 {rec_row['title']} - {rec_row['artist_name']} (相似度: {sims[i]:.4f})")

if __name__ == "__main__":
    recommend_demo()