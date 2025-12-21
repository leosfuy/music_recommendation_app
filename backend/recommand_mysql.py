# 用 MySQL + CNN 模型 + embedding 進行歌曲推薦
# （模型已載入，embedding 由 precompute 階段產生）
import os
import pymysql
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

# ================= MySQL 設定 =================
DB_CONFIG = {
    "host": "localhost",
    "user": "appuser",
    "password": "AppUser123",
    "database": "music_recommendation",
    "charset": "utf8mb4"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bestcnn_model.pth")

# ============================================

# ================= 模型定義 =================
class MusicCNN(nn.Module):
    def __init__(self):
        super(MusicCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(24, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, 128)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x / x.norm(dim=1, keepdim=True)
# ============================================

def recommend_demo():
    print("📂 連線 MySQL 資料庫...")
    conn = pymysql.connect(**DB_CONFIG)

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
        print("❌ MySQL 中沒有 embedding，請先執行 precompute")
        return


    # ===== embedding → numpy matrix =====
    def to_array(blob):
        return np.frombuffer(blob, dtype=np.float32)

    matrix = np.stack(df["embedding"].apply(to_array).values)

    # ===== 隨機選一首歌當 target =====
    target_row = df.sample(1).iloc[0]
    target_idx = df.index.get_loc(target_row.name)

    print("-" * 50)
    print(f"🎵 目標歌曲: {target_row['title']} - {target_row['artist_name']}")
    print("-" * 50)

    # ===== cosine similarity =====
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
