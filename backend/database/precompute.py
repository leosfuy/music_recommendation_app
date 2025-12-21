import os
import json # &numpy 把 json 轉數字矩陣
import numpy as np 
import torch #模型
import torch.nn as nn
from tqdm import tqdm
import mysql.connector
from dotenv import load_dotenv #載環境

load_dotenv()

# ================= 設定區 =================(資料量)
MODEL_PATH = "bestcnn_model.pth"
MAX_LEN = 500 #每首歌時間序列統一成500(啥是時間序列????)
BATCH_SIZE = 32 #一次處理32首
# =========================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

# 1. 模型結構（要跟 train 完全一樣） 定義模型
class MusicCNN(nn.Module):
    def __init__(self):
        super(MusicCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(24, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, 128)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x / x.norm(dim=1, keepdim=True)

# 2. 批次預處理:把 DB 的 JSON 字串變成模型可以吃的 Tensor(查Tensor是啥)
def preprocess_batch(rows):
    tensors = []
    valid_indices = []
    for i, row in enumerate(rows):
        try:
            # row = (song_id, segments_timbre, segments_pitches)
            t = np.array(json.loads(row["timbre_segments"]), dtype=np.float32)
            p = np.array(json.loads(row["pitches_segments"]), dtype=np.float32)

            combined = np.concatenate([t, p], axis=1)  # (Time, 24)

            curr_len = combined.shape[0]
            if curr_len > MAX_LEN:
                combined = combined[:MAX_LEN, :]
            else:
                padding = np.zeros((MAX_LEN - curr_len, 24), dtype=np.float32)
                combined = np.vstack((combined, padding))

            tensors.append(combined.T)   # (24, MAX_LEN)
            valid_indices.append(i)
        except:
            continue

    if not tensors:
        return None, []

    return torch.tensor(np.array(tensors)), valid_indices  # (B,24,MAX_LEN)

def ensure_embedding_column(cursor):
    cursor.execute("""
        SELECT COUNT(*) AS c
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'songs'
          AND COLUMN_NAME = 'embedding'
    """)
    exists = cursor.fetchone()["c"]   # dictionary cursor 要用 key 取值

    if not exists:
        cursor.execute("ALTER TABLE songs ADD COLUMN embedding BLOB NULL")


def main():
    print("🚀 [Step 1] 連線 MySQL & 準備欄位...")
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    ensure_embedding_column(cursor)
    conn.commit()

    print("🚀 [Step 2] 載入模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("device =", device)

    print("🚀 [Step 3] 計算全庫 Embedding...")

    # 先拿總數（進度條用）
    cursor.execute("SELECT COUNT(*) AS cnt FROM songs")
    total = cursor.fetchone()["cnt"]
    progress = tqdm(total=total)

    last_id = 0  # 已經處理到的歌的id

    while True:
        cursor.execute("""
            SELECT id, timbre_segments, pitches_segments
            FROM songs
            WHERE id > %s
            ORDER BY id
            LIMIT %s
        """, (last_id, BATCH_SIZE))

        rows = cursor.fetchall()
        if not rows:
            break

        batch_tensors, valid_indices = preprocess_batch(rows)

        if batch_tensors is not None:
            batch_tensors = batch_tensors.to(device)
            with torch.no_grad():
                embeddings = model(batch_tensors).cpu().numpy()

            update_list = []
            for i, emb in enumerate(embeddings):
                orig_idx = valid_indices[i]
                s_id = rows[orig_idx]["id"]
                update_list.append((emb.astype(np.float32).tobytes(), s_id))

            cursor.executemany(
                "UPDATE songs SET embedding=%s WHERE id=%s",
                update_list
            )
            conn.commit()

        last_id = rows[-1]["id"]
        progress.update(len(rows))



if __name__ == "__main__":
    main()
