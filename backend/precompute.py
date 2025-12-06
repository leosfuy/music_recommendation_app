#模型預處理
import sqlite3
import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
import os

# ================= 設定區 =================
DB_FILE = "training_data.db"
MODEL_PATH = "bestcnn_model.pth" # 確保你有這個訓練好的模型
MAX_LEN = 500
BATCH_SIZE = 32
# =========================================

# 1. 模型結構
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

# 2. 資料處理
def preprocess_batch(rows):
    tensors = []
    valid_indices = []
    for i, row in enumerate(rows):
        try:
            # 注意索引：[0]id, [1]seg_timbre, [2]seg_pitches
            t = np.array(json.loads(row[1]), dtype=np.float32)
            p = np.array(json.loads(row[2]), dtype=np.float32)
            combined = np.concatenate([t, p], axis=1)
            
            curr_len = combined.shape[0]
            if curr_len > MAX_LEN:
                combined = combined[:MAX_LEN, :]
            else:
                padding = np.zeros((MAX_LEN - curr_len, 24), dtype=np.float32)
                combined = np.vstack((combined, padding))
            tensors.append(combined.T)
            valid_indices.append(i)
        except:
            continue
    if not tensors: return None, []
    return torch.tensor(np.array(tensors)), valid_indices

def main():
    print("🚀 [Step 1] 準備資料庫結構...")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 檢查並新增 embedding 欄位
    cursor.execute("PRAGMA table_info(training_songs)")
    columns = [info[1] for info in cursor.fetchall()]
    if "embedding" not in columns:
        cursor.execute("ALTER TABLE training_songs ADD COLUMN embedding BLOB")
        conn.commit()

    print("🚀 [Step 2] 載入模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicCNN().to(device)
    # 如果找不到模型會報錯，請確認 bestcnn_model.pth 在同目錄
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("🚀 [Step 3] 開始計算全庫 Embedding...")
    cursor.execute("SELECT count(*) FROM training_songs")
    total = cursor.fetchone()[0]
    
    # 只撈需要的三個欄位
    cursor.execute("SELECT song_id, segments_timbre, segments_pitches FROM training_songs")
    
    progress = tqdm(total=total)
    
    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows: break
        
        batch_tensors, valid_indices = preprocess_batch(rows)
        
        if batch_tensors is not None:
            batch_tensors = batch_tensors.to(device)
            with torch.no_grad():
                embeddings = model(batch_tensors).cpu().numpy()
            
            update_list = []
            for i, emb in enumerate(embeddings):
                orig_idx = valid_indices[i]
                s_id = rows[orig_idx][0]
                update_list.append((emb.tobytes(), s_id))
            
            conn.executemany("UPDATE training_songs SET embedding = ? WHERE song_id = ?", update_list)
            conn.commit()
            
        progress.update(len(rows))
        
    conn.close()
    print("\n✅ 所有歌曲特徵計算完成！現在可以進行推薦")

if __name__ == "__main__":
    main()