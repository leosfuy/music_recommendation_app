import sqlite3
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # 進度條工具

# ================= 設定區 =================
DB_FILE = "training_data.db"
TRIPLET_CSV = "triplets.csv"
MODEL_SAVE_PATH = "bestcnn_model.pth"

# 設定最大時間長度 (Frame數)，太長切掉，太短補0
# 根據 MSD 資料，通常 500~1000 左右比較剛好
MAX_LEN = 500 
BATCH_SIZE = 16  # 如果記憶體不夠，改小一點 (例如 8)
EPOCHS = 10      # 訓練幾輪
# =========================================

# 1. 定義 Dataset (負責把資料從 DB 挖出來變矩陣)
class MSDTripletDataset(Dataset):
    def __init__(self, csv_file, db_file):
        self.triplets = pd.read_csv(csv_file)
        self.db_file = db_file
        
    def __len__(self):
        return len(self.triplets)
    
    # 這是最關鍵的函數：把 JSON 字串變成 PyTorch Tensor
    def preprocess(self, timbre_json, pitches_json):
        try:
            # 解析 JSON -> Numpy
            t = np.array(json.loads(timbre_json), dtype=np.float32) # (Time, 12)
            p = np.array(json.loads(pitches_json), dtype=np.float32) # (Time, 12)
            
            # 把 Timbre 和 Pitches 疊在一起 -> (Time, 24)
            combined = np.concatenate([t, p], axis=1)
            
            # 統一長度處理 (Padding / Truncating)
            curr_len = combined.shape[0]
            if curr_len > MAX_LEN:
                combined = combined[:MAX_LEN, :] # 太長切掉
            else:
                # 太短補 0
                padding = np.zeros((MAX_LEN - curr_len, 24), dtype=np.float32)
                combined = np.vstack((combined, padding))
            
            # 轉置：PyTorch Conv1d 需要 (Channel, Time) -> (24, 500)
            return torch.tensor(combined.T)
            
        except Exception as e:
            # 萬一資料有壞掉的，回傳全 0 矩陣避免程式崩潰
            return torch.zeros((24, MAX_LEN), dtype=torch.float32)

    def __getitem__(self, idx):
        # 1. 拿出一組題目 (Anchor, Positive, Negative)
        row = self.triplets.iloc[idx]
        ids = [row['anchor_id'], row['positive_id'], row['negative_id']]
        
        tensors = []
        
        # 2. 連線 DB (每次讀取都連線一次確保 Thread Safe)
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        for song_id in ids:
            cursor.execute("SELECT segments_timbre, segments_pitches FROM training_songs WHERE song_id=?", (song_id,))
            result = cursor.fetchone()
            
            if result:
                tensors.append(self.preprocess(result[0], result[1]))
            else:
                # 找不到就給全 0
                tensors.append(torch.zeros((24, MAX_LEN), dtype=torch.float32))
                
        conn.close()
        
        # 回傳三個 Tensor: Anchor, Positive, Negative
        return tensors[0], tensors[1], tensors[2]

# 2. 定義 CNN 模型 (廚師)
class MusicCNN(nn.Module):
    def __init__(self):
        super(MusicCNN, self).__init__()
        # 輸入 Channel = 24 (12 Timbre + 12 Pitch)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(24, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2), # 長度變一半
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # 強制壓扁成 (Batch, 256, 1)
        )
        
        self.fc = nn.Linear(256, 128) # 最終輸出 128 維的 Embedding

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # 拉直
        x = self.fc(x)
        # L2 Normalize (對於計算 Cosine Similarity 很重要)
        return x / x.norm(dim=1, keepdim=True)

# 3. 訓練主程式
if __name__ == "__main__":
    print("🔥 載入 Dataset...")
    dataset = MSDTripletDataset(TRIPLET_CSV, DB_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("🧠 初始化模型...")
    model = MusicCNN()
    
    # 這是專門給三元組用的 Loss Function
    # margin=1.0 代表：希望 正樣本距離 比 負樣本距離 近至少 1.0
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🚀 開始訓練 (共 {EPOCHS} 輪)...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        
        # 使用 tqdm 顯示進度條
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for anchor, positive, negative in progress_bar:
            optimizer.zero_grad()
            
            # 1. 算出三個向量
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)
            
            # 2. 計算 Loss (希望 A跟P近，A跟N遠)
            loss = criterion(emb_a, emb_p, emb_n)
            
            # 3. 反向傳播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成! 平均 Loss: {avg_loss:.4f}")
        
        # 每一輪都存一次檔，以防電腦當機
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\n🎉 訓練完成！模型已儲存為 {MODEL_SAVE_PATH}")
    print("現在你可以用這個模型來做推薦系統了！")