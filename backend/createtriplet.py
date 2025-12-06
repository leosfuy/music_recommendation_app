import sqlite3
import pandas as pd
import random
import os

# ================= 設定區 =================
DB_FILE = "training_data.db"
OUTPUT_CSV = "triplets.csv"
NUM_TRIPLETS = 10000  # 生成 1 萬組訓練資料
# =========================================

def generate_triplet_csv():
    print(f"📂 連線至 {DB_FILE} 準備生成訓練清單...")
    
    if not os.path.exists(DB_FILE):
        print("❌ 找不到資料庫！請確認 training_data.db 存在。")
        return

    conn = sqlite3.connect(DB_FILE)
    
    # 1. 只需要撈 ID 和 作者名 (不用撈 segments，那樣太慢)
    print("⏳ 正在讀取歌曲清單...")
    df = pd.read_sql("SELECT song_id, artist_name FROM training_songs", conn)
    conn.close()
    
    print(f"📊 資料庫共有 {len(df)} 首歌。正在整理作者歸戶...")

    # 2. 把每個作者的歌整理成字典: {'artist': ['id1', 'id2'], ...}
    artist_groups = df.groupby('artist_name')['song_id'].apply(list).to_dict()
    
    # 剔除只有一首歌的作者 (因為他沒辦法當 Positive，找不到同作者的第二首歌)
    valid_artists = [a for a, songs in artist_groups.items() if len(songs) >= 2]
    
    if len(valid_artists) < 2:
        print("❌ 有效作者數量不足，無法生成三元組！")
        return

    triplets = []
    print(f"🚀 開始生成 {NUM_TRIPLETS} 組訓練資料...")
    
    # 3. 隨機生成題目
    for i in range(NUM_TRIPLETS):
        try:
            # --- 步驟 A: 選主角 (Anchor) ---
            anchor_artist = random.choice(valid_artists)
            songs_of_anchor = artist_groups[anchor_artist]
            anchor_song = random.choice(songs_of_anchor)
            
            # --- 步驟 B: 選同作者的歌 (Positive) ---
            positive_song = random.choice(songs_of_anchor)
            # 確保不要選到自己 (雖然機率很低)
            while positive_song == anchor_song:
                positive_song = random.choice(songs_of_anchor)
                
            # --- 步驟 C: 選不同作者的歌 (Negative) ---
            negative_artist = random.choice(valid_artists)
            while negative_artist == anchor_artist:
                negative_artist = random.choice(valid_artists)
            
            negative_song = random.choice(artist_groups[negative_artist])
            
            # 加入清單
            triplets.append([anchor_song, positive_song, negative_song])
            
        except Exception as e:
            continue

    # 4. 存檔
    triplet_df = pd.DataFrame(triplets, columns=['anchor_id', 'positive_id', 'negative_id'])
    triplet_df.to_csv(OUTPUT_CSV, index=False)
    
    print("-" * 30)
    print(f"✅ 成功生成：{OUTPUT_CSV}")
    print(f"📝 內容範例 (前 3 行)：")
    print(triplet_df.head(3))
    print("-" * 30)

if __name__ == "__main__":
    generate_triplet_csv()