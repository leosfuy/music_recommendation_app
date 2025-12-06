#.sql轉.db檔
import sqlite3
import os
import re
import sys

# ================= 設定區 =================

SQL_FILE_PATH = "/Users/leeangel/Desktop/資料庫專題/MSD40000.sql"
DB_FILE_PATH = "training_data.db"
# =========================================

def rebuild_db_with_titles():
    print(f"🚀 正在重建資料庫 (加入 Title 欄位)...")
    
    # 1. 刪除舊的資料庫 (因為我們要改表格結構)
    if os.path.exists(DB_FILE_PATH):
        os.remove(DB_FILE_PATH)
        print("🗑️ 舊資料庫已刪除，準備重建...")
        
    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()

    # 2. 建立包含 title 的新表格
    cursor.execute('''
        CREATE TABLE training_songs (
            song_id TEXT PRIMARY KEY,
            title TEXT,        
            artist_name TEXT,
            segments_timbre TEXT,
            segments_pitches TEXT,
            embedding BLOB     -- 預留給之後存 Embedding 用
        )
    ''')
    
    value_pattern = re.compile(r"\((.*?)\)")
    
    count = 0
    skipped = 0
    
    print("⏳ 開始讀取大檔案並匯入... (請稍候)")
    
    try:
        with open(SQL_FILE_PATH, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if "INSERT INTO" in line or "VALUES" in line:
                    clean_line = line.replace("\\'", "''").replace('\\"', '"')
                    matches = value_pattern.findall(clean_line)
                    
                    for match in matches:
                        try:
                            if not match.endswith(','): match += ','
                            match_fixed = match.replace("NULL", "None")
                            data_tuple = eval(f"({match_fixed})")
                            
                            # --- 欄位抓取 (根據你的 SQL 結構) ---
                            if len(data_tuple) >= 16:
                                song_id = data_tuple[1]
                                artist = data_tuple[2]
                                title = data_tuple[3]   
                                seg_timbre = data_tuple[14]
                                seg_pitches = data_tuple[15]
                                
                                # 確保關鍵資料存在
                                if song_id and seg_timbre and seg_pitches:
                                    # 插入資料庫 (embedding 先給 None)
                                    cursor.execute(
                                        "INSERT INTO training_songs (song_id, title, artist_name, segments_timbre, segments_pitches) VALUES (?, ?, ?, ?, ?)", 
                                        (song_id, title, artist, seg_timbre, seg_pitches)
                                    )
                                    count += 1
                            
                        except Exception:
                            skipped += 1
                            continue

                if count % 5000 == 0 and count > 0:
                    sys.stdout.write(f"\r已匯入: {count} 筆 | 跳過: {skipped} 筆")
                    sys.stdout.flush()

    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        
    finally:
        conn.commit()
        conn.close()
        print(f"\n\n🎉 資料庫重建完成！現在裡面有 {count} 首歌")

if __name__ == "__main__":
    rebuild_db_with_titles()