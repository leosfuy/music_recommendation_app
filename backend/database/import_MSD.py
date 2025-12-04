import os
import mysql.connector
import numpy as np
import json
import sys
import hdf5_getters

def read_song(path):
    """
    讀取單一 .h5 檔並萃取 MSS(音色/音高統計值)
    path: .h5 路徑
    return: dict (可直接 INSERT 到 MySQL)
    """

    # 開啟 HDF5
    h5 = hdf5_getters.open_h5_file_read(path)
    try:
        # --- 取得 segments 資料（真正用於相似度） ---
        seg_timbre = hdf5_getters.get_segments_timbre(h5)       # shape: (N, 12)
        seg_pitches = hdf5_getters.get_segments_pitches(h5)     # shape: (N, 12)

        # 安全檢查：避免空陣列造成錯誤
        if len(seg_timbre) == 0 or len(seg_pitches) == 0:
            h5.close()
            return None

        # --- 計算統計特徵（要存入 MySQL 的特徵） ---
        timbre_mean = np.mean(seg_timbre, axis=0).tolist()       # 12 維
        timbre_std = np.std(seg_timbre, axis=0).tolist()         # 12 維
        pitches_mean = np.mean(seg_pitches, axis=0).tolist()     # 12 維
        pitches_std = np.std(seg_pitches, axis=0).tolist()       # 12 維

        # --- 組成字典 ---
        data = {
            "song_id": hdf5_getters.get_song_id(h5).decode(),
            "artist_name": hdf5_getters.get_artist_name(h5).decode(),
            "title": hdf5_getters.get_title(h5).decode(),
            "year": int(hdf5_getters.get_year(h5)),

            "duration": float(hdf5_getters.get_duration(h5)),
            "tempo": float(hdf5_getters.get_tempo(h5)),
            "loudness": float(hdf5_getters.get_loudness(h5)),
            "key": int(hdf5_getters.get_key(h5)),
            "mode": int(hdf5_getters.get_mode(h5)), #1大調/0小調

            "timbre_mean": timbre_mean,
            "timbre_std": timbre_std,
            "pitches_mean": pitches_mean,
            "pitches_std": pitches_std
        }

        return data
    finally:
        h5.close()

def insert_song(connection, data):
    cursor = connection.cursor()    # 建立 SQL 游標

    # SQL 插入語法（使用佔位符%s避免 SQL injection）
    sql = """
    INSERT INTO songs (
        song_id, artist_name, title, year,
        duration, tempo, loudness, `key`, mode,
        timbre_mean, timbre_std, pitches_mean, pitches_std
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    # 執行 SQL 插入
    cursor.execute(sql, (
        data["song_id"], data["artist_name"], data["title"], data["year"],
        data["duration"], data["tempo"], data["loudness"],
        data["key"], data["mode"],
        json.dumps(data["timbre_mean"]),  
        json.dumps(data["timbre_std"]),
        json.dumps(data["pitches_mean"]),
        json.dumps(data["pitches_std"])
    ))

    connection.commit()  # 寫入資料庫

MSDRootDir = "./msd_targz" #原始資料位置 將msd_targz放在music_recommendation_app裡 讓terminal路徑在music_recommendation_app

#main
if __name__ == "__main__":
    #建立連線
    try:
        connection = mysql.connector.connect(host="localhost", # MySQL 主機位置(localhost = 127.0.0.1 = 本機)
                                             port="3306",
                                             user="root",
                                             password="12345678")
    except mysql.connector.Error as err:
        print("資料庫連線失敗：", err)
        sys.exit(1)

    cursor = connection.cursor() #建立游標
    cursor.execute("CREATE DATABASE IF NOT EXISTS `MSD`;") #建資料庫
    cursor.execute("USE `MSD`") #使用MSD這個資料庫
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id INT AUTO_INCREMENT PRIMARY KEY,      -- 自動遞增唯一 ID (自訂ID)
            song_id VARCHAR(50) UNIQUE,             -- MSD 裡的唯一 song_id
            artist_name VARCHAR(255),               -- 歌手名稱
            title VARCHAR(255),                     -- 歌名
            year INT,                               -- 發行年份
                   
            duration FLOAT,                         -- 歌曲長度 (秒)
            tempo FLOAT,                            -- 節奏 BPM
            loudness FLOAT,                         -- 音量 dB
            `key` INT,                              -- 調性
            `mode` INT,                             -- 大調/小調
                   
            timbre_mean JSON,                       -- 音色平均 (12 維)
            timbre_std  JSON,                       -- 音色標準差 (12 維)
            pitches_mean JSON,                      -- 音高平均 (12 維)
            pitches_std  JSON                       -- 音高標準差 (12 維)
        );
        """)
    if not os.path.exists(MSDRootDir):
        print("路徑不存在：", MSDRootDir)
    else:
        # os.walk：會遞迴逐層讀取資料夾與子資料夾 root:現在所在的資料夾路徑 dirs:root 內的所有子資料夾名稱（list） files:root 內所有檔案名稱（list）
        for root, dirs, files in os.walk(MSDRootDir):
            for f in files:
                if f.endswith(".h5"):  # 只處理 .h5 檔案
                    path = os.path.join(root, f)  # 取得完整檔案路徑
                    try:
                        data = read_song(path)        # 讀取 .h5 內容
                        if data is None:
                            print("跳過空資料：", path)
                            continue

                        insert_song(connection, data)         # 插入 MySQL
                        print("匯入成功：", data["song_id"])
                    except Exception as e:
                        print("匯入失敗：", path, "錯誤：", e)
                        

        cursor.close()
        connection.close()

