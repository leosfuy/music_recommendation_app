import librosa
import numpy as np
import json

def extract_features(audio_path):
    """
    讀取音檔並提取能與 Million Song Dataset 進行比較的核心特徵。
    
    Args:
        audio_path (str): 音檔的路徑 (例如 'my_song.mp3' 或 'input.wav')
        
    Returns:
        dict: 包含特徵數據的字典
    """
    try:
        # 1. 載入音檔
        # sr=22050 是 librosa 的預設採樣率，也是許多音樂分析的標準
        y, sr = librosa.load(audio_path, sr=22050)
        
        print(f"正在處理: {audio_path} | 長度: {len(y)/sr:.2f} 秒")

        # ---------------------------------------------------------
        # 2. 提取特徵 (Feature Extraction)
        # ---------------------------------------------------------

        # A. MFCC (Mel-frequency cepstral coefficients) - 音色特徵
        # MSD 中包含類似的 Timbre 數據。MFCC 是語音和音樂處理中最常用的音色特徵。
        # n_mfcc=13 是標準設定，足以捕捉大部份音色資訊。
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # B. Chroma (色譜圖) - 音高/和聲特徵
        # 這代表了音樂的 12 個半音 (C, C#, D...) 的能量分佈，不管八度音高。
        # 這對於比較兩首歌是否使用相似的和弦進行非常有用。
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # C. Spectral Contrast - 頻譜對比度
        # 用於區分音樂的「紋理」，例如區分強烈的節奏音樂與平滑的音樂。
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # D. Tempo - 節奏 (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # ---------------------------------------------------------
        # 3. 數據聚合 (Aggregation / Statistics)
        # ---------------------------------------------------------
        # Librosa 提取出來的是「每一幀(frame)」的數據（時間序列）。
        # 為了存入資料庫並做快速比對，我們通常取「平均值(Mean)」和「變異數(Var)」。
        # 這能把一整首歌壓縮成一個一維的向量。

        features = {
            "tempo": float(tempo), # 轉為 float 以便存入 DB
            
            # MFCC 統計數據 (13個數值的平均 與 13個數值的變異數)
            "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
            "mfcc_var": np.var(mfcc, axis=1).tolist(),
            
            # Chroma 統計數據 (12個音的平均能量)
            "chroma_mean": np.mean(chroma, axis=1).tolist(),
            
            # Contrast 統計數據
            "contrast_mean": np.mean(contrast, axis=1).tolist()
        }

        return features

    except Exception as e:
        print(f"處理音檔時發生錯誤: {e}")
        return None

# --- 使用範例 ---
# 假設你有一個檔案叫 'test_song.wav' (請換成你實際的檔案路徑)
# file_path = 'test_song.wav' 
# data = extract_features(file_path)

# if data:
#     print("特徵提取成功！數據結構如下：")
#     print(json.dumps(data, indent=4))