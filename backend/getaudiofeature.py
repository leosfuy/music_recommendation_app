import librosa
import numpy as np

def extract_features_from_audio(file_path):
    """
    輸入音檔路徑，輸出對應的特徵向量
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)  # 讀取音檔
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Loudness
    loudness = np.mean(librosa.amplitude_to_db(np.abs(librosa.stft(y))))
    
    # Timbre (使用 MFCC 代替)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    if mfcc.shape[1] == 0:
        timbre_mean = np.zeros(12)
    else:
        timbre_mean = np.mean(mfcc, axis=1)

    # Pitches (chroma 平均值)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    if chroma.shape[1] == 0:
        pitch_mean = np.zeros(12)
    else:
        pitch_mean = np.mean(chroma, axis=1)

        # Key & mode (粗略估計)
    if np.all(pitch_mean == 0):
        key = -1   # 無法判斷
        mode = -1
    else:
        key = int(np.argmax(pitch_mean))  # 0~11 對應 C~B
        mode = 1 if pitch_mean[key] > 0 else 0  # 簡單判斷 major/minor

    
    # 整理成向量
    feature_vector = [
        tempo,
        loudness,
        key,
        mode
    ]
    feature_vector.extend(timbre_mean.tolist())
    feature_vector.extend(pitch_mean.tolist())

    
    return np.array(feature_vector)

# ------------------------
# 範例：使用者上傳音樂
# ------------------------
audio_path = "/Users/leeangel/Downloads/je-sais-que-la-terre-est-plate.mp3"  # 或 wav
user_features = extract_features_from_audio(audio_path)
print("使用者歌曲特徵向量：", user_features)
print("向量長度：", len(user_features))

