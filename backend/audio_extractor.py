import json
import numpy as np
import librosa
import os

# Krumhansl-Schmuckler key profiles (簡易 key/mode 估計用)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key_mode(chroma_mean_12: np.ndarray):
    """
    輸出：key (0~11), mode (1=major, 0=minor)
    注意：這只是近似，無法保證跟 MSD/Echo Nest 一模一樣。
    """
    v = chroma_mean_12.astype(float)
    if np.allclose(v, 0):
        return 0, 1  # fallback: C major

    v = v / (np.linalg.norm(v) + 1e-9)

    best_score = -1e9
    best_key = 0
    best_mode = 1

    # 12 個轉調比較
    for k in range(12):
        maj = np.roll(_MAJOR_PROFILE, k)
        minr = np.roll(_MINOR_PROFILE, k)

        maj = maj / (np.linalg.norm(maj) + 1e-9)
        minr = minr / (np.linalg.norm(minr) + 1e-9)

        s_maj = float(np.dot(v, maj))
        s_min = float(np.dot(v, minr))

        if s_maj > best_score:
            best_score = s_maj
            best_key = k
            best_mode = 1
        if s_min > best_score:
            best_score = s_min
            best_key = k
            best_mode = 0

    return best_key, best_mode


def audio_extractor_msd_like(audio_path, sr=22050, hop_length=512, n_mfcc=12):
    # 1) Load
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    duration = float(len(y) / sr)

    # 2) Track-level: tempo, loudness
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    tempo = float(np.asarray(tempo).item())

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    loudness = float(librosa.amplitude_to_db(rms, ref=np.max).mean())

    # 3) Frame-level features (用來做 pitches/timbre)
    # pitches: chroma (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)  # (12, T)

    # timbre: MFCC (12)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)  # (12, T)

    # 4) Segment boundary：用 onset 近似 segment（跟你原本想法一致，但輸出改成 DB 欄位）
    seg_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True)
    if len(seg_frames) == 0:
        seg_frames = np.array([0], dtype=int)

    # 加入結尾
    T = chroma.shape[1]
    seg_frames = np.unique(np.clip(seg_frames, 0, T - 1))
    seg_frames = np.append(seg_frames, T)

    pitches_segments = []
    timbre_segments = []

    for i in range(len(seg_frames) - 1):
        a, b = int(seg_frames[i]), int(seg_frames[i + 1])
        if b <= a:
            continue

        pitch_vec = np.mean(chroma[:, a:b], axis=1)
        # MSD pitches 通常是相對量，做個歸一化避免尺度差
        pitch_vec = pitch_vec / (np.max(pitch_vec) + 1e-9)

        timbre_vec = np.mean(mfcc[:, a:b], axis=1)

        pitches_segments.append(pitch_vec.astype(float).tolist())
        timbre_segments.append(timbre_vec.astype(float).tolist())

    # 5) 統計：mean/std（用 segments 來算，跟你 DB 的設計最容易對齊）
    if len(pitches_segments) == 0:
        pitches_mean = np.zeros(12, dtype=float)
        pitches_std = np.zeros(12, dtype=float)
    else:
        P = np.array(pitches_segments, dtype=float)  # (S, 12)
        pitches_mean = P.mean(axis=0)
        pitches_std = P.std(axis=0)

    if len(timbre_segments) == 0:
        timbre_mean = np.zeros(n_mfcc, dtype=float)
        timbre_std = np.zeros(n_mfcc, dtype=float)
    else:
        M = np.array(timbre_segments, dtype=float)  # (S, 12)
        timbre_mean = M.mean(axis=0)
        timbre_std = M.std(axis=0)

    # 6) key/mode：用全曲平均 chroma 去估（近似）
    chroma_mean = np.mean(chroma, axis=1)
    key, mode = estimate_key_mode(chroma_mean)

    # 7) 回傳：欄位「完全對齊你 MySQL 的 songs_subset」
    return {
        "duration": duration,
        "tempo": tempo,
        "loudness": loudness,
        "key": int(key),
        "mode": int(mode),
        "timbre_mean": timbre_mean.astype(float).tolist(),
        "timbre_std": timbre_std.astype(float).tolist(),
        "pitches_mean": pitches_mean.astype(float).tolist(),
        "pitches_std": pitches_std.astype(float).tolist(),
        # 這兩個建議用 json.dumps 存進 mediumtext
        "timbre_segments": timbre_segments,
        "pitches_segments": pitches_segments,
    }


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))  # backend 資料夾
    audio = os.path.join(base, "test_data", "test_song.wav")
    data = audio_extractor_msd_like(audio)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    out_path = os.path.join(base, "msd_features.json")  # 你也可以改成 .txt
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("已輸出到：", out_path)