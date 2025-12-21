import torch #模型
import torch.nn as nn
import numpy as np
from database.precompute import MusicCNN

MODEL_PATH = "bestcnn_model.pth"

#檔案被import時就會執行
#載入模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def getEmbedding(timbre_segments:list, pitches_segments:list):
    combined = np.concatenate([timbre_segments, pitches_segments], axis=1)  # (Time, 24)
    
    tensor = torch.from_numpy(combined.T).unsqueeze(0).to(device).float() # (1, 24, Time)

    with torch.no_grad(): # 不紀錄梯度，單純forward
        embedding = model(tensor)   # (1, 128)

    return embedding.squeeze(0).cpu().numpy()  # (128,)
