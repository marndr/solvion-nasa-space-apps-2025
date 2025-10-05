import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import xarray as xr
import pandas as pd
import numpy as np
from model import build_transformer

NAME = "" # <-- model name

FILE_PATH_SRC = ""
FILE_PATH_TGT = ""

SRC_SEQ_LEN = 24
TGT_SEQ_LEN = 6

class WeatherPairDataset(Dataset):
    def __init__(self, data_list, src_seq_len, tgt_seq_len):
        self.src, self.tgt = data_list
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.length = min(len(self.src), len(self.tgt)) - (src_seq_len + tgt_seq_len)

    def __getitem__(self, idx):
        src_seq = self.src[idx : idx + self.src_seq_len]
        tgt_seq = self.tgt[idx + self.src_seq_len : idx + self.src_seq_len + self.tgt_seq_len]
        return src_seq, tgt_seq
    
class CubicLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        return torch.mean(torch.abs(error) ** 3)

ds_src = xr.open_dataset(FILE_PATH_SRC)
ds_tgt = xr.open_dataset(FILE_PATH_TGT)

df_src = ds_src.to_dataframe()
df_tgt = ds_tgt.to_dataframe()

src_tensor = torch.tensor(df_src.values, dtype=torch.float32)
tgt_tensor = torch.tensor(df_tgt.values, dtype=torch.float32)

dataset = WeatherPairDataset([src_tensor, tgt_tensor], SRC_SEQ_LEN, TGT_SEQ_LEN)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

dataset = df_src.values.astype(np.float32)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

SRC_DIM = df_src.shape[1]
TGT_DIM = df_tgt.shape[1]

model = build_transformer(
    src_dim = SRC_DIM,
    tgt_dim = TGT_DIM,
    src_seq_len = SRC_SEQ_LEN,
    tgt_seq_len = TGT_SEQ_LEN,
    d_model = 64,
    n = 3,
    h = 4,
    dropout_rate = 0,
    d_ff = 256
).to("cuda" if torch.cuda.is_available() else "cpu")

device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = CubicLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()

        src_mask = tgt_mask = None      # <-- no masking

        enc_output = model.encode(src, src_mask)
        dec_output = model.decode(enc_output, src_mask, tgt, tgt_mask)
        output = model.projection(dec_output)

        loss = criterion(output, tgt)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), f"models/transformer_{NAME}.pth")