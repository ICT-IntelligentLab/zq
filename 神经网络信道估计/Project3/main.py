import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------ 工具函数 ------------------
def mapd_index(h_tensor):
    cqi = (h_tensor ** 2).sum(dim=1)  # |h|^2
    thresholds = [0.1054, 0.2231, 0.3567, 0.5108, 0.6931,
                  0.9163, 1.2040, 1.6094, 2.3026]
    index = torch.full_like(cqi, 9, dtype=torch.long)
    for i, th in enumerate(thresholds):
        mask = (cqi < th) & (index == 9)
        index[mask] = i
    return index


def generate_data(num_samples=10000, pilot_lens=8, snr_ranges=(-10, 30)):
    pattern = np.array([1], dtype=np.complex64)
    base_pattern = np.tile(pattern, int(np.ceil(pilot_lens / 1)))[:pilot_lens]
    x = np.tile(base_pattern, (num_samples, 1))

    h_real = np.random.randn(num_samples) / np.sqrt(2)
    h_image = np.random.randn(num_samples) / np.sqrt(2)
    h = h_real + 1j * h_image

    snr_db = np.random.uniform(snr_ranges[0], snr_ranges[1], size=num_samples)
    snr_linear = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1 / snr_linear / 2)

    noise = (np.random.randn(num_samples, pilot_lens) +
             1j * np.random.randn(num_samples, pilot_lens)) * noise_std[:, None]

    y = h[:, None] * x + noise
    input_features = np.stack([y.real, y.imag], axis=2)

    h_label = np.stack([h_real, h_image], axis=1)
    interval = mapd_index(torch.tensor(h_label, dtype=torch.float32))+0.5

    return (torch.tensor(input_features, dtype=torch.float32),
            interval.float(),
            torch.tensor(snr_db, dtype=torch.float32).unsqueeze(1))


def generate_textdata(num_samples_text=1000, pilot_lens_text=8, snr_ranges=(-10, 30)):
    pattern = np.array([1], dtype=np.complex64)
    base_pattern = np.tile(pattern, int(np.ceil(pilot_lens_text / 1)))[:pilot_lens_text]
    x = np.tile(base_pattern, (num_samples_text, 1))

    h_real = np.random.randn(num_samples_text) / np.sqrt(2)
    h_image = np.random.randn(num_samples_text) / np.sqrt(2)
    h = h_real + 1j * h_image

    snr_db = np.random.uniform(snr_ranges[0], snr_ranges[1], size=num_samples_text)
    snr_linear = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1 / snr_linear / 2)

    noise_a = (np.random.randn(num_samples_text, pilot_lens_text) +
               1j * np.random.randn(num_samples_text, pilot_lens_text)) * noise_std[:, None]
    noise_b = (np.random.randn(num_samples_text, pilot_lens_text) +
               1j * np.random.randn(num_samples_text, pilot_lens_text)) * noise_std[:, None]

    y_a = h[:, None] * x + noise_a
    y_b = h[:, None] * x + noise_b

    input_features_a = np.stack([y_a.real, y_a.imag], axis=2)
    input_features_b = np.stack([y_b.real, y_b.imag], axis=2)

    h_label = np.stack([h_real, h_image], axis=1)+0.5
    interval = mapd_index(torch.tensor(h_label, dtype=torch.float32))

    return (torch.tensor(input_features_a, dtype=torch.float32), interval.float(),
            torch.tensor(snr_db, dtype=torch.float32).unsqueeze(1),
            torch.tensor(input_features_b, dtype=torch.float32), interval.float(),
            torch.tensor(snr_db, dtype=torch.float32).unsqueeze(1))


# ------------------ 模型 ------------------
class ChannelEstimatorReg(nn.Module):
    def __init__(self, snr_aware=True):
        super().__init__()
        self.snr_aware = snr_aware
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.fc_snr = nn.Linear(1, 16) if snr_aware else None
        self.fc = nn.Sequential(
            nn.Linear(64 + (16 if snr_aware else 0), 128), nn.ReLU(),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(64, 32, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(32, 1, bias=True),
        )

    def forward(self, x, snr=None):
        x = x.permute(0, 2, 1)
        feat = self.cnn(x).permute(0, 2, 1)
        lstm_out, _ = self.lstm(feat)
        lstm_feat = lstm_out[:, -1, :]
        if self.snr_aware and snr is not None:
            snr_feat = F.relu(self.fc_snr(snr))
            feat_all = torch.cat([lstm_feat, snr_feat], dim=-1)
        else:
            feat_all = lstm_feat
        return self.fc(feat_all)


# ------------------ 训练与评估 ------------------
def train(model, x, y_float, snr, device='cpu', epochs=1, lr=1e-3, batch_size=100):
    model = model.to(device)
    loader = DataLoader(TensorDataset(x, y_float, snr), batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        total = 0
        for xb, yb, sb in loader:
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            opt.zero_grad()
            out = model(xb, sb)
            loss = loss_fn(out.squeeze(1), yb)   # <---- 关键修改
            loss.backward()
            opt.step()
            total += loss.item()
        #print(f"Epoch {ep+1}/{epochs}, Loss: {total/len(loader):.6f}")
    return model


def evaluate(model, x, y_float, snr, device='cpu'):
    model.eval()
    with torch.no_grad():
        x, y_float, snr = x.to(device), y_float.to(device), snr.to(device)
        pred = model(x, snr)
        mse = F.mse_loss(pred.squeeze(1), y_float).item()  # <---- 关键修改
    return mse, pred.cpu()


# ------------------ 主程序 ------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rounds = 6
    local_epochs = 50
    pilot_len = 256
    sample_per_a = 10000
    sample_per_b = 10000
    sample_test_per = 10
    snr_range = (5, 5)
    snr_range_test = (5, 5)

    alice_model = ChannelEstimatorReg(True).to(device)
    bob_model   = ChannelEstimatorReg(True).to(device)

    for r in range(rounds):
        print(f"\n--- Round {r + 1} ---")
        xa, ya, sa = generate_data(sample_per_a, pilot_len, snr_range)
        xb, yb, sb = generate_data(sample_per_b, pilot_len, snr_range)
        test_xa, test_ya, test_sa, test_xb, test_yb, test_sb = generate_textdata(
            sample_test_per, pilot_len, snr_range_test)

        alice_model = train(alice_model, xa, ya, sa, device, epochs=local_epochs)
        bob_model   = train(bob_model, xb, yb, sb, device, epochs=local_epochs)

        acc_a, pred_a = evaluate(alice_model, test_xa, test_ya, test_sa, device)
        acc_b, pred_b = evaluate(bob_model, test_xb, test_yb, test_sb, device)
        print(pred_a)
        print(pred_b)
        ab_diff = (pred_a.int() != pred_b.int()).sum().item()
        print(f"Alice MSE: {acc_a:.4f} | Bob MSE: {acc_b:.4f} | Diff elems: {ab_diff}")
