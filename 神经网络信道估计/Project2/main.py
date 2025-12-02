import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def muse_ratio(ax: torch.Tensor, bx: torch.Tensor) -> torch.Tensor:

    a_c = torch.complex(ax[:, 0], ax[:, 1])
    b_c = torch.complex(bx[:, 0], bx[:, 1])
    return torch.mean(torch.abs(a_c - b_c) ** 2 / torch.abs(b_c) ** 2)


pattern = np.array([1,-1,1,-1], dtype=np.complex64)

def generate_data(num_samples=10000, pilot_lens=8, snr_ranges=(-10, 30)):
    base_pattern = np.tile(pattern, int(np.ceil(pilot_lens / 4)))[:pilot_lens]
    x = np.tile(base_pattern, (num_samples, 1))

    h_real = np.random.randn(num_samples) / np.sqrt(2)
    h_image = np.random.randn(num_samples) / np.sqrt(2)
    h = h_real + 1j * h_image

    snr_db = np.random.uniform(snr_ranges[0], snr_ranges[1], size=num_samples)
    snr_linear = 10 ** (snr_db / 10)
    noise_power =1/ snr_linear
    noise_std = np.sqrt(noise_power / 2)

    noise = (np.random.randn(num_samples, pilot_lens) + 1j * np.random.randn(num_samples, pilot_lens)) * noise_std[:,
                                                                                                         np.newaxis]
    yc = h[:, np.newaxis] * x + noise
    #yc = yc.reshape(yc.shape[0], -1, 4).mean(axis=2)
    #print(yc)
    input_features = np.stack([yc.real, yc.imag,x.real], axis=1)

    h_label = np.stack([h_real, h_image], axis=1)


    return (torch.tensor(input_features, dtype=torch.float32),
            torch.tensor(h_label, dtype=torch.float32))



def generate_textdata(num_samples_text=10000, pilot_lens_text=8, snr_ranges=(-10, 30)):
    base_pattern = np.tile(pattern, int(np.ceil(pilot_lens_text /4)))[:pilot_lens_text]
    x = np.tile(base_pattern, (num_samples_text, 1))

    h_real = np.random.randn(num_samples_text) / np.sqrt(2)
    h_image = np.random.randn(num_samples_text) / np.sqrt(2)
    h = h_real + 1j * h_image

    snr_db = np.random.uniform(snr_ranges[0], snr_ranges[1], size=num_samples_text)
    snr_linear = 10 ** (snr_db / 10)
    noise_power =1/ snr_linear
    noise_std = np.sqrt(noise_power / 2)

    noise_a = (np.random.randn(num_samples_text, pilot_lens_text) + 1j * np.random.randn(num_samples_text, pilot_lens_text)) * noise_std[:, np.newaxis]
    noise_b = (np.random.randn(num_samples_text, pilot_lens_text) + 1j * np.random.randn(num_samples_text, pilot_lens_text)) * noise_std[:, np.newaxis]

    y_a = h[:, np.newaxis] * x + noise_a
    y_b = h[:, np.newaxis] * x + noise_b
    #y_a = y_a.reshape(y_a.shape[0], -1, 4).mean(axis=2)
    #y_b = y_b.reshape(y_b.shape[0], -1, 4).mean(axis=2)

    input_features_a = np.stack([y_a.real, y_a.imag,x.real], axis=1)

    input_features_b = np.stack([y_b.real, y_b.imag,x.real], axis=1)


    h_label = np.stack([h_real,h_image], axis=1)


    return (torch.tensor(input_features_a, dtype=torch.float32),
            torch.tensor(h_label, dtype=torch.float32),
            torch.tensor(input_features_b, dtype=torch.float32),
            torch.tensor(h_label, dtype=torch.float32))

# CNN模型（四通道）
class CNNModel(nn.Module):
    def __init__(self, pilot_lens=8, dropout_rate=0.001):
        super(CNNModel, self).__init__()
        self.pilot_len = pilot_lens
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=2, padding=1, stride=1,bias=False),

            nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=2,bias=False),

            nn.AdaptiveAvgPool1d(8))
        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512, 128, bias=True),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(128, 64, bias=True),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(64, 2, bias=True))


    def forward(self, x):
            x = x.view(-1, 3, self.pilot_len)
            x = self.conv(x)
            x = self.fc(x)
            return x


def train(model, x, yv, devices='cpu', epochs=1, lr=1e-5, weight_decay=0.01, batch_size=100):
    model = model.to(devices)
    dataset = TensorDataset(x, yv)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn =nn.MSELoss()

    model.train()
    for ep in range(epochs):
        total_loss = 0
        for x, yv in dataloader:
            x, yv = x.to(devices), yv.to(devices)
            optimizer.zero_grad()
            output = model(x)
            loss=2*loss_fn(output,yv)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss=total_loss/batch_size
        if epochs <= 10 or ep == epochs - 1:
            print(f"Epoch {ep + 1}/{epochs}, Loss: {avg_loss:.6f}")
    return model

# 测试过程
def evaluate(model, x, yu, devices='cpu'):
    model.eval()
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        x, yu = x.to(devices), yu.to(devices)
        pared = model(x)
        loss = 2*loss_fn(pared, yu)
    return loss, pared.cpu()

def mapd(pared):
    cqi =(pared ** 2).sum(dim=1)
    cqi=np.sqrt(cqi)
    thresholds = [0.4270, 0.6368, 0.8325, 1.0481,1.3386]

    index = torch.full_like(cqi, 6, dtype=torch.long)

    for ix, th in enumerate(thresholds):
        masks = (cqi < th) & (index == 6)
        index[masks] = ix + 1
    return index,cqi

# 主程序
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rounds = 40
    local_epochs = 50
    pilot_len =64
    sample_per_a = 1000
    sample_per_b = 1000
    sample_test_per = 10000
    snr_range = (10, 10)
    snr_range_test = (10, 10)


    alice_model = CNNModel(pilot_lens=64 ).to(device)
    bob_model = CNNModel(pilot_lens=64 ).to(device)

    for r in range(rounds):
        print(f"\n--- Round {r + 1} ---")

        xa, ya = generate_data(num_samples=sample_per_a, pilot_lens=pilot_len, snr_ranges=snr_range)
        xb, yb = generate_data(num_samples=sample_per_b, pilot_lens=pilot_len, snr_ranges=snr_range)


        alice_model = train(alice_model, xa, ya, device, epochs=local_epochs)
        bob_model = train(bob_model, xb, yb, device, epochs=local_epochs)


        test_xa, test_ya, test_xb, test_yb = generate_textdata(num_samples_text=sample_test_per,
                                                               pilot_lens_text=pilot_len, snr_ranges=snr_range_test)
        mse_a, pared_a = evaluate(alice_model, test_xa, test_ya, device)
        mse_b, pared_b = evaluate(bob_model, test_xb, test_yb, device)

        # pared_b 是一个 torch.Tensor
        interval_a,cqi_a = mapd(pared_a)
        interval_b,cqi_b= mapd(pared_b)

        muse_a=muse_ratio(pared_a, test_ya)
        muse_b = muse_ratio(pared_b, test_yb)
        muse_ab = muse_ratio(pared_a, pared_b)
        # 计绝对误差小于0.5的数量
        print(f"Alice MISE: {muse_a:.6f} | Bob MISE: {muse_b:.6f} | Alice-Bob MISE: {muse_ab:.6f}")

        num_different = (interval_a != interval_b).sum().item()
        print(f"不同元素个数: {num_different}")

        #for case_value, base, scale, offset in [
         #   (1, 0.0, 0.1823, 0),
        #    (2, 0.1823, 0.2232, 1),
       #     (3, 0.4055, 0.2876, 2),
       #     (4, 0.6931, 0.4055, 3),
       #    (5, 1.0986, 0.6932, 4),
        #    (6, 1.7918, 1.4, 5),
       # ]:
        for case_value, base, scale, offset in [
            (1, 0.0, 0.4270, 0),
            (2, 0.4270, 0.2098, 1),
            (3,0.6368, 0.1957, 2),
            (4, 0.8325, 0.2156, 3),
            (5, 1.0481, 0.2905, 4),
            (6, 1.3386, 0.48, 5),
        ]:


            mask = (interval_a == case_value)
            a = torch.abs(cqi_a - base)
            x = a / scale

            # 特殊处理 case 10 的上限
            if case_value == 6:
                x = torch.where(x > 0.9999, torch.tensor(0.999, device=x.device), x)

            cqi_a = torch.where(mask, x + offset, cqi_a)

            mask = (interval_b == case_value)
            a = torch.abs(cqi_b - base)
            x = a / scale

            # 特殊处理 case 10 的上限
            if case_value == 6:
                x = torch.where(x > 0.999, torch.tensor(0.99, device=x.device), x)

            cqi_b = torch.where(mask, x + offset, cqi_b)

        ab_abs_error = torch.abs(cqi_a - cqi_b)

        # 统计绝对误差小于0.5的数量
        count_below_threshold = torch.sum(ab_abs_error >= 0.5)
        print(f"不同元素个数: {count_below_threshold}")



















