import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

    input_features = np.stack([yc.real, yc.imag,x.real], axis=1)

    h_label = np.stack([h_real, h_image], axis=1)
    hh=torch.tensor(h_label)

    interval = mapd(hh).unsqueeze(1).float()-0.5

    return torch.tensor(input_features, dtype=torch.float32), interval



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



    input_features_a = np.stack([y_a.real, y_a.imag,x.real], axis=1)

    input_features_b = np.stack([y_b.real, y_b.imag,x.real], axis=1)


    h_label = np.stack([h_real,h_image], axis=1)
    hh = torch.tensor(h_label)

    interval = mapd(hh).unsqueeze(1).float()-0.5

    return (torch.tensor(input_features_a, dtype=torch.float32), interval,
            torch.tensor(input_features_b, dtype=torch.float32), interval)

# CNN模型（四通道）
class CNNModel(nn.Module):
    def __init__(self, pilot_lens=8, dropout_rate=0.001):
        super(CNNModel, self).__init__()
        self.pilot_len = pilot_lens
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=2, padding=0, stride=1,bias=False),

            nn.Conv1d(32, 64, kernel_size=3, padding=0, stride=2,bias=False),

            nn.AdaptiveAvgPool1d(16))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512, bias=True),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(512, 128, bias=True),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(128, 64, bias=True),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(64, 1, bias=True))


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
            loss=loss_fn(output,yv)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss=total_loss/batch_size
        if epochs <= 10 or ep == epochs - 1:
            print(f"Epoch {ep + 1}/{epochs}, Loss: {avg_loss:.6f}")
    return model

def fed_avg(models, num_samples):
    total_samples = sum(num_samples)
    avg_model = {}
    for key in models[0].state_dict().keys():
        weighted_sum = sum(model.state_dict()[key] * (num_samples[ix] / total_samples)
                           for ix, model in enumerate(models))
        avg_model[key] = weighted_sum
    return avg_model

# 测试过程
def evaluate(model, x, yu, devices='cpu'):
    model.eval()
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        x, yu = x.to(devices), yu.to(devices)
        pared = model(x)
        loss = loss_fn(pared, yu)
    return loss, pared.cpu()

def mapd(pared):
    cqi = (pared ** 2).sum(dim=1)

    thresholds = [0.1823, 0.4055, 0.6931, 1.0986, 1.7918]

    index = torch.full_like(cqi, 6, dtype=torch.long)

    for ix, th in enumerate(thresholds):
        masks = (cqi < th) & (index == 6)
        index[masks] = ix + 1
    return index

def mapd11(pared):
    cqi = (pared ** 2).sum(dim=1)
    thresholds = [0.1054, 0.2231, 0.3567, 0.5108, 0.6931,
                  0.9163, 1.2040, 1.6094, 2.3026]

    index = torch.full_like(cqi, 10, dtype=torch.long)

    for ix, th in enumerate(thresholds):
        masks = (cqi < th) & (index == 10)
        index[masks] = ix + 1

    return index

# 主程序
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rounds = 30
    local_epochs = 50
    pilot_len =64
    sample_per_a = 1000
    sample_per_b = 1000
    sample_test_per = 10000
    snr_range = (10, 10)
    snr_range_test = (10, 10)


    alice_model = CNNModel(pilot_lens=pilot_len ).to(device)
    bob_model = CNNModel(pilot_lens=pilot_len ).to(device)

    for r in range(rounds):
        print(f"\n--- Round {r + 1} ---")

        xa, ya = generate_data(num_samples=sample_per_a, pilot_lens=pilot_len, snr_ranges=snr_range)
        xb, yb = generate_data(num_samples=sample_per_b, pilot_lens=pilot_len, snr_ranges=snr_range)


        alice_model = train(alice_model, xa, ya, device, epochs=local_epochs)
        bob_model = train(bob_model, xb, yb, device, epochs=local_epochs)

         #global_weights = fed_avg([alice_model, bob_model],[sample_per_a, sample_per_b])
         #alice_model.load_state_dict(global_weights)
         #bob_model.load_state_dict(global_weights)

        #alice_model.load_state_dict(torch.load("alice_model_1.pth", map_location=device))
        #bob_model.load_state_dict(torch.load("bob_model_1.pth", map_location=device))

        #alice_model.eval()
        #bob_model.eval()
        test_xa, test_ya, test_xb, test_yb = generate_textdata(num_samples_text=sample_test_per,
                                                               pilot_lens_text=pilot_len, snr_ranges=snr_range_test)
        muse_a, pared_a = evaluate(alice_model, test_xa, test_ya, device)
        muse_b, pared_b = evaluate(bob_model, test_xb, test_yb, device)
        # pared_b 是一个 torch.Tensor
        mask=(pared_a<0.001)
        pared_a[mask]=0.01
        mask = (pared_a > 9.999)
        pared_a[mask] = 9.99
        mask = (pared_b < 0.001)
        pared_b[mask] = 0.01
        mask = (pared_b > 9.999)
        pared_b[mask] = 9.99
        #print(pared_a)

        # 计算绝对误差
        ab_abs_error = torch.abs(pared_a - pared_b)
        #ab_mse = torch.mean((pared_a - pared_b) ** 2)

        # 统计绝对误差小于0.5的数量
        count_below_threshold = torch.sum(ab_abs_error > 0.5)
        #print(f"Alice MISE: {muse_a:.6f} | Bob MISE: {muse_b:.6f} | Alice-Bob MISE: {ab_mse:.6f}")

        a_int = torch.floor(pared_a)
        b_int = torch.floor(pared_b)  # 先取下整
        s_int = torch.floor(test_ya)
        num_different = (a_int != b_int).sum().item()
        print(f"不同元素个数: {num_different}")
        print(f"不同元素个数: {count_below_threshold}")




        #if count_below_threshold.item() < 120 and count_below_threshold1 < 15:
            #print(f" 提前停止于 Round {r + 1}，保存模型")
            #torch.save(alice_model.state_dict(), f"alice_model_3.pth")
            #torch.save(bob_model.state_dict(), f"bob_model_3.pth")
            #break
        #for i in range(len(pared_a)):
          #print(f"第{i + 1}个元素:")
          #print(f"数组1: {pared_a[i].item():.4f}")
          #print(f"数组2: {pared_b[i].item():.4f}")
          #print("-" * 20)

          # 阈值


        #print(pared_a)
        #print(pared_b)
       # dist = torch.full_like(pared_a, 2.0)  # 先全部设为2
       # a_sd=pared_a-a_int
        # ---- 计算 dist ----
       # mask_mid = pared_a > 0

        #dist[mask_mid] = torch.where(a_sd[mask_mid] < 0.5, torch.tensor(0.0), dist[mask_mid])
       # dist[mask_mid] = torch.where(a_sd[mask_mid] > 0.5, torch.tensor(1.0), dist[mask_mid])
        #a_ss=abs(a_sd-0.5)
       # a_ss1=abs(a_ss-0.5)

        #b_sd = pared_b - b_int
        #b_ss = abs(b_sd - 0.5)
        # 四个掩码
       # mask0 = (dist == 0)
        #mask1 = (dist == 1)

        # ---- dist == 0 :  b_int 为偶数则 +1 ----
       # even0 = (b_sd > 0.5)
        #even1 = (b_ss>a_ss1)
        #b_int[mask0 & even0 & even1] += 1


        # ---- dist == 1 :  b_int 为奇数则 -1 ----
        #odd0 = (b_sd < 0.5)
        #odd1 = (b_ss > a_ss1)
        #b_int[mask1 & odd0 & odd1] -= 1


        #print(b_int)

        #num_different = (a_int != b_int).sum().item()
        #print(f"不同元素个数: {num_different}")





