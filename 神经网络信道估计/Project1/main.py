import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights


# 定义模型（使用预训练的ResNet50）
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)


# 修改ResNet50的全连接层，使其输出15个类别（15种蔬菜）
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),  # 添加Dropout层以防止过拟合
    nn.Linear(2048, 15)  # 输出15个类别
)

# 设置设备GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 将模型移到设备（GPU或CPU）
model.to(device)

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(20),  # 随机旋转
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])

# 数据集路径
train_dir = r"C:\Users\zq0520\Downloads\Vegetable Images\train"
val_dir = r"C:\Users\zq0520\Downloads\Vegetable Images\validation"
test_dir = r"C:\Users\zq0520\Downloads\Vegetable Images\test"

# 使用ImageFolder导入数据集
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
val_data = datasets.ImageFolder(root=val_dir, transform=transform)
test_data = datasets.ImageFolder(root=test_dir, transform=transform)

# 使用DataLoader加载数据
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)



# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)  # 使用Adam优化器，加入L2正则化

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)  # 每10个epoch降低学习率

# 训练过程
epochs = 60  # 设置训练的epoch数

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # 正向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 统计训练集损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 打印训练集的损失和准确率
    train_accuracy = correct / total
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")

    # 验证过程
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # 打印验证集损失和准确率
    val_accuracy = val_correct / val_total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # 更新学习率
    scheduler.step()

# 测试模型
model.eval()
test_correct = 0
test_total = 0
test_loss = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

# 输出测试集的准确率和损失
test_accuracy = test_correct / test_total
print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_accuracy:.4f}")

# 可选：保存最终模型
torch.save(model.state_dict(), 'final_vegetable_classifier.pth')

