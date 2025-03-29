import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from tqdm import tqdm  # 新增导入

# 配置参数
DATA_ROOT = './ImageNet_mini'  # 包含train/val/test子目录
NUM_CLASSES = 20
BATCH_SIZE = 32
EPOCHS = 30
MODEL_SAVE_PATH = 'resnet50_20cls.pth'

# 1. 数据预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. 加载数据集
train_dataset = datasets.ImageFolder(
    os.path.join(DATA_ROOT, 'train'),
    transform=train_transform
)
val_dataset = datasets.ImageFolder(
    os.path.join(DATA_ROOT, 'val'),
    transform=test_transform
)
test_dataset = datasets.ImageFolder(
    os.path.join(DATA_ROOT, 'test'),
    transform=test_transform
)

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE)
test_loader = DataLoader(test_dataset, BATCH_SIZE)


# 3. 初始化模型
def create_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


model = create_model().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# 4. 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# 5. 增强的训练函数（添加进度条）
def train_model():
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        # 训练阶段带进度条
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 添加训练进度条
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} [Train]',
                         bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 实时更新进度条信息
            train_bar.set_postfix({
                'loss': f"{running_loss / (train_bar.n + 1):.3f}",
                'acc': f"{100 * correct / total:.2f}%"
            })

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 验证阶段带进度条
        val_loss, val_acc = evaluate(val_loader, mode='Validation')
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%')
        print('-' * 50)

    # 可视化训练过程（保持不变）
    visualize_training(history)


# 6. 增强的评估函数（添加进度条）
def evaluate(loader, mode='Validation'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # 添加评估进度条
    eval_bar = tqdm(loader, desc=f'{mode:10}',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    with torch.no_grad():
        for inputs, labels in eval_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条信息
            eval_bar.set_postfix({
                'loss': f"{running_loss / (eval_bar.n + 1):.3f}",
                'acc': f"{100 * correct / total:.2f}%"
            })

    return running_loss / len(loader), 100 * correct / total


# 7. 增强的测试函数（添加进度条）
def test_model():
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss, test_acc = evaluate(test_loader, mode='Testing')
    print(f'\nTest Results:')
    print(f'Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%')


# 8. 可视化函数（保持不变）
def visualize_training(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.show()


# 9. 推理函数
def inference(image_path):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    # 预处理
    img = Image.open(image_path).convert('RGB')
    img = test_transform(img).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)

    # 获取类别名称
    class_names = test_dataset.classes
    return class_names[pred.item()]


# 执行流程
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 训练与验证
    train_model()

    # 测试
    test_model()

    # 示例推理
    sample_image = './ImageNet_mini/test/n01484850/n01484850_17.JPEG'
    print(f'Prediction: {inference(sample_image)}')