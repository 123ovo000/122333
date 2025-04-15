import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, log_loss
import matplotlib.pyplot as plt

# 加载数据
data_path = "C:/Users/33207/Desktop/ex2data1(1).txt"  # 根据需要修改路径
data = pd.read_csv(data_path, header=None)

# 数据预处理
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分训练集、验证集和测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25,
                                                  random_state=42)  # 0.25 x 0.8 = 0.2

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 模型训练与损失计算
clf = SGDClassifier(loss="log_loss", max_iter=1, learning_rate='constant', eta0=0.01, random_state=42, warm_start=True)
train_losses, val_losses = [], []

max_epochs = 100
for epoch in range(max_epochs):
    clf.fit(X_train_scaled, y_train)

    # 计算训练集和验证集上的损失
    train_loss = log_loss(y_train, clf.predict_proba(X_train_scaled))
    val_loss = log_loss(y_val, clf.predict_proba(X_val_scaled))

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{max_epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("Training and Validation Loss per Epoch")
plt.show()

# 测试模型并计算评价指标
y_pred = clf.predict(X_test_scaled)
y_scores = clf.decision_function(X_test_scaled)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_scores)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_scores)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()