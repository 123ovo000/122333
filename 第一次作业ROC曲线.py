import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def compute_cost(X, y, theta, lambda_=0):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * (np.sum(errors ** 2) + lambda_ * np.sum(theta[1:] ** 2))
    return cost


def gradient_descent(X_train, y_train, X_test, y_test, theta, alpha=0.01, num_iters=500, lambda_=0):
    m = len(y_train)
    J_history_train = []
    J_history_test = []

    for i in range(num_iters):
        gradients = (1 / m) * X_train.T.dot(X_train.dot(theta) - y_train)
        if len(theta) > 1:
            gradients[1:] += (lambda_ / m) * theta[1:]
        theta -= alpha * gradients

        # 记录训练集和测试集的损失
        J_history_train.append(compute_cost(X_train, y_train, theta, lambda_))
        J_history_test.append(compute_cost(X_test, y_test, theta, lambda_))

    return theta, J_history_train, J_history_test


# 加载数据
file_path = r"C:\Users\33207\Desktop\regress_data1.xlsx"
data = pd.read_excel(file_path)

# 假设数据中有两列：'人口' 和 '收益'
X = data[['人口']].values
y = data['收益'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据归一化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 添加偏置项
X_b_train = np.c_[np.ones((len(X_train_scaled), 1)), X_train_scaled]
X_b_test = np.c_[np.ones((len(X_test_scaled), 1)), X_test_scaled]

# 初始化参数
theta_initial = np.zeros((2, 1))

# 超参数设置
alpha = 0.01
num_iters = 500

# 梯度下降（无正则化）
theta_final_no_reg, J_history_train_no_reg, J_history_test_no_reg = gradient_descent(X_b_train, y_train.reshape(-1, 1),
                                                                                     X_b_test, y_test.reshape(-1, 1),
                                                                                     theta_initial.copy(), alpha,
                                                                                     num_iters, lambda_=0)

# 梯度下降（有正则化）
lambda_ = 0.1
theta_final_with_reg, J_history_train_with_reg, J_history_test_with_reg = gradient_descent(X_b_train,
                                                                                           y_train.reshape(-1, 1),
                                                                                           X_b_test,
                                                                                           y_test.reshape(-1, 1),
                                                                                           theta_initial.copy(), alpha,
                                                                                           num_iters, lambda_=lambda_)

# 绘制结果
plt.figure(figsize=(18, 6))

# 无正则化
plt.subplot(1, 3, 1)
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, X_b_train.dot(theta_final_no_reg), color='red', label='Fitted line (No Regularization)')
plt.xlabel('人口')
plt.ylabel('收益')
plt.title('Linear Regression Fit (Training Data, No Regularization)')
plt.legend()

# 有正则化
plt.subplot(1, 3, 2)
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, X_b_train.dot(theta_final_with_reg), color='green', label='Fitted line (With Regularization)')
plt.xlabel('人口')
plt.ylabel('收益')
plt.title(f'Linear Regression Fit (Training Data, With Regularization λ={lambda_})')
plt.legend()

# 损失变化曲线
plt.subplot(1, 3, 3)
plt.plot(range(num_iters), J_history_train_no_reg, label='Train Loss (No Regularization)')
plt.plot(range(num_iters), J_history_test_no_reg, label='Test Loss (No Regularization)')
plt.plot(range(num_iters), J_history_train_with_reg, label='Train Loss (With Regularization)', linestyle='--')
plt.plot(range(num_iters), J_history_test_with_reg, label='Test Loss (With Regularization)', linestyle='--')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.legend()

plt.tight_layout()
plt.show()

print("最终参数 (无正则化):", theta_final_no_reg.ravel())
print("最终参数 (有正则化):", theta_final_with_reg.ravel())