import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 数据
true_labels = np.array([
    [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1],
    [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
    [0, 0, 1], [0, 1, 0]
])

predict_scores = np.array([
    [0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8],
    [0.4, 0.2, 0.4], [0.6, 0.3, 0.1], [0.4, 0.2, 0.4], [0.4, 0.1, 0.5],
    [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]
])

# 绘制每个类别的ROC曲线
plt.figure()
for i in range(3):
    fpr, tpr, _ = roc_curve(true_labels[:, i], predict_scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

# 绘制平均ROC曲线
y_true = np.argmax(true_labels, axis=1)
y_score = predict_scores
fpr_micro, tpr_micro, _ = roc_curve(y_true, y_score[:, 1], pos_label=1)
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, label=f'Micro-average ROC curve (AUC = {roc_auc_micro:.2f})', linestyle='--')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()