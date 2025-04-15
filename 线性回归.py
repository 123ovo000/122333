import numpy as np
from sklearn.metrics import precision_recall_curve,roc_curve,auc
import matplotlib.pyplot as plt

y_ture = np.array([1,0,0,1,0,0,1,1,0,0])
y_scores = np.array([0.90,0.40,0.20,0.60,0.50,0.40,0.70,0.40,0.65,0.35])

precision,recall, _ = precision_recall_curve(y_ture,y_scores)
auc_pr = auc(recall,precision)

fpr,tpr, _ = roc_curve(y_ture,y_scores)
auc_roc = auc(fpr,tpr)

plt.figure(figsize=(10,6))
plt.plot(recall,precision,label=f'PR curve (AUC = {auc_pr:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('pr_curve.png')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(fpr,tpr,label=f'ROC curve (AUC = {auc_pr:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('roc_curve.png')
plt.show()

