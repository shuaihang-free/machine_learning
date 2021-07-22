from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

y_true = ["cat", "dog", "cat", "cat", "dog", "rebit"]
y_pred = ["dog", "dog", "rebit", "cat", "dog", "cat"]
C2= confusion_matrix(y_true, y_pred, labels=["dog", "rebit", "cat"])
print("C2:\n",C2)
