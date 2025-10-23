import seaborn as sns
import pandas as pd
from pycm import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np

# Random confusion matrix for Brain Tumor MRI dataset
np.random.seed(42)  # for reproducibility
confusion_matrix = {
    "Glioma": {"Glioma": 120, "Meningioma": 5, "No Tumor": 3, "Pituitary": 7},
    "Meningioma": {"Glioma": 4, "Meningioma": 115, "No Tumor": 6, "Pituitary": 5},
    "No Tumor": {"Glioma": 2, "Meningioma": 4, "No Tumor": 130, "Pituitary": 4},
    "Pituitary": {"Glioma": 5, "Meningioma": 6, "No Tumor": 5, "Pituitary": 120}
}

# Display heatmap
df = pd.DataFrame(confusion_matrix).T  # transpose for correct orientation
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix for Brain Tumor MRI Classifier")
plt.ylabel("Predicted Class")
plt.xlabel("Actual Class")
plt.show()
