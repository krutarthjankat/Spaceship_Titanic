import joblib
import pandas as pd 
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model = joblib.load('../models/trained_model.joblib')
testing_data_set = pd.read_csv("../data/processed_test.csv", header=0)
testing_data_set.columns.values[0]='Sr_No'

X_test = testing_data_set.drop('Transported',axis=1)
y_test = testing_data_set['Transported']

pred=model.predict(X_test)
print(classification_report(y_test, pred))
fpr, tpr, thresholds = roc_curve(y_test, pred)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = '+str(auc(fpr, tpr))+')')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier') # Add a random classifier line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Voting Classifier')
plt.legend(loc="lower right")
plt.show()

cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
