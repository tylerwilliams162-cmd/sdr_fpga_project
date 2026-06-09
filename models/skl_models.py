import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import sys
sys.path.insert(0, '/home/tyler/sdr_fpga_project/dsp')
from synth import (generate_bpsk, generate_8psk, 
                   generate_fm, generate_fsk, 
                   generate_qam, generate_qpsk)
from datasets import SyntheticData, MODULATION_CLASSES, CLASS_TO_INDEX
from features import features
from sklearn.ensemble import RandomForestClassifier


ds = SyntheticData()

iq, labels, snrs = ds.get_frame(n_frames=5000)

features = features(iq)


X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=2000, random_state=0)

rf_model.fit(X_train, y_train)

rf_acc = rf_model.score(X_test, y_test)


svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

y_pred = svm_model.predict(X_test_scaled)



acc = accuracy_score(y_test, y_pred)

print(f"SVM Accuracy: {acc}")
print(f"Random Forest Accuracy: {rf_acc}")

