# ==============================
# Tanzila Islam
# Email: tanzilamohita@gmail.com
# Date: 5/27/2024
# ===============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ConvCGP import create_model
import os

# Using GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # model will be trained on GPU 0, if it is -1, GPU will not use for training

C = 3
data = "HDRA"
# Load data
X = pd.read_csv(f'{data}_X_C{C}.csv', index_col=0)
Y = pd.read_csv(f'{data}_Y.csv', index_col=0)
print(X.shape)
# Drop missing values
Y = Y.dropna()
X = X.loc[Y.index]

# Preprocess X
X.columns = np.arange(0, len(X.columns))
Y.columns = np.arange(0, len(Y.columns))

# Define callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.00001)

index = 0

# Training and evaluating on X
for i in range(0, 1):  # Adjust range for the number of traits
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y[i], test_size=0.2, random_state=42)
    X_train = np.expand_dims(X_train, axis=2)
    X_valid = np.expand_dims(X_valid, axis=2)

    model_X = create_model(input_shape=(X_train.shape[1], 1))

    history_X = model_X.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_valid, y_valid),
                              callbacks=[es, reduce_lr])

    mse_X = model_X.evaluate(X_valid, y_valid)
    print(f'\nMSE in prediction for trait {i} on X =', mse_X)

    y_hat_X = model_X.predict(X_valid)
    corr_X = np.corrcoef(y_valid, y_hat_X[:, 0])[0, 1]
    print(f'\nCorrelation for trait {i} on X =', corr_X)

    index += 1
