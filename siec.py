#plik 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# plik 3
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense

df = pd.read_hdf('atm_neutrino_classA.h5',key='y')

# Plik 2

print(df.head())
print(df.columns)

input_energy_cols=['jmuon_E', 'jmuon_JENERGY_ENERGY', 'jmuon_JENERGY_CHI2', 'jmuon_JENERGY_NDF',
                    'jmuon_JGANDALF_NUMBER_OF_HITS', 'jmuon_JSHOWERFIT_ENERGY', 'jmuon_AASHOWERFIT_ENERGY']
input_dir_cols = ['dir_x', 'dir_y', 'dir_z', 'jmuon_likelihood']

target_energy_cols = ['energy']
target_dir_cols = ['dir_x', 'dir_y', 'dir_z']

X_energy = df[input_energy_cols].values
X_dir = df[input_dir_cols].values
y_energy = df[target_energy_cols].values
y_dir = df[target_dir_cols].values

print(f"Kształt wejściowy energii: {X_energy.shape}")
print(f"Kształt wyjściowy kierunku: {y_dir.shape}\n")

X_en_train, X_en_test, X_dir_train, X_dir_test, y_en_train, y_en_test, y_dir_train, y_dir_test = train_test_split(X_energy, X_dir, y_energy, y_dir, test_size=0.2, random_state=42)

print(f'Liczba próbek treningowych: {len(X_en_train)}')
print(f'Liczba próbek testowych: {len(X_en_test)}\n')

scaler_energy = StandardScaler()

X_en_train_scaled = scaler_energy.fit_transform(X_en_train)
X_en_test_scaled = scaler_energy.transform(X_en_test)

scaler_dir = StandardScaler()
X_dir_train_scaled = scaler_dir.fit_transform(X_dir_train)
X_dir_test_scaled = scaler_dir.transform(X_dir_test)

print("Statystyki dla X_energy (Train) przed skalowaniem:")
print(f"Mean: {np.mean(X_en_train[:,0]):.2f}, Std: {np.std(X_en_train[:,0]):.2f}\n")

print("Statystyki dla X_energy (Train) po skalowaniu:")
print(f"Mean: {np.mean(X_en_train_scaled[:,0]):.2f}, Std: {np.std(X_en_train_scaled[:,0]):.2f}\n")

print("Kształt en przed reshape:", X_en_train_scaled.shape)
print("Kształt en przed reshape:", X_en_test_scaled.shape)

X_en_train_cnn = X_en_train_scaled.reshape(X_en_train_scaled.shape[0], X_en_train_scaled.shape[1], 1)
X_en_test_cnn = X_en_test_scaled.reshape(X_en_test_scaled.shape[0], X_en_test_scaled.shape[1], 1)

print("Kształt en po reshape:", X_en_train_cnn.shape)
print("Kształt en po reshape:", X_en_test_cnn.shape)

print("\nKształt dir przed reshape:", X_dir_train_scaled.shape)
print("Kształt dir przed reshape:", X_dir_test_scaled.shape)

X_dir_train_cnn = X_dir_train_scaled.reshape(X_dir_train_scaled.shape[0], X_dir_train_scaled.shape[1], 1)
X_dir_test_cnn = X_dir_test_scaled.reshape(X_dir_test_scaled.shape[0], X_dir_test_scaled.shape[1], 1)

print("Kształt dir po reshape:", X_dir_train_cnn.shape)
print("Kształt dir po reshape:", X_dir_test_cnn.shape)

# Plik 3

model = Sequential()
model.add(Input(shape=(4,1)))
print("sth")
model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
print("sth")
model.add(Flatten())
print("sth")

model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='linear'))
print("sth4")
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

history = model.fit(x=X_dir_train_cnn, y=y_dir_train, epochs=50, batch_size=64, validation_split=0.2, verbose=1)
print("sth5")
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(loss) + 1)

plt.figure(figsize=(12, 8))
plt.plot(epochs_range, loss, 'y',label='Training Loss')
plt.plot(epochs_range, val_loss, 'r',label='Validation Loss')
plt.yscale('log')
plt.title("Learning Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()