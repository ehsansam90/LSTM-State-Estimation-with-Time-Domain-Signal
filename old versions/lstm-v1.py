import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import model_selection
import tensorflow
import tensorflow.keras as keras
from tensorflow.keras import regularizers
"""
this is for experimenting, using LSTMs and DROPBEAR.

my theory is that the adam optimizer is interacting poorly with the data,
because the data will look so much different in different sections, the 
learning rate tuning is thrown out of whack and creates nan values.

so now i'm using minibatched SGD, with training pulled out semi-randomly
next i'll probably make it completely randomized pulling from the time-series
"""
#%% load data
f = open('data_6_with_FFT.json')
data = json.load(f)
f.close()

acc = np.array(data['acceleration_data'])
acc_t = np.array(data['time_acceleration_data'])
pin = np.array(data['measured_pin_location'])
pin_t = np.array(data['measured_pin_location_tt'])

#%%preprocess
downsampling_factor = 64

# scaling data, which means that it must be unscaled to be useful
acc_scaler = sk.preprocessing.StandardScaler()
acc_scaler.fit(acc.reshape(-1, 1))
acc = acc_scaler.fit_transform(acc.reshape(-1, 1)).flatten()
pin_scaler = sk.preprocessing.StandardScaler()
pin_scaler.fit(pin.reshape(-1,1))
pin = pin_scaler.fit_transform(pin.reshape(-1,1)).flatten()
acc = acc[::downsampling_factor]
acc_t = acc_t[::downsampling_factor]


# plt.figure()
# plt.plot(acc_t, acc)

# plt.figure()
# plt.plot(pin_t, pin)

train_len = 100
acc_reshape = np.reshape(acc[:acc.size//train_len*train_len].T, (-1, train_len))
t_reshape = np.reshape(acc_t[:acc_t.size//train_len*train_len].T, (-1, train_len))
t = t_reshape[:, -1]
y = np.array([pin[(np.abs(pin_t - v)).argmin()] for v in t])
# del t_reshape

# X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(acc_reshape, y, test_size = .2)
X_train, y_train = acc_reshape, y
X_train = X_train.reshape(-1, train_len, 1).astype(np.float32)
# X_test = X_test.reshape(-1, train_len, 1).astype(np.float32)
y_train = y_train.reshape(-1, 1).astype(np.float32)
# y_test = y_test.reshape(-1, 1).astype(np.float32)


# np.random.seed(42)

# n_steps = 50
# series = generate_time_series(10000, n_steps + 1)
# X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
# X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
# X_test, y_test = series[9000:, :n_steps], series[9000:, -1]
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath = "./model_saves",
        save_weights_only = True,
        monitor = "loss",
        mode = "min",
        save_freq = 1,
        save_best_only = True)

dense_top = keras.layers.Dense(1)
# dense_top.set_weights([np.ones((10,1)), np.ones(1)])
model = keras.models.Sequential([
    keras.layers.LSTM(130, return_sequences = True, input_shape = [None, 1]),
    keras.layers.LSTM(130, return_sequences = True),
    # keras.layers.LSTM(1, return_sequences = False)])
    dense_top])
# optimizer = keras.optimizers.Adam(clipvalue = .01)
optimizer = keras.optimizers.SGD(clipvalue = .5)
model.compile(loss = "mean_squared_error",
              optimizer = optimizer,
              metrics = ['accuracy'])

#model.fit(X_train, y_train, epochs = 15, callbacks = [model_checkpoint_callback])
#%%
from math import isnan
from numpy.random import randint
batch_size = 32
sudo_epochs = 200 # no real epochs, but see total_trains uses an equivalent
total_trains = sudo_epochs * X_train.shape[0]//batch_size
broken_on_error = False
for i in range(total_trains):
    if(i  % (total_trains//sudo_epochs) == 0):
        print("finished epoch " + str(i//(total_trains//sudo_epochs)) + " out of " + str(sudo_epochs))
    indices = np.array([randint(0, X_train.shape[0]) for i in range(batch_size)]) #maybe repeats? who cares.
    X_mini = X_train[indices]
    y_mini = y_train[indices]
    hist = model.fit(X_mini, y_mini, epochs=1, batch_size=batch_size,\
                     callbacks = [model_checkpoint_callback], verbose = 0)
    if(isnan(hist.history['loss'][0])):
        print("training failure")
        # broken_on_error = True; break
        model.load_weights("./model_saves")

print('broken on error: ' + str(broken_on_error))

acc = acc.reshape(1, -1, 1)

pred_pin = model.predict(acc)
pin = pin_scaler.inverse_transform(pin.reshape(-1,1)).flatten()
pred_pin = pin_scaler.inverse_transform(pred_pin.reshape(-1,1)).flatten()
from sklearn.metrics import mean_squared_error

# mse = mean_squared_error(pin, pred_pin)
plt.figure()
plt.title("LSTM prediction of pin location, MSE = ")
plt.plot(pin_t, pin, label = "actual pin location")
plt.plot(acc_t, pred_pin.flatten(), label = "predicted pin location")
plt.xlabel("Training time [seconds]")
plt.ylabel("pin location")
plt.legend()
#%%
import scipy as sp
from scipy import signal
# f, ts, Sxx = signal.spectrogram(pred_pin, window=('tukey', 5000),nperseg=10000, noverlap=5000)
# plt.figure(figsize = (6.5, 3))
# plt.pcolormesh(ts, f, Sxx, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.tight_layout()
# plt.show()

from scipy.fft import fft, fftfreq
# Number of sample points
N = pred_pin.shape[0]
# sample spacing
T = 1
y = pred_pin
yf =  2.0/N * np.abs(fft(y)[1:N//2])
xf = fftfreq(N, T)[:N//2]
period = 1/xf[np.argmax(yf)]
print("period: " + str(period))
plt.figure(figsize=(6.5, 4))
plt.plot(xf[1:],yf)
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title("FFT of vibration data")
plt.tight_layout()
plt.savefig('fft.png', dpi = 500)
plt.show()