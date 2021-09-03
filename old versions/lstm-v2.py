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

at downsample factors less than 64, LSTM training produces NaN values during
training (probably a problem in backpropagation, but it isn't gradient explosion).
at downsampling factor = 64, training is much more well-behaved, but i'm keeping
my safeguards in place in case it does; SGD optimizer (not adam), and personally-made
minibatch training that resets if the model NaNs out. 

Also interesting to note is that at below downsampling factor 64, the model 
predictions were periodic. I only tested at downsampling 16, it had periodicity 
of 34.6 samples.

here i'm implementing a completely stocastic training algorithm, which trains
in minibatches of 32, which at each step are taken randomly from the time
sequence data. 
"""
#%% load data
f = open('data_6_with_FFT.json')
data = json.load(f)
f.close()

acc = np.array(data['acceleration_data'])
acc_t = np.array(data['time_acceleration_data'])
pin = np.array(data['measured_pin_location'])
pin_t = np.array(data['measured_pin_location_tt'])

#%% preprocess
ds = 64 # downsampling factor

# scaling data, which means that it must be unscaled to be useful
acc_scaler = sk.preprocessing.StandardScaler()
acc_scaler.fit(acc.reshape(-1, 1))
acc = acc_scaler.fit_transform(acc.reshape(-1, 1)).flatten()
pin_scaler = sk.preprocessing.StandardScaler()
pin_scaler.fit(pin.reshape(-1,1))
pin_transform = pin_scaler.fit_transform(pin.reshape(-1,1)).flatten().astype(np.float32)


X = np.reshape(acc[:acc.size//ds*ds], (acc.size//ds, ds)).T
acc_t_reshape = np.reshape(acc_t[:acc_t.size//ds*ds], (acc_t.size//ds, ds)).T
y = np.array([pin_transform[(np.abs(pin_t - v)).argmin()] for v in acc_t])
y = np.reshape(y[:y.size//ds*ds], (y.size//ds, ds)).T
#%% create model
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
    keras.layers.LSTM(130, return_sequences = True),
    # keras.layers.LSTM(1, return_sequences = False)])
    dense_top])
#%% train model
# optimizer = keras.optimizers.Adam(clipvalue = .01)
def last_time_step_mse(y_true, y_pred):
    return keras.metrics.mean_squared_error(y_true[:, -1], y_pred[:, -1])
optimizer = keras.optimizers.SGD(clipvalue = .5)
model.compile(loss = "mean_squared_error",
              optimizer = optimizer,
              metrics = [last_time_step_mse])
from math import isnan
from numpy.random import randint
train_len = 200
batch_size = 32
total_batches = 1000

runs = X.shape[0] # i'm treating each split of the data as a different 'run' of the experiment
run_size = X.shape[1]
for i in range(total_batches):
    indices = [(randint(0, runs), randint(0, run_size - train_len)) for i in range(batch_size)]
    X_mini = np.expand_dims(np.array([X[index[0],index[1]:index[1]+train_len] for index in indices]), 2)
    y_mini = np.array([y[index[0],index[1]+train_len] for index in indices])
    hist = model.fit(X_mini, y_mini, epochs=1, batch_size=batch_size,\
                     callbacks = [model_checkpoint_callback], verbose = 0)
    if(isnan(hist.history['loss'][0])):
        print("training failure")
        # broken_on_error = True; break
        model.load_weights("./model_saves")
    if(i % 100 == 0):
        print("{} batches completed".format(i))
print("training complete")
#%% evaluate model
from sklearn.metrics import mean_squared_error

X_test = X[0]
test_times = acc_t_reshape[0]
pred_pin = pin_scaler.inverse_transform(model.predict(X_test.reshape(1, -1, 1)))
# mse = mean_squared_error(pin, pred_pin)
plt.figure()
plt.title("LSTM prediction of pin location")
plt.plot(pin_t, pin, label = "actual pin location")
plt.plot(test_times, pred_pin.flatten(), label = "predicted pin location")
plt.xlabel("Training time [seconds]")
plt.ylabel("pin location")
plt.legend()
#%% garbage
# import scipy as sp
# from scipy import signal
# f, ts, Sxx = signal.spectrogram(pred_pin, window=('tukey', 5000),nperseg=10000, noverlap=5000)
# plt.figure(figsize = (6.5, 3))
# plt.pcolormesh(ts, f, Sxx, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.tight_layout()
# plt.show()

# from scipy.fft import fft, fftfreq
# # Number of sample points
# N = pred_pin.shape[0]
# # sample spacing
# T = 1
# y = pred_pin
# yf =  2.0/N * np.abs(fft(y)[1:N//2])
# xf = fftfreq(N, T)[:N//2]
# period = 1/xf[np.argmax(yf)]
# print("period: " + str(period))
# plt.figure(figsize=(6.5, 4))
# plt.plot(xf[1:],yf)
# plt.grid()
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power')
# plt.title("FFT of vibration data")
# plt.tight_layout()
# plt.savefig('fft.png', dpi = 500)
# plt.show()