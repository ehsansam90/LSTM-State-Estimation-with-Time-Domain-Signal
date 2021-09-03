import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import model_selection
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
import joblib
"""
experimenting with my toy model on the DROPBEAR dataset
"""
#%% load data
def load_and_preprocess():
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
    sav_X = open('./pickles/X_data', 'wb')
    sav_y = open('./pickles/y_data', 'wb')
    sav_t = open('./pickles/t_data', 'wb')
    
    pickle.dump(X, sav_X)
    pickle.dump(y, sav_y)
    pickle.dump(acc_t_reshape, sav_t)
    joblib.dump(acc_scaler, './pickles/acc_scaler', compress=True)
    joblib.dump(pin_scaler, './pickles/pin_scaler', compress=True)
    
    sav_X.close()
    sav_y.close()
    sav_t.close()

# print("preprocessing data...")
# load_and_preprocess()

print("loading data...")
load_X = open("./pickles/X_data", 'rb')
load_y = open("./pickles/y_data", 'rb')
load_t = open("./pickles/t_data", 'rb')

X = pickle.load(load_X)
y = pickle.load(load_y)
acc_t_reshape = pickle.load(load_t)
pin_scaler = joblib.load('./pickles/pin_scaler')
acc_scaler = joblib.load('./pickles/pin_scaler')

load_X.close()
load_y.close()
load_t.close()
#%% toy models - period and amplitude

print("loading toy models...")
period_model = keras.models.load_model('period')
amplitude_model = keras.models.load_model('amplitude')
frequency_model = keras.models.load_model('frequency')

#%% model is a splice of the period and amplitude for the lower two layers
print("making model...")
t_units = 20
a_units = 10
tw1, tu1, tb1 = frequency_model.layers[0].get_weights()
aw1, au1, ab1 = amplitude_model.layers[0].get_weights()
tw2, tu2, tb2 = frequency_model.layers[1].get_weights()
aw2, au2, ab2 = amplitude_model.layers[1].get_weights()

w1 = np.append(tw1[:,:t_units],aw1[:,:a_units], axis=1)
w1 = np.append(w1, tw1[:,t_units:t_units*2], axis=1)
w1 = np.append(w1, aw1[:,a_units:a_units*2], axis=1)
w1 = np.append(w1, tw1[:,t_units*2:t_units*3], axis=1)
w1 = np.append(w1, aw1[:,a_units*2:a_units*3], axis=1)
w1 = np.append(w1, tw1[:,t_units*3:], axis=1)
w1 = np.append(w1, aw1[:,a_units*3:], axis=1)

u1 = np.insert(tu1, t_units, np.zeros((a_units, t_units)), axis=1)
u1 = np.insert(u1, 2*t_units+a_units, np.zeros((a_units, t_units)), axis=1)
u1 = np.insert(u1, 3*t_units+2*a_units, np.zeros((a_units, t_units)), axis=1)
u1 = np.insert(u1, 4*t_units+3*a_units, np.zeros((a_units, t_units)), axis=1)
u1_ = np.insert(au1, 0, np.zeros((t_units, a_units)), axis=1)
u1_ = np.insert(u1_, t_units+a_units, np.zeros((t_units, a_units)), axis=1)
u1_ = np.insert(u1_, 2*t_units+2*a_units, np.zeros((t_units, a_units)), axis=1)
u1_ = np.insert(u1_, 3*t_units+3*a_units, np.zeros((t_units, a_units)), axis=1)
u1 = np.append(u1, u1_, axis=0)

b1 = np.append(tb1[:t_units], ab1[:a_units])
b1 = np.append(b1, tb1[t_units:t_units*2])
b1 = np.append(b1, ab1[a_units:a_units*2])
b1 = np.append(b1, tb1[t_units*2:t_units*3])
b1 = np.append(b1, ab1[a_units*2:a_units*3])
b1 = np.append(b1, tb1[t_units*3:])
b1 = np.append(b1, ab1[a_units*3:])

w2 = np.insert(tw2, t_units, np.zeros((a_units, t_units)), axis=1)
w2 = np.insert(w2, 2*t_units+a_units, np.zeros((a_units, t_units)), axis=1)
w2 = np.insert(w2, 3*t_units+2*a_units, np.zeros((a_units, t_units)), axis=1)
w2 = np.insert(w2, 4*t_units+3*a_units, np.zeros((a_units, t_units)), axis=1)
w2_ = np.insert(aw2, 0, np.zeros((t_units, a_units)), axis=1)
w2_ = np.insert(w2_, t_units+a_units, np.zeros((t_units, a_units)), axis=1)
w2_ = np.insert(w2_, 2*t_units+2*a_units, np.zeros((t_units, a_units)), axis=1)
w2_ = np.insert(w2_, 3*t_units+3*a_units, np.zeros((t_units, a_units)), axis=1)
w2 = np.append(w2, w2_, axis=0)

u2 = np.insert(tu2, t_units, np.zeros((a_units, t_units)), axis=1)
u2 = np.insert(u2, 2*t_units+a_units, np.zeros((a_units, t_units)), axis=1)
u2 = np.insert(u2, 3*t_units+2*a_units, np.zeros((a_units, t_units)), axis=1)
u2 = np.insert(u2, 4*t_units+3*a_units, np.zeros((a_units, t_units)), axis=1)
u2_ = np.insert(au2, 0, np.zeros((t_units, a_units)), axis=1)
u2_ = np.insert(u2_, t_units+a_units, np.zeros((t_units, a_units)), axis=1)
u2_ = np.insert(u2_, 2*t_units+2*a_units, np.zeros((t_units, a_units)), axis=1)
u2_ = np.insert(u2_, 3*t_units+3*a_units, np.zeros((t_units, a_units)), axis=1)
u2 = np.append(u2, u2_, axis=0)

b2 = np.append(tb2[:t_units], ab2[:a_units])
b2 = np.append(b2, tb2[t_units:t_units*2])
b2 = np.append(b2, ab2[a_units:a_units*2])
b2 = np.append(b2, tb2[t_units*2:t_units*3])
b2 = np.append(b2, ab2[a_units*2:a_units*3])
b2 = np.append(b2, tb2[t_units*3:])
b2 = np.append(b2, ab2[a_units*3:])


dense_top = keras.layers.TimeDistributed(keras.layers.Dense(1))
model = keras.models.Sequential([
    keras.layers.LSTM(t_units+a_units, return_sequences = True, input_shape = [None, 1], trainable=False),
    keras.layers.LSTM(t_units+a_units, return_sequences = True, trainable=False),
    keras.layers.LSTM(20, return_sequences = True),
    keras.layers.LSTM(20, return_sequences = True),
    # keras.layers.LSTM(1, return_sequences = False)])
    dense_top])
model.layers[0].set_weights([w1,u1,b1])
model.layers[1].set_weights([w2,u2,b2])
# model.layers[0].set_weights(frequency_model.layers[0].get_weights())
# model.layers[1].set_weights(frequency_model.layers[1].get_weights())
# model.layers[0].set_weights(amplitude_model.layers[0].get_weights())
# model.layers[1].set_weights(amplitude_model.layers[1].get_weights())
#%% train model
from stocastic_sdg_fit import fit
print("training model...")
model = fit(X, y, model, total_batches = 600)

print("training complete.")
#%% evaluate model
print("plotting...")
from sklearn.metrics import mean_squared_error

X_test = X[0]
y_test = pin_scaler.inverse_transform(y[0])
test_times = acc_t_reshape[0]
pred_pin = pin_scaler.inverse_transform(model.predict(X_test.reshape(1, -1, 1)))
# mse = mean_squared_error(pin, pred_pin)
plt.figure()
plt.title("LSTM prediction of pin location")
plt.plot(test_times, y_test, label = "actual pin location")
plt.plot(test_times, pred_pin.flatten(), label = "predicted pin location")
plt.xlabel("Time [seconds]")
plt.ylabel("pin location")
plt.legend()
#%% save model
print("saving model...")
model.save("pin_loc")
#%%
# pred_freq = frequency_model.predict(X_test.reshape(1, -1, 1))

# plt.figure()
# plt.plot(test_times, pred_freq.flatten())
# plt.xlabel("Time [seconds]")
# plt.ylabel("pin location")

