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
I think I'm getting somewhere, but training is a bitch.
I'm making the model non-sequential, and this will allow me to turn layers
on and off during training. I want to iterate the model into correctness.
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
    # pin contains some nan values
    from math import isnan
    for i in range(len(pin)):
        if(isnan(pin[i])):
            pin[i] = pin[i-1]
    
    ds = 64 # downsampling factor
    
    # scaling data, which means that it must be unscaled to be useful
    acc_scaler = sk.preprocessing.StandardScaler()
    acc_scaler.fit(acc.reshape(-1, 1))
    acc = acc_scaler.fit_transform(acc.reshape(-1, 1)).flatten()
    pin_scaler = sk.preprocessing.StandardScaler()
    pin_scaler.fit(pin.reshape(-1,1))
    pin_transform = pin_scaler.fit_transform(pin.reshape(-1,1)).flatten().astype(np.float32)
    
    

    y = np.array([pin_transform[(np.abs(pin_t - v)).argmin()] for v in acc_t])
    # remove data from before initial excitement (at 1.5 seconds)
    acc = acc[acc_t > 1.5]
    y = y[acc_t > 1.5]
    acc_t = acc_t[acc_t > 1.5]
    
    #reshape/downsample
    X = np.reshape(acc[:acc.size//ds*ds], (acc.size//ds, ds)).T
    acc_t_reshape = np.reshape(acc_t[:acc_t.size//ds*ds], (acc_t.size//ds, ds)).T
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

plt.figure()
plt.plot(acc_t_reshape.flatten(), y.flatten())

load_X.close()
load_y.close()
load_t.close()
#%% toy models - period and amplitude

print("loading toy models...")
# period_model = keras.models.load_model('period')
amplitude_model = keras.models.load_model('amplitude')
frequency_model = keras.models.load_model('frequency')
f_units = 20
a_units = 10

#%% model is a splice of the period and amplitude for the lower two layers
print("making model...")
m_units = 10
n_units = 5
l_units = 15
x_input = keras.Input(shape = [None, 1], name = 'x')

f = keras.Sequential([
    keras.layers.LSTM(f_units, return_sequences = True, input_shape = [None, 1]),
    keras.layers.LSTM(f_units, return_sequences = True)],
    name='f')(x_input)
# f.layers[0].set_weights(frequency_model.layers[0].get_weights())
# f.layers[1].set_weights(frequency_model.layers[1].get_weigts())
a = keras.Sequential([
    keras.layers.LSTM(a_units, return_sequences = True, input_shape = [None, 1]),
    keras.layers.LSTM(a_units, return_sequences = True)],
    name='a')(x_input)
# a.layers[0].set_weights(amplitude_model.layers[0].get_weights())
# a.layers[1].set_weights(amplitude_model.layers[1].get_weights())
m = keras.layers.LSTM(m_units, name='m', return_sequences=True)(f)
n = keras.layers.LSTM(n_units, name='n', return_sequences=True)(a)
concat = keras.layers.Concatenate(axis=2, name='concat')([m,n])
l = keras.layers.LSTM(l_units, name='l', return_sequences=True)(concat)
d = keras.layers.TimeDistributed(keras.layers.Dense(1), name='d')(l)

model = keras.Model(
    inputs=[x_input],
    outputs=[d]
)
# model.get_layer('f').layers[0].set_weights(frequency_model.layers[0].get_weights())
# model.get_layer('f').layers[1].set_weights(frequency_model.layers[1].get_weights())
# model.get_layer('a').layers[0].set_weights(amplitude_model.layers[0].get_weights())
# model.get_layer('a').layers[1].set_weights(amplitude_model.layers[1].get_weights())

#%% train model
from stocastic_sdg_fit import fit
def plot_evaluate(savfig = False, savpath = "plot.png"):
    from sklearn.metrics import mean_squared_error
    X_test = X[0]
    y_test = pin_scaler.inverse_transform(y[0])
    test_times = acc_t_reshape[0]
    y_pred = pin_scaler.inverse_transform(model.predict(X_test.reshape(1, -1, 1))).flatten()
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared = False)
    
    plt.figure(figsize=(4.2,3))
    plt.title("LSTM prediction of pin location")
    plt.plot(test_times, y_pred, label = "predicted pin location")
    plt.plot(test_times, y_test, label = "actual pin location",alpha=.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Pin location [m]")
    plt.ylim((0.045, .23))
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(savpath, dpi=800)
    print("mean squared error: " + str(mse))
    print("root mean squared error: " + str(rmse))



model.get_layer('f').trainable = False
model.get_layer('a').trainable = False
model.get_layer('m').trainable = True
model.get_layer('n').trainable = False
model.get_layer('l').trainable = True
model.get_layer('d').trainable = True
model = fit(X, y, model, total_batches = 600)
plot_evaluate()
model.get_layer('f').trainable = False
model.get_layer('a').trainable = False
model.get_layer('m').trainable = False
model.get_layer('n').trainable = True
model.get_layer('l').trainable = True
model.get_layer('d').trainable = True
model = fit(X, y, model, total_batches = 600)
plot_evaluate()
model.get_layer('f').trainable = False
model.get_layer('a').trainable = False
model.get_layer('m').trainable = True
model.get_layer('n').trainable = True
model.get_layer('l').trainable = True
model.get_layer('d').trainable = True
model = fit(X, y, model, total_batches = 600)
plot_evaluate()
model.get_layer('f').trainable = True
model.get_layer('a').trainable = True
model.get_layer('m').trainable = True
model.get_layer('n').trainable = True
model.get_layer('l').trainable = True
model.get_layer('d').trainable = True
model = fit(X, y, model, total_batches = 4000)
plot_evaluate()
print("training complete.")
#%% evaluate model
print("plotting...")

#%% save model
print("saving model...")
model.save("pin_loc_slightly_overfit")
#%%
# pred_freq = frequency_model.predict(X_test.reshape(1, -1, 1))

# plt.figure()
# plt.plot(test_times, pred_freq.flatten())
# plt.xlabel("Time [seconds]")
# plt.ylabel("pin location")
