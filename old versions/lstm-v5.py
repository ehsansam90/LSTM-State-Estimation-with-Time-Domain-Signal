import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import tensorflow.keras as keras
import joblib
"""
After doing some edits to training, preprocessing, it no longer looks like
pretraining with the toy sets in necessary. Coming towards my final architecture,
it will be LSTM layers 30->30->15->15->Dense 1.

There are still a few more things that I want to do.
    1. Increase the amount of validation runs
    2. Decrease learning rate and see if that has an effect on training of sequentialized model
    3. Downey wants me to make the algorithm work through a rolling window. I don't
       see why but I'll do it.
    4. I need to see the extent to which the pretrained lower layers are required
       because that would be a pain to work with if we plan on going through the
       grid search for lowest error.
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
amplitude_model = keras.models.load_model('amplitude')
frequency_model = keras.models.load_model('frequency')
#%% model is a splice of the period and amplitude for the lower two layers
print("making model...")
f_units = 20
a_units = 10
m_units = 10
n_units = 5
l_units = 15
x_input = keras.Input(shape = [None, 1], name = 'x')

f = keras.Sequential([
    keras.layers.LSTM(f_units, return_sequences = True, input_shape = [None, 1]),
    keras.layers.LSTM(f_units, return_sequences = True)],
    name='f')(x_input)
a = keras.Sequential([
    keras.layers.LSTM(a_units, return_sequences = True, input_shape = [None, 1]),
    keras.layers.LSTM(a_units, return_sequences = True)],
    name='a')(x_input)
m = keras.layers.LSTM(m_units, name='m', return_sequences=True)(f)
n = keras.layers.LSTM(n_units, name='n', return_sequences=True)(a)
concat = keras.layers.Concatenate(axis=2, name='concat')([m,n])
l = keras.layers.LSTM(l_units, name='l', return_sequences=True)(concat)
d = keras.layers.TimeDistributed(keras.layers.Dense(1), name='d')(l)

model = keras.Model(
    inputs=[x_input],
    outputs=[d]
)
model.get_layer('f').layers[0].set_weights(frequency_model.layers[0].get_weights())
model.get_layer('f').layers[1].set_weights(frequency_model.layers[1].get_weights())
model.get_layer('a').layers[0].set_weights(amplitude_model.layers[0].get_weights())
model.get_layer('a').layers[1].set_weights(amplitude_model.layers[1].get_weights())
# model = keras.models.Sequential([
#     keras.layers.LSTM(30, return_sequences = True, input_shape = [None, 1], trainable=False),
#     keras.layers.LSTM(30, return_sequences = True, trainable=False),
#     keras.layers.LSTM(15, return_sequences = True),
#     keras.layers.LSTM(15, return_sequences = True),
#     keras.layers.TimeDistributed(keras.layers.Dense(1))
# ])

#%% train model
frame = 0 # i want to make a gif of training 
from stocastic_sdg_fit import fit
def plot_evaluate(plot_fig = True, savfig = False, savpath = "plot.png"):
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
    if(savfig):    
        plt.savefig(savpath, dpi=800)
    if(not plot_fig):
        plt.close()
    print("mean squared error: " + str(mse))
    print("root mean squared error: " + str(rmse))
    return rmse, mse

def fit(X, y, model, filepath = "./model_saves/model", train_len=100, batch_size=32, total_batches=1200,
        evaluate_every=None):
    global frame
    print("training model...")
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath = filepath,
            save_weights_only = True,
            monitor = "loss",
            mode = "min",
            save_freq = 1,
            save_best_only = True)
    model.save_weights(filepath)
    def last_time_step_mse(y_true, y_pred):
        return keras.metrics.mean_squared_error(y_true[:, -1], y_pred[:, -1])
    optimizer = keras.optimizers.SGD(clipvalue = .5, learning_rate=0.01, momentum=.9)
    model.compile(loss = "mean_squared_error",
                  optimizer = optimizer,
                  metrics = [last_time_step_mse])
    from math import isnan
    from numpy.random import randint
    
    runs = X.shape[0] # i'm treating each split of the data as a different 'run' of the experiment
    run_size = X.shape[1]
    best_rmse = None
    for i in range(total_batches):
        if(i % 100 == 0):
            print("{} mini-batches completed".format(i))
        indices = [(randint(1, runs), randint(0, run_size - train_len)) for i in range(batch_size)]
        X_mini = np.expand_dims(np.array([X[index[0],index[1]:index[1]+train_len] for index in indices]), 2)
        y_mini = np.array([y[index[0],index[1]+train_len] for index in indices])
        hist = model.fit(X_mini, y_mini, epochs=1, batch_size=batch_size,\
                         callbacks = [model_checkpoint_callback], verbose = 0)
        if(isnan(hist.history['loss'][0])):
            print("training failure")
            # broken_on_error = True; break
            model.load_weights(filepath)
        if(evaluate_every != None and i % evaluate_every == 0 and i != 0):
            rmse = plot_evaluate(savfig=True, savpath="./plots/plt" + str(frame))[0]
            frame += 1
            if(best_rmse == None or rmse < best_rmse):
                model.save_weights("./model_saves/pin_loc" + str(i))
                best_rmse = rmse
            
    return model

model.get_layer('f').trainable = False
model.get_layer('a').trainable = False
model.get_layer('m').trainable = True
model.get_layer('n').trainable = False
model.get_layer('l').trainable = True
model.get_layer('d').trainable = True
model = fit(X, y, model, total_batches = 600)
plot_evaluate(savfig=True, savpath="./plots/" + str(frame)); frame +=1;
model.get_layer('f').trainable = False
model.get_layer('a').trainable = False
model.get_layer('m').trainable = False
model.get_layer('n').trainable = True
model.get_layer('l').trainable = True
model.get_layer('d').trainable = True
model = fit(X, y, model, total_batches = 600)
plot_evaluate(savfig=True, savpath="./plots/" + str(frame)); frame +=1;
model.get_layer('f').trainable = False
model.get_layer('a').trainable = False
model.get_layer('m').trainable = True
model.get_layer('n').trainable = True
model.get_layer('l').trainable = True
model.get_layer('d').trainable = True
model = fit(X, y, model, total_batches = 600)
plot_evaluate(savfig=True, savpath="./plots/" + str(frame)); frame +=1;
model.get_layer('f').trainable = True
model.get_layer('a').trainable = True
model.get_layer('m').trainable = True
model.get_layer('n').trainable = True
model.get_layer('l').trainable = True
model.get_layer('d').trainable = True
model = fit(X, y, model, total_batches = 1500, evaluate_every=400)
plot_evaluate()
print("training complete.")
model.save("pretrained_split")
#%% turn LSTM sequential
# so this was back when i had t instead of f
print("making sequential model")
t_units = f_units
seq_model = keras.models.Sequential([
    keras.layers.LSTM(30, return_sequences = True, input_shape = [None, 1], trainable=False),
    keras.layers.LSTM(30, return_sequences = True, trainable=False),
    keras.layers.LSTM(15, return_sequences = True),
    keras.layers.LSTM(15, return_sequences = True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])

tw1, tu1, tb1 = model.get_layer('f').layers[0].get_weights()
aw1, au1, ab1 = model.get_layer('a').layers[0].get_weights()
tw2, tu2, tb2 = model.get_layer('f').layers[1].get_weights()
aw2, au2, ab2 = model.get_layer('a').layers[1].get_weights()

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

seq_model.layers[0].set_weights([w1,u1,b1])
seq_model.layers[1].set_weights([w2,u2,b2])
# do the same for M and N layers / third layer of sequential
tw2, tu2, tb2 = model.get_layer('m').get_weights()
aw2, au2, ab2 = model.get_layer('n').get_weights()

w2 = np.insert(tw2, m_units, np.zeros((n_units, f_units)), axis=1)
w2 = np.insert(w2, 2*m_units+n_units, np.zeros((n_units, f_units)), axis=1)
w2 = np.insert(w2, 3*m_units+2*n_units, np.zeros((n_units, f_units)), axis=1)
w2 = np.insert(w2, 4*m_units+3*n_units, np.zeros((n_units, f_units)), axis=1)
w2_ = np.insert(aw2, 0, np.zeros((m_units, a_units)), axis=1)
w2_ = np.insert(w2_, m_units+n_units, np.zeros((m_units, a_units)), axis=1)
w2_ = np.insert(w2_, 2*m_units+2*n_units, np.zeros((m_units, a_units)), axis=1)
w2_ = np.insert(w2_, 3*m_units+3*n_units, np.zeros((m_units, a_units)), axis=1)
w2 = np.append(w2, w2_, axis=0)

u2 = np.insert(tu2, m_units, np.zeros((n_units, m_units)), axis=1)
u2 = np.insert(u2, 2*m_units+n_units, np.zeros((n_units, m_units)), axis=1)
u2 = np.insert(u2, 3*m_units+2*n_units, np.zeros((n_units, m_units)), axis=1)
u2 = np.insert(u2, 4*m_units+3*n_units, np.zeros((n_units, m_units)), axis=1)
u2_ = np.insert(au2, 0, np.zeros((m_units, n_units)), axis=1)
u2_ = np.insert(u2_, m_units+n_units, np.zeros((m_units, n_units)), axis=1)
u2_ = np.insert(u2_, 2*m_units+2*n_units, np.zeros((m_units, n_units)), axis=1)
u2_ = np.insert(u2_, 3*m_units+3*n_units, np.zeros((m_units, n_units)), axis=1)
u2 = np.append(u2, u2_, axis=0)

b2 = np.append(tb2[:m_units], ab2[:n_units])
b2 = np.append(b2, tb2[m_units:m_units*2])
b2 = np.append(b2, ab2[n_units:n_units*2])
b2 = np.append(b2, tb2[m_units*2:m_units*3])
b2 = np.append(b2, ab2[n_units*2:n_units*3])
b2 = np.append(b2, tb2[m_units*3:])
b2 = np.append(b2, ab2[n_units*3:])

seq_model.layers[2].set_weights([w2,u2,b2])
seq_model.layers[3].set_weights(model.get_layer('l').get_weights())
seq_model.layers[4].set_weights(model.get_layer('d').get_weights())
#%% train with sequential model
model = seq_model
plot_evaluate(plot_fig=True, savfig=False)
model = fit(X, y, model, total_batches = 1500, evaluate_every=400)
model.load_weights("./model_saves/pin_loc800")
plot_evaluate(savfig=True, savpath="./plots/" + str(frame)); frame +=1;
model.save("pretrained_sequential")
#%% rolling window comparison
# it's slower, it's less accurate, but it's a rolling window
def plot_evaluate_windowed(window_size=100, roll=10, plot_fig = True, savfig = False, savpath = "plot.png"):
    from sklearn.metrics import mean_squared_error
    X_test = X[0]
    X_windows = np.array(X[i-100:i] for i in range(window_size, X_test.size//window_size//roll*roll, roll))
    y_test = pin_scaler.inverse_transform(y[0])
    y_test_windows = np.array(y[i-100:i] for i in range(window_size, X_test.size//window_size//roll*roll, roll))
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
    if(savfig):    
        plt.savefig(savpath, dpi=800)
    if(not plot_fig):
        plt.close()
    print("mean squared error: " + str(mse))
    print("root mean squared error: " + str(rmse))
    return rmse, mse
#%% plots
model = keras.models.load_model("pretrained_plit")
keras.utils.plot_model(model, "split_diagram.png", show_shapes=True, dpi=500)
