def preprocess():
    import json
    import pickle
    import numpy as np
    import sklearn as sk
    import joblib
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
    from sklearn import preprocessing
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
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(X, y, acc_t_reshape)
    X_train = np.expand_dims(X_train, 2)
    X_test = np.expand_dims(X_test, 2)
    y_train = np.expand_dims(y_train, 2)
    y_test = np.expand_dims(y_test, 2)
    t_train = np.expand_dims(t_train, 2)
    t_test = np.expand_dims(t_test, 2)
    
    
    sav_X_train = open('./pickles/X_train', 'wb')
    sav_y_train = open('./pickles/y_train', 'wb')
    sav_t_train = open('./pickles/t_train', 'wb')
    sav_X_test = open('./pickles/X_test', 'wb')
    sav_y_test = open('./pickles/y_test', 'wb')
    sav_t_test = open('./pickles/t_test', 'wb')
    
    pickle.dump(X_train, sav_X_train)
    pickle.dump(y_train, sav_y_train)
    pickle.dump(t_train, sav_t_train)
    pickle.dump(X_test, sav_X_test)
    pickle.dump(y_test, sav_y_test)
    pickle.dump(t_test, sav_t_test)
    
    joblib.dump(acc_scaler, './pickles/acc_scaler', compress=True)
    joblib.dump(pin_scaler, './pickles/pin_scaler', compress=True)
    
    sav_X_train.close()
    sav_y_train.close()
    sav_t_train.close()
    sav_X_test.close()
    sav_y_test.close()
    sav_t_test.close()

def save_model_weights_as_json(model, savpath="model_weights.json"):
    import json
    i = 0
    data = {}
    layer_weights = []
    for layer in model.layers:
        layer_weights = [a.tolist() for a in layer.get_weights()]
        data["layer" + str(i)] = layer_weights
        i += 1
    with open(savpath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    preprocess()