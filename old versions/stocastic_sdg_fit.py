import numpy as np
import tensorflow.keras as keras
# does not inverse transform for error values
def fit(X, y, model, filepath = "./model_saves/model", train_len=100, batch_size=32, total_batches=1200,
        evaluate_every=None):
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
            plot_evaluate(X, y, model)
    return model