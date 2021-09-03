# LSTM-State-Estimation-with-Time-Domain-Signal
It's a mouthful isn't it?

# Goals
Experiment with LSTM recurrent neural networks to see if a vibration signal in time domain can be used to perform a state prediction. 
We use the DROPBEAR dataset, which has an accelerometer connected to a vibrating beam. The state of the system is the location of a roller along the beam.
# Methodology
Previous iterations and experimentations can be seen in /old_versions. The following is what I found to work the best:
## Data Preprocessing
Acceleration data is downsampled by 64. So as to not lose data, each splice of the total data is saved and used for training. State (roller position) comes with different time values. I align them with the acceleration data by taking state data point for every acceleration value. Slight improvement could be made by using linear interpolation. Some state values from the raw data are null; those are replaced with the previous value. 
## The Toy Models
When I couldn't get the acceleration/roller position model to converge, I created a toy dataset with sinewave signals of varying frequency, amplitude. When I got good results from LSTMs to simply predict frequency and amplitude, I had the idea to use these toy models as the pretrained lower layers to the real model. Maybe with the improvements I made to training, that isn't need needed anymore, but it definitely improves the time to convergence: especially frequency, because that is what physics tells us roller position effects. Generating the toy models can be seen in frequency-prediction-lstm.py
## Training the Real Model
The frequency and amplitude models were put in parallel to form the pretrained lower layers. Another LSTM layer was placed on top of each, then the outputs of the parallel LSTM sequences were concatinated. Another LSTM was placed on top, then a dense top. I froze and unfroze training of the layers in a sequence which can be seen in lstm-v6.py. The idea was to help 'merge' the two models together, but it probably does nothing. 
## Sequentializing the Models
I didn't like the parellel layers (for one thing, it makes it hard to implement this in hardware, the next step of this project). There are some not-too-complex formulas for combining parellel layers, (see sequentialize.py), and I followed those to make the final model. It has shape LSTM(30 units) -> LSTM(30 units) -> LSTM -> (30 units) -> LSTM(15 units) -> Dense Top, but of course, you can train any shape you want. One side effect of sequentializing the model, is that it creates zero spaces in the matrices of the new layers. I trained it again, trying to take advantage of these new weights, but actually the RMSE didn't decrease that much. 
## Results
RMSE ~= .006 were the best results I got. I'm sure there are things I could to make it better. I never experimented with increased the amount of units, or tuning hyperparameters, for instance.
