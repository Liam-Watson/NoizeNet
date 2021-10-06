import matplotlib   
import numpy as np
from collections import Counter
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import librosa as lib
import os
import soundfile as sf #For writing
import matplotlib.pyplot as plt
import gc
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import sklearn as skl
import pandas as pd
import utils
import librosa.display


# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

#######################################################################################################################

class NoizeNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(NoizeNet, self).__init__()

            
        self.hidden_dim = hidden_dim
        self.num_layers = n_layers
        self.output_size = output_size
        self.input_size = input_size


        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        # self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        #TODO: Test RNN vs LSTM
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        # self.hidden = (torch.zeros(1,1,self.hidden_dim), torch.zeros(1,1,self.hidden_dim)) #We need a tuple for a LSTM


        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
    # def forward(self, x, hidden, c0): #LSTM!
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)
        if (train_on_gpu):
            x.cuda()
        else:
            x.cpu()
        # get RNN outputs
        # r_out, hidden = self.rnn(x , hidden)
        # r_out, (hidden, c0) = self.lstm(x , hidden) #LSTM!
        #####LSTM
        h_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).cuda() #hidden state
        c_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).cuda() #internal state
        # Propagate input through LSTM
        r_out, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hidden = hn
        #####LSTM

        # shape output to be (batch_size*seq_length, hidden_dim)
        if (train_on_gpu):
            hidden.cuda()
        r_out = r_out.view(-1, self.hidden_dim)

        # get final output
        output = self.fc(r_out)

        return output, hidden

#####################################################################################################################

# decide on hyperparameters
n_steps = 1
input_size=1
output_size=1
hidden_dim=50
n_layers=2

# instantiate an RNN
noizeNet = NoizeNet(input_size, output_size, hidden_dim, n_layers)
print(noizeNet)
######################################################################################################################

# MSE loss and Adam optimizer with a learning rate of 0.01
# criterion = nn.MSELoss()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(noizeNet.parameters(), lr=0.001)

######################################################################################################################

# train the RNN
def train(noizeNet, n_steps, AUDIO_DIR, genreTracks, step_size=1, duration=5):
    count = 0
    fileCount = 0
    for f in genreTracks[0]:
        print(f)
        fileCount+=1
        hidden = None
        c0 = None #LSTM!
        fileData, sr = lib.load("/home/liam/Desktop/University/2021/MAM3040W/thesis/works/playground/wavs/fma_small/000/" + f, mono=True, duration = duration)
        batch_size = (int)(duration*sr/n_steps)
        number_of_steps = len(fileData)-batch_size
        print("BATCH SIZE:",  batch_size, "\nNUMBER OF BATCHES:", n_steps ,"\nNUMBER OF STEPS: ", number_of_steps)
        for batch_i in (range(0,number_of_steps)):
            # defining the training data

            if(train_on_gpu):
                noizeNet.cuda()

            time_steps = np.linspace((int)(batch_i), (int)((batch_i + batch_size)), (int)(batch_size))
            # print(data.size)
            data = fileData[(batch_i): batch_size + batch_i]
            # data = np.resize(data, (seq_length+1), 1)
            data = np.resize(data,((batch_size), 1))
            # print(data.size, batch_i + batch_size, batch_i)
            # data.resize((seq_length + 1, 1))  # input_size=1

            x = data[:-1]
            y = data[1:]

            # convert data into Tensors
            x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
            y_tensor = torch.Tensor(y).unsqueeze(0)
            if(train_on_gpu):
                    x_tensor, y_tensor = x_tensor.cuda(), y_tensor.cuda()
            # print(x_tensor.size())
            # outputs from the rnn
            prediction, hidden = noizeNet(x_tensor, hidden)

            # prediction, hidden, c0 = noizeNet(x_tensor, hidden, c0) #LSTM!

            ## Representing Memory ##
            # make a new variable for hidden and detach the hidden state from its history
            # this way, we don't backpropagate through the entire history
            hidden = hidden.data
            # c0 = c0.data #LSTM!

            # calculate the loss
            loss = criterion(prediction, y_tensor)
            # zero gradients
            optimizer.zero_grad()
            # perform backprop and update weights
            loss.backward()
            optimizer.step()

            if(int((batch_i/number_of_steps)) % 100 == 0 and batch_i % 100 == 0):
                print("PROGRESS:\t", round((batch_i/number_of_steps)*100, 2), "%"  , sep="")
                print("Prediction dimensions:\t", prediction.cpu().data.numpy().flatten()[-5:],"\nY:\t", y[-5:].flatten() ,"\nX:\t" , x[-5:].flatten() ,sep="")
                print('Loss: ', loss.item(), "\t num:", batch_i, "\t File:", fileCount)
        del prediction
        del hidden
        del x_tensor
        del y_tensor
        del data
        del x
        del y
        gc.collect()
    return noizeNet

###################################################################################################################
def predict(noizeNet, duration=5, n_steps=30):
    print("PREDICTING...")
    noizeNet.eval()
    hidden = None
    
    y, sr = lib.load("/home/liam/Desktop/University/2021/MAM3040W/thesis/works/playground/wavs/fma_small/000/" + files[1], mono=True, duration = duration)
    data = y
    # music = np.zeros(shape=y.shape)
    # data = np.resize(data,((seq_length+1), 1))
    data = np.random.normal(-1,1,y.shape)

    batch_size = (int)(duration*sr/n_steps)
    number_of_steps = len(data)-batch_size
    # music = y[0 : batch_size]
    music = []
    next = data[batch_size]
    fileData = data
    print("BATCH SIZE:", batch_size ,sep="\t")
    print("NUMBER OF STEPS:", number_of_steps , sep="\t")
    # data = data[:batch_size]
    for batch_i in (range(0, number_of_steps)):
        
        # defining the training data

        if(train_on_gpu):
            noizeNet.cuda()

        # time_steps = np.linspace((int)(batch_i), (int)((batch_i + batch_size)), (int)(batch_size))
        # print(data.size)
        data = data[1: batch_size-1]
        data = np.append(data, next)
        # print("data",data[-1])
        # data = np.resize(data, (seq_length+1), 1)
        data = np.resize(data,((batch_size), 1))
        # print(data.size, sr*batch_i, step_size*batch_i)
        # data.resize((seq_length + 1, 1))  # input_size=1

        x = data

        # convert data into Tensors
        x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
        if(train_on_gpu):
                x_tensor = x_tensor.cuda()

        prediction, hidden = noizeNet(x_tensor, hidden)
        
        music = np.append(music,(prediction.cpu().data.numpy().flatten())[-1])
        next = prediction.cpu().data.numpy().flatten()[-1]
        if(int((batch_i/number_of_steps)) % 100 == 0 and batch_i % 100 == 0):
            print("PROGRESS:\t", round(((batch_i)/number_of_steps)*100, 2), "%"  , sep="")
            print("Prediction dimensions:\t", prediction.cpu().size(), "\t" ,prediction.cpu().data.numpy().flatten().size, "\nMusic dimensions:\t", music.size ,sep="")
       
        hidden = hidden.data

    sf.write('/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/outputSoundFile.wav', music, 22050,format="WAV")
    time_steps = np.linspace(0, len(music), len(music))
    plt.plot(time_steps, music,"b.",  markersize=0.1)
    plt.show()
    return music
###################################################################################################################



#Get metadata for fma dataset
AUDIO_DIR = "data/fma_small/"

tracks = utils.load('data/fma_metadata/tracks.csv')
# genres = utils.load('data/fma_metadata/genres.csv') #Not needed
# features = utils.load('data/fma_metadata/features.csv') #Not needed
# echonest = utils.load('data/fma_metadata/echonest.csv') #Not needed 

small = tracks['set', 'subset'] <= 'small'
genre1 = tracks['track', 'genre_top'] == 'Instrumental'
genre2 = tracks['track', 'genre_top'] == 'Hip-Hop' #We can set multilpe genres bellow as (genre1 | genre2)
genreTracks = tracks.loc[small & (genre1), ('track', 'number')]
X = tracks.loc[small & (genre1)]
for g in X :
    print(g)

#Set if we want to train new model or load and predict with saved model
TRAIN = True

if TRAIN:
    n_steps = 30
    print_every = 5
    step_size =  1
    duration = 5
    print("TRAINING...")
    trained_rnn = train(noizeNet, n_steps,AUDIO_DIR, genreTracks, step_size=step_size, duration = duration)
    torch.save(trained_rnn.state_dict(), "/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/smallTrainingV1LSTM.pt")
    predict(noizeNet=trained_rnn, duration=duration, n_steps=n_steps)

    

else:
    n_steps = 30
    input_size=1
    output_size=1
    hidden_dim=50
    step_size =  1
    n_layers=1
    # instantiate an RNN
    noizeNet = NoizeNet(input_size, output_size, hidden_dim, n_layers)
    # model = TheModelClass(*args, **kwargs)
    noizeNet.load_state_dict(torch.load("/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/trainedBtchv2.pt"))
    predict(noizeNet=noizeNet, duration=duration, n_steps=n_steps)

#TODO This is how to plot
# fig, ax = plt.subplots(nrows=3, sharex=True)
# ax[0].set(title='Envelope view, mono')
# ax[0].label_outer()
# lib.display.waveshow(y, sr=sr, ax=ax[0])
# plt.show()

