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
import noise


# torch.autograd.detect_anomaly(True) #Check for errors and return a stack trace. (Used to debug nan loss)

# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

#######################################################################################################################

class NoizeNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, LSTMBool):
        super(NoizeNet, self).__init__()

        self.LSTMBool = LSTMBool    
        self.hidden_dim = hidden_dim
        self.num_layers = n_layers
        self.output_size = output_size
        self.input_size = input_size




        #TODO: Test RNN vs LSTM
        if LSTMBool:
            self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
            self.hidden = (torch.zeros(1,1,self.hidden_dim), torch.zeros(1,1,self.hidden_dim).cuda()) #We need a tuple for a LSTM
        else:
            # define an RNN with specified parameters
            # batch_first means that the first dim of the input and output will be the batch_size
            self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden, c0=None):
    # def forward(self, x, hidden, c0): #LSTM!

        batch_size = x.size(0)
        if (train_on_gpu):
            x.cuda()
        else:
            x.cpu()
        

        if self.LSTMBool:
            # x (batch_size, seq_length, input_size)
            # hidden (n_layers, batch_size, hidden_dim)
            # r_out (batch_size, time_step, hidden_size)
            # r_out, (hidden, c0) = self.lstm(x , hidden) #LSTM without hidden and internal state
            h_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).cuda() #hidden state
            c_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).cuda() #internal state
            # Propagate input through LSTM
            r_out, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
            hidden = hn
        else:
            # get RNN outputs
            r_out, hidden = self.rnn(x , hidden)

        # shape output to be (batch_size*seq_length, hidden_dim)
        if (train_on_gpu):
            hidden.cuda()
        r_out = r_out.view(-1, self.hidden_dim)

        # get final output
        output = self.fc(r_out)

        if self.LSTMBool:
            return output, (hidden, cn)
        else:
            return output, hidden
        
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        #THIS CODE IS FROM UDACITY
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden



######################################################################################################################

# train the RNN
def train(noizeNet, n_steps, AUDIO_DIR, genreTracks, LSTM ,step_size=1, duration=5, numberOfTracks=1, clip=5):
    fileCount = 0
    noizeNet.train()
    for id in genreTracks:
        fileCount+=1
        if(fileCount > numberOfTracks):
            break 

        filename = utils.get_audio_path(AUDIO_DIR, id) #Get the actual path to the file from the id
        

        hidden = None
        if LSTM:
            c0 = None #LSTM!
        fileData, sr = lib.load(filename, mono=True, duration = duration)
        batch_size = (int)(duration*sr/n_steps)
        number_of_steps = len(fileData)-batch_size
        print("BATCH SIZE:",  batch_size, "\nNUMBER OF BATCHES:", n_steps ,"\nNUMBER OF STEPS: ", number_of_steps)
        if(np.isnan(np.sum(fileData))):
            print("NAN ON FILE:\t", filename, "--------------------")
            break
        for batch_i in (range(0,number_of_steps, step_size)):
            # defining the training data

            if(train_on_gpu):
                noizeNet.cuda()

            # time_steps = np.linspace((int)(batch_i), (int)((batch_i + batch_size)), (int)(batch_size))
            data = fileData[(batch_i): batch_size + batch_i]
            data = np.resize(data,((batch_size), 1))    
            # print(data.size, batch_i + batch_size, batch_i)
            # data.resize((seq_length + 1, 1))  # input_size=1

            x = data[:-1]
            y = data[1:]
            if(np.isnan(np.sum(x))):
                print("NAN ON FILE:\t", filename, "XXX--------------------")
                break
            if(np.isnan(np.sum(y))):
                print("NAN ON FILE:\t", filename, "YYY--------------------")
                break
            # convert data into Tensors
            x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
            y_tensor = torch.Tensor(y)
            if(train_on_gpu):
                    x_tensor, y_tensor = x_tensor.cuda(), y_tensor.cuda()
            
            if LSTM:
                prediction, (hidden, c0) = noizeNet(x_tensor, (hidden, c0)) #LSTM!
            else:
                # outputs from the rnn
                prediction, hidden = noizeNet(x_tensor, hidden)

            ## Representing Memory ##
            # make a new variable for hidden and detach the hidden state from its history
            # this way, we don't backpropagate through the entire history
            hidden = hidden.data

            if LSTM:
                c0 = c0.data #LSTM!


            # zero gradients
            optimizer.zero_grad()

            # calculate the loss
            
            loss = criterion(prediction, y_tensor)
            # if(np.isnan(loss)):
            #     break
            
            # perform backprop and update weights
            loss.backward()

            torch.nn.utils.clip_grad_norm_(noizeNet.parameters(), clip) #Clip gradient
            optimizer.step()

            if(int((batch_i/number_of_steps)) % 100 == 0 and batch_i % 100 == 0):
                print("PROGRESS:\t", round((batch_i/number_of_steps)*100, 2), "%"  , sep="")
                print("P:\t", prediction.cpu().data.numpy().flatten()[-5:],"\nY:\t", y[-5:].flatten() ,"\nX:\t" , x[-5:].flatten() ,sep="")
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
def genPerlin(x):
    tmp = []
    for xx in x:
        tmp.append(noise.pnoise1(xx))

    return tmp


def predict(noizeNet, genreTrack ,duration=1, n_steps=30, LSTMBool=False, predictDuration = 30):
    print("PREDICTING...")
    noizeNet.eval()
    hidden = None
    if LSTMBool:
        c0 = None
    filePath = utils.get_audio_path(AUDIO_DIR, genreTrack) #Get the actual path to the file from the id
    y, sr = lib.load(filePath, mono=True, duration = duration)
    
    data = y

    #data = np.random.normal(-1,1,y.shape)
    #data = genPerlin(np.linspace(0,1,y.size))
    
    batch_size = (int)(duration*sr/n_steps)
    # number_of_steps = len(data)-batch_size
    number_of_steps = (sr*predictDuration) - batch_size

    music = []
    next = data[-step_size:]
    sf.write('/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/predictionSeed.wav', np.append(data, next), sr,format="WAV")
    fileData = data
    print("BATCH SIZE:", batch_size ,sep="\t")
    print("NUMBER OF STEPS:", number_of_steps , sep="\t")
    for batch_i in (range(0, number_of_steps, step_size)):

        if(train_on_gpu):
            noizeNet.cuda()

        # time_steps = np.linspace((int)(batch_i), (int)((batch_i + batch_size)), (int)(batch_size))
        data = data[step_size: batch_size-step_size]
        data = np.append(data, next)
        data = np.resize(data,((batch_size), 1))

        x = data

        # convert data into Tensors
        x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
        if(train_on_gpu):
                x_tensor = x_tensor.cuda()

        prediction, (hidden, c0) = noizeNet(x_tensor, (hidden, c0))
        
        music = np.append(music,(prediction.cpu().data.numpy().flatten())[-step_size:])
        next = prediction.cpu().data.numpy().flatten()[-step_size:]
        if(int((batch_i/number_of_steps)) % 100 == 0 and batch_i % 100 == 0):
            print("PROGRESS:\t", round(((batch_i)/number_of_steps)*100, 2), "%"  , sep="")
            print("Prediction dimensions:\t", prediction.cpu().size(), "\t" ,prediction.cpu().data.numpy().flatten().size, "\nMusic dimensions:\t", music.size ,sep="")
       
        hidden = hidden.data
        c0 = c0.data

    sf.write('/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/outputSoundFile.wav', music, 22050,format="WAV")
    time_steps = np.linspace(0, len(music), len(music))
    plt.plot(time_steps, music,"b.",  markersize=0.1)
    plt.show()
    return music
###################################################################################################################

def generateModelName(n_steps = 30, print_every = 5, step_size =  1, duration = 5, numberOfTracks = 1, clip = 5, LSTMBool=False, hidden_dim=50,n_layers=1):
    return "n_steps="+ str(n_steps) + "__" +"print_every="+ str(print_every) + "__" +"step_size="+ str(step_size) + "__" +"duration="+ str(duration) + "__" +  \
    "numberOfTracks="+ str(numberOfTracks) + "__" +  "clip="+ str(clip) + "__" +  "LSTMBool="+ str(LSTMBool) + "hidden_dim="+ str(hidden_dim) + "__" +"n_layers="+ str(n_layers) +"__"+ "lr=" + str(lr) + ".pt"

#####################################################################################################################

# decide on hyperparameters
# n_steps = 1
input_size=1
output_size=1
hidden_dim=100
n_layers=1
LSTMBool = True
# instantiate an RNN
noizeNet = NoizeNet(input_size, output_size, hidden_dim, n_layers, LSTMBool)
print(noizeNet)
######################################################################################################################

# MSE loss and Adam optimizer with a learning rate of 0.01
# criterion = nn.MSELoss()
lr=0.001
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(noizeNet.parameters(), lr=lr)

#Get metadata for fma dataset
AUDIO_DIR = "data/fma_small/"

tracks = utils.load('data/fma_metadata/tracks.csv')

small = tracks['set', 'subset'] <= 'small'
genre1 = tracks['track', 'genre_top'] == 'Instrumental'
genre2 = tracks['track', 'genre_top'] == 'Hip-Hop' #We can set multilpe genres bellow as (genre1 | genre2)
genreTracks = list(tracks.loc[small & (genre1),('track', 'genre_top')].index)


#Set if we want to train new model or load and predict with saved model
TRAIN = False

n_steps = 10 #The number of full frame steps to be taken to complete training
print_every = 5 
step_size =  1000 #The step size taken by each training frame 
duration = 30 #The duration of the training segment
predictDuration = 30 #The duration of the predicted song is seconds
numberOfTracks = 1 #The number of tracks to be trained on
clip = 1 #Gradient clipping
if TRAIN:
    print("TRAINING...")
    trained_rnn = train(noizeNet, n_steps,AUDIO_DIR, genreTracks, LSTMBool ,step_size=step_size, duration = duration, numberOfTracks=numberOfTracks, clip=clip)
    torch.save(trained_rnn.state_dict(), "/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/" + generateModelName(n_steps, print_every, step_size, duration, numberOfTracks, clip, LSTMBool, hidden_dim,n_layers))
    predict(trained_rnn, genreTracks[-1] ,duration=duration, n_steps=n_steps, LSTMBool=LSTMBool, predictDuration = predictDuration)

    

else:
    # instantiate an RNN
    noizeNet = NoizeNet(input_size, output_size, hidden_dim, n_layers, LSTMBool)
    # model = TheModelClass(*args, **kwargs)
    noizeNet.load_state_dict(torch.load("/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/n_steps=10__print_every=5__step_size=1000__duration=30__numberOfTracks=1__clip=1__LSTMBool=Truehidden_dim=100__n_layers=1__lr=0.001.pt"))
    predict(noizeNet, genreTracks[-1] ,duration=duration, n_steps=n_steps, LSTMBool=LSTMBool, predictDuration = predictDuration)

#TODO This is how to plot
# fig, ax = plt.subplots(nrows=3, sharex=True)
# ax[0].set(title='Envelope view, mono')
# ax[0].label_outer()
# lib.display.waveshow(y, sr=sr, ax=ax[0])
# plt.show()



