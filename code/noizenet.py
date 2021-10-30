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

#Print if we are able to use a GPU
if(train_on_gpu):
    print('Processing on GPU.')
else:
    print('No GPU available.')

scaler = sklearn.preprocessing.StandardScaler()

def time_blocks_to_fft_blocks(blocks_time_domain):
    # FFT blocks initialized
    fft_blocks = []
    # for block in blocks_time_domain:
    # Computes the one-dimensional discrete Fourier Transform and returns the complex nD array
    # i.e The truncated or zero-padded input, transformed from time domain to frequency domain.
    fft_block = np.fft.fft(blocks_time_domain)
    # Joins a sequence of blocks along frequency axis.
    new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
    fft_blocks = (new_block) #Scale data by mp3 maximum requency
    print(fft_blocks.size)
    return fft_blocks

def fft_to_time(blocks_ft_domain):
    # Time blocks initialized
    time_blocks = []
    # for block in blocks_ft_domain:
    if blocks_ft_domain.shape[0] % 2 == 0:
        num_elems = (int)(blocks_ft_domain.shape[0] / 2)
    else:
        blocks_ft_domain = blocks_ft_domain[0:-1]
        num_elems = (int)(blocks_ft_domain.shape[0] / 2)
    # Extracts real part of the amplitude corresponding to the frequency
    real_chunk = blocks_ft_domain[0:num_elems]
    # Extracts imaginary part of the amplitude corresponding to the frequency
    imag_chunk = blocks_ft_domain[num_elems:]
    # Represents amplitude as a complex number corresponding to the frequency
    new_block = real_chunk + 1.0j * imag_chunk
    # Computes the one-dimensional discrete inverse Fourier Transform and returns the transformed
    # block from frequency domain to time domain
    time_block = np.fft.ifft(new_block*20000) #Re-scale data
    # Joins a sequence of blocks along time axis.
    time_blocks = (time_block)
    return time_blocks

def get_batches(arr, batch_size, seq_length):
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr)//batch_size_total
    
    # Keep only enough elements to fill a batch
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

#######################################################################################################################

class NoizeNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, LSTMBool, dropout_prob):
        super(NoizeNet, self).__init__()

        self.LSTMBool = LSTMBool  
        self.hidden_dim = hidden_dim
        self.num_layers = n_layers
        self.output_size = output_size
        self.input_size = input_size
        self.dropout_prob = dropout_prob


        #TODO: Test RNN vs LSTM
        if LSTMBool:
            self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=dropout_prob,batch_first=True)
            self.hidden = (torch.zeros(1,1,self.hidden_dim).cuda(), torch.zeros(1,1,self.hidden_dim).cuda()) #We need a tuple for a LSTM
        else:
            # define an RNN with specified parameters
            # batch_first means that the first dim of the input and output will be the batch_size
            self.rnn = nn.RNN(input_size, hidden_dim, n_layers, dropout=dropout_prob, batch_first=True)

        self.dropout = nn.Dropout(dropout_prob)

        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden, c0=None):
        batch_size = x.size(0)
        if (train_on_gpu):
            x.cuda()
        else:
            x.cpu()
        

        if self.LSTMBool:
            h_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).cuda() #hidden state
            c_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).cuda() #internal state
            # Propagate input through LSTM
            r_out, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
            hidden = hn
        else:
            # get RNN outputs
            r_out, hidden = self.rnn(x , hidden)

        r_out = self.dropout(r_out) #Dropout

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
            (hidden, c0) = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            (hidden, c0) = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        
        return (hidden, c0)



######################################################################################################################

# train the RNN
def train(noizeNet, n_steps, AUDIO_DIR, genreTracks, LSTM ,step_size=1, duration=5, numberOfTracks=1, clip=5):
    fileCount = 0 #Used for displaying the file that is currently being trained on
    noizeNet.train() #Set the model to training mode
    lossArr = [] #Array used to plot loss over time

    if(train_on_gpu):
        noizeNet.cuda() #Move the model to GPU if available

    #Loop over all the files in our filtered list
    for id in genreTracks: 
        #Stop training after n files
        fileCount+=1
        if(fileCount > numberOfTracks):
            break 

        filename = utils.get_audio_path(AUDIO_DIR, id) #Get the actual path to the file from the id
        

        # hidden = None
        # if LSTM:
        #     c0 = None #LSTM!
        
        fileData, sr = lib.load(filename, mono=True, duration = duration)

        # fileData = time_blocks_to_fft_blocks(fileData)

        fileData = scaler.fit_transform(fileData.reshape(-1, 1))
        # x,y = sliding_windows(fileData, 10)
        data= fileData
        batch_size = (int)(duration*sr/n_steps)
        # number_of_steps = len(fileData)-batch_size
        number_of_steps = 1
        # print("BATCH SIZE:",  batch_size, "\nNUMBER OF BATCHES:", n_steps ,"\nNUMBER OF STEPS: ", number_of_steps)

        if(np.isnan(np.sum(fileData))):
            print("NAN ON FILE:\t", filename)
            break
        #FOR TESTING
        batch_i = 0
        for e in range(0,100):
        # for batch_i in (range(0,number_of_steps, step_size)):
            (hidden, c0) = noizeNet.init_hidden(batch_size)
        # for x, y in get_batches(fileData, 500, 50):
            
            # defining the training data
            data = fileData[(batch_i): batch_size + batch_i]
            # data = np.resize(data,((batch_size), 1)) 
            data = data.reshape(batch_size,1)  
            # data = data.reshape(len(data),1)    
            x = data[:-1]
            y = data[1:]

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
            lossArr.append(loss.item())
            # perform backprop and update weights
            loss.backward()
            
            torch.nn.utils.clip_grad_value_(noizeNet.parameters(), clip) #Clip gradient
            optimizer.step()



            if(int((e*batch_i)) % 1000 == 0):
                print("PROGRESS:\t", round((fileCount*batch_i/(number_of_steps*numberOfTracks))*100, 2), "%"  , sep="")
                print("P:\t", prediction.cpu().data.numpy().flatten()[-5:],"\nY:\t", y[-5:].flatten() ,"\nX:\t" , x[-5:].flatten() ,sep="")
                print('Loss: ', loss.item(), "\t num:", batch_i, "\t File:", fileCount)
                # val_h = NoizeNet.init_hidden(batch_size)
                # val_losses = []
                # NoizeNet.eval()
                # for e in range(0,2):
                #     # for batch_i in (range(0,number_of_steps, step_size)):

                #     # One-hot encode our data and make them Torch tensors
                #     x = one_hot_encode(x, n_chars)
                #     x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                #     # Creating new variables for the hidden state, otherwise
                #     # we'd backprop through the entire training history
                #     val_h = tuple([each.data for each in val_h])
                    
                #     inputs, targets = x, y
                #     if(train_on_gpu):
                #         inputs, targets = inputs.cuda(), targets.cuda()

                #     output, val_h = net(inputs, val_h)
                #     val_loss = criterion(output, targets.view(batch_size*seq_length).long())
                
                #     val_losses.append(val_loss.item())
                
                # NoizeNet.train() # reset to train mode after iterationg through validation data
        # del prediction
        # del hidden
        # del x_tensor
        # del y_tensor
        # del data
        # del x
        # del y
        # gc.collect()
        # time_steps = np.linspace((int)(batch_i), (int)((batch_i + batch_size)), (int)(batch_size))
        # plt.plot(time_steps[1:], prediction.cpu().detach().numpy(), markersize=0.1)
        # plt.show()
    plt.plot(lossArr)
    plt.show()
    return noizeNet

###################################################################################################################
def genPerlin(x):
    tmp = []
    for xx in x:
        tmp.append(noise.pnoise1(xx))

    return tmp


def predict(noizeNet, genreTrack ,duration=1, n_steps=30, LSTMBool=False, predictDuration = 30, step_size=1):
    print("PREDICTING...")
    noizeNet.eval()
    hidden = None
    
    if LSTMBool:
        c0 = None
        (hidden, c0) = noizeNet.init_hidden(1)
    filePath = utils.get_audio_path(AUDIO_DIR, genreTrack) #Get the actual path to the file from the id
    y, sr = lib.load(filePath, mono=True, duration = duration)
    y = scaler.fit_transform(y.reshape(-1,1))
    data = y
    # data = time_blocks_to_fft_blocks(y)

    #data = np.random.normal(-1,1,y.shape)
    # data = genPerlin(np.linspace(0,1,y.size))
    batch_size = (int)(sr*duration/n_steps)



    number_of_steps = (int)(sr*predictDuration/step_size)

    music = []
    next = data[-1:]

    sf.write('/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/predictionSeed.wav', np.append( data[step_size: batch_size], next), sr,format="WAV")
    print("BATCH SIZE:", batch_size ,sep="\t")
    print("NUMBER OF STEPS:", number_of_steps , sep="\t")
    data = data[0: batch_size]
    for batch_i in (range(0, number_of_steps)):
        (hidden, c0) = noizeNet.init_hidden(1)
        if(train_on_gpu):
            noizeNet.cuda()
        data = next
        # data = data[batch_i] #Shrink data
        # data = np.append(data, next)
        # data = np.resize(data,(batch_size, 1))
        data = data.reshape(len(data),1)
        # data = data.reshape(batch_size,1)
        x = data
        if(np.isnan(np.sum(data))):
            print("data contains NAN", data)

        x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
        if(train_on_gpu):
                x_tensor = x_tensor.cuda()

        if(LSTMBool):
            prediction, (hidden, c0) = noizeNet(x_tensor, (hidden, c0))
            c0 = c0.data
        else:
            prediction, hidden = noizeNet(x_tensor, hidden)
        hidden = hidden.data
        
        
        music = np.append(music,(prediction.cpu().data.numpy().flatten())[-step_size:])
        next = prediction.cpu().data.numpy().flatten()[-step_size:]
        
        # print(music.size, next.size)
        # print(next, data.size, data[-5:])
        # print(prediction)
        if(int((batch_i)) % 100 == 0):
            # time_steps = np.linspace((int)(batch_i), (int)((batch_i + batch_size)), (int)(batch_size))
            # plt.plot(time_steps, prediction.cpu().detach().numpy(), markersize=0.1)
            fig, ax = plt.subplots(nrows=2)
            ax[0].plot(prediction.cpu().data.numpy())
            ax[1].plot(scaler.inverse_transform(prediction.cpu().data.numpy()))
            print(np.average(prediction.cpu().data.numpy()))
            print(prediction.cpu().data.numpy().flatten()[-1:])
            print(x)
            # ax[1].plot(fft_to_time(prediction.cpu().detach().numpy()))
            # plt.show()
            
            print("PROGRESS:\t", round(((batch_i)/number_of_steps)*100, 2), "%"  , sep="")
            print("Prediction dimensions:\t", prediction.cpu().size(), "\t" ,prediction.cpu().data.numpy().flatten().size, "\nMusic dimensions:\t", music.size ,sep="")
       
    # print(fft_to_time(music))  
    print(music)
    music = scaler.inverse_transform(music)
    music = prediction.cpu().detach().numpy()
    music = abs(fft_to_time(music))#.astype('float32')
    print(music)
    sf.write('/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/outputSoundFile.wav', (music), 22050,format="WAV")
    time_steps = np.linspace(0, len(music), len(music))
    plt.plot(time_steps, music,"b.",  markersize=0.1)
    plt.show()
    return music

def predict2(noizeNet, genreTrack ,duration=1, n_steps=30, LSTMBool=False, predictDuration = 30, step_size=1):
    print("PREDICTING...")
    noizeNet.eval()

    if LSTMBool:
        (hidden, c0) = noizeNet.init_hidden(1)
    filePath = utils.get_audio_path(AUDIO_DIR, genreTrack) #Get the actual path to the file from the id
    y, sr = lib.load(filePath, mono=True, duration = duration)
    sf.write('/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/Predict2Seed.wav', (y), 22050,format="WAV")
    y = scaler.fit_transform(y.reshape(-1,1))
 
    batch_size = (int)(sr*duration/n_steps)

    number_of_steps = (int)(sr*duration/step_size)
    print()
    music = []
    for batch_i in (range(0, number_of_steps)):
        if(train_on_gpu):
            noizeNet.cuda()
        data = y[batch_i]
        print("y",y.shape)
        print("data",data.shape)
        data = data.reshape(len(data),1)
        print("data val",data)
        x = data
        # print(x)

        x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
        if(train_on_gpu):
                x_tensor = x_tensor.cuda()

        if(LSTMBool):
            prediction, (hidden, c0) = noizeNet(x_tensor, (hidden, c0))
            c0 = c0.data
        else:
            prediction, hidden = noizeNet(x_tensor, hidden)
        hidden = hidden.data
        
        
        music = np.append(music,(prediction.cpu().data.numpy().flatten())[-step_size:])
        print(music.shape)
        # print(next, data.size, data[-5:])
        # print(prediction)
        if(int((batch_i)) % 100 == 0):
            # print((prediction.cpu().data.numpy()))
            # print(x)
            # print(prediction.cpu().data.numpy().flatten()[-1:])
            print("PROGRESS:\t", round(((batch_i)/number_of_steps)*100, 2), "%"  , sep="")
            print("Prediction dimensions:\t", prediction.cpu().size(), "\t" ,prediction.cpu().data.numpy().flatten().size, "\nMusic dimensions:\t", music.size ,sep="")
       
    # print(fft_to_time(music))  
    # print(music)
    # music = scaler.inverse_transform(music)
    # music = music.cpu().data.numpy().flatten()
    # music = abs(fft_to_time(music))#.astype('float32')
    print(music)
    sf.write('/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/outputSoundFilePredict2.wav', (music), 22050,format="WAV")
    time_steps = np.linspace(0, len(music), len(music))
    plt.plot(time_steps, music,"b.",  markersize=1)
    plt.show()
    return music

def predict3(noizeNet, seeded ,duration=1, n_steps=30, LSTMBool=False, predictDuration = 30, step_size=1):
    print("PREDICTING...")
    noizeNet.eval()

    if LSTMBool:
        (hidden, c0) = noizeNet.init_hidden(1)

    seeded = scaler.fit_transform(np.array([seeded]).reshape(-1,1))
 
    
    number_of_steps = (int)(22050*predictDuration/step_size)
    music = []

    for batch_i in (range(0, number_of_steps)):
        if(train_on_gpu):
            noizeNet.cuda()
        data = seeded
        data = data.reshape(1,1)
        x = data

        x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
        if(train_on_gpu):
                x_tensor = x_tensor.cuda()

        if(LSTMBool):
            prediction, (hidden, c0) = noizeNet(x_tensor, (hidden, c0))
            c0 = c0.data
        else:
            prediction, hidden = noizeNet(x_tensor, hidden)
        hidden = hidden.data
        
        seeded = prediction.cpu().data.numpy()
        print(seeded)
        music = np.append(music,(prediction.cpu().data.numpy().flatten())[-step_size:])
        # print(music.shape)
        # print(next, data.size, data[-5:])
        # print(prediction)
        if(int((batch_i)) % 10000 == 0):
            # print((prediction.cpu().data.numpy()))
            # print(x)
            # print(prediction.cpu().data.numpy().flatten()[-1:])
            print("PROGRESS:\t", round(((batch_i)/number_of_steps)*100, 2), "%"  , sep="")
            print("Prediction dimensions:\t", prediction.cpu().size(), "\t" ,prediction.cpu().data.numpy().flatten().size, "\nMusic dimensions:\t", music.size ,sep="")
       
    # print(fft_to_time(music))  
    # print(music)
    # music = scaler.inverse_transform(music)
    # music = music.cpu().data.numpy().flatten()
    # music = abs(fft_to_time(music))#.astype('float32')
    print(music)
    sf.write('/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/outputSoundFilePrediction3.wav', (music), 22050,format="WAV")
    time_steps = np.linspace(0, len(music), len(music))
    plt.plot(time_steps, music,"b.",  markersize=1)
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
dropout_prob = 0.5

# instantiate an RNN
noizeNet = NoizeNet(input_size, output_size, hidden_dim, n_layers, LSTMBool, dropout_prob)
print(noizeNet)
######################################################################################################################

# MSE loss and Adam optimizer with a learning rate of 0.01
# criterion = nn.MSELoss()
lr=0.0001
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(noizeNet.parameters(), lr=lr)

#Get metadata for fma dataset
AUDIO_DIR = "data/fma_small/"

tracks = utils.load('data/fma_metadata/tracks.csv')

small = tracks['set', 'subset'] <= 'small'
genre1 = tracks['track', 'genre_top'] == 'Instrumental'
# genre2 = tracks['track', 'genre_top'] == 'Hip-Hop' #We can set multilpe genres bellow as (genre1 | genre2)
genreTracks = list(tracks.loc[small & (genre1),('track', 'genre_top')].index)


#Set if we want to train new model or load and predict with saved model
TRAIN = True

n_steps = 1 #The number of full frame steps to be taken to complete training
print_every = 5 
step_size =  1 #The step size taken by each training frame 
duration = 1 #The duration of the training segment
predictDuration = 1#The duration of the predicted song is seconds
numberOfTracks = 1 #The number of tracks to be trained on
clip = 5 #Gradient clipping
seedDuration = 1


if TRAIN:
    print("TRAINING...")
    trained_rnn = train(noizeNet, n_steps,AUDIO_DIR, genreTracks, LSTMBool ,step_size=step_size, duration = duration, numberOfTracks=numberOfTracks, clip=clip)
    # torch.save(trained_rnn.state_dict(), "/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/" + generateModelName(n_steps, print_every, step_size, duration, numberOfTracks, clip, LSTMBool, hidden_dim,n_layers))
    duration = seedDuration
    predicted = predict2(trained_rnn, genreTracks[-2] ,duration=duration, n_steps=n_steps, LSTMBool=LSTMBool, predictDuration = predictDuration)
    predict3(noizeNet, predicted[-2] ,duration=duration, n_steps=n_steps, LSTMBool=LSTMBool, predictDuration = predictDuration)

    

else:
    # instantiate an RNN
    noizeNet = NoizeNet(input_size, output_size, hidden_dim, n_layers, LSTMBool, dropout_prob)
    noizeNet.load_state_dict(torch.load("/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/good model/n_steps=1__print_every=5__step_size=1__duration=10__numberOfTracks=100__clip=5__LSTMBool=Truehidden_dim=200__n_layers=1__lr=0.0001.pt"))
    duration = seedDuration
    predicted = predict2(noizeNet, genreTracks[-2] ,duration=duration, n_steps=n_steps, LSTMBool=LSTMBool, predictDuration = predictDuration)
    predict3(noizeNet, genreTracks[-2] ,duration=duration, n_steps=n_steps, LSTMBool=LSTMBool, predictDuration = predictDuration)






