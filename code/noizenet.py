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

def time_to_fft(blocks_time_domain):
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
    #Note this code was taken fomr Udacity course on machine learning. 
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
            # h_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).cuda() #hidden state
            # c_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).cuda() #internal state
            h_0 = hidden
            c_0 = c0
            # Propagate input through LSTM
            state = (h_0, c_0)
            r_out, (hn, cn) = self.lstm(x, state) #lstm with input, hidden, and internal state
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
        # hidden = torch.autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)).cuda() #hidden state
        # c0 = torch.autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)).cuda() #internal state
        return hidden, c0



######################################################################################################################

# train the RNN
def train(noizeNet, n_steps, AUDIO_DIR, genreTracks, LSTM ,step_size=1, duration=5, numberOfTracks=1, clip=5, fft_bool=False):
    fileCount = 0 #Used for displaying the file that is currently being trained on
    noizeNet.train() #Set the model to training mode
    lossArr = [] #Array used to plot loss over time
    val_losses = []
    if(train_on_gpu):
        noizeNet.cuda() #Move the model to GPU if available

    val_file = utils.get_audio_path(AUDIO_DIR, genreTracks[-1])
    val_data, sr = lib.load(val_file, mono=True, duration = duration)
    if fft_bool:
        print("We are training on FFT data")
        val_data = time_to_fft(val_data)
    val_data = scaler.fit_transform(val_data.reshape(-1,1)) #Scale data
    hidden = None
    c0 = None
    hidden, c0 = noizeNet.init_hidden(1)
    #Loop over all the files in our filtered list
    for id in genreTracks: 
        #Stop training after n files
        fileCount+=1
        if(fileCount > numberOfTracks):
            break 

        filename = utils.get_audio_path(AUDIO_DIR, id) #Get the actual path to the file from the id
        
        fileData, sr = lib.load(filename, mono=True, duration = duration)
        fileData = fileData.reshape(-1,1)
        if fft_bool:
            fileData = time_to_fft(fileData)
        fileData = scaler.fit_transform(fileData)

        data = fileData
        batch_size = (int)(duration*sr/n_steps) #Find the size of the window that slides across the input song
        number_of_steps = (int)(duration*sr)-batch_size #Assumes step size of one as a larger step size produced poor results

        if(np.isnan(np.sum(fileData))):
            print("NAN ON FILE:\t", filename)
            break
        
        for batch_i in (range(0,number_of_steps, step_size)):
            for e in range(0,1):
                # for batch_i in (range(0,number_of_steps, step_size)):
                # if LSTM:
                #     (hidden, c0) = noizeNet.init_hidden(batch_size)
                # for x, y in get_batches(fileData, 500, 50):
                
                # defining the training data
                data = fileData[(batch_i): batch_size + batch_i]
                # data = np.resize(data,((batch_size), 1)) 
                # data = data.reshape(batch_size,1)  
                # data = data.reshape(len(data),1)    
                x = data[:-1] #Select all but the last element in the input data
                y = data[1:] #Select all but the first element in the input data. Essentially a forward shift in time

                # convert data into Tensors
                x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
                y_tensor = torch.Tensor(y)
                if(train_on_gpu):
                        x_tensor, y_tensor = x_tensor.cuda(), y_tensor.cuda()
                
                if LSTM:
                    prediction, (hidden, c0) = noizeNet(x_tensor, hidden, c0) #LSTM!
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


                #Validation step
                if(int((batch_i)) % 1000 == 0):
                    if LSTM:
                        (val_h, val_c) = noizeNet.init_hidden(1)
                    else:
                        val_h = None

                    
                    noizeNet.eval()
                    val_x = val_data[:-1]
                    val_y = val_data[1:]

                    

                    for e in range(0,1):
                        val_x, val_y = torch.from_numpy(val_x), torch.from_numpy(val_y)
                        
                        
                        inputs, targets = val_x.reshape(-1,1).unsqueeze(0), val_y.reshape(-1,1)
                        if(train_on_gpu):
                            inputs, targets = inputs.cuda(), targets.cuda()
                        if LSTM:
                            output, (val_h, val_c) = noizeNet(inputs, val_h, val_c)
                        else:
                            output, val_h = noizeNet(inputs, val_h)
                        val_loss  = criterion(output, targets)

                        val_losses.append(val_loss.item())

                        if LSTM:
                            val_h = val_h.data
                            val_c = val_c.data
                        else:
                            val_h = val_h.data


                    print("Training Progress:\t", round((fileCount*e/(number_of_steps*numberOfTracks))*100, 2), "%"  , sep="")
                    print("P:\t", prediction.cpu().data.numpy().flatten()[-5:],"\nY:\t", y[-5:].flatten() ,"\nX:\t" , x[-5:].flatten() ,sep="")
                    print('Training Loss: ', loss.item(), "Validation loss: ", val_loss.item() , "\t num:", batch_i, "\t File:", fileCount)
                    noizeNet.train() # reset to train mode after validation
            
            #Clean unused variables to ensure memory is kept free
            # del prediction
            # del hidden
            # del x_tensor
            # del y_tensor
            # del data
            # del x
            # del y
            # gc.collect()
        fig, ax = plt.subplots(nrows=2)
        ax[0].plot(lossArr)
        ax[0].set(title="Training loss", ylabel="Loss", xlabel="Epochs")
        ax[1].plot(val_losses)
        ax[1].set(title="Validation loss", ylabel="Loss", xlabel="Epochs")
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
    # y = time_to_fft(y)
    y = scaler.fit_transform(y.reshape(-1,1))

    data = y
    

    #data = np.random.normal(-1,1,y.shape)
    # data = genPerlin(np.linspace(0,1,y.size))
    batch_size = (int)(sr*duration/n_steps)



    number_of_steps = (int)(sr*predictDuration/step_size)

    music = []
    next = data[batch_size-1]

    sf.write('/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/predictionSeed.wav', np.append( data[step_size: batch_size], next), sr,format="WAV")
    print("BATCH SIZE:", batch_size ,sep="\t")
    print("NUMBER OF STEPS:", number_of_steps , sep="\t")
    # data = data[0: batch_size-1]
    hidden = None
    c0 = None
    hidden, c0 = noizeNet.init_hidden(1)
    for batch_i in (range(0, number_of_steps)):
        if(train_on_gpu):
            noizeNet.cuda()
        data = data[batch_i: batch_size-1 + batch_i]
        data = np.append(data,next)
        x = data.reshape(-1,1)

        print(data.shape)
        if(np.isnan(np.sum(data))):
            print("data contains NAN", data)

        x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
        if(train_on_gpu):
                x_tensor = x_tensor.cuda()

        if(LSTMBool):
            prediction, (hidden, c0) = noizeNet(x_tensor, hidden, c0)
            c0 = c0.data
        else:
            prediction, hidden = noizeNet(x_tensor, hidden)
        hidden = hidden.data
        
        
        music = np.append(music,(prediction.cpu().data.numpy().flatten())[-1])
        next = prediction.cpu().data.numpy().flatten()[-1]
        
        if(int((batch_i)) % 100 == 0):
            # fig, ax = plt.subplots(nrows=2)
            # ax[0].plot(prediction.cpu().data.numpy())
            # ax[1].plot(scaler.inverse_transform(prediction.cpu().data.numpy()))
            # print(np.average(prediction.cpu().data.numpy()))
            # print(prediction.cpu().data.numpy().flatten()[-1:])
            # print(x)
            # ax[1].plot(fft_to_time(prediction.cpu().detach().numpy()))
            # plt.show()
            
            print("PROGRESS:\t", round(((batch_i)/number_of_steps)*100, 2), "%"  , sep="")
            print("Prediction dimensions:\t", prediction.cpu().size(), "\t" ,prediction.cpu().data.numpy().flatten().size, "\nMusic dimensions:\t", music.size ,sep="")
       
    # print(fft_to_time(music))  
    print(music)
    music = scaler.inverse_transform(music.reshape(-1,1))
    # music = prediction.cpu().detach().numpy()
    # music = abs(fft_to_time(music))#.astype('float32')
    # print(music)
    sf.write('/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/outputSoundFile.wav', (music), 22050,format="WAV")
    time_steps = np.linspace(0, len(music), len(music))
    plt.plot(time_steps, music,"b.",  markersize=0.1)
    plt.show()
    return music

def predict2(noizeNet, genreTrack ,duration=1, n_steps=30, LSTMBool=False, predictDuration = 30, step_size=1):
    print("PREDICTING...")
    noizeNet.eval()

    if LSTMBool:
        hidden, c0 = noizeNet.init_hidden(1)
    else:
        hidden = None
    filePath = utils.get_audio_path(AUDIO_DIR, genreTrack) #Get the actual path to the file from the id
    y, sr = lib.load(filePath, mono=True, duration = duration)
    sf.write('/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/Predict2Seed.wav', y, 22050,format="WAV")
    y = scaler.fit_transform(y.reshape(-1,1))
 
    batch_size = (int)(sr*duration/n_steps)

    number_of_steps = (int)(sr*duration/step_size)
    music = []
    for batch_i in (range(0, number_of_steps)):
        if(train_on_gpu):
            noizeNet.cuda()
        data = y[batch_i]
        data = data.reshape(-1,1)
        x = data


        x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
        if(train_on_gpu):
                x_tensor = x_tensor.cuda()

        if(LSTMBool):
            prediction, (hidden, c0) = noizeNet(x_tensor, hidden, c0)
            c0 = c0.data
        else:
            prediction, hidden = noizeNet(x_tensor, hidden)
        hidden = hidden.data
        
        print(prediction.cpu().data.numpy().shape)
        music = np.append(music,(prediction.cpu().data.numpy().flatten())[-step_size:])
        if(int((batch_i)) % 1000 == 0):
            print("PROGRESS:\t", round(((batch_i)/number_of_steps)*100, 2), "%"  , sep="")
            print("Prediction dimensions:\t", prediction.cpu().size(), "\t" ,prediction.cpu().data.numpy().flatten().size, "\nMusic dimensions:\t", music.size ,sep="")
       
    sf.write('/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/outputSoundFilePredict2.wav', (music), 22050,format="WAV")
    time_steps = np.linspace(0, len(music), len(music))
    plt.plot(time_steps, music,"b.",  markersize=1)
    plt.show()
    if LSTMBool:
        return music, (hidden, c0)
    else: 
        return music, hidden

def predict3(noizeNet, seeded ,duration=1, n_steps=30, LSTMBool=False, predictDuration = 30, step_size=1, hidden = None, c0 = None):
    print("PREDICTING...")
    noizeNet.eval()
    number_of_steps = (int)(22050*predictDuration/step_size)
    music = []

    for batch_i in (range(0, number_of_steps)):
        if(train_on_gpu):
            noizeNet.cuda()
        seeded = np.array([seeded]).reshape(-1,1)
        data = seeded
        if batch_i % 200 == 0:
            data = seeded*np.random.normal(-1,1)
        data = data.reshape(-1,1)
        x = data
        
        x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
        print("x_tensor",x_tensor)
        if(train_on_gpu):
                x_tensor = x_tensor.cuda()

        if (LSTMBool):
            prediction, (hidden, c0) = noizeNet(x_tensor, hidden, c0)
            c0 = c0.data
        else:
            prediction, hidden = noizeNet(x_tensor, hidden)
        hidden = hidden.data
        
        seeded = prediction.cpu().data.numpy()[-1:step_size:]
        music = np.append(music,(prediction.cpu().data.numpy().flatten())[-step_size:])
        if(int((batch_i)) % 10000 == 0):
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
n_layers=2
LSTMBool = True
dropout_prob = 0.5

# instantiate an RNN
noizeNet = NoizeNet(input_size, output_size, hidden_dim, n_layers, LSTMBool, dropout_prob)
print(noizeNet)
######################################################################################################################

# MSE loss and Adam optimizer with a learning rate of 0.01
criterion = nn.MSELoss()
lr=0.0001
# criterion = nn.L1Loss()
# optimizer = torch.optim.Adam(noizeNet.parameters(), lr=lr)
# optimizer = torch.optim.SGD(noizeNet.parameters(),lr=lr)
optimizer = torch.optim.Adadelta(noizeNet.parameters())

#Get metadata for fma dataset
AUDIO_DIR = "data/fma_small/"

tracks = utils.load('data/fma_metadata/tracks.csv')

small = tracks['set', 'subset'] <= 'small'
genre1 = tracks['track', 'genre_top'] == 'Instrumental'
# genre2 = tracks['track', 'genre_top'] == 'Hip-Hop' #We can set multilpe genres bellow as (genre1 | genre2)
genreTracks = list(tracks.loc[small & (genre1),('track', 'genre_top')].index)


#Set if we want to train new model or load and predict with saved model
TRAIN = True

n_steps = 30 #The number of full frame steps to be taken to complete training
print_every = 5 
step_size =  1 #The step size taken by each training frame 
duration = 1 #The duration of the training segment
predictDuration = 1 #The duration of the predicted song is seconds
numberOfTracks = 1 #The number of tracks to be trained on
clip = 1 #Gradient clipping
seedDuration = 1


if TRAIN:
    print("TRAINING...")
    trained_rnn = train(noizeNet, n_steps,AUDIO_DIR, genreTracks, LSTMBool ,step_size=step_size, duration = duration, numberOfTracks=numberOfTracks, clip=clip)
    torch.save(trained_rnn.state_dict(), "/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/" + generateModelName(n_steps, print_every, step_size, duration, numberOfTracks, clip, LSTMBool, hidden_dim,n_layers))
    duration = seedDuration
    # predict(trained_rnn, genreTracks[-1] ,duration=duration, n_steps=n_steps, LSTMBool=LSTMBool, predictDuration = predictDuration, step_size=step_size)
    if LSTMBool:
        predicted, (hidden, c0) = predict2(trained_rnn, genreTracks[-2] ,duration=duration, n_steps=n_steps, LSTMBool=LSTMBool, predictDuration = predictDuration)
        predict3(noizeNet, predicted[-1] ,duration=duration, n_steps=n_steps, LSTMBool=LSTMBool, predictDuration = predictDuration,  hidden=hidden, c0 =c0)
    else: 
        predicted, hidden = predict2(trained_rnn, genreTracks[-2] ,duration=duration, n_steps=n_steps, LSTMBool=LSTMBool, predictDuration = predictDuration)
        predict3(noizeNet, predicted[-1] ,duration=duration, n_steps=n_steps, LSTMBool=LSTMBool, predictDuration = predictDuration,  hidden=hidden)
    

    

else:
    # instantiate an RNN
    noizeNet = NoizeNet(input_size, output_size, hidden_dim, n_layers, LSTMBool, dropout_prob)
    noizeNet.load_state_dict(torch.load("/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/good model/n_steps=1__print_every=5__step_size=1__duration=10__numberOfTracks=100__clip=5__LSTMBool=Truehidden_dim=200__n_layers=1__lr=0.0001.pt"))
    duration = seedDuration
    if LSTMBool:
        predicted, (hidden, c0) = predict2(noizeNet, genreTracks[-2] ,duration=duration, n_steps=n_steps, LSTMBool=LSTMBool, predictDuration = predictDuration)
        predict3(noizeNet, predicted[-1] ,duration=duration, n_steps=n_steps, LSTMBool=LSTMBool, predictDuration = predictDuration,  hidden=hidden, c0 =c0)
    else: 
        predicted, hidden = predict2(noizeNet, genreTracks[-2] ,duration=duration, n_steps=n_steps, LSTMBool=LSTMBool, predictDuration = predictDuration)
        predict3(noizeNet, predicted[-1] ,duration=duration, n_steps=n_steps, LSTMBool=LSTMBool, predictDuration = predictDuration,  hidden=hidden)






