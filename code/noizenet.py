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

# import wandb
# wandb.init(project='noizenet', entity='liamwatson')
# %matplotlib inline
#############################################################################

# # 2. Save model inputs and hyperparameters
# config = wandb.config
# config.learning_rate = 0.01

# # 3. Log gradients and model parameters
# wandb.watch(model)
# for batch_idx, (data, target) in enumerate(train_loader):
#   ...
#   if batch_idx % args.log_interval == 0:
#     # 4. Log metrics to visualize performance
#     wandb.log({"loss": loss})

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

files = os.listdir("/home/liam/Desktop/University/2021/MAM3040W/thesis/works/playground/wavs/fma_small/000/")
# print(files)
duration = 30
y, sr = lib.load("/home/liam/Desktop/University/2021/MAM3040W/thesis/works/playground/wavs/fma_small/000/" + files[0], mono=True, duration = duration)
# for x in range(y.size):
#     if y[x] == 0:
#         y[x] = 0
#     elif abs(y[x]*10) < 1:
#         y[x] = y[x]*10
#     else:
#         y[x] = 1
# print(y, sr)
# print(np.mean(y))
sf.write('/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/uhh.wav', y, sr,format="wav")


seq_length = sr*duration

time_steps = np.linspace(0, seq_length , seq_length + 1)
data = y
data = np.resize(data,((seq_length+1), 1))
x = data[:-1]
y = data[1:]
plt.plot(time_steps[1:], x, 'r.', label='input, x', markersize=0.1) # x
plt.plot(time_steps[1:], y, 'b.', label='target, y', markersize=0.1) # y

plt.legend(loc='best')
plt.show(block = False)
plt.pause(.001)

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

        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)
        if (train_on_gpu):
            x.cuda()
        else:
            x.cpu()
        # get RNN outputs
        r_out, hidden = self.rnn(x , hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        if (train_on_gpu):
            hidden.cuda()
        else:
            hidden
        r_out = r_out.view(-1, self.hidden_dim)

        # get final output
        output = self.fc(r_out)

        return output, hidden


#####################################################################################################################

# decide on hyperparameters
n_steps = 50
input_size=1
output_size=1
hidden_dim=50
n_layers=1

# instantiate an RNN
noizeNet = NoizeNet(input_size, output_size, hidden_dim, n_layers)
print(noizeNet)
######################################################################################################################

# MSE loss and Adam optimizer with a learning rate of 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(noizeNet.parameters(), lr=0.01)

######################################################################################################################

# train the RNN
def train(noizeNet, n_steps, print_every, data, sr):
    # initialize the hidden state
    hidden = None
    music = []
    for batch_i in (range(0, n_steps)):
        # defining the training data

        if(train_on_gpu):
            noizeNet.cuda()

        time_steps = np.linspace((int)((sr*batch_i)/n_steps), (int)((sr*(batch_i+1))/n_steps), (int)(sr/n_steps))
        # data = np.resize((seq_length+1), 1)
        # data[0, sr]
        # data.resize((seq_length + 1, 1))  # input_size=1

        x = data[:-1]
        y = data[1:]

        # convert data into Tensors
        x_tensor = torch.Tensor(x).unsqueeze(0)  # unsqueeze gives a 1, batch_size dimension
        y_tensor = torch.Tensor(y)
        if(train_on_gpu):
                x_tensor, y_tensor = x_tensor.cuda(), y_tensor.cuda()
        # print(x_tensor.size())
        # outputs from the rnn
        prediction, hidden = noizeNet(x_tensor, hidden)

        ## Representing Memory ##
        # make a new variable for hidden and detach the hidden state from its history
        # this way, we don't backpropagate through the entire history
        hidden = hidden.data

        # calculate the loss
        loss = criterion(prediction, y_tensor)
        # zero gradients
        optimizer.zero_grad()
        # perform backprop and update weights
        loss.backward()
        optimizer.step()

        # display loss and predictions
        #if batch_i % print_every == 0:
        print('Loss: ', loss.item())
        print(len(x))
        # plt.plot(time_steps[1:], x[(int)((sr*step)/n_steps), (int)(sr*(step+1)/n_steps)], 'r.')  # input
        # prediction
        music = (prediction.cpu().data.numpy().flatten())
        large_time = np.linspace(0, seq_length , seq_length + 1)
        # plt.plot(time_steps[1:], prediction.cpu().data.numpy().flatten()[(int)((sr*batch_i)/n_steps)+1: (int)(sr*(batch_i+1)/n_steps)], 'g.', markersize=0.2)  # predictions
        plt.plot(large_time[1:], prediction.cpu().data.numpy().flatten(), 'g.', markersize=0.2)
        plt.show(block = False)
        
        plt.pause(.001)
    # sf.write("./uhh.wav", np.array(music) ,sr)
    # print(music.type(), np.array(music).type())
    sf.write('/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/outputSoundFile.wav', music, 22050,format="WAV")
    return noizeNet

###################################################################################################################

n_steps = 75
print_every = 5
test_input = torch.Tensor(data).unsqueeze(0)
print(test_input.size())
trained_rnn = train(noizeNet, n_steps, print_every, data = data, sr = sr)

