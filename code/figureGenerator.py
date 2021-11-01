import numpy as np
import librosa as lib
import matplotlib.pyplot as plt
import librosa.display
import utils

AUDIO_DIR = "/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/data/fma_small/"

tracks = utils.load('/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/data/fma_metadata/tracks.csv')

small = tracks['set', 'subset'] <= 'small'
genre1 = tracks['track', 'genre_top'] == 'Instrumental'
# genre2 = tracks['track', 'genre_top'] == 'Hip-Hop' #We can set multilpe genres bellow as (genre1 | genre2)
genreTracks = list(tracks.loc[small & (genre1),('track', 'genre_top')].index)

def time_to_fft(blocks_time_domain):
    # FFT blocks initialized
    fft_blocks = []
    # for block in blocks_time_domain:
    # Computes the one-dimensional discrete Fourier Transform and returns the complex nD array
    # i.e The truncated or zero-padded input, transformed from time domain to frequency domain.
    fft_block = np.fft.fft(blocks_time_domain)
    # Joins a sequence of blocks along frequency axis.
    new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
    fft_blocks = (new_block)
    return fft_blocks

def fft_to_blocks(blocks_ft_domain):

    # Time blocks initialized
    time_blocks = []
    # for block in blocks_ft_domain:
    num_elems = (int)(blocks_ft_domain.shape[0] / 2)
    # Extracts real part of the amplitude corresponding to the frequency
    real_chunk = blocks_ft_domain[0:num_elems]
    # Extracts imaginary part of the amplitude corresponding to the frequency
    imag_chunk = blocks_ft_domain[num_elems:]
    # Represents amplitude as a complex number corresponding to the frequency
    new_block = real_chunk + 1.0j * imag_chunk
    # Computes the one-dimensional discrete inverse Fourier Transform and returns the transformed
    # block from frequency domain to time domain
    time_block = np.fft.ifft(new_block)
    # Joins a sequence of blocks along time axis.
    time_blocks = (time_block)
    return time_blocks


def genClassicalVsInstrumental():
    AUDIO_DIR = "/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/data/fma_small/"
    filename = utils.get_audio_path(AUDIO_DIR, genreTracks[0])
    y, sr = lib.load("/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/Maarten Schellekens - Shattered Innocence.mp3", mono=True)
    fig, ax = plt.subplots(nrows=2, sharex=False)
    ax[0].set(xlim=[0, 5], title='Envelope view Classical song Maarten Schellekens - Shattered Innocence')
    ax[1].set(xlim=[0, 5], title='Envelope view Instrumental song')

    ax[1].set_ylabel("Amplitude")
    ax[0].set_ylabel("Amplitude")
    lib.display.waveshow(y, sr=sr, ax=ax[0], marker='.')
    y, sr = lib.load(filename, mono=True)
    lib.display.waveshow(y, sr=sr, ax=ax[1], marker='.')


def genClassicalVsInstrumentalFFT():
    AUDIO_DIR = "/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/data/fma_small/"
    filename = utils.get_audio_path(AUDIO_DIR, genreTracks[0])
    y, sr = lib.load("/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/Maarten Schellekens - Shattered Innocence.mp3", mono=True)
    y = time_to_fft(y)
    fig, ax = plt.subplots(nrows=2, sharex=False)
    ax[0].set(xlim=[0, 5], title='Envelope view Classical song Maarten Schellekens - Shattered Innocence under FFT')
    ax[1].set(xlim=[0, 5], title='Envelope view Instrumental song under FFT')


    lib.display.waveshow(y, sr=sr, ax=ax[0], marker='.')
    y, sr = lib.load(filename, mono=True)
    y = time_to_fft(y)
    lib.display.waveshow(y, sr=sr, ax=ax[1], marker='.')
    ax[1].set_ylabel("Density")
    ax[0].set_ylabel("Density")
    ax[1].set_xlabel("Frequency")
    ax[0].set_xlabel("Frequency")

def genFigsFromArbitraryFile(filename):
    y, sr = lib.load(filename, mono=True)
    fig, ax = plt.subplots(nrows=3, sharex=False)
    
    ax[0].set(title='Envelope view')
    # ax[0].label_outer()
    ax[0].set_ylabel("Amplitude")
    lib.display.waveshow(y, sr=sr, ax=ax[0])

    ax[1].set(title='Wave plot')
    # ax[1].label_outer()
    ax[1].set(xlim=[0.0, 0.01], ylim=[-1, 1])
    ax[1].set_ylabel("Amplitude")
    librosa.display.waveshow(y, sr=sr, ax=ax[1], marker='.')


    ax[2].set(title='Linear Spec gram')
    ax[2].label_outer()
    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D1, y_axis='linear', x_axis='time',
                                   sr=sr, ax=ax[2])

    # hop_length = 1024
    # D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)),
    #                             ref=np.max)
    # librosa.display.specshow(D2, y_axis='log', sr=sr, hop_length=hop_length,
    #                          x_axis='time', ax=ax[1])   

def getFileName(track):
    AUDIO_DIR = "/home/liam/Desktop/University/2021/MAM3040W/thesis/writeup/code/data/fma_small/"
    filename = utils.get_audio_path(AUDIO_DIR, track)
    return filename

# genFigsFromArbitraryFile(getFileName(genreTracks[-1]))
genFigsFromArbitraryFile("outputSoundFilePrediction3.wav")
# genClassicalVsInstrumental()
# genClassicalVsInstrumentalFFT()

                   
plt.show()