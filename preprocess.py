from scipy.io import wavfile
from scipy.signal import resample
from lab1_feature_extraction import *
import numpy as np
import os

# TODO modify this in a class
def read(filename):
    if not filename:
        raise ValueError("No audio files found in '{}'.".format(filename))
    else:
        samplerate, data = wavfile.read(filename) # sample rate = 48000, data[~, 2L]
        # downsample
        data = resample(data, 20000)
        # winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22
        channal_1 = mfcc(data[:, 0])
        channal_2 = mfcc(data[:, 1])

        return np.concatenate(channal_1, channal_2)


if __name__ == '__main__':
    read('')