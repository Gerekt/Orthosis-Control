import tensorflow as tf
import scipy as sc
import os
import numpy as np
import time

def get_label(file_path):
  part = tf.strings.split(input=file_path, sep=os.path.sep)
  return part[-2]

def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1) #tfl.squeeze

def get_scipy_spectrogram(waveform):
    print()
    print(waveform.shape)
    startTime = time.time()
    #waveform = tf.cast(waveform, dtype=tf.float32)
    _, _, spectrogram = sc.signal.spectrogram(waveform, 16000, nperseg=256, noverlap=128, nfft=256, mode="magnitude", window=sc.signal.windows.hann(256))
    spectrogram = tf.cast(spectrogram, dtype=tf.float32)
    spectrogram = spectrogram[..., tf.newaxis]
    print("End time of spectrogram: ", (time.time()-startTime)*1000)
    return spectrogram