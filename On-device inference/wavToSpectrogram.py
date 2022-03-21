import tensorflow as tf
import scipy as sc
import os
import numpy as np
import time
import python_speech_features

commands = ['up', 'stop', 'no', 'right', 'left', 'down', 'go', 'off', 'on', 'yes', '_unknown_'] #Has to be done this way in order to keep the order of commands

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

def get_tflite_spectrogram(signal, frame_length, frame_step, fft_length):
  """TF-Lite-compatible version of tf.abs(tf.signal.stft())."""
  def _hann_window():
    return tf.reshape(
      tf.constant(
          (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(0, 1.0, 1.0 / frame_length))
          ).astype(np.float32),
          name='hann_window'), [1, frame_length])

  def _dft_matrix(dft_length):
    """Calculate the full DFT matrix in NumPy."""
    # See https://en.wikipedia.org/wiki/DFT_matrix
    omega = (0 + 1j) * 2.0 * np.pi / float(dft_length)
    # Don't include 1/sqrt(N) scaling, tf.signal.rfft doesn't apply it.
    return np.exp(omega * np.outer(np.arange(dft_length), np.arange(dft_length)))

  def _rdft(framed_signal, fft_length):
    """Implement real-input Discrete Fourier Transform by matmul."""
    # We are right-multiplying by the DFT matrix, and we are keeping only the
    # first half ("positive frequencies").  So discard the second half of rows,
    # but transpose the array for right-multiplication.  The DFT matrix is
    # symmetric, so we could have done it more directly, but this reflects our
    # intention better.
    complex_dft_matrix_kept_values = _dft_matrix(fft_length)[:(
        fft_length // 2 + 1), :].transpose()
    real_dft_matrix = tf.constant(
        np.real(complex_dft_matrix_kept_values).astype(np.float32),
        name='real_dft_matrix')
    imag_dft_matrix = tf.constant(
        np.imag(complex_dft_matrix_kept_values).astype(np.float32),
        name='imaginary_dft_matrix')
    signal_frame_length = tf.shape(framed_signal)[-1]
    half_pad = (fft_length - signal_frame_length) // 2
    padded_frames = tf.pad(
        framed_signal,
        [
            # Don't add any padding in the frame dimension.
            [0, 0],
            # Pad before and after the signal within each frame.
            [half_pad, fft_length - signal_frame_length - half_pad]
        ],
        mode='CONSTANT',
        constant_values=0.0)
    real_stft = tf.matmul(padded_frames, real_dft_matrix)
    imag_stft = tf.matmul(padded_frames, imag_dft_matrix)
    return real_stft, imag_stft

  def _complex_abs(real, imag):
    return tf.sqrt(tf.add(real * real, imag * imag))

  startTime = time.time()
  framed_signal = tf.signal.frame(signal, frame_length, frame_step)
  print("hann window type is:", type(_hann_window()))
  print("Framed signal type is:", type(framed_signal))
  windowed_signal = framed_signal * _hann_window()
  real_stft, imag_stft = _rdft(windowed_signal, fft_length)
  stft_magnitude = _complex_abs(real_stft, imag_stft)
  print("Time needed to calculate tflite spectrogram: ", (time.time() - startTime)*1000)
  return stft_magnitude

def get_mfcc(waveform):
  print()
  print("MFCC Calc")
  startTime = time.time()
  mfcc = python_speech_features.base.mfcc(waveform, samplerate=48000, winlen=0.025, winstep=0.01, numcep=16, nfilt=26, nfft=2048, preemph = 0.0, ceplifter=0, appendEnergy = False, winfunc=np.hanning)
  print("End time of MFCC calc: ", (time.time()-startTime)*1000)
  print("Shape of mfcc: ", mfcc.shape)
  print("Length of mfcc: ", len(mfcc))
  return mfcc

def get_spectrogram_inference(waveform):
      # Zero-padding for an audio waveform with less than 16,000 samples.
  print()
  print(waveform.shape)
  startTime = time.time()
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  #print("Spectogram is:", spectrogram)
  # Obtain the magnitude of the STFT.
  print("End time of STFT, start of ABS: ", (time.time()-startTime)*1000)
  startTime = time.time()
  spectrogram = tf.abs(spectrogram)
  #print("Spectogram absolute/magnitude is:", spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  print("End time of ABS, start of reformat: ", (time.time()-startTime)*1000)
  startTime = time.time()
  spectrogram = spectrogram[..., tf.newaxis]
  #print("Spectogram with added channel dimension is:", spectrogram)
  print("End time of spectrogram calculation: ", (time.time()-startTime)*1000)
  print()
  return spectrogram

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

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  print()
  startTime = time.time()
  #print("Start spectrogram calc time: ",startTime)
  input_len = 16000
  waveform = waveform[:input_len]
  #print("Waveform is:", waveform)
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  print("End time of padding, start of casting: ", (time.time() - startTime)*1000)
  startTime = time.time()
  #print("Zero padding of waveform:", zero_padding)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  print("End time of casting, start of concat: ", (time.time() -startTime)*1000)
  startTime = time.time()
  #print("Waveform after float32 casting is:", waveform)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  #print("Equal length concat is:", equal_length)
  # Convert the waveform to a spectrogram via a STFT.
  print("End time of concat, start of STFT: ", (time.time()-startTime)*1000)
  startTime = time.time()
  spectrogram = tf.signal.stft(
      equal_length, frame_length=1024, frame_step=512)
  #print("Spectogram is:", spectrogram)
  # Obtain the magnitude of the STFT.
  print("End time of STFT, start of ABS: ", (time.time()-startTime)*1000)
  startTime = time.time()
  spectrogram = tf.abs(spectrogram)
  #print("Spectogram absolute/magnitude is:", spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  print("End time of ABS, start of reformat: ", (time.time()-startTime)*1000)
  startTime = time.time()
  spectrogram = spectrogram[..., tf.newaxis]
  #print("Spectogram with added channel dimension is:", spectrogram)
  print("End time of spectrogram calculation: ", (time.time()-startTime)*1000)
  print()
  return spectrogram

def get_spectrogram_and_label(file_path):
  label = get_label(file_path)
  print("File path is: ", file_path)
  label_id = tf.argmax(label == commands)
  print("The label ID is: ", label_id.numpy())

  audio_binary = tf.io.read_file(file_path)
  #print("Audio Binaray is: ", audio_binary)
  waveform = decode_audio(audio_binary)

  spectrogram = get_spectrogram(waveform)

  return spectrogram, label_id
