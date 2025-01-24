"""
Sample Audio Generation
"""
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate',value=100)  #Voice rate
text = "Sample Audio generated using Python."

output_file = "sample_audio.wav"

engine.save_to_file(text,output_file)
engine.runAndWait()

"""
Uniform Noise Generation - Once noise is generated, make this snippet inactive
as the randomness of noise will change the results everytime the code runs
"""
import numpy as np
from soundfile import write

sample_rate = 44100
duration = 4
amplitude = 0.5

white_noise = amplitude * np.random.uniform(-1,1,sample_rate*duration)
write("noise_sample.wav",white_noise,sample_rate)

"""
Merging Two Audio Signals
"""
from pydub import AudioSegment

audio1 = AudioSegment.from_wav("sample_audio.wav","wav")
audio2 = AudioSegment.from_wav("noise_sample.wav","wav")
audio3 = audio1.overlay(audio2)
audio3.export("merged_sample.wav","wav")

"""
Denoising Signal
"""
import matplotlib.pyplot as plt
import librosa

Audio_to_be_filtered = "merged_sample.wav"
x,sr = librosa.load(Audio_to_be_filtered)
#Plots the audio wave form in time domain
plt.figure(figsize=(14,5))
librosa.display.waveshow(x,sr=sr)
#Short-Time Fourier Transform - To visualize the frequency domain waveform 
#for the loaded audio data in time domain; where sr is the sample rate
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))

plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
plt.colorbar()
##plt.show()

noise_start_time = 0.0
noise_end_time = 4.0
threshold = 40
noise_segment = x[int(noise_start_time + sr):int(noise_end_time + sr)]
stft_noise = librosa.stft(noise_segment,n_fft = 1024,hop_length=512)
stft_audio = librosa.stft(x,n_fft=1024,hop_length=512)

noise_magnitude = np.mean(np.abs(stft_noise),axis=1)
audio_magnitude = np.abs(stft_audio)
mask = (audio_magnitude >threshold * noise_magnitude[:,np.newaxis])

stft_audio_denoised = stft_audio * mask

audio_denoised = librosa.istft(stft_audio_denoised,hop_length=512)
plt.figure()
librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft_audio_denoised),ref=np.max),sr=sr,hop_length=512,x_axis='time',y_axis='hz')
plt.title("Denoised signal")
write("Denoised.wav",audio_denoised,sr)
##plt.show()

"""
Filtering Signal
"""
from scipy.signal import butter,lfilter,filtfilt
import wave

denoised_sgnl = wave.open("Denoised.wav", 'r')
signal = denoised_sgnl.readframes(-1)
denoised_sgnl.close()
sound_sgnl = np.frombuffer(signal, dtype='int16')
framerate = denoised_sgnl.getframerate()
channels = denoised_sgnl.getnchannels()
width = denoised_sgnl.getsampwidth()
time_stamp = np.linspace(start=0,stop=len(sound_sgnl)/framerate,
                         num=len(sound_sgnl))

plt.figure(figsize=(12, 6))
plt.plot(time_stamp,sound_sgnl, label='Time Domain Denoised')
plt.title('Denoised')
plt.ylabel('Amplitude')
plt.xlabel('Time (seconds)')
plt.legend()

fs = 20000  
fc = 2000
order = 6
b, a = butter(order, [fc/(fs/2)], btype='lowpass')
filtered_signal = filtfilt(b,a,sound_sgnl)
scale_factor = 30000/max(abs(filtered_signal))
filtered_signal = filtered_signal * scale_factor
plt.figure(figsize=(12, 6))
plt.plot(time_stamp,filtered_signal, label='Filtered Signal')
plt.title('Filtered sound wave')
plt.ylabel('Amplitude')
plt.xlabel('Time (seconds)')
plt.legend()

regenerated_audio= wave.open('Filtered_signal.wav','wb')
regenerated_audio.setnchannels(channels)
regenerated_audio.setsampwidth(width)
regenerated_audio.setframerate(framerate)
regenerated_audio.writeframes(np.int16(filtered_signal))
print("Audio saved successfully")

#amplification
result_audio = AudioSegment.from_wav("Filtered_signal.wav")
amplified_audio = result_audio+2
amplified_audio.export("Result.wav","wav")
plt.show()
