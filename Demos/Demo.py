#Antes de correr el programa, se asume que ya se cumplen con todos los requisitos y dependencias (nemo, torch, etc.)

import IPython.display as ipd
import librosa
import librosa.display
import numpy as np
import torch
from matplotlib import pyplot as plt

# Reduce logging messages for this notebook
from nemo.utils import logging
logging.setLevel(logging.ERROR)

from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel
from nemo.collections.tts.helpers.helpers import regulate_len

# Load the models from NGC
fastpitch = FastPitchModel.from_pretrained("tts_en_fastpitch").eval().cuda()
hifigan = HifiGanModel.from_pretrained("tts_hifigan").eval().cuda()
sr = 22050

# Define a helper function to go from string to audio
def str_to_audio(inp, pace=1.0, durs=None, pitch=None):
    with torch.no_grad():
        tokens = fastpitch.parse(inp)
        spec, _, durs_pred, _, pitch_pred, *_ = fastpitch(text=tokens, durs=durs, pitch=pitch, speaker=None, pace=pace)
        audio = hifigan.convert_spectrogram_to_audio(spec=spec).to('cpu').numpy()
    return spec, audio, durs_pred, pitch_pred

# Define a helper function to plot spectrograms with pitch and display the audio
def display_pitch(audio, pitch, sr=22050, durs=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    spec = np.abs(librosa.stft(audio[0], n_fft=1024))
    # Check to see if pitch has been unnormalized
    if torch.abs(torch.mean(pitch)) <= 1.0:
        # Unnormalize the pitch with LJSpeech's mean and std
        pitch = pitch * 65.72037058703644 + 214.72202032404294
    # Check to see if pitch has been expanded to the spec length yet
    if len(pitch) != spec.shape[0] and durs is not None:
        pitch = regulate_len(durs, pitch.unsqueeze(-1))[0].squeeze(-1)
    # Plot and display audio, spectrogram, and pitch
    ax.plot(pitch.cpu().numpy()[0], color='cyan', linewidth=1)
    librosa.display.specshow(np.log(spec+1e-12), y_axis='log')
    ipd.display(ipd.Audio(audio, rate=sr))
    plt.show()

input_string = input('Ingresar string a reproducir [eng]: ')
_, audio, *_ = str_to_audio(input_string, pace=1)
ipd.display(ipd.Audio(audio, rate=sr))
