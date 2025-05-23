import numpy as np
from scipy.signal import butter


N_CHANNELS = 2

# EEG constants ---------------
MIN_EEG_FREQUENCY = 0.5
MAX_EEG_FREQUENCY = 40
N_EEG_FREQUENCIES = 512
EEG_FREQUENCIES = np.linspace(MIN_EEG_FREQUENCY, MAX_EEG_FREQUENCY, N_EEG_FREQUENCIES)
SAMPLE_RATE = 256
SEGMENT_LEN_S = 5 
CHANNEL_IDS = (0, 1, 2, 3)
SAMPLE_EEG_PATH = "./samples/eeg_samples/2min_16hz.csv"
BANDPASS_FILTER = butter(4, (MIN_EEG_FREQUENCY, MAX_EEG_FREQUENCY), 'bp', output='sos', fs=SAMPLE_RATE)

# Spectrogram constants -------
SPECTROGRAM_WIDTH = 512
SPECTROGRAM_HEIGHT = 512
SPECTROGRAM_SHIFT = 30  # shift frequencies to higher notes
SPECTROGRAM_POWER = 4  # for converting it to audio
SPECTROGRAM_MAX_VALUE = 30e6
MEL_NOTES = [51, 66, 76, 91]
NOTE_MASK = np.load('./samples/c2_to_c6_mask.npy')

# Model constants -------------
RIFFUSION_CHECKPOINT = "riffusion/riffusion-model-v1"
SCHEDULER_OPTIONS = [
    "DPMSolverMultistepScheduler",
    "PNDMScheduler",
    "DDIMScheduler",
    "LMSDiscreteScheduler",
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
]
TEXT_PROMPT = 'piano, dreamy tune, An ethereal, dynamic soundscape, Layered textures of soft chimes, subtle arpeggios, and flowing pads create a sense of fluid motion,'
TEXT_NEGATIVE_PROMPT = 'bell, A harsh, static soundscape that disrupts brain activity during REM sleep. The composition features sharp, jarring frequencies outside the theta range, paired with dissonant, chaotic melodies that evoke tension and unease. Abrasive textures of clashing tones, erratic beats, and fragmented noise create a sense of disorder and agitation, contrasting with the fluidity of REM sleep. Sudden tonal shifts and irregular rhythms '
DENOISING_STRENGTH = 0.65
GUIDANCE_SCALE = 7.0
INFERENCE_STEPS = 15
SEED = 42

# Audio constants -------------
AUDIO_SAMPLE_RATE = 44100
MIN_AUDIO_FREQUENCY = 0
MAX_AUDIO_FREQUENCY = 10000
PLAYBACK_SLEEP_TIME_S = 4.98
DESIRED_DB = -20
CROSSFADE_SAVE_MS = 500
ANTISPIKE_THRESHOLD = 20e6
DEFAULT_SAVE_AUDIO_FOLDER = 'uncombined'
