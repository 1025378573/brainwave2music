import time
from pydub import AudioSegment
import torch
from src.data.eeg_features import extract_features
from src.data.spectral_transform import combine_spectrograms, transform_spectrogram
from src.data.utils import save_pydub_audio_file, produce_audio_from_spectrogram_with_torch, apply_audio_filters
from src.data.torch_utils import SpectrogramConverter
from src.data.riffusion import load_stable_diffusion_img2img_pipeline
from src.data.sample_gen import get_offline_eeg_segments
from src.parameters import ChannelParameters
from src.constants import N_CHANNELS

from pydub import AudioSegment
import time
import os

# 常量配置
AUDIO_SAMPLE_RATE = 44100
MIN_AUDIO_FREQUENCY = 0
MAX_AUDIO_FREQUENCY = 10000
PLAYBACK_SLEEP_TIME_S = 4.98
DESIRED_DB = -20
CROSSFADE_SAVE_MS = 500
ANTISPIKE_THRESHOLD = 20e6
DEFAULT_SAVE_AUDIO_FOLDER = 'uncombined'

def generate_audio_from_segments(n_segments: int):
    """
    Generate audio by processing EEG segments into spectrograms, transforming them,
    and combining the resulting audio segments.

    Parameters:
        n_segments (int): Number of EEG segments to process.

    Returns:
        AudioSegment: Combined audio segment generated from EEG data.
    """
    # Initialize components and parameters
    converter = SpectrogramConverter()
    riffusion_model = load_stable_diffusion_img2img_pipeline(device='mps' if torch.backends.mps.is_available() else "cuda" )
    eeg_segments = get_offline_eeg_segments()
    parameters = {i: ChannelParameters() for i in range(N_CHANNELS)}

    # Debugging: Print the number of samples in EEG segments
    print(f"Number of samples in segment 0: {len(eeg_segments[0])}")
    print(f"Number of samples in segment 1: {len(eeg_segments[1])}")

    # Initialize an empty AudioSegment to concatenate results
    combined_audio = AudioSegment.empty()

    # Ensure the output folder exists
    os.makedirs(DEFAULT_SAVE_AUDIO_FOLDER, exist_ok=True)

    # Process each EEG segment
    for i, segment in enumerate(eeg_segments[:n_segments]):
        start = time.time()

        try:
            # Step 1: Extract spectrograms for each channel
            spectrograms = []
            for ch in range(N_CHANNELS):
                spectrogram = extract_features(segment, ch=ch, channel_params=parameters[ch])
                spectrograms.append(spectrogram)

            # Step 2: Combine spectrograms from all channels
            combined_spectrogram = combine_spectrograms(spectrograms)

            # Step 3: Transform the spectrogram using the Riffusion model
            transformed_spectrogram = transform_spectrogram(
                combined_spectrogram,
                riffusion_model=riffusion_model,
                measure_difference=True
            )

            # Step 4: Generate audio from the spectrogram
            audio = produce_audio_from_spectrogram_with_torch(transformed_spectrogram, converter)

            # Step 5: Apply audio filters
            audio = apply_audio_filters(audio)

            # Step 6: Normalize volume to desired dB level
            audio = audio.apply_gain(DESIRED_DB - audio.dBFS)

            # Step 7: Check for spikes and apply crossfade if necessary
            if max(audio.get_array_of_samples()) > ANTISPIKE_THRESHOLD:
                print(f"Spike detected in segment {i + 1}, applying crossfade.")
                audio = audio.fade_in(CROSSFADE_SAVE_MS).fade_out(CROSSFADE_SAVE_MS)

            # Step 8: Save individual audio segment
            segment_path = os.path.join(DEFAULT_SAVE_AUDIO_FOLDER, f"segment_{i + 1}.wav")
            audio.export(segment_path, format="wav")
            print(f"Segment {i + 1} saved to {segment_path}")

            # Step 9: Concatenate the processed audio
            combined_audio += audio

            end = time.time()
            print(f'Processed segment {i + 1} in {end - start:.2f} s')

        except Exception as e:
            # Catch any errors and print debug information
            print(f"Error processing segment {i + 1}: {e}")

    # Return the combined audio segment
    return combined_audio

if __name__ == "__main__":
    n_segments = 8  # Specify the number of segments you want to process
    combined_audio = generate_audio_from_segments(n_segments) 
    save_pydub_audio_file(combined_audio, './gen_musci_new.wav')