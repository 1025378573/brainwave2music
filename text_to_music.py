from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import scipy
import os
USE_DIFFUSION_DECODER = False  # True: use diffusion decoder, False: use VQ-VAE decoder

def generate_music_from_text(description, device=None):
    """
    Generate music from a given text description and save it as a WAV file.
    
    Args:
        description (str): The text description of the music to generate.
        device (torch.device, optional): The device to use (e.g., "cuda", "mps", "cpu").
    """
    # Set the device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load processor and model
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)

    # Prepare inputs
    inputs = processor(
        text=[description],  # Use provided description
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate audio
    audio_values = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,  # Adjust for flexibility
        max_new_tokens=512   # Adjust for output length
    )
    print('Saving...')
    # Get sampling rate
    sampling_rate = model.config.audio_encoder.sampling_rate

    # check if output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Save the audio
    filename = './output/final_text.wav'
    audio_data = audio_values[0, 0].detach().cpu().numpy()  # Ensure the data is on the CPU
    scipy.io.wavfile.write(filename, rate=sampling_rate, data=audio_data)
    print(f"Music is saved as {filename}")
