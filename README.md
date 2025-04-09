# Brainwave2Music

A system that converts EEG signals into music with multiple generation modes.

## System Overview

Brainwave2Music is an innovative music generation system that can:
- Convert EEG signals into music
- Generate music from text descriptions
- Combine EEG signals with text descriptions to create music

## Core Workflow

### 1. EEG Signal Processing Pipeline
```
EEG Data → Wavelet Transform → Spectrogram → Diffusion Model → Audio Signal
```
- **Data Loading**: Load EEG data from CSV files
- **Wavelet Processing**: Apply Wavelet Transform to extract time-frequency features
- **Spectrogram Generation**: Convert wavelet coefficients to spectrograms
- **Diffusion Processing**: Use Stable Diffusion model to enhance spectrograms
- **Audio Conversion**: Transform enhanced spectrograms to audio signals

### 2. Text-to-Music Pipeline
```
Text Description → MusicGen Model → Audio Generation
```
- **Text Processing**: Process user input descriptions
- **Model Generation**: Use MusicGen model to generate music
- **Audio Output**: Save generated music as WAV files

### 3. Hybrid Mode Pipeline
```
EEG Signal + Text Description → Demucs Processing → Diffusion Model → MusicGen → Final Audio
```
- **Signal Processing**: Process EEG signals using Wavelet Transform
- **Audio Separation**: Use Demucs for audio processing
- **Diffusion Enhancement**: Apply Stable Diffusion for quality improvement
- **Music Generation**: Combine with text descriptions
- **Final Output**: Generate enhanced musical composition

## Main Features

1. **EEG to Music Conversion**
   - Processes EEG signals using Wavelet Transform
   - Uses Stable Diffusion for spectrogram enhancement
   - Generates high-quality spectrograms
   - Converts to audio signals

2. **Text to Music Generation**
   - Uses MusicGen model
   - Supports music generation from text descriptions
   - Adjustable generation parameters

3. **Hybrid Mode**
   - Combines EEG signals with text descriptions
   - Uses Demucs for audio separation
   - Applies Stable Diffusion for quality enhancement
   - Supports multi-channel processing

## System Structure

```
brainwave2music/
├── app.py                 # Main entry point
├── eeg_to_music.py       # EEG signal processing
├── melody_to_music.py    # Melody generation
├── melody.py            # Audio processing
└── text_to_music.py    # Text to music conversion
```

## Usage

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the Program**
```bash
python app.py
```

3. **Choose Mode**
- Text mode: Generate music from text descriptions
- Melody mode: Generate music from EEG signals

## Technical Features

- **Signal Processing**: Uses Wavelet Transform for EEG signal processing
- **AI Models**: 
  - Stable Diffusion for spectrogram enhancement
  - MusicGen for text-to-music generation
  - Demucs for audio separation
- **Audio Processing**: Supports multi-channel audio processing
- **Visualization**: Generates spectrograms

## System Requirements

- Python 3.8+
- PyTorch
- CUDA/MPS support (optional)
- Other dependencies listed in requirements.txt

## File Descriptions

- `app.py`: Main entry point, provides user interface
- `eeg_to_music.py`: Core EEG signal processing
- `melody_to_music.py`: Melody generation module
- `melody.py`: Audio processing tools
- `new.py`: New implementation of EEG signal processing
- `text_to_music.py`: Text to music conversion module

## Contributing

Pull requests are welcome to improve the system.

## License

MIT License

## Contact

For any questions, please submit an Issue on GitHub.

