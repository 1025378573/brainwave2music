import torch
import torchaudio
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from eeg_to_music import generate_audio_from_segments
import os

def generate_music_from_melody(description, n_segments=5):
    # ==========================
    # Audio Generation from EEG Segments
    # ==========================

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # 从 EEG 数据生成音频
    combined_audio_segment = generate_audio_from_segments(n_segments)

    # 保存生成的音频用于后续调试
    temp_audio_path = "./temp_generated_audio.wav"
    combined_audio_segment.export(temp_audio_path, format="wav")
    print(f"Intermediate audio saved to: {temp_audio_path}")

    # 加载生成的音频
    waveform, sample_rate = torchaudio.load(temp_audio_path)

    # 将音频数据转换为 Tensor
    wav = waveform.mean(dim=0)  # 如果是多通道音频，转为单通道
    wav = wav.to(torch.float32).to(device)  # 加载到指定设备

    # ==========================
    # MusicGen Part
    # ==========================

    # 加载 MusicGen 模型和处理器
    processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
    model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody").to(device)

    # 准备输入
    inputs = processor(
        audio=wav.unsqueeze(0),  # 增加批次维度 -> [1, num_samples]
        sampling_rate=sample_rate,
        text=[description],
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 生成新音乐
    generated_audio = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=5.0,  # 控制生成的灵活性
        max_new_tokens=768   # 控制生成长度
    )

    # check if output directory exists
    os.makedirs('output', exist_ok=True)
    # 保存生成的音乐
    generated_audio_path = "./output/final_melody.wav"
    torchaudio.save(generated_audio_path, generated_audio.squeeze(0).to("cpu"), sample_rate)
    print(f"Generated music saved to: {generated_audio_path}")

if __name__ == "__main__":
    description = "An ethereal, dynamic soundscape designed to enhance the brain's activity during the REM sleep stage. The music incorporates gentle, oscillating frequencies in the theta range (4 to 8 Hz), with light, airy melodies that evoke a dreamlike quality. Layered textures of soft chimes, subtle arpeggios, and flowing pads create a sense of fluid motion, mirroring the vividness and creativity of REM sleep. Gradual tonal shifts and delicate rhythms enhance the immersive experience without disrupting the sleep cycle, fostering an environment of relaxation and imaginative dreaming. Ideal for facilitating lucid dreams and mental rejuvenation."
    generate_music_from_melody(description)
