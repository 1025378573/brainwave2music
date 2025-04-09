from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio
import torch
import torchaudio
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration


# 检查是否支持 MPS
device = torch.device("mps") if torch.has_mps else torch.device("cpu")
print(f"Using device: {device}")

# 替换以下路径为你的 .wav 文件路径
wav_file_path = "/Users/chenyanting/Desktop/test/pure/gen_musci.wav"

# 加载自己的 .wav 音频
waveform, sample_rate = torchaudio.load(wav_file_path)

# 将音频数据转换为 Tensor 并进行标准化
wav = waveform.mean(dim=0)  # 如果是多通道音频，取平均转为单通道
wav = wav.to(torch.float32).to(device)  # 将音频加载到 MPS 设备

# 加载 Demucs 模型
demucs = pretrained.get_model('htdemucs').to(device)  # 将模型加载到 MPS 设备

# 将音频转换为 Demucs 模型所需的采样率和通道数
wav = convert_audio(wav[None], sample_rate, demucs.samplerate, demucs.audio_channels).to(device)

# 使用 Demucs 分离音频
demucs_result = apply_model(demucs, wav[None])

# 打印结果的形状
print(f"Demucs result shape: {demucs_result.shape}")

# 合成所有音轨并转换为单通道
combined_music = demucs_result.sum(dim=1).squeeze(0)  # 合并 4 个音轨 -> [channels, samples]
combined_music = combined_music.mean(dim=0, keepdim=True).to("cpu")  # 转换为单通道 -> [1, num_samples]

# 打印最终形状，确保符合 [1, num_samples]
print(f"Combined music shape for MusicGen: {combined_music.shape}")

# ==========================
# MusicGen 模型部分
# ==========================

# 加载 MusicGen 模型和处理器
processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody").to(device)

# 准备输入
inputs = processor(
    audio=combined_music,
    sampling_rate=demucs.samplerate,
    text=["An ethereal, dynamic soundscape designed to enhance the brain's activity during the REM sleep stage. The music incorporates gentle, oscillating frequencies in the theta range (4–8 Hz), with light, airy melodies that evoke a dreamlike quality. Layered textures of soft chimes, subtle arpeggios, and flowing pads create a sense of fluid motion, mirroring the vividness and creativity of REM sleep. Gradual tonal shifts and delicate rhythms enhance the immersive experience without disrupting the sleep cycle, fostering an environment of relaxation and imaginative dreaming. Ideal for facilitating lucid dreams and mental rejuvenation."],  # 自定义描述
    padding=True,
    return_tensors="pt",
)

# 强制切换到 CPU 执行生成
inputs = {k: v.to("cpu") for k, v in inputs.items()}
model = model.to("cpu")

# 生成新音乐
generated_audio = model.generate(
    **inputs,
    do_sample=True,
    guidance_scale=3.0,  # 控制生成的灵活性
    max_new_tokens=768   # 控制生成长度
)

# 保存生成的音乐
generated_audio_path = "./generated_music.wav"
torchaudio.save(generated_audio_path, generated_audio.squeeze(0).to("cpu"), demucs.samplerate)
print(f"Generated music saved to: {generated_audio_path}")
