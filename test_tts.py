import sherpa_onnx
import soundfile as sf

# model_dir = "./models/vits-icefall-en_US-ljspeech-medium"
model_dir = "./models/vits-piper-hi_IN-priyamvada-medium"

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            # model=f"{model_dir}/model.onnx",
            model=f"{model_dir}/hi_IN-priyamvada-medium.onnx",
            tokens=f"{model_dir}/tokens.txt",
            data_dir=model_dir,   # ✅ THIS is the fix
        ),
        num_threads=8,
    ),
)

tts = sherpa_onnx.OfflineTts(tts_config)

# audio = tts.generate("Hello, this should now work properly. how are you doing today?")
audio = tts.generate(" मेरा नाम रोहन है  और मैं ठीक हूँ। आप कैसे हैं?")

sf.write("output.wav", audio.samples, audio.sample_rate)