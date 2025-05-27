import sys
import openvino_genai
from pathlib import Path
from utils import optimum_cli, read_audio

DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_STT_DEVICE = "NPU"

if __name__ == "__main__":
    model_dir=DEFAULT_WHISPER_MODEL.split("/")[-1]
    if not Path(model_dir).exists():
        optimum_cli(model_id=DEFAULT_WHISPER_MODEL, output_dir=Path(model_dir))

    ov_config = dict()
    ov_config["CACHE_DIR"] = "./cache"

    pipe = openvino_genai.WhisperPipeline(
        models_path=model_dir, 
        device=DEFAULT_STT_DEVICE, 
        **ov_config
    )

    # test run
    sample_audio_path = "sample.wav"
    raw_audio = read_audio(sample_audio_path)
    result = pipe.generate(raw_audio, task="transcribe", return_timestamps=False)
    print(f"Successfully init {DEFAULT_WHISPER_MODEL} on {DEFAULT_STT_DEVICE}. Sample result: {result}")
    sys.exit(0)
