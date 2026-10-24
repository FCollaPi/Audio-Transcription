# Audio-Transcription

A small, dependency-light CLI tool that transcribes audio files (mp3, wav,
m4a, flac, …) into **plain text**, **SRT** subtitles or **WebVTT** captions
using open-source [Whisper](https://github.com/openai/whisper) models through
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (a CTranslate2-based
implementation that runs well on CPU **and** GPU).

> This repository was developed for academic use and is designed to be cited in
> a thesis. See [How to cite](#how-to-cite).

## Features

- Transcribes any audio/video format that FFmpeg can decode (via PyAV).
- Runs fully offline after the model is downloaded once.
- Works on CPU (int8) or NVIDIA GPU (float16).
- Outputs `txt`, `srt` (subtitles) or `vtt` (WebVTT captions).
- Voice-activity detection (VAD) to skip silence — on by default.
- Optional translation of any language to English (`--task translate`).

## Requirements

- python 3.11
- [Anaconda](https://www.anaconda.com/) (optional; recommended for managing python environment and easy installation of GPU dependencies)
- NVIDIA GPU + CUDA 12 / cuDNN 9 for GPU execution (optional; CPU works fine).

## Installation

Using the provided conda environment file (recommended for GPU usage, matches `environment.yml`):

```bash
conda env create -f environment.yml
conda activate audio-transcription
```

Or simply install the runtime dependency:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python transcribe.py audio.mp3
python transcribe.py audio.wav --model small --language en
python transcribe.py podcast.m4a --model medium --format srt --output podcast.srt
python transcribe.py meeting.mp3 --device cuda --compute-type float16
```

If no `--output` is given, the result is written next to the input with the new
extension (e.g. `audio.mp3` → `audio.txt`).

### Options

| Option              | Default      | Description |
|---------------------|--------------|-------------|
| `--model`           | `small`      | Model size: `tiny`, `base`, `small`, `medium`, `large-v1/v2/v3`, `.en` variants, `distil-large-v3`. Larger = more accurate but slower. |
| `--device`          | `auto`       | `cpu`, `cuda` or `auto`. |
| `--compute-type`    | `int8`       | Quantization: `int8`, `int8_float16`, `float16`, `float32`. |
| `--language`        | auto-detect  | Force a language code, e.g. `en`, `pt`, `ja`, `es`. |
| `--task`            | `transcribe` | `transcribe` or `translate` (any language → English). |
| `--format`          | `txt`        | `txt`, `srt` or `vtt`. |
| `--output`          | automatic    | Output file path. |
| `--no-vad-filter`   | (VAD on)     | Disable voice-activity filtering of silence. |

## How it works

1. The `WhisperModel` from faster-whisper loads a CTranslate2-converted Whisper
   model (downloaded from the Hugging Face Hub on first use).
2. Audio is decoded with PyAV and resampled to 16 kHz mono.
3. When VAD is enabled, a Silero VAD model (run via ONNX Runtime) skips silent
   parts.
4. faster-whisper (CTranslate2) generates segments, which this script renders
   as TXT, SRT or VTT.

## Sources, licences and attribution

This tool was coded on top of the following open-source projects.
The parts that were adapted/derived are called out explicitly:

| Component | Repository | Licence | Used for |
|-----------|-----------|---------|----------|
| OpenAI Whisper | [openai/whisper](https://github.com/openai/whisper) | MIT | `format_timestamp()` and the TXT/SRT/VTT writers (adapted from `whisper/utils.py`), plus `--model`/`--language`/`--task` semantics |
| faster-whisper | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT | Transcription engine (`WhisperModel`, `transcribe()`) |
| CTranslate2 | [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) | MIT | Fast inference engine (dependency of faster-whisper) |
| ONNX Runtime | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | MIT | Runs the Silero VAD model for silence filtering |
| Silero VAD | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | MIT | Voice-activity detection when `--vad-filter` is enabled |
| PyAV | [PyAV-Org/PyAV](https://github.com/PyAV-Org/PyAV) | BSD-3-Clause | Audio decoding (bundles FFmpeg) |
| Hugging Face Hub | [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 | Model & tokenizer downloads |
| Hugging Face Tokenizers | [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | Apache-2.0 | Tokenizer encoding |
| Whisper model weights | [`openai/whisper-*`](https://huggingface.co/openai), [`Systran/faster-whisper-*`](https://huggingface.co/Systran) | MIT | Speech-to-text weights |
| distil-whisper (`distil-large-v3`) | [huggingface/distil-whisper](https://github.com/huggingface/distil-whisper) | MIT | Optional distilled model |

Full attribution plus the required licence notices (as required by MIT/BSD for
the incorporated code) are in **[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)**.

## How to cite

### This repository

[![DOI](https://zenodo.org/badge/1431879836.svg)](https://doi.org/10.5281/zenodo.23611203)

BibTeX:

```bibtex
@software{audio_transcription,
    author       = {Felipe Colla Pinheiro},
    title        = {{audio-transcription}: a minimal CLI transcriber for audio files using Whisper models},
    year         = {2026},
    publisher    = {Zenodo},
    version      = {1.0.0},
    url          = {https://github.com/FCollaPi/audio-transcription},
    doi          = {10.5281/zenodo.23611204},
    note         = {Built on faster-whisper and OpenAI Whisper, both MIT licensed}
}
```

### Underlying work you should also cite

Whisper (the ASR model and paper):

```bibtex
@inproceedings{radford2023robust,
  title     = {Robust Speech Recognition via Large-Scale Weak Supervision},
  author    = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg
               and McLeavey, Christine and Sutskever, Ilya},
  booktitle = {International Conference on Machine Learning},
  publisher = {PMLR},
  year      = {2023},
  note      = {arXiv:2212.04356}
}
```

faster-whisper (the inference library used):

```bibtex
@software{faster_whisper,
  author    = {{SYSTRAN}},
  title     = {{faster-whisper}: Whisper transcription with {CTranslate2}},
  year      = {2023},
  url       = {https://github.com/SYSTRAN/faster-whisper},
  note      = {MIT licensed}
}
```

## License

MIT — see [`LICENSE`](LICENSE). The original license texts and credits for all
incorporated third-party code are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).