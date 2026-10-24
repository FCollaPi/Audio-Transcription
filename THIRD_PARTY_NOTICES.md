# Third-party notices and attribution

This tool was coded with the assistance of AI on top of the following open-source
repositories. The table below maps each source to the parts of this codebase
that derive from or depend on it.

| # | Source / Repository | Licence | Relation to this project |
|---|---------------------|---------|---------------------------|
| 1 | [openai/whisper](https://github.com/openai/whisper) | MIT | `format_timestamp()` and the TXT/SRT/VTT writers in `transcribe.py` are adapted from `whisper/utils.py`. Also the `--model` / `--language` / `--task` semantics. |
| 2 | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT | Core transcription engine: `WhisperModel(model, device, compute_type)` and `model.transcribe(language, task, vad_filter)`. CLI argument design mirrors its API. |
| 3 | [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) | MIT | Fast Transformer inference engine used underneath faster-whisper. |
| 4 | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | MIT | Executes the Silero VAD model used by faster-whisper for voice-activity detection. |
| 5 | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | MIT | VAD model used when `--vad-filter` is enabled (on by default). |
| 6 | [PyAV-Org/PyAV](https://github.com/PyAV-Org/PyAV) | BSD-3-Clause | Audio decoding (bundles FFmpeg libs); used by faster-whisper to load audio. |
| 7 | [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 | Downloads CTranslate2 Whisper model weights and tokenizers from the Hugging Face Hub. |
| 8 | [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | Apache-2.0 | Tokenizer encoding used by faster-whisper. |
| 9 | Whisper model weights (e.g. [`openai/whisper-*`](https://huggingface.co/openai), [`Systran/faster-whisper-*`](https://huggingface.co/Systran)) | MIT | Speech-to-text weights used at runtime. |
| 10 | [huggingface/distil-whisper](https://github.com/huggingface/distil-whisper) (model `distil-large-v3`) | MIT | Optional distilled model. |
| 11 | NVIDIA CUDA Toolkit / cuDNN (`nvidia::cudatoolkit`, `nvidia::cudnn` in `environment.yml`) | NVIDIA proprietary EULA (not OSI open source) | GPU execution libraries. Redistribution is restricted by NVIDIA's terms. |

All license texts below are reproduced where a license requires it (MIT/BSD
code incorporated into this repository). Apache-2.0 components are not
redistributed here (they are installed from PyPI on the user's machine), so a
link to their license text is given instead.

## 1. OpenAI Whisper — MIT License (Copyright (c) 2022 OpenAI)

The following code in `transcribe.py` was adapted from `whisper/utils.py`
from the [openai/whisper](https://github.com/openai/whisper) repository:

- `format_timestamp()`
- `write_txt()`
- `write_srt()`
- `write_vtt()`

The original license text is reproduced below, as required by the MIT License.

```
MIT License

Copyright (c) 2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. SYSTRAN faster-whisper — MIT License (Copyright (c) 2023 SYSTRAN)

```
MIT License

Copyright (c) 2023 SYSTRAN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
## 3. OpenNMT/CTranslate2 — MIT License

```
MIT License

Copyright (c) 2018-     SYSTRAN.
Copyright (c) 2019-     The OpenNMT Authors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 4. Microsoft ONNX Runtime — MIT License (Copyright (c) Microsoft Corporation)

```
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 5. Silero VAD — MIT License (Copyright (c) 2020-present Silero Team)

```
MIT License

Copyright (c) 2020-present Silero Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
## 6. PyAV — BSD-3-Clause License

```
Copyright retained by original committers. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the project nor the names of its contributors may be
      used to endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

Note: PyAV bundles FFmpeg/libav libraries, which are distributed under the
LGPL/GPL family of licences. This only matters if you yourself redistribute the
built binary wheels of PyAV; it has no effect on using this script as-is.

## 7. Hugging Face Hub — Apache License 2.0

Used at runtime (not copied into this repository). Licence text:
https://www.apache.org/licenses/LICENSE-2.0 and
https://github.com/huggingface/huggingface_hub/blob/main/LICENSE

## 8. Hugging Face Tokenizers — Apache License 2.0

Used at runtime (not copied into this repository). Licence text:
https://www.apache.org/licenses/LICENSE-2.0 and
https://github.com/huggingface/tokenizers/blob/main/LICENSE

## 9. Whisper model weights — MIT License

The CTranslate2 model weights downloaded at runtime (e.g.
`Systran/faster-whisper-small`) are conversions of OpenAI's Whisper models,
released under the MIT License (© 2022 OpenAI).

## 10. distil-whisper (`distil-large-v3`) — MIT License

Copyright 2023 The OpenAI Authors and The HuggingFace Inc. team.

The `distil-large-v3` model is released under the MIT License
(© 2023 The OpenAI Authors and The HuggingFace Inc. team), same terms as the
licences reproduced above. See
https://github.com/huggingface/distil-whisper/blob/main/LICENSE.

## 11. NVIDIA CUDA Toolkit / cuDNN — proprietary licence

`environment.yml` pins `nvidia::cudatoolkit=12.0.0` and `nvidia::cudnn=9.23.1.3`
from the NVIDIA conda channel. These are not open-source and are governed by
NVIDIA's End User Licence Agreement
(`https://www.nvidia.com/en-us/drivers/nvidia-license/networking/`).
Redistribution of those binaries outside your environment is not permitted.

---

This file covers all code incorporated into or invoked by this repository.
This project itself is MIT licensed; see [LICENSE](LICENSE).
---