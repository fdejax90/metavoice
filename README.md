# Data Engineer: Take home project

## Introduction
The goal of this project is to evaluate your knowledge and skills in the design and implementation  of a scalable data pre-processing pipeline.

## Problem statement
- Reads audio data being populated by Metavoice product, `Studio`, into a CloudFlare R2 bucket
- Runs two data transformation steps on the audio files:
  - Transcription - use [Whisper](https://github.com/openai/whisper)
  - Tokenisation - use mock code [here](https://gist.github.com/sidroopdaska/364e9f493d8dd9584eb9e1e9cae5715c)
- Stores the results using the example schema below. 
   ```<id - relative path of audio file>, <transcription>, <token array>```

## Requirements
- Install `ffmpeg` by following instructions [here](https://www.hostinger.com/tutorials/how-to-install-ffmpeg)
- Use pipenv to install the required packages:
  ```pipenv install```
- Go to where the `main.py` file is located and run:
```python main.py ```

## Notes
For scalability, I decided to read the audio file with a given chunk_size, and so preprocess the audio file in chunks. 
This is to avoid memory issues when dealing with large audio files.
The script is broken after a while (probably an audio file it does not like) as it shows:

```pydub.exceptions.CouldntDecodeError: Decoding failed. ffmpeg returned error code: 1```

I think there is a better solution, but by lack of time and not 100% sure if that feasible, that would be to:
- create a HuggingFace [loading-script](https://huggingface.co/docs/datasets/audio_dataset#loading-script)
- And so we could use the HF Dataset API to load the audio files and preprocess it.
- For the Whisper model, HF provide useful functions to [preprocess](https://huggingface.co/learn/audio-course/chapter1/preprocessing) it:
   ```
  from transformers import WhisperFeatureExtractor
  feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
  ... 
  ```


