# SNAC Audio-Text Dataset Preprocessing

A GPU-accelerated preprocessing pipeline for speech-to-text datasets using SNAC (Simplified Neural Audio Codec) encoding. This code combines text tokenization with multi-level audio codec tokens for training speech synthesis models.

## Overview

This preprocessing function takes audio-text paired datasets and converts them into a unified token sequence suitable for training models like snorTTS or Indic TTS. It processes both text and audio data, encoding audio using SNAC's hierarchical codec structure.

## Features

- GPU-Accelerated Processing: Utilizes CUDA for fast SNAC encoding
- Automatic Audio Resampling: Converts audio to SNAC's required 24kHz sample rate
- Multi-Level Audio Tokenization: Flattens SNAC's hierarchical codec tokens
- Flexible Text Column Detection: Automatically finds transcription columns
- Error Handling: Graceful error handling with debug output
- Batched Processing: Efficient batch processing with configurable batch size

## Requirements
```bash
pip install torch librosa numpy snac
```

### Dependencies

- torch - PyTorch for GPU computation
- librosa - Audio processing and resampling
- numpy - Numerical operations
- snac - SNAC audio codec model

## Prerequisites

1. CUDA-enabled GPU: Required for GPU acceleration
2. Tokenizer: Must have a tokenizer object defined in your environment
3. Dataset: Audio-text paired dataset with columns:
   - Audio column named 'audio' containing:
     - 'array': audio waveform data
     - 'sampling_rate': original sample rate
   - Text column (one of): 'transcription', 'text', or 'sentence'

## Model Setup

The code uses the pre-trained SNAC model:
```python
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda")
snac_model.eval()
```

## Configuration

### Key Parameters

- AUDIO_OFFSET: 128266 - Offset added to audio tokens to separate them from text tokens
- Max Sequence Length: 2048 - Maximum combined sequence length
- Target Sample Rate: 24000 Hz - Required by SNAC
- Batch Size: 32 - Configurable based on GPU memory

## Usage
```python
# Ensure your tokenizer is defined
# tokenizer = YourTokenizer.from_pretrained(...)

# Process your dataset
processed_dataset = train_data.map(
    preprocess_function,
    batched=True,
    batch_size=32,
    remove_columns=train_data.column_names,
    desc="🚀 GPU-Accelerated SNAC Encoding"
)

print(f"📊 Final Processed Dataset Count: {len(processed_dataset)}")
```

## Processing Pipeline

### Step-by-Step Breakdown

1. Text Tokenization
   - Encodes text using the tokenizer
   - Adds special tokens (BOS/EOS)

2. Audio Resampling
   - Checks original sample rate
   - Resamples to 24kHz if necessary using librosa

3. SNAC Encoding
   - Converts audio to GPU tensor
   - Encodes using SNAC model
   - Returns hierarchical codec tokens

4. Token Flattening
   - Flattens multi-level SNAC codes
   - Adds offset (128266) to distinguish from text tokens
   - Creates continuous token sequence

5. Sequence Concatenation
   - Combines text tokens + audio tokens
   - Truncates to maximum length (2048)

## Token Structure

Text tokens followed by offset audio tokens:
[text_token_1, text_token_2, ..., audio_token_1+offset, audio_token_2+offset, ...]

### Example Token Ranges

- Text tokens: 0 - 128265
- Audio tokens: 128266+ (after offset)

## Troubleshooting

### Common Issues

Empty dataset after processing:
```python
# Reduce batch size to see error messages
processed_dataset = train_data.map(
    preprocess_function,
    batched=True,
    batch_size=1,  # Start with 1 to debug
    ...
)
```

CUDA out of memory:
- Reduce batch_size
- Process smaller audio chunks
- Clear GPU cache: torch.cuda.empty_cache()

Tokenizer not defined:
```python
# Define your tokenizer before running
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("your-model-name")
```

Wrong audio column format:
- Ensure dataset follows HuggingFace audio format
- Check column names match expected format

## Output Format

The function returns a dictionary with:
```python
{
    "input_ids": [[token_seq_1], [token_seq_2], ...],
    "labels": [[token_seq_1], [token_seq_2], ...]  # Same as input_ids
}
```

## Performance Tips

1. GPU Memory: Monitor with nvidia-smi
2. Batch Size: Start with 32, adjust based on GPU memory
3. Error Logging: First 3 errors are printed for debugging
4. Dataset Inspection: Verify column names before processing

## Citation

If using SNAC, please cite:
```bibtex
@article{snac2024,
  title={SNAC: Simplified Neural Audio Codec},
  author={Siuzdak, Hubert},
  year={2024}
}
```

## License

Check the SNAC model license at: https://huggingface.co/hubertsiuzdak/snac_24khz

## Additional Notes

- Processing speed depends on audio length and GPU capability
- The 7-token pattern refers to SNAC's multi-level encoding structure
- Designed for compatibility with snorTTS/Indic TTS architectures
