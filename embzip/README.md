# EmbZip

A Python package for efficiently compressing and decompressing embeddings using Product Quantization.

## Installation

```bash
pip install embzip
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### Quantize embeddings

```python
import torch
import embzip

# Create sample embeddings
embeddings = torch.randn(1000, 768)  # 1000 vectors with 768 dimensions

# Quantize the embeddings with default parameters
quantized = embzip.quantize(embeddings)

# Quantize with custom M parameter (controls compression level)
# Lower M = higher compression but lower accuracy
quantized_high_compression = embzip.quantize(embeddings, m=24)  # more compressed
quantized_high_quality = embzip.quantize(embeddings, m=96)      # less compressed
```

### Save and load compressed embeddings

```python
import torch
import embzip

# Create sample embeddings
embeddings = torch.randn(1000, 768)  # 1000 vectors with 768 dimensions

# Save embeddings to a file with default parameters
embzip.save(embeddings, "embeddings.ezip")

# Save with custom M parameter
embzip.save(embeddings, "embeddings_higher_compression.ezip", m=24)

# Load embeddings from a file
loaded_embeddings = embzip.load("embeddings.ezip")

# Check similarity between original and reconstructed embeddings
similarity = torch.nn.functional.cosine_similarity(
    embeddings.view(-1), loaded_embeddings.view(-1), dim=0
)
print(f"Cosine similarity: {similarity.item()}")
```

## Parameter Tuning

The `m` parameter controls the number of sub-quantizers used in Product Quantization:

- Lower `m`: Higher compression, lower quality/accuracy
- Higher `m`: Lower compression, higher quality/accuracy

The default `m` is calculated as `dimension // 16`.

## Requirements

- Python 3.6+
- PyTorch
- FAISS

## Development

To run tests:
```bash
pytest
``` 