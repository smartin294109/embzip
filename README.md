# `embzip`  🪨

A Python package for efficiently compressing and decompressing embeddings using Product Quantization.

## Installation

```bash
pip install embzip
```

## Usage

### Quantize embeddings

```python
import torch
import embzip

# Create sample embeddings
embeddings = torch.randn(1000, 768)  # 1000 vectors with 768 dimensions

# Quantize the embeddings
quantized = embzip.quantize(embeddings)
```

### Save and load compressed embeddings

```python
import torch
import embzip

# Create sample embeddings
embeddings = torch.randn(1000, 768)  # 1000 vectors with 768 dimensions

# Save embeddings to a file
embzip.save(embeddings, "embeddings.ezip")

# Load embeddings from a file
loaded_embeddings = embzip.load("embeddings.ezip")

# Check similarity between original and reconstructed embeddings
similarity = torch.nn.functional.cosine_similarity(
    embeddings.view(-1), loaded_embeddings.view(-1), dim=0
)
print(f"Cosine similarity: {similarity.item()}")
```

## Requirements

- Python 3.6+
- PyTorch
- FAISS 