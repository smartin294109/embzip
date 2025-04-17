import torch
import os
import time
from embzip.core import quantize, save, load

def test_embzip():
    print("Testing EmbZip package...")
    
    # Create sample embeddings
    dim = 768
    num_vectors = 1000
    print(f"Creating {num_vectors} random vectors with {dim} dimensions...")
    embeddings = torch.randn(num_vectors, dim)
    
    # Test quantization
    print("\nTesting quantization...")
    start_time = time.time()
    quantized = quantize(embeddings)
    quant_time = time.time() - start_time
    print(f"Quantization completed in {quant_time:.4f} seconds")
    
    # Calculate similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        embeddings.view(-1), quantized.view(-1), dim=0
    )
    print(f"Cosine similarity: {cos_sim.item():.4f}")
    
    # Test save and load
    print("\nTesting save and load...")
    test_file = "test_embeddings.ezip"
    
    # Save embeddings
    start_time = time.time()
    save(embeddings, test_file)
    save_time = time.time() - start_time
    print(f"Embeddings saved to '{test_file}' in {save_time:.4f} seconds")
    
    # Check file size
    orig_size = embeddings.element_size() * embeddings.nelement()
    file_size = os.path.getsize(test_file)
    compression_ratio = orig_size / file_size
    print(f"Original size: {orig_size / 1024:.2f} KB")
    print(f"Compressed size: {file_size / 1024:.2f} KB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Load embeddings
    start_time = time.time()
    loaded = load(test_file)
    load_time = time.time() - start_time
    print(f"Embeddings loaded in {load_time:.4f} seconds")
    
    # Calculate similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        embeddings.view(-1), loaded.view(-1), dim=0
    )
    print(f"Cosine similarity: {cos_sim.item():.4f}")
    
    # Clean up
    os.remove(test_file)
    print(f"Test file '{test_file}' deleted")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_embzip() 