import torch
import faiss
import os
import pickle

def calculate_default_m(d: int) -> int:
    """
    Calculate default M parameter based on embedding dimension.
    
    Args:
        d: Embedding dimension
    
    Returns:
        Default M value for PQ
    """
    return d // 8 // 2

def quantize(embeddings: torch.Tensor, device: torch.device = None, m: int = None, nbits: int = 4) -> torch.Tensor:
    """
    Quantize embeddings using Product Quantization (PQ).
    
    Args:
        embeddings: Input embeddings tensor
        device: Target device for the output tensor
        m: Number of sub-quantizers to use (default: dimension/16)
        nbits: Number of bits per sub-quantizer (default: 4)
    
    Returns:
        Quantized embeddings tensor
    """
    if device is None:
        device = embeddings.device
        
    d = embeddings.size(1)
    if m is None:
        m = calculate_default_m(d)
        
    index = faiss.index_factory(
        d, f"IDMap2,PQ{m}x{nbits}", faiss.METRIC_INNER_PRODUCT
    )
    index.train(embeddings.cpu())
    code = index.index.sa_encode(embeddings.cpu().numpy())
    return torch.tensor(index.index.sa_decode(code)).to(device)

def save(embeddings: torch.Tensor, path: str, device: torch.device = None, m: int = None, nbits: int = 4) -> None:
    """
    Compress and save embeddings to a file.
    
    Args:
        embeddings: Embeddings tensor to save
        path: Output file path
        device: Device to use for quantization
        m: Number of sub-quantizers to use (default: dimension/16)
        nbits: Number of bits per sub-quantizer (default: 4)
    """
    if device is None:
        device = embeddings.device
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    
    d = embeddings.size(1)
    if m is None:
        m = calculate_default_m(d)
    
    # Create and train the index
    index = faiss.index_factory(
        d, f"IDMap2,PQ{m}x{nbits}", faiss.METRIC_INNER_PRODUCT
    )
    index.train(embeddings.cpu())
    
    # Add IDs and embeddings to the index
    ids = torch.arange(len(embeddings)).numpy()
    index.add_with_ids(embeddings.cpu().numpy(), ids)
    
    # Encode embeddings
    codes = index.index.sa_encode(embeddings.cpu().numpy())
    
    # Save the index configuration, codes, and metadata
    data = {
        'codes': codes,
        'index': index,
        'shape': embeddings.shape,
        'm': m,
        'nbits': nbits
    }
    
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load(path: str, device: str = None) -> torch.Tensor:
    """
    Load and decompress embeddings from a file.
    
    Args:
        path: Input file path
        device: Target device for the output tensor
    
    Returns:
        Reconstructed embeddings tensor
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    codes = data['codes']
    index = data['index']
    
    # Decode the embeddings
    reconstructed = torch.tensor(index.index.sa_decode(codes)).to(device)
    return reconstructed 