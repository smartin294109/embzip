import torch
import faiss
import os
import pickle

def quantize(embeddings: torch.Tensor, device: torch.device = None) -> torch.Tensor:
    """
    Quantize embeddings using Product Quantization (PQ).
    
    Args:
        embeddings: Input embeddings tensor
        device: Target device for the output tensor
    
    Returns:
        Quantized embeddings tensor
    """
    if device is None:
        device = embeddings.device
        
    d = embeddings.size(1)
    M = d // 8 // 2
    nbits = 4
    index = faiss.index_factory(
        d, f"IDMap2,PQ{M}x{nbits}", faiss.METRIC_INNER_PRODUCT
    )
    index.train(embeddings.cpu())
    code = index.index.sa_encode(embeddings.cpu().numpy())
    return torch.tensor(index.index.sa_decode(code)).to(device)

def save(embeddings: torch.Tensor, path: str, device: torch.device = None) -> None:
    """
    Compress and save embeddings to a file.
    
    Args:
        embeddings: Embeddings tensor to save
        path: Output file path
        device: Device to use for quantization
    """
    if device is None:
        device = embeddings.device
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    d = embeddings.size(1)
    M = d // 8 // 2
    nbits = 4
    
    # Create and train the index
    index = faiss.index_factory(
        d, f"IDMap2,PQ{M}x{nbits}", faiss.METRIC_INNER_PRODUCT
    )
    index.train(embeddings.cpu())
    
    # Add IDs and embeddings to the index
    ids = torch.arange(len(embeddings)).numpy()
    index.add_with_ids(embeddings.cpu().numpy(), ids)
    
    # Encode embeddings
    codes = index.index.sa_encode(embeddings.cpu().numpy())
    
    # Save the index and codes
    data = {
        'codes': codes,
        'index': index,
        'shape': embeddings.shape
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
    shape = data['shape']
    
    # Decode the embeddings
    reconstructed = torch.tensor(index.index.sa_decode(codes)).to(device)
    return reconstructed 