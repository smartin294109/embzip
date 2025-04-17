import torch
import faiss

def quantize(embeddings: torch.Tensor, device: torch.device) -> torch.Tensor:
    d = embeddings.size(1)
    M = d // 8 // 2
    # M = d // (3 * 64 * 1)
    nbits = 4
    index = faiss.index_factory(
        d, f"IDMap2,PQ{M}x{nbits}", faiss.METRIC_INNER_PRODUCT
    )
    index.train(embeddings.cpu())
    code = index.index.sa_encode(embeddings.cpu().numpy())
    return torch.tensor(index.index.sa_decode(code)).to(device)