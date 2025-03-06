from typing import Dict

import numpy as np
import faiss


def build_search_index(embeddings, config: Dict):
    """Build and save FAISS index for fast similarity search."""
    print("Building search index...")

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance

    # Add vectors to the index
    index.add(embeddings.astype(np.float32))

    # Save the index
    index_path = str(config["index_path"]) + "/" + config["index_file_name"]
    faiss.write_index(index, index_path)
    print(f"Search index saved to {index_path}")
