import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from contextlib import asynccontextmanager

from ..embedding_service.image_processor import get_preprocessor
from ..embeddings.models import get_embedding_model
from ..indexing.index import get_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global preprocessor, embedding_model, index

    # Initialize on startup
    preprocessor = get_preprocessor(
        preprocessor_type="standard", image_size=(224, 224), normalize=True
    )

    embedding_model = get_embedding_model(
        model_type="resnet", embedding_dim=512, pretrained=True
    )
    embedding_model.eval()

    index = get_index(index_type="faiss", dim=512, metric="cosine")

    yield

    # Cleanup on shutdown (if needed)
    # Add cleanup code here if necessary


app = FastAPI(title="Visual Search System", lifespan=lifespan)

# Global objects
preprocessor = None
embedding_model = None
index = None


class SearchResult(BaseModel):
    image_id: int
    distance: float
    metadata: Optional[Dict] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_time: float


@app.post("/embeddings")
async def generate_embedding(
    file: UploadFile = File(...),
) -> Dict[str, Union[str, List[float]]]:
    """Generate embedding for uploaded image."""
    try:
        # Read and preprocess image
        image = Image.open(file.file)
        tensor = preprocessor.preprocess(image)

        # Generate embedding
        with torch.no_grad():
            embedding = embedding_model.get_embedding(tensor.unsqueeze(0))

        return {"status": "success", "embedding": embedding.squeeze().tolist()}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating embedding: {str(e)}"
        )


@app.post("/search")
async def search_similar(
    file: UploadFile = File(...), *, k: int = 10
) -> SearchResponse:
    """Search for similar images."""
    try:
        # Read and preprocess image
        image = Image.open(file.file)
        tensor = preprocessor.preprocess(image)

        # Generate embedding
        with torch.no_grad():
            embedding = embedding_model.get_embedding(tensor.unsqueeze(0))

        # Search index
        distances, indices = index.search(embedding.numpy(), k=k)

        # Format results
        results = []
        for idx, (distance, image_id) in enumerate(zip(distances[0], indices[0])):
            results.append(
                SearchResult(
                    image_id=int(image_id),
                    distance=float(distance),
                    metadata={"rank": idx + 1},
                )
            )

        return SearchResponse(
            results=results, query_time=0.0  # TODO: Add actual timing
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error searching similar images: {str(e)}"
        )


@app.post("/index/add")
async def add_to_index(
    image_id: int,
    file: UploadFile = File(...),
) -> Dict[str, str]:
    """Add new image to the index."""
    try:
        # Read and preprocess image
        image = Image.open(file.file)
        tensor = preprocessor.preprocess(image)

        # Generate embedding
        with torch.no_grad():
            embedding = embedding_model.get_embedding(tensor.unsqueeze(0))

        # Add to index
        index.add(embedding.numpy(), np.array([image_id]))

        return {"status": "success"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error adding image to index: {str(e)}"
        )
