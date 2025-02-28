# Visual Search System

A modular visual search system that uses deep learning to find similar images based on a query image. The system supports flexible embedding methods, various nearest neighbor search algorithms, and customizable re-ranking strategies.

## System Architecture

The system consists of two main services:

1. **Embedding Generation Service**
   - Image preprocessing
   - Embedding generation using contrastive learning
   - Flexible model architecture support

2. **Indexing Service**
   - Embedding storage and indexing
   - Nearest neighbor search
   - Customizable re-ranking pipeline

## Project Structure

```
visual_search_system/
├── src/
│   ├── data/              # Data handling and preprocessing
│   ├── embeddings/        # Embedding generation models
│   ├── indexing/          # Index management and search
│   ├── evaluation/        # Metrics and evaluation tools
│   ├── api/              # FastAPI service endpoints
│   └── utils/            # Common utilities
├── tests/                # Unit and integration tests
├── configs/             # Configuration files
└── notebooks/           # Jupyter notebooks for experimentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Development

1. Run tests:
```bash
pytest tests/
```

2. Start the API server:
```bash
uvicorn src.api.main:app --reload
```

## Features

- Flexible embedding model architecture
- Multiple nearest neighbor search methods (FAISS, Annoy, etc.)
- Customizable re-ranking pipeline
- Comprehensive evaluation metrics
- RESTful API interface

## Evaluation Metrics

### Offline Metrics
- Mean Reciprocal Rank (MRR)
- Recall@K
- Precision@K
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)

### Online Metrics
- Click-through Rate (CTR)
- User engagement metrics

## Contributing

1. Format code:
```bash
black src/ tests/
isort src/ tests/
```

2. Run type checking:
```bash
mypy src/
```