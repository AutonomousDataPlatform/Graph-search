# Graph Search Server

Search system for autonomous driving datasets using Neo4j graph database and CLIP embeddings.

## Features

- Text, image, and hybrid search
- Vector search using Neo4j
- Multilingual support (100+ languages)
- Advanced filters (weather, time, objects, etc.)

## Installation

### Prerequisites

- Python 3.10+
- Neo4j 5.x with vector index support

### Setup

1. Install base dependencies first (to avoid conflicts):
```bash
pip install "typing-extensions>=4.10.0"
```

2. Install PyTorch (tested version: 2.7.1):

**For CPU-only (Windows/Linux/Mac):**
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For CUDA 12.1 (NVIDIA GPU):**
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Note:** PyTorch 2.7.1 is the tested and working version on Windows. Other versions may have DLL loading issues.

See [PyTorch Get Started](https://pytorch.org/get-started/locally/) for more CUDA versions.

3. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

Create `.env` file:

```env
# Neo4j Database Connection
NEO4J_URL="bolt://your-neo4j-host:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your-password-here"

# Application Settings
DEVICE="cpu"  # Options: 'cuda' or 'cpu'
LOG_LEVEL="INFO"  # Options: DEBUG, INFO, WARNING, ERROR

# API Settings
API_TITLE="Graph Search API"
API_VERSION="0.1.0"
API_HOST="0.0.0.0"
API_PORT="5000"
```

### Run

```bash
python search.py
```

Server runs at `http://localhost:5000`

## API

### POST `/search`

Search endpoint supporting text, image, and hybrid modes.

**Parameters:**
- `query`: Text query (optional)
- `image`: Image file (optional)
- `image_weight`: Image weight for hybrid mode (default: 0.6)
- `text_weight`: Text weight for hybrid mode (default: 0.4)
- `datasets`: JSON array of datasets (empty = all)
- `object_filters`: JSON object `{"car": 2, "pedestrian": 1}`
- `weather`, `time_of_day`, `road_condition`: JSON arrays
- `traffic_density`, `pedestrian_density`: JSON arrays
- `limit`: Results per page (default: 100)
- `page`: Page number (default: 1)

**Example:**

```bash
curl -X POST http://localhost:5000/search \
  -F "query=car on highway" \
  -F "datasets=[\"KITTI\", \"BDD100k\"]" \
  -F "limit=100"
```

**Response:**

```json
{
  "results": [
    {
      "node_id": "4:abc123:xyz",
      "dataset": "BDD100k",
      "filepath": "path/to/image.jpg",
      "image_url": "https://dataset-url/path/to/image.jpg",
      "score": 0.892,
      "scores": {"image": 0.895, "text": 0.887},
      "objects": {"composition": {"car": 3}, "total": 3},
      "environment": {"weather": "rainy", "time_of_day": "night"},
      "context": {"traffic_density": "medium"}
    }
  ],
  "page": 1,
  "has_more": true
}
```

### GET `/search/categories`

Get available object categories.

### GET `/filters/options`

Get available filter options.

### GET `/health`

Health check endpoint.