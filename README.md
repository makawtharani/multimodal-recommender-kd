# multimodal-recommender-kd

# Multimodal Recommendation Engine with Knowledge Distillation

A powerful multimodal recommendation system that combines images, text, and graph signals through a teacher-student knowledge distillation framework. The system uses a full-modality "teacher" model with cross-modal attention and distills knowledge to a lightweight LightGCN "student" model for efficient deployment.

## Features

- **Multimodal Fusion**: Combines images, text, and graph signals with cross-modal attention
- **Knowledge Distillation**: Transfers knowledge from complex teacher to efficient student model
- **Modality Dropout**: Robust performance even when images or text are missing
- **Real-time Deployment**: Lightweight student model for production environments
- **Comprehensive Evaluation**: HR@K and NDCG@K metrics for thorough assessment

## Architecture

```
┌─────────────────┐    Knowledge     ┌─────────────────┐
│   Teacher Model │    Distillation   │  Student Model  │
│                 │    ──────────────▶│                 │
│ • ResNet/CLIP   │                   │ • LightGCN      │
│ • BERT          │                   │ • Distill Layer │
│ • Cross-modal   │                   │ • Modality      │
│   Attention     │                   │   Dropout       │
│ • Multimodal    │                   │                 │
│   GNN           │                   │                 │
└─────────────────┘                   └─────────────────┘
```

## Directory Structure

```
modality-dropout-distillation/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example             # Configuration template
│
├── data/
│   ├── raw/                 # Raw Amazon Beauty dataset
│   ├── processed/           # Processed train/val/test data
│   └── make_dataset.py      # Data preprocessing script
│
├── models/
│   ├── teacher/             # Teacher model components
│   ├── student/             # Student model components
│   └── losses.py            # Loss functions
│
├── train/
│   ├── train_teacher.py     # Teacher training script
│   └── train_student.py     # Student training with KD
│
├── evaluate/
│   ├── metrics.py           # Evaluation metrics
│   └── eval_runner.py       # Evaluation pipeline
│
├── configs/
│   ├── teacher.yaml         # Teacher configuration
│   ├── student.yaml         # Student configuration
│   └── sweep.yaml           # Hyperparameter sweeps
│
├── scripts/                 # Utility scripts
│   ├── download_data.sh
│   └── launch_sweep.sh
│
└── utils/                   # Utility modules
    ├── seed.py              # Reproducibility
    ├── logging.py           # Logging utilities
    └── serialization.py     # Model serialization
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd modality-dropout-distillation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 2. Data Preparation

```bash
# Process the Amazon Beauty dataset
python data/make_dataset.py
```

This will:
- Download the Amazon Beauty dataset from Hugging Face
- Clean and filter the data
- Extract text and image features
- Create train/validation/test splits
- Save processed data to `data/processed/`

### 3. Training

```bash
# Train the teacher model
python train/train_teacher.py

# Train the student model with knowledge distillation
python train/train_student.py
```

### 4. Evaluation

```bash
# Evaluate both models
python evaluate/eval_runner.py --model teacher
python evaluate/eval_runner.py --model student
```

## Dataset

The project uses the Amazon Beauty dataset from Hugging Face:
- **Reviews**: User ratings, text reviews, and timestamps
- **Metadata**: Product information, images, and descriptions
- **Scale**: Filtered to users/items with ≥5 interactions

### Data Loading Example

```python
from datasets import load_dataset

# Load reviews
reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)

# Load metadata
metadata = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty", split="full", trust_remote_code=True)
```

## Model Architecture

### Teacher Model
- **Image Encoder**: ResNet50 or CLIP Vision Transformer
- **Text Encoder**: BERT for review and product text
- **Graph Component**: GNN for user-item interactions
- **Fusion**: Cross-modal attention mechanism

### Student Model
- **Core**: LightGCN for efficient graph convolution
- **Distillation**: Knowledge transfer from teacher
- **Modality Dropout**: Random masking during training

## Configuration

Configuration files are located in `configs/`:

- `teacher.yaml`: Teacher model hyperparameters
- `student.yaml`: Student model and distillation settings
- `sweep.yaml`: Hyperparameter search ranges

## Evaluation Metrics

- **HR@K**: Hit Ratio at K (5, 10, 20)
- **NDCG@K**: Normalized Discounted Cumulative Gain at K
- **Robustness**: Performance with missing modalities

## Modality Dropout

The system implements random modality dropout during training:
- **Image Dropout**: Randomly mask image features
- **Text Dropout**: Randomly mask text features
- **Benefits**: Improved robustness and generalization

## Acknowledgments

- Amazon Beauty dataset from [McAuley-Lab](https://github.com/McAuley-Lab/Amazon-Reviews-2023)
- Built on PyTorch, PyTorch Geometric, and Hugging Face Transformers
- Inspired by LightGCN and knowledge distillation literature

## License

This project is licensed under the MIT License - see the LICENSE file for details. 