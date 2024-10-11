# Bilingual Word Embedding Alignment

This project implements a bilingual word embedding alignment method using the Procrustes alignment technique. It focuses on aligning English and Hindi word embeddings using a bilingual lexicon, enabling effective translation and semantic similarity evaluation.

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Ablation Study](#ablation-study)
- [Evaluation Metrics](#evaluation-metrics)

## Getting Started

This section will guide you through setting up the project.

### Prerequisites

- Python 3.x
- Required libraries:
  - NumPy
  - SciPy
  - scikit-learn
 
# Installation
1. You can download FastText embeddings(English and Hindi) from this site https://fasttext.cc/
2. Download the bilingual dictionary (English to Hindi) from here https://dl.fbaipublicfiles.com/arrival/dictionaries/en-hi.txt

3. For more information visit this https://github.com/facebookresearch/MUSE

# Usage
1.**Limit Vocabulary**: Restrict to the top 100,000 words in each language
```python
  english_embeddings = limit_vocabulary(english_embeddings, 100000)
  hindi_embeddings = limit_vocabulary(hindi_embeddings, 100000)
```

2. **Extract Bilingual Lexicon**: Create word translation pairs from the MUSE dataset:
```python
bilingual_dict = extract_bilingual_lexicon(muse_dataset)
Align Embeddings: Perform Procrustes alignment:
aligned_X, W = procrustes_alignment(english_embeddings, hindi_embeddings, bilingual_dict)
```

3. **Evaluate Translation**: Evaluate translation precision:
```python
precision_at_1, precision_at_5 = evaluate_precision(bilingual_dict, english_embeddings, hindi_embeddings, W)
```

# Ablation study
1. Conduct an ablation study by evaluating different training dictionary sizes (5k, 10k, 20k):
```python
for size in [5000, 10000, 20000]:
    bilingual_lexicon = extract_bilingual_lexicon(bilingual_dict, size)
    aligned_X, W = procrustes_alignment(english_embeddings, hindi_embeddings, bilingual_lexicon)
    precision_at_1, precision_at_5 = evaluate_precision(bilingual_lexicon, english_embeddings, hindi_embeddings, W)
    print(f"Size: {size}, Precision@1: {precision_at_1}, Precision@5: {precision_at_5}")
```


# Evaluation-metrics
```python
- Precision@1: Proportion of correctly translated words in the top prediction.
- Precision@5: Proportion of correctly translated words in the top 5 predictions.
```





