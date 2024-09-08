# Self-attention-transformers
 Text Sequence prediction using self-attention transformers

# Project Overview
This project focuses on implementing a Transformer-based language model designed to predict the next character in a text sequence

## Objectives

1. **Implement a Transformer Model**: Develop a language model using the Transformer architecture, which includes embedding layers, positional encoding, and a series of Transformer encoder layers.
2. **Train and Evaluate**: Train the model on a large text dataset and evaluate its performance using perplexity and log probability metrics.
3. **Optimization**: Ensure the model achieves a perplexity of 7 or lower and trains within 10 minutes.

## Data

- **Training Set**: The first 100,000 characters from the `text8` dataset.
- **Development Set**: 500 characters from a different section of the dataset to validate the model.

## Model Description

### Transformer Architecture

The Transformer model used in this project consists of several key components:

1. **Embedding Layers**: Convert character indices into dense vectors of a fixed size (`d_model`).
2. **Positional Encoding**: Adds positional information to the embeddings to account for the order of characters.
3. **Transformer Encoder Layers**: Apply self-attention mechanisms to process the sequence of embeddings.
4. **Final Dense Layer**: Projects the output from the Transformer encoder to the size of the vocabulary.
5. **Log Softmax**: Applies the log-softmax function to the final outputs to get log probabilities.

### Model Parameters

- **Hidden Size (D_HIDDEN)**: 128
- **Embedding Size (D_MODEL)**: 64
- **Internal Dimension (D_INTERNAL)**: 16
- **Number of Heads (NUM_HEADS)**: 16
- **Number of Layers (NUM_LAYERS)**: 6
- **Sequence Length**: 20
- **Dropout Rate**: 0.001
- **Learning Rate**: 0.002
- **Epochs**: 20
- **Batch Size**: 128

## Implementation

### Files

- **`models.py`**: Contains the model definitions and training functions.
  - `Transformer`: The core Transformer model, including embedding, positional encoding, and Transformer encoder layers.
  - `NeuralLanguageModel`: Integrates the Transformer model to provide character prediction functionality.
  - `UniformLanguageModel`: A baseline model that assigns equal probability to each character.

### Key Classes and Functions

- **`Example`**: Represents a chunk of text with its corresponding input and output tensors.
- **`CustomDataset`**: A PyTorch Dataset class for loading and batching text data.
- **`train_lm`**: Function to train the Transformer model using the provided text data and return a trained `NeuralLanguageModel`.
