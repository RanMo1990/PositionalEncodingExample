# Positional-Encoding

**Designed a Transformer-Style Model with Learnable Positional Encoding on Structured Synthetic Data using PyTorch**

This repository demonstrates how to implement a **learnable positional encoding** layer in PyTorch and evaluate it—alongside several fixed and hybrid encoding schemes—on a custom synthetic dataset where absolute token position is critical for correct prediction.

---

## Introduction

Positional encoding gives sequence models information about the order of tokens. Traditional approaches (e.g., sinusoidal) use fixed functions, but a **learnable** encoding lets the model discover optimal position representations from data. To showcase this, we generate sequences with a rigid template—special start, middle and end markers interleaved with arithmetic token patterns—and pad the rest. This structure forces any successful model to pay close attention to both token values and their exact positions.

---

## Features

- **Learnable Positional Encoding** that adapts during training  
- **Structured Dummy Dataset** combining special markers and numeric patterns  
- **Transformer-Style Architecture** integrating embedding, PE, multi-head attention, and feed-forward layers  
- **Multiple PE Strategies** compared: Sinusoidal, SinInit, Learnable, RoPE, and LRoPE  
- **Grid Experiments** across varying layer depths and training epochs  
- **Evaluation** by average test loss on held-out data  

---

## Implementation Overview

1. **Structured Dummy Dataset**  
   Sequences follow a fixed template: a start marker, a small arithmetic sub-sequence, a middle marker, the continuation of that arithmetic progression, an end marker, and zero-padding to a uniform length. This design guarantees that identical token values serve different semantic roles purely by position.

2. **Learnable Positional Encoding**  
   Rather than pre-computing position embeddings, we represent them as trainable model parameters. During each forward pass, these learned embeddings are added to the token input embeddings, allowing the network to refine positional signals based on the task.

3. **Transformer-Style Model**  
   A simple encoder stack processes the embedded inputs. Each layer applies multi-head self-attention—now augmented by learned positional cues—followed by a feed-forward network. The final layer projects back to the vocabulary space for next-token prediction.

4. **PE Comparison Framework**  
   By swapping the learnable PE for other schemes—classic sinusoidal, sininit (sinusoidal initialized then trainable), RoPE and LRoPE—we measure how each encoding performs under identical model configurations.

---

## Training and Evaluation

- **Loss Function**: Cross-entropy loss for next-token prediction  
- **Optimizer**: Adam for efficient gradient-based updates  
- **Train/Test Split**: 80% training, 20% held-out testing  
- **Experimental Grid**:  
  - Attention layer depths: 1, 4, 16  
  - Training epochs: 1, 4, 16  
  - PE types: sinusoidal, sininit, learnable, RoPE, LRoPE  

Models are trained on the synthetic dataset and then evaluated on test sequences to compute the average test loss, enabling fair comparison across encoding strategies and model capacities.

---

## Conclusion

- **Learnable Positional Encoding** consistently yields the lowest test loss, demonstrating superior adaptability to tasks with strict positional requirements.  
- **SinInit** strikes a balance between inductive bias and flexibility, closely following the learnable baseline.  
- **RoPE/LRoPE** underperform in our settings but may excel in longer or more relative-position-oriented settings.  

To fully understand each method’s strengths, we next explore the interaction of **model depth × training epochs** in a comprehensive follow-up study.

---

## Getting Started

1. **Clone** this repository into your local environment.  
2. **Install** dependencies with `pip install torch numpy pandas matplotlib scikit-learn`.  
3. **Open** and run `PositionalEncodingExample.ipynb` in Jupyter to reproduce all experiments and visualizations.

## Required Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


For a guided walkthrough, see the 5-minute screencast linked in the project description:
https://www.youtube.com/watch?v=lyb0D6np8uE
---

## Author

**Ran Mo**  
ML Internship Technical Interview Submission  
ranmo@iu.edu  
May 5, 2025  
