# models.py

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List, Any
from utils import Indexer
import math

# Define fixed parameters

SEQ_START = " "

# following D_HIDDEN > D_MODEL > D_INTERNAL = D_K
D_HIDDEN = 128  # hidden layer
D_MODEL = 64  # embedding
D_INTERNAL = 16  # d_k, d_q and d_v basically
NUM_HEADS = 16

VOCAB_SIZE = 27     # 26 letters + space
SEQUENCE_LENGTH = 20
USE_NORM = True

EPSILON = 10**-6
DROPOUT = 0.001
LEARNING_RATE = 0.002
WARMUP_STEPS = 40

NUM_EPOCHS = 20
HIDDEN_SIZE = 16
BATCH_SIZE = 128
NUM_LAYERS = 6


seen_labels = set()
INSPECTION = False


def inspect(x, label, show_val: bool = False):
    if INSPECTION and label not in seen_labels:
        seen_labels.add(label)
        label = "-----------" + "TENSOR # inspect: " + label + "-----------"
        sep = "_" * len(label)
        print(label)
        print(x.size())
        print(sep)
        if show_val:
            print(x)
            print(sep)


class LanguageModel(object):
    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


# =============================================== MY MODEL CODE HERE ======================================


class Example(object):
    def __init__(self, text: str, vocab_index: Indexer):
        self.input = SEQ_START + text[:-1]
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in self.input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = text
        self.output_indexed = np.array([vocab_index.index_of(ci) for ci in self.output])
        self.output_tensor = torch.LongTensor(self.output_indexed)


class CustomDataset(Dataset):
    def __init__(self, text: str, vocab_index: Indexer):
        labels = list(text[0+i:SEQUENCE_LENGTH+i] for i in range(0, len(text), SEQUENCE_LENGTH))
        labels = labels[:-1]    # Remove last element since it might not have required length
        self.examples = [Example(text, vocab_index) for text in labels]
        self.data = [example.input_tensor for example in self.examples]
        self.labels = [example.output_tensor for example in self.examples]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.data[idx]
        return data, label


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_size, n_heads, n_layers, seq_len, dropout):
        super(Transformer, self).__init__()
        # embedding layers
        self.enc_embedding = InputEmbeddings(vocab_size, d_model)
        # positional encoding layers
        self.enc_pe = PositionalEncoding(d_model, seq_len)
        # encoder/decoder layers
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, hidden_size, dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        # final dense layer
        self.dense = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

    def forward(self, x):
        inspect(x, " Model forward in")
        x = self.enc_embedding(x).permute(1, 0, 2)
        x = self.enc_pe(x)
        x = self.encoder(x, mask=self.mask)
        x = self.dense(x)
        x = self.log_softmax(x)
        inspect(x, " Model forward out")
        return x


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len=20, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def compute_loss(loss_function: Any, model: Transformer, dataloader: DataLoader):
    model.eval()
    for X_batch, Y_batch in dataloader:
        Y_predictions = model.forward(X_batch)
        loss = loss_function(Y_predictions.permute(1, 2, 0), Y_batch)
    return loss


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, voc_size: int, vocab_index: Indexer, model: Transformer):
        self.voc_size = voc_size
        self.model = model
        self.vocab_index = vocab_index

    def get_next_char_log_probs(self, context):
        if len(context) < SEQUENCE_LENGTH:
            context_last_idx = len(context)-1 if len(context) else 0
            context += SEQ_START*(SEQUENCE_LENGTH-len(context))
            assert len(context) == SEQUENCE_LENGTH
        else:
            context += SEQ_START
            context = context[len(context)-SEQUENCE_LENGTH:]
            context_last_idx = len(context) - 1
            assert len(context) == SEQUENCE_LENGTH

        x = Example(context, self.vocab_index).input_tensor
        x = x.unsqueeze(0)
        inspect(x, "Model inference forward", show_val=True)
        y = self.model.forward(x)
        inspect(y, "Model inference forward output", show_val=True)
        # result = np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)
        return y[context_last_idx][0].detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        log_prob_from_single_probs = 0.0
        for i in range(0, len(next_chars)):
            next_char_log_probs = self.get_next_char_log_probs(context + next_chars[0:i])
            log_prob_from_single_probs += next_char_log_probs[self.vocab_index.index_of(next_chars[i])]
        return log_prob_from_single_probs

# =============================================== MY MODEL CODE ENDS ======================================


def train_lm(args, train_text, dev_text, vocab_index: Indexer):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    train_ds = CustomDataset(train_text, vocab_index)
    print(len(train_ds))
    train_dataloader = DataLoader(
        dataset=train_ds,
        sampler=SubsetRandomSampler(np.arange(len(train_ds))),
        batch_size=BATCH_SIZE,
    )

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        hidden_size=D_HIDDEN,
        n_heads=NUM_HEADS,
        n_layers=NUM_LAYERS,
        seq_len=SEQUENCE_LENGTH,
        dropout=DROPOUT)

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.NLLLoss()

    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        for X_batch, Y_batch in train_dataloader:
            Y_predictions = model.forward(X_batch)
            loss = loss_function(Y_predictions.permute(1, 2, 0), Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch: {epoch}, {train_loss=:.3f}")

    model.batched = False
    model.eval()

    return NeuralLanguageModel(VOCAB_SIZE, vocab_index, model)

