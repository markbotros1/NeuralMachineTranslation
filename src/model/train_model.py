import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import torch
import shutil
from sklearn.model_selection import train_test_split
from nmt import NMT
from train_helpers import train_and_evaluate
from data.data_processing import read_corpus, create_embed_matrix
from data.vocab import Vocab

# Data file paths
source_path = os.path.join(os.getcwd(), "data", "source.txt")
target_path = os.path.join(os.getcwd(), "data", "target.txt")
test_path = os.path.join(os.getcwd(), "data", "source_test.txt")

# Read src and tgt data from .txt files
print("reading corpus")
src_sents = read_corpus(source_path, "src")
tgt_sents = read_corpus(target_path, "tgt")

# Split src and tgt data into training and validation sets
print("creating training and validation sets")
train_data_src, val_data_src, train_data_tgt, val_data_tgt = train_test_split(src_sents, tgt_sents, test_size=0.045922, random_state=42)
train_data = list(zip(train_data_src, train_data_tgt))
val_data = list(zip(val_data_src, val_data_tgt))

# Create Vocab objects for src and tgt data
print("creating vocab")
src_vocab = Vocab.from_corpus(src_sents, 20000, 2)
tgt_vocab = Vocab.from_corpus(tgt_sents, 20000, 2)

# Set and/or change model parameters
config = {
    "embed_size": 350,
    "hidden_size": 512,
    "epochs": 10,
    "train_batch_size": 32,
    "clip_grad": 2,
    "lr": 1e-3,
    "dropout": .2
}

# Pretrain word embeddings (comment following lines if pretrained embeddings not preferred)
print("creating embedding matrices")
pretrained_src = create_embed_matrix(src_vocab, src_sents, config["embed_size"], 100)
pretrained_tgt = create_embed_matrix(tgt_vocab, tgt_sents, config["embed_size"], 100)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_save_path="NMT_model.ckpt"

# Build model using config parameters
model = NMT(
    config["embed_size"],
    config["hidden_size"],
    src_vocab,
    tgt_vocab,
    config["dropout"],
    device=device,
    pretrained_source=pretrained_src,
    pretrained_target=pretrained_tgt
)


if __name__ == "__main__":
    model.to(device)
    model.train
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    print("training model")
    train_and_evaluate(
        model,
        train_data,
        val_data,
        optimizer,
        config['epochs'],
        config['train_batch_size'],
        config['clip_grad'],
        model_save_path
    )

    shutil.move("NMT_model.ckpt","models/model.ckpt")

