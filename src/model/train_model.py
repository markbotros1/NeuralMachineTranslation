import os
import sys
import torch
from nmt import NMT
from train_helpers import train_and_evaluate
from sklearn.model_selection import train_test_split
from data.data_processing import read_corpus, create_embed_matrix
from data.vocab import Vocab


# Data file paths
source_path = os.path.join(os.getcwd(), "data", "source.txt")
target_path = os.path.join(os.getcwd(), "data", "target.txt")
test_path = os.path.join(os.getcwd(), "data", "source_test.txt")

# Read src and tgt data from .txt files
src_sents = read_corpus(source_path, "src")
tgt_sents = read_corpus(target_path, "tgt")

# Split src and tgt data into training and validation sets
train_data_src, val_data_src, train_data_tgt, val_data_tgt = train_test_split(src_sents, tgt_sents, test_size=0.045922, random_state=42)
train_data = list(zip(train_data_src, train_data_tgt))
val_data = list(zip(val_data_src, val_data_tgt))

# Create Vocab objects for src and tgt data
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
pretrained_src = create_embed_matrix(src_vocab, src_sents, config["embed_size"], 100)
pretrained_tgt = create_embed_matrix(tgt_vocab, tgt_sents, config["embed_size"], 100)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Build model using config parameters
model = NMT(
    config["embed_size"],
    config["hidden_size"],
    src_vocab,
    tgt_vocab,
    config["dropout"],
    device=device,
    pretrained_src=pretrained_src,
    pretrained_tgt=pretrained_tgt
)

log_every = 100
valid_niter = 500
model_save_path="NMT_model.ckpt"


if __name__ == "__main__":
    # model.to(device)
    # model.train
    # optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # train_and_evaluate(
    #     model,
    #     train_data,
    #     val_data,
    #     optimizer,
    #     config['epochs'],
    #     config['train_batch_size'],
    #     config['clip_grad'],
    #     log_every,
    #     valid_niter,
    #     model_save_path
    # )
    print(os.getcwd())
    # shutil.move("NMT_model.ckpt","drive/MyDrive/P3-NeuralMachineTranslation/models/mod_ab.ckpt")