import torch
from transformers import Trainer, TrainingArguments
import wandb
from datasets import load_dataset
from transformer_model import Transformer_E, multi_transformer_E
from musdb_dataset import hug_musdbhq, multi_hug_musdbhq
import yaml
import argparse


def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    # assert references.dim() == 4
    # assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = torch.sum(torch.square(references), dim=(-2,-1))
    den = torch.sum(torch.square(references - estimates), dim=(-2,-1))
    num += delta
    den += delta
    scores = 10 * torch.log10(num / den)
    return scores

# B * C * fq * T

def compute_metrics(pred):

    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions

    preds = torch.tensor(preds, dtype=torch.float32)
    if preds.dim() == 5:
        preds = preds[:,3]
        labels = labels[:,3]
    batch, ch, freq, length = preds.shape
    preds = preds.view(batch,-1,2,freq,length)
    preds = torch.view_as_complex(preds.permute(0,1,3,4,2).contiguous())

    n_fft = 1024  
    preds = preds.view(-1,freq,length)
    signal = torch.istft(preds,
                        n_fft = n_fft,
                        window = torch.hann_window(n_fft).to(preds.real),
                        )  # B C * F * T  -> B C * L
    signal = signal.reshape(batch,ch//2,-1)

    signal_distortion_ratio = new_sdr(signal,torch.tensor(labels)).mean(0)

    return {
        'signal distortion ratio': signal_distortion_ratio,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training splitter.')
    parser.add_argument("--conf", type=str, default="config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    if config["report_to_wandb"]:
        wandb.init(
        project="simple_song_splitter",
        name = config["train_name"]
        )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = multi_transformer_E(
        n_block = config["n_block"],
        n_fft = config["n_fft"],
        d_inner = config["d_inner"],
        n_head = config["n_head"],
        d_k = config["d_k"],
        d_v = config["d_v"],
        dropout = config["dropout"],
        )

    model.to(device)

    print("model ready")

    musdb_train,musdb_valid = load_dataset("danjacobellis/musdb18HQ",split=["train","validation[:25%]"])

    dataset_train = multi_hug_musdbhq(musdb_train, duration = 300032/44100)
    dataset_valid = multi_hug_musdbhq(musdb_valid, duration = 300032/44100)

    print("dataset ready")
    
    training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=300,                 # model saving step.
    num_train_epochs=config["epochs"],              # total number of training epochs
    learning_rate=config["learning_rate"],               # learning_rate
    per_device_train_batch_size=config["batch_size"],  # batch size per device during training
    per_device_eval_batch_size=config["batch_size"],   # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.1,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    eval_strategy='steps', # evaluation strategy to adopt during training
    eval_steps = 300,            # evaluation step.
    load_best_model_at_end = True,
    report_to = "wandb" if config["report_to_wandb"] else "none"
    )

    trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=dataset_train,         # training dataset
    eval_dataset=dataset_valid,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
    )

    # train model
    trainer.train()

    torch.save(model.state_dict(), config["output_path"])
