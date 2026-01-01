import torch
from transformers import Trainer, TrainingArguments
import wandb
from datasets import load_dataset
from transformer_model import Transformer_E
from musdb_dataset import hug_musdbhq
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

    preds = torch.view_as_complex(preds.permute(0,2,3,1))

    n_fft = 1024  
    signal = torch.istft(preds,
                        n_fft = n_fft,
                        window = torch.hann_window(n_fft).to(preds),
                        )  # B * F * T  -> B * L

    batch = 8
    total_sdr = 0

    for i in range(signal.shape[0] // batch):
        total_sdr += new_sdr(signal[i*batch:(i+1)*batch,:],torch.tensor(labels[i*batch:(i+1)*batch,:])) * batch

    remain = signal.shape[0] % batch

    if remain > 0:
        total_sdr += new_sdr(signal[-remain:,:],torch.tensor(labels[-remain:,:])) * remain

    signal_distortion_ratio = total_sdr / signal.shape[0]


    return {
        'signal distortion ratio': signal_distortion_ratio,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training splitter.')
    parser.add_argument("--conf", type=str, default="config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    wandb.init(
    project="simple_song_splitter",
    name = config["train_name"]
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Transformer_E(
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

    musdb18hq = load_dataset("danjacobellis/musdb18HQ")

    dataset_train = hug_musdbhq(musdb18hq["train"], duration = 300032/44100)
    dataset_valid = hug_musdbhq(musdb18hq["validation"][:50], duration = 300032/44100)

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
    weight_decay=0,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    eval_strategy='steps', # evaluation strategy to adopt during training
    eval_steps = 300,            # evaluation step.
    load_best_model_at_end = True,
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
