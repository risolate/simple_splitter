import torch
from transformers import AutoConfig, Trainer, TrainingArguments
import wandb
from datasets import load_dataset
from transformer_model import Transformer_E
from musdb_dataset import hug_musdbhq

wandb.init(
    project="simple_song_splitter",
    name = "stft_encoder18_test"
)

##train param
learning_rate = 5e-2
epochs = 1
output_path = './result/12_30_test_hqdata.pt'

## transformer param
n_block = 6
d_inner = 2048
d_k = 512
d_v = 512
n_head = 8
dropout = 0.1

## stft param
n_fft = 1024

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Transformer_E(
    n_block = n_block,
    n_fft = n_fft,
    d_inner = d_inner,
    n_head = n_head,
    d_k = d_k,
    d_v = d_v,
    dropout = dropout,
    )

model.to(device)

print("model ready")

# dataset = MusDBDataset_test()
# dataset_train, dataset_valid = train_test_split(dataset,test_size=0.1)

### Musdbhq 데이터셋 코드
# train_path = []
# valid_path = []

# for path in glob("musdb18hq/train/*/mixture.wav"):
#     train_path.append(path)

# for path in glob("musdb18hq/test/*/mixture.wav"):
#     valid_path.append(path)

# dataset_train = MusDBhqDataset(train_path, duration = 300032/44100)
# dataset_valid = MusDBhqDataset(valid_path, duration = 300032/44100)

### Musdb 데이터셋 코드
# train_mus = musdb.DB(root="musdb18",subsets ="train")
# valid_mus = musdb.DB(root="musdb18",subsets="test")

# dataset_train = MusDBDataset(train_mus,duration=5)
# dataset_valid = MusDBDataset(valid_mus,duration=5)

###huggingface musdb 데이터셋 코드

musdb18hq = load_dataset("danjacobellis/musdb18HQ")

dataset_train = hug_musdbhq(musdb18hq["train"], duration = 300032/44100)
dataset_valid = hug_musdbhq(musdb18hq["validation"], duration = 300032/44100)

print("dataset ready")


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
    epsilon = torch.finfo(torch.float32).eps

    for i in range(signal.shape[0] // batch):
        # total_sdr += sdr(signal[i*batch:(i+1)*batch,:],torch.tensor(labels[i*batch:(i+1)*batch,:])+epsilon) * batch
        total_sdr += new_sdr(signal[i*batch:(i+1)*batch,:],torch.tensor(labels[i*batch:(i+1)*batch,:])) * batch

    remain = signal.shape[0] % batch

    if remain > 0:
        # total_sdr += sdr(signal[-remain:,:],torch.tensor(labels[-remain:,:])) * remain
        total_sdr += new_sdr(signal[-remain:,:],torch.tensor(labels[-remain:,:])) * remain

    signal_distortion_ratio = total_sdr / signal.shape[0]


    return {
        'signal distortion ratio': signal_distortion_ratio,
    }

training_args = TrainingArguments(
output_dir='./results',          # output directory
save_total_limit=5,              # number of total save model.
save_steps=300,                 # model saving step.
num_train_epochs=epochs,              # total number of training epochs
learning_rate=learning_rate,               # learning_rate
per_device_train_batch_size=8,  # batch size per device during training
per_device_eval_batch_size=8,   # batch size for evaluation
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

torch.save(model.state_dict(), output_path)