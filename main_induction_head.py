import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append('..')
import torch
import random
import numpy as np
from random import choice, choices
from mamba_lm import MambaLM, MambaLMConfig

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"> Using {device} device")

n_vocab = 16
epochs = 5
epoch_sz = 8192
batch_size = 8
seq_length = 256
prefix_len = 10
d_model=64
d_state=16
n_layers=2
dt_rank=16
d_conv=4
report_every=100
learning_rate=1e-3
model_path = f'saves/model.pth'

class InductionData:
    def __init__(self, batch, n_vocab, seq_len, prefix_len):
        """
        Generates synthetic data of the form:
        ... S M .... S M
        where S is an 'induction token' and M is the token to memorize / recall

        n_vocab: token alphabet size
        seq_len: total sequence length to generate
        prefix_len: region where first S should occur
        ind_tok: the 'special token' as per mamba's implementation
        """
        assert prefix_len < seq_len - 4
        self.B = batch
        self.V = n_vocab
        self.L = seq_len
        self.P = prefix_len
        self.vocab = list(range(self.V))
        self.ind_tok = self.V

    def gen(self):
        """
        Section E.1 from https://arxiv.org/pdf/2212.14052.pdf 
        Training consists of randomly generating data every step
        """
        memory_tok = choice(self.vocab)
        ind_pos = choice(range(self.P))
        
        cadence = [self.ind_tok, memory_tok]
        pre = choices(self.vocab, k=ind_pos)
        noise = choices(self.vocab, k=self.L-ind_pos-2) 
        seq = pre + cadence + noise + cadence 
        return seq

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        for _ in range(self.B):
            batch.append(self.gen())
        return torch.tensor(batch).to(device)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_checkpoint(filepath, model):
    print(f"> Loading model from: {filepath}")
    try:
        loaded_checkpoint = torch.load(filepath, map_location=device)
        loaded_epoch = loaded_checkpoint['epoch']
        loaded_model = model
        loaded_model.load_state_dict(loaded_checkpoint['model_state'])
        print("> Loaded model")
        return True, loaded_epoch, loaded_model
    except Exception as e:
        print("> Cannot load model")
        return False, 0, model

def train():
    config = MambaLMConfig(d_model=d_model, n_layers=n_layers, dt_rank=dt_rank, d_conv=d_conv, d_state=d_state, vocab_size=n_vocab+1)
    model = MambaLM(config).to(device)
    train_data = InductionData(batch_size, n_vocab, seq_length, prefix_len)
    it = iter(train_data)

    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    step = 0
    model.train()
    for epoch in range(epochs):
        loss_sum = 0
        for b in range(epoch_sz):
            batch = next(it)
            optim.zero_grad()
            out = model(batch[:,:-1])
            pred = out[:,-1,:]
            targ = batch[:,-1]
            loss = criterion(pred, targ)
            loss.backward()
            loss_sum += loss
            optim.step()
            step += 1
            if step % report_every == 0:
                print(f'{step=}, {loss.item():3.3f}')

        epoch_loss = (loss_sum / epoch_sz).item()
        print(f'{epoch=}, {epoch_loss=:3.3f}')

        # If mamba has perfectly solved the inductin head task
        if loss.item() < 0.001:
            print('Training completed at epoch no.', epoch+1)
            print('Saving model checkpoint...')
            checkpoint = {
                'epoch': epoch+1,
                'model_state': model.state_dict()
            }
            torch.save(checkpoint, model_path)
            break

def infer():
    criterion = torch.nn.CrossEntropyLoss()
    config = MambaLMConfig(d_model=d_model, n_layers=n_layers, dt_rank=dt_rank, d_conv=d_conv, d_state=d_state, vocab_size=n_vocab+1)
    model = MambaLM(config).to(device)
    isLoaded, _, model = load_checkpoint(f'saves/model.pth', model)
    if (not isLoaded):
        return
    eval_data = InductionData(1, n_vocab, 128, prefix_len)
    it = iter(eval_data)

    # Evaluation loop
    model.eval()
    batch = next(it)
    out = model(batch[:,:-1])
    pred = out[:,-1,:]
    targ = batch[:,-1]
    loss = criterion(pred, targ)
    print(f'Loss: {loss.item():3.3f}')
    print('Input:', batch[:,:-1])
    print('Predicted output:', torch.argmax(out[:,-1,:], dim=-1))

def prepare_folders():
    try:
        os.makedirs("./saves/")
    except:
        pass

if __name__ == "__main__":
    seed_everything(534)
    prepare_folders()

    # train()
    infer()