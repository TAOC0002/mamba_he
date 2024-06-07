import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import sys
import urllib.request
from zipfile import ZipFile
sys.path.append('..')

import shutil

from time import perf_counter

import torch
import torch.nn.functional as F
from torcheval.metrics.text import Perplexity
import torch.nn as nn
from mamba_lm import from_pretrained
from mamba_lm import MambaLM, MambaLMConfig

from transformers import AutoTokenizer

import datasets

import numpy as np
import random


# Automated device selection based on available backends
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available() and False
        else "cpu"
    )

print(f"> Using {device} device")

def listdir_nohidden(path):
    files = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            files.append(f"{path}/{f}")
    return files

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_checkpoint(filepath, model, scheduler, optimizer):
    print(f"> Loading model from: {filepath}")
    try:
        loaded_checkpoint = torch.load(filepath, map_location=device)

        loaded_epoch = loaded_checkpoint['epoch']
        loaded_model = model
        loaded_scheduler = scheduler
        loaded_optimizer = optimizer

        loaded_model.load_state_dict(loaded_checkpoint['model_state'])
        if scheduler is not None:
            loaded_scheduler.load_state_dict(loaded_checkpoint['scheduler_state'])
        if optimizer is not None:
            loaded_optimizer.load_state_dict(loaded_checkpoint['optimizer_state'])
        
        print("> Loaded model")
        return True, loaded_epoch, loaded_model, loaded_scheduler, loaded_optimizer
    except Exception as e:
        print("> Cannot load model")
        return False, 0, model, scheduler, optimizer

def calculate_perplexity(loss):
    return math.exp(loss)

def load_enwiki8_dataset():
    url = "http://mattmahoney.net/dc/enwik8.zip"
    urllib.request.urlretrieve(url, "saves/enwik8.zip")

    with ZipFile("saves/enwik8.zip") as f:
        data = f.read("enwik8").decode("utf-8")

    return data

def encode_dataset(tokenizer, text_data, seq_length):
    def batch_encode(tokenizer, text_data, batch_size=1000):
        # Tokenize in batches
        batched_input_ids = []
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i+batch_size]
            inputs = tokenizer(batch, add_special_tokens=True, truncation=True,
                               padding='max_length', max_length=seq_length,
                               return_tensors='pt')
            batched_input_ids.append(inputs['input_ids'])
        return torch.cat(batched_input_ids)

    # Assuming enwiki8_data is a list of sentences
    input_ids = batch_encode(tokenizer, text_data)
    global vocab_size
    vocab_size = len(tokenizer.vocab)
    print(f"vocab_size = {vocab_size}")
    encoded_inputs = input_ids.long()

    # must compare with raw `input_ids` instead of tokenized `encoded_inputs`
    attention_mask = (input_ids != tokenizer.pad_token_id).type(encoded_inputs.dtype)
    print(f"attention_mask.shape = {attention_mask.shape}")
    print(f"encoded_inputs.shape = {encoded_inputs.shape}")

    return encoded_inputs, attention_mask

def train(pretrained=False):
    # Training parameters
    '''
    epochs - number of epochs during training
    batch_size - size of a single batch during training
    seq_length - number of tokens in model's context during training
    learning_rate - initial learning rate of the training
    model_path - path to the saved weights; if empty it'll save there new weights during training
    backup_path - path to the backup of a model. if None - no backup is created
    '''
    epochs = 25
    batch_size = 128 #32 for 24GB and 130m model
    seq_length = 128
    learning_rate = 1e-2
    model_path = f'saves/model.pth'
    backup_path = f"saves/model-b.pth"
    best_path = f"saves/model-best.pth"
    log_path = f"saves/log.txt"

    ## Load datasets
    # encoded_inputs_file = 'saves/encoded_inputs_mamba.pt'
    # attention_mask_file = 'saves/encoded_attention_mamba.pt'
    # if os.path.exists(encoded_inputs_file):
    #     print("Loading pre-tokenized data...")
    #     encoded_inputs = torch.load(encoded_inputs_file)
    #     attention_mask = torch.load(attention_mask_file)
    # else:
    #     print(f"Download and extract enwiki8 data")
    #     enwiki8_data = load_enwiki8_dataset()
    #     print("Tokenizer in action ...")
    #     encoded_inputs, attention_mask = encode_dataset(tokenizer, enwiki8_data, seq_length)
    #     torch.save(encoded_inputs, encoded_inputs_file)
    #     torch.save(attention_mask, attention_mask_file)
    #     print(f"finished tokenizing data")

    
    # dataset = datasets.load_dataset('wikitext', 'wikitext-2-v1')
    dataset = datasets.load_dataset('text', data_files={'train': ['../Mamba_SSM/enwiki8_train.txt'],
                                                        'validation': ['../Mamba_SSM/enwiki8_eval.txt']})

    # Usage of custom txt datasets
    '''
    In order to load custom training data add filepaths to the list
    For example to use one txt file change the name of the file in the command below:

    dataset = datasets.load_dataset('text', data_files={'train': ['austen-emma.txt']})

    For more files add them to the list after comma

    https://huggingface.co/docs/datasets/v1.2.1/loading_datasets.html
    '''

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') # EleutherAI/gpt-neox-20b
    # Add eos tokens
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    # Map tokenizer to the dataset
    tokenize_data = lambda example, tokenizer: {'tokens': tokenizer.tokenize(example['text'], truncation=True)} 
    tokenized_dataset = dataset.map(tokenize_data, remove_columns=['text'], 
        fn_kwargs={'tokenizer': tokenizer})
    

    # custom dataset

    # Prepare and load tokenizer's vocabulary for later use
    vocab = tokenizer.vocab
    print(f"Vocab size: {len(vocab)}")

    
    # Select the wanted model
    '''
    If pretrained==True - the script loads pretrained mamba weights specified by the string.
    If pretrained==False - the script creates a new MambaLM model with parameters specified in config variable
    '''
    if pretrained:
        model = from_pretrained('state-spaces/mamba-130m').to(device)
    else:
        config = MambaLMConfig(d_model=16, n_layers=2, dt_rank=1, d_conv=4, vocab_size=len(tokenizer.vocab))
        model = MambaLM(config).to(device)

    # Create optimizer and pass the model
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                            optim,
                                                            mode='min',
                                                            factor=0.1, #factor by which the lr is multiplied
                                                            patience=2,
                                                        )

    # Load previously trained weights
    ''' 
    If the model is the same it will load previous weights located in specified path
    If the model differs or the path is empty it'll skip loading and train from scratch
    '''
    _, epoch, model, scheduler, optim = load_checkpoint(model_path, model, scheduler, optim)

    
    # Create data loader functions
    def get_data(dataset, vocab, batch_size):
        data = []                                   
        for example in dataset:
            if example['tokens']:
                tokens = [vocab[token] for token in example['tokens']]
                data.extend(tokens)
        
        data = torch.LongTensor(data)              
        num_batches = data.shape[0] // batch_size 
        data = data[:num_batches * batch_size]                       
        data = data.view(batch_size, num_batches)
        return data     

    def get_batch(data, seq_len, idx):
        src = data[:, idx:idx+seq_len]
        target = data[:, idx+1:idx+seq_len+1]
        return src, target

    # Get data and apply tokenizer to the dataset
    train_data = get_data(tokenized_dataset['train'], vocab, batch_size)
    valid_data = get_data(tokenized_dataset['validation'], vocab, batch_size)
    print(f"Train data length before: {train_data.shape[-1]//seq_length}")
    print(f"Valid data length before: {valid_data.shape[-1]//seq_length}")
    metric=Perplexity()
    
    # Training loop
    leap = 1
    best_perplex = math.inf
    t0_start = perf_counter()
    for z in range(epoch, epochs):
        avg_loss = 0
        print(f"\n> Epoch {z+1}/{epochs}")

        t2_start = perf_counter()
        model.train()
        for idx in range(0,train_data.shape[-1]-seq_length,leap):
            t1_start = perf_counter()

            input, output = get_batch(train_data, seq_length, idx)
            output = output.reshape(-1)
            input = input.to(device)
            output = output.to(device)

            logits = model(input)

            # If the batch is not complete - skip
            if (logits.view(-1, logits.size(-1)).shape[0] != output.view(-1).shape[0]):
                print(logits.view(-1, logits.size(-1)).shape[0], output.view(-1).shape[0])
                print("skip")
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), output)
                avg_loss += loss.item()

                optim.zero_grad()
                loss.backward()
                optim.step()

                t1_stop = perf_counter()

                # Print the progress during training
                batch_no = (idx+1)//leap + 1

            # Increment idx
            if idx + seq_length >= train_data.shape[-1]:
                idx = 0
                break

        print(f"> Train Batch: {batch_no}/{train_data.shape[-1]//leap+1} loss: {avg_loss/(batch_no):.5f} time: {t1_stop-t1_start:.2f} sec ", end="")
        # Update schedulers
        # optim.step()
        scheduler.step(avg_loss/batch_no)

        ## EVALUATION ##
        model.eval()
        with torch.no_grad():
            eval_loss = 0
            for idx in range(0,valid_data.shape[-1]-seq_length,leap):
                t1_start = perf_counter()
                input, output = get_batch(valid_data, seq_length, idx)
                output = output.reshape(-1).unsqueeze(1)
                input = input.to(device)
                logits = model(input).reshape(output.shape[0],-1).unsqueeze(1).cpu()
                metric.update(logits, output)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), output.squeeze(1))
                eval_loss += loss.item()
                batch_no = (idx+1)//leap + 1
                t1_stop = perf_counter()
            print(f"> Eval Batch: {batch_no}/{valid_data.shape[-1]//leap+1} loss: {eval_loss/(batch_no):.5f} time: {t1_stop-t1_start:.2f} sec ", end="")
            eval_loss = eval_loss / batch_no

        # Save the model
        checkpoint = {
            'epoch': z,
            'model_state': model.state_dict(),
            'optimizer_state': optim.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        }
        # Create backup file
        if backup_path is not None and os.path.isfile(model_path):
            shutil.copyfile(model_path, backup_path)
        torch.save(checkpoint, model_path)

        perplex_torch = metric.compute()
        perplex = calculate_perplexity(eval_loss)
        metric.reset()
        if perplex <= best_perplex:
            print('Saving the best checkpoint...')
            best_perplex = perplex
            torch.save(checkpoint, best_path)
        print('Current perplexity:', perplex, 'Current perplexity [torch]:', perplex_torch, ', best perplexity:', best_perplex)
            
        t2_stop = perf_counter()
        print(f"\n> Epoch time: {t2_stop - t2_start:.3f} seconds at epoch {z+1}")

    t0_stop = perf_counter()
    print(f"\n> Finished training in: {t0_stop-t0_start} seconds")


# Sample generation based on trained model
def my_gen(pretrained=False):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    # Load the model
    if pretrained:
        model = from_pretrained('state-spaces/mamba-130m').to(device)
    else:
        config = MambaLMConfig(d_model=16, n_layers=2, dt_rank=1, d_conv=4, vocab_size=len(tokenizer.vocab))
        model = MambaLM(config).to(device)

    # Load weights
    isLoaded, _, model, *_ = load_checkpoint(f'saves/model-best.pth', model, None, None)
    if (not isLoaded):
        return

    model.eval()
    # Generate text based on prompt
    output = model.generate(tokenizer, "Proudhon's philosophy of property is complex: it was developed in a number of works over his "
                            , num_tokens=1
                            , sample=False
                            , temperature=1
                            , top_k=None)

    print(f"Answer: {output}")

def prepare_folders():
    try:
        os.makedirs("./saves/")
    except:
        pass

if __name__ == "__main__":
    seed_everything(534)
    prepare_folders()

    train(pretrained=False)
    my_gen(pretrained=False)