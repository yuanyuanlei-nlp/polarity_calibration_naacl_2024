

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'


import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import math
import random
import time
import datetime
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import Counter
from scipy.optimize import linear_sum_assignment
from math import floor
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from accelerate import Accelerator



''' hyper-parameters '''

max_source_length = 512
max_target_length = 512
no_decay = ['bias', 'layer_norm.weight']
weight_decay = 1e-2
valid_steps = 1024
num_epochs = 10
batch_size = 1
gradient_accumulation_steps = 32
warmup_proportion = 0.05
lr = 1e-5



''' custom dataset '''

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

class custom_dataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        file_path = self.file_paths[idx]

        with open(file_path, "r") as in_file:
            input_summaries = json.load(in_file)

        positive_summaries = input_summaries['positive']
        negative_summaries = input_summaries['negative']

        if positive_summaries[-1] == ".":
            input_text = "Summarize the following customer reviews: " + positive_summaries + " " + negative_summaries
        else:
            input_text = "Summarize the following customer reviews: " + positive_summaries + ". " + negative_summaries

        if len(tokenizer(input_text).input_ids) > max_source_length:
            extra = len(tokenizer(input_text).input_ids) - max_source_length
            positive_summaries = tokenizer.decode(tokenizer(positive_summaries).input_ids[:-extra])
            input_text = "Summarize the following customer reviews: " + positive_summaries + ". " + negative_summaries


        source_input_ids = tokenizer(input_text, max_length=max_source_length, truncation=True, return_tensors="pt").input_ids


        with open(file_path.replace("conflicting_opinions_summaries", "min_10_max_100_revs_filt_complete"), "r") as in_file:
            label_summaries = json.load(in_file)

        verdict = label_summaries['website_summaries'][0]['verdict']
        pros = ". ".join(label_summaries['website_summaries'][0]['pros']) + "."
        cons = ". ".join(label_summaries['website_summaries'][0]['cons']) + "."
        label_text = verdict + " " + pros + " " + cons

        target_input_ids = tokenizer(label_text, max_length=max_target_length, truncation=True, return_tensors="pt").input_ids


        dict = {"source_input_ids": source_input_ids, "target_input_ids": target_input_ids}

        return dict



''' evaluate '''

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True, split_summaries=True)

def evaluate(model, eval_dataloader, verbose):

    model.eval()

    rouge1_list = []
    rouge2_list = []
    rougeL_list = []
    rougeLsum_list = []
    eval_loss_list = []

    for step, batch in enumerate(eval_dataloader):

        source_input_ids = batch['source_input_ids'][0]
        target_input_ids = batch['target_input_ids'][0]

        source_input_ids = source_input_ids.to(device)
        target_input_ids = target_input_ids.to(device)

        loss = model(input_ids=source_input_ids, labels=target_input_ids).loss
        eval_loss_list.append(loss.item())

        outputs = model.generate(source_input_ids, max_new_tokens=max_target_length, num_beams=5, no_repeat_ngram_size=3)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        label_text = tokenizer.decode(target_input_ids[0]).replace("</s>", "")

        rouge_metric = scorer.score(label_text, generated_text)
        rouge1_list.append(rouge_metric['rouge1'][2])
        rouge2_list.append(rouge_metric['rouge2'][2])
        rougeL_list.append(rouge_metric['rougeL'][2])
        rougeLsum_list.append(rouge_metric['rougeLsum'][2])


    rouge1 = sum(rouge1_list) / len(rouge1_list) if len(rouge1_list) != 0 else 0
    rouge2 = sum(rouge2_list) / len(rouge2_list) if len(rouge2_list) != 0 else 0
    rougeL = sum(rougeL_list) / len(rougeL_list) if len(rougeL_list) != 0 else 0
    rougeLsum = sum(rougeLsum_list) / len(rougeLsum_list) if len(rougeLsum_list) != 0 else 0
    rouge = (rouge1 + rouge2 + rougeL + rougeLsum) / 4
    eval_loss = sum(eval_loss_list) / len(eval_loss_list)

    if verbose:
        print("rouge1 is {:}, rouge2 is {:}, rougeL is {:}, rougeLsum is {:}, rouge is {:}, eval_loss is {:}".format(rouge1, rouge2, rougeL, rougeLsum, rouge, eval_loss))

    return rouge1, rouge2, rougeL, rougeLsum, rouge, eval_loss



''' train '''


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()


seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.use_deterministic_algorithms(True)


model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
model.cuda()


param_all = list(model.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_all if (not any(nd in n for nd in no_decay))],
     'lr': lr, 'weight_decay': weight_decay},
    {'params': [p for n, p in param_all if (any(nd in n for nd in no_decay))],
     'lr': lr, 'weight_decay': 0.0},]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-8)


train_path = "./AmaSum/conflicting_opinions_summaries/train/"
train_files_names = os.listdir(train_path)
train_file_paths = []
for file_i in range(len(train_files_names)):
    train_file_paths.append(train_path + train_files_names[file_i])

valid_path = "./AmaSum/conflicting_opinions_summaries/valid/"
valid_files_names = os.listdir(valid_path)
valid_file_paths = []
for file_i in range(len(valid_files_names)):
    valid_file_paths.append(valid_path + valid_files_names[file_i])

test_path = "./AmaSum/conflicting_opinions_summaries/test/"
test_files_names = os.listdir(test_path)
test_file_paths = []
for file_i in range(len(test_files_names)):
    test_file_paths.append(test_path + test_files_names[file_i])


train_dataset = custom_dataset(train_file_paths)
dev_dataset = custom_dataset(valid_file_paths)
test_dataset = custom_dataset(test_file_paths)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


num_train_steps = num_epochs * len(train_dataloader) // gradient_accumulation_steps # scheduler.step_with_optimizer = True by default
warmup_steps = int(warmup_proportion * num_train_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)


accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)


best_rouge = 0

for epoch_i in range(num_epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i, num_epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0
    num_batch = 0 # number of batch to calculate average loss
    total_num_batch = 0 # number of batch in this epoch


    for batch in train_dataloader:

        if total_num_batch % valid_steps == 0:

            # valid every valid_steps, actual update steps = valid_steps / gradient_accumulation_steps

            elapsed = format_time(time.time() - t0)
            avg_loss = total_loss / num_batch if num_batch != 0 else 0
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    loss average: {:.3f}'.format(total_num_batch, len(train_dataloader), elapsed, avg_loss))

            total_loss = 0
            num_batch = 0

            rouge1, rouge2, rougeL, rougeLsum, rouge, eval_loss = evaluate(model = model, eval_dataloader = dev_dataloader, verbose = 1)

            if rouge > best_rouge:
                torch.save(model.state_dict(), "./saved_models/base_model_best_rouge_AmaSum.ckpt")
                best_rouge = rouge


        model.train()

        source_input_ids = batch['source_input_ids'][0]
        target_input_ids = batch['target_input_ids'][0]

        with accelerator.accumulate(model):

            loss = model(input_ids=source_input_ids, labels=target_input_ids).loss

            total_loss += loss.item()
            num_batch += 1
            total_num_batch += 1

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


    # valid at the end of each epoch

    elapsed = format_time(time.time() - t0)
    avg_loss = total_loss / num_batch if num_batch != 0 else 0
    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    loss average: {:.3f}'.format(total_num_batch, len(train_dataloader), elapsed, avg_loss))

    total_loss = 0
    num_batch = 0

    rouge1, rouge2, rougeL, rougeLsum, rouge, eval_loss = evaluate(model = model, eval_dataloader = dev_dataloader, verbose = 1)

    if rouge > best_rouge:
        torch.save(model.state_dict(), "./saved_models/base_model_best_rouge_AmaSum.ckpt")
        best_rouge = rouge


# test

print("Best rouge is: {:}".format(best_rouge))
model.load_state_dict(torch.load("./saved_models/base_model_best_rouge_AmaSum.ckpt", map_location=device))
rouge1, rouge2, rougeL, rougeLsum, rouge, eval_loss = evaluate(model = model, eval_dataloader = test_dataloader, verbose = 1)







# stop here
