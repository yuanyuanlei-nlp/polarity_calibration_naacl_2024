
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
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
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import math
import random
import time
import datetime
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
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
num_epochs = 5
batch_size = 1
gradient_accumulation_steps = 32
warmup_proportion = 0
lr = 1e-6
num_return_sequences = 1
top_p = 1.0
lambda_polarity = 1.0
lambda_similarity = 0.5
lambda_fluency = 0.2
lambda_task = 0.5
polarity_logits_flag = True



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()



''' target polarity score, calculated from input text '''

with open("./NeuS_data/target_polarity_score/target_polarity_score_save_dict.json", "r") as f:
    target_polarity_score_save_dict = json.load(f)

with open("./NeuS_data/target_polarity_score/target_polarity_score_logits_save_dict.json", "r") as f:
    target_polarity_score_logits_save_dict = json.load(f)



''' polarity model used to calculate polarity reward '''

polarity_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
polarity_model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base")
polarity_model.load_state_dict(torch.load("./saved_models/polarity_model.ckpt", map_location=device)) # load polarity model
polarity_model.cuda()
polarity_model.eval()


''' text similarity model used to calculate content similarity reward '''

similarity_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
similarity_model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base")
similarity_model.load_state_dict(torch.load("./saved_models/similarity_model.ckpt", map_location=device)) # load similarity model
similarity_model.cuda()
similarity_model.eval()


''' language fluency model used to calculate language fluency reward '''

fluency_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
fluency_model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base")
fluency_model.load_state_dict(torch.load("./saved_models/fluency_model.ckpt", map_location=device)) # load fluency model
fluency_model.cuda()
fluency_model.eval()



''' custom dataset '''

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

class custom_dataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        file_path = self.file_paths[idx]
        file_name = file_path.split("/")[-1][:-5]

        with open(file_path, "r") as in_file:
            data = json.load(in_file)

        source_list = data['source']
        source_text = "Summarize: " + source_list[0] + " </s> " + source_list[1] + " </s> " + source_list[2]
        source_input_ids = tokenizer(source_text, max_length=max_source_length, truncation=True, return_tensors="pt").input_ids # no padding

        target_text = data['target']
        target_input_ids = tokenizer(target_text, max_length=max_target_length, truncation=True, return_tensors="pt").input_ids

        # target polarity score

        target_polarity_score = target_polarity_score_save_dict[file_name]
        target_polarity_score_logits = target_polarity_score_logits_save_dict[file_name]
        target_polarity_score = torch.tensor(target_polarity_score)
        target_polarity_score_logits = torch.tensor(target_polarity_score_logits)

        dict = {"source_input_ids": source_input_ids, "target_input_ids": target_input_ids,
                "target_polarity_score": target_polarity_score, "target_polarity_score_logits": target_polarity_score_logits}

        return dict



''' evaluate '''

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True, split_summaries=True)

def evaluate(model, eval_dataloader, verbose):

    model.eval()

    target_polarity_score_list = []
    polarity_score_list = []
    rouge1_list = []
    rouge2_list = []
    rougeL_list = []
    rougeLsum_list = []

    for step, batch in enumerate(eval_dataloader):

        source_input_ids = batch['source_input_ids'][0]
        target_input_ids = batch['target_input_ids'][0]

        source_input_ids = source_input_ids.to(device)
        target_input_ids = target_input_ids.to(device)

        label_text = tokenizer.decode(target_input_ids[0]).replace("</s>", "")

        # target polarity score

        target_polarity_score = batch['target_polarity_score'][0].item()
        target_polarity_score_list.append(target_polarity_score)

        # generate text

        outputs = model.generate(source_input_ids, max_new_tokens=max_target_length, num_beams=5, no_repeat_ngram_size=3)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # calculate polarity distance

        polarity_inputs = polarity_tokenizer(generated_text, truncation=True, max_length=512, return_tensors="pt")
        polarity_inputs = polarity_inputs.to(device)
        polarity_outputs = polarity_model(**polarity_inputs)
        polarity_score = softmax_1(polarity_outputs.logits)[0, 1].item() / 2 + softmax_1(polarity_outputs.logits)[0, 2].item()
        polarity_score_list.append(polarity_score)

        # calculate rouge score

        rouge_metric = scorer.score(label_text, generated_text)
        rouge1_list.append(rouge_metric['rouge1'][2])
        rouge2_list.append(rouge_metric['rouge2'][2])
        rougeL_list.append(rouge_metric['rougeL'][2])
        rougeLsum_list.append(rouge_metric['rougeLsum'][2])


    rmse = mean_squared_error(target_polarity_score_list, polarity_score_list, squared=False)
    mae = mean_absolute_error(target_polarity_score_list, polarity_score_list)
    polarity_distance = (rmse + mae) / 2

    rouge1 = sum(rouge1_list) / len(rouge1_list) if len(rouge1_list) != 0 else 0
    rouge2 = sum(rouge2_list) / len(rouge2_list) if len(rouge2_list) != 0 else 0
    rougeL = sum(rougeL_list) / len(rougeL_list) if len(rougeL_list) != 0 else 0
    rougeLsum = sum(rougeLsum_list) / len(rougeLsum_list) if len(rougeLsum_list) != 0 else 0
    rouge = (rouge1 + rouge2 + rougeL + rougeLsum) / 4

    metric = rouge / polarity_distance

    if verbose:
        print("rmse is {:}, mae is {:}, polarity distance is {:}, rouge1 is {:}, rouge2 is {:}, rougeL is {:}, rougeLsum is {:}, rouge is {:}, metric is {:}".format(rmse, mae, polarity_distance, rouge1, rouge2, rougeL, rougeLsum, rouge, metric))

    return rmse, mae, polarity_distance, rouge1, rouge2, rougeL, rougeLsum, rouge, metric



''' train '''


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.use_deterministic_algorithms(True)


model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
model.load_state_dict(torch.load("./saved_models/base_model_best_NeuS.ckpt", map_location=device)) # load base summarizer
model.cuda()

softmax_1 = nn.Softmax(dim = 1)
softmax_1.cuda()


param_all = list(model.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_all if (not any(nd in n for nd in no_decay))],
     'lr': lr, 'weight_decay': weight_decay},
    {'params': [p for n, p in param_all if (any(nd in n for nd in no_decay))],
     'lr': lr, 'weight_decay': 0.0},]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-8)


train_path = "./NeuS_data/process_json/train/"
train_files_names = os.listdir(train_path)
train_file_paths = []
for file_i in range(len(train_files_names)):
    train_file_paths.append(train_path + train_files_names[file_i])

valid_path = "./NeuS_data/process_json/val/"
valid_files_names = os.listdir(valid_path)
valid_file_paths = []
for file_i in range(len(valid_files_names)):
    valid_file_paths.append(valid_path + valid_files_names[file_i])

test_path = "./NeuS_data/process_json/test/"
test_files_names = os.listdir(test_path)
test_file_paths = []
for file_i in range(len(test_files_names)):
    test_file_paths.append(test_path + test_files_names[file_i])


train_dataset = custom_dataset(train_file_paths)
dev_dataset = custom_dataset(dev_file_paths)
test_dataset = custom_dataset(test_file_paths)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


num_train_steps = num_epochs * len(train_dataloader) // gradient_accumulation_steps # scheduler.step_with_optimizer = True by default
warmup_steps = int(warmup_proportion * num_train_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)


accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)


best_polarity_distance = 10000
best_metric = -10000


for epoch_i in range(num_epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i, num_epochs))
    print('Training...')

    t0 = time.time()
    total_polarity_reward = 0
    total_similarity_reward = 0
    total_fluency_reward = 0
    total_task_loss = 0
    num_batch = 0 # number of batch to calculate average loss
    total_num_batch = 0  # number of batch in this epoch


    for batch in train_dataloader:

        if total_num_batch % valid_steps == 0:

            # valid every valid_steps, actual update steps = valid_steps / gradient_accumulation_steps

            elapsed = format_time(time.time() - t0)
            avg_polarity_reward = total_polarity_reward / num_batch if num_batch != 0 else 0
            avg_similarity_reward = total_similarity_reward / num_batch if num_batch != 0 else 0
            avg_fluency_reward = total_fluency_reward / num_batch if num_batch != 0 else 0
            avg_task_loss = total_task_loss / num_batch if num_batch != 0 else 0
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    polarity reward average: {:.3f}'.format(total_num_batch, len(train_dataloader), elapsed, avg_polarity_reward))
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    similarity reward average: {:.3f}'.format(total_num_batch, len(train_dataloader), elapsed, avg_similarity_reward))
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    fluency reward average: {:.3f}'.format(total_num_batch, len(train_dataloader), elapsed, avg_fluency_reward))
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    task loss average: {:.3f}'.format(total_num_batch, len(train_dataloader), elapsed, avg_task_loss))

            total_polarity_reward = 0
            total_similarity_reward = 0
            total_fluency_reward = 0
            total_task_loss = 0
            num_batch = 0

            rmse, mae, polarity_distance, rouge1, rouge2, rougeL, rougeLsum, rouge, metric = evaluate(model = model, eval_dataloader = dev_dataloader, verbose = 1)

            if polarity_distance < best_polarity_distance:
                torch.save(model.state_dict(), "./saved_models/calibrated_model_best_polarity_distance_NeuS.ckpt")
                best_polarity_distance = polarity_distance

            if metric > best_metric:
                torch.save(model.state_dict(), "./saved_models/calibrated_model_best_metric_NeuS.ckpt")
                best_metric = metric


        # train

        model.eval()

        source_input_ids = batch['source_input_ids'][0]
        target_input_ids = batch['target_input_ids'][0]

        label_text = tokenizer.decode(target_input_ids[0]).replace("</s>", "")

        # target polarity score logits

        target_polarity_score = batch['target_polarity_score'][0].item()
        target_polarity_score_logits = batch['target_polarity_score_logits'][0].item()

        # baseline generated text

        baseline_outputs = model.generate(input_ids=source_input_ids, max_new_tokens=max_target_length, num_beams=5, no_repeat_ngram_size=3)
        baseline_generated_text = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)

        # top-p sampling generated text

        outputs = model.generate(input_ids=source_input_ids, max_new_tokens=max_target_length,
                                 do_sample=True, top_p=top_p, top_k=0, num_return_sequences=num_return_sequences) # deactivate top-k sampling by setting top_k=0
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


        if baseline_generated_text == "" or generated_text == "":

            # exclude empty text

            baseline_polarity_reward = 0
            polarity_reward = -1
            baseline_similarity_reward = 0
            similarity_reward = -1
            baseline_fluency_reward = 0
            fluency_reward = -1

        else:

            # calculate polarity reward

            polarity_inputs = polarity_tokenizer(baseline_generated_text, truncation=True, max_length=512, return_tensors="pt")
            polarity_inputs = polarity_inputs.to(device)
            polarity_outputs = polarity_model(**polarity_inputs)
            baseline_polarity_score_logits = polarity_outputs.logits[0, 1].item() / 2 + polarity_outputs.logits[0, 2].item()
            baseline_polarity_score = softmax_1(polarity_outputs.logits)[0, 1].item() / 2 + softmax_1(polarity_outputs.logits)[0, 2].item()

            polarity_inputs = polarity_tokenizer(generated_text, truncation=True, max_length=512, return_tensors="pt")
            polarity_inputs = polarity_inputs.to(device)
            polarity_outputs = polarity_model(**polarity_inputs)
            polarity_score_logits = polarity_outputs.logits[0, 1].item() / 2 + polarity_outputs.logits[0, 2].item()
            polarity_score = softmax_1(polarity_outputs.logits)[0, 1].item() / 2 + softmax_1(polarity_outputs.logits)[0, 2].item()

            if polarity_logits_flag:
                baseline_polarity_reward = - abs(baseline_polarity_score_logits - target_polarity_score_logits)
                polarity_reward = - abs(polarity_score_logits - target_polarity_score_logits)
            else:
                baseline_polarity_reward = - abs(baseline_polarity_score - target_polarity_score)
                polarity_reward = - abs(polarity_score - target_polarity_score)

            # calculate similarity reward

            sent_pair = [(baseline_generated_text, label_text)]
            similarity_encoding = similarity_tokenizer(sent_pair, truncation='only_second', max_length=512, return_tensors="pt")
            baseline_similarity_reward = similarity_model(input_ids=similarity_encoding.input_ids.cuda(), attention_mask=similarity_encoding.attention_mask.cuda()).logits.item() / 5

            sent_pair = [(generated_text, label_text)]
            similarity_encoding = similarity_tokenizer(sent_pair, truncation='only_second', max_length=512, return_tensors="pt")
            similarity_reward = similarity_model(input_ids=similarity_encoding.input_ids.cuda(), attention_mask=similarity_encoding.attention_mask.cuda()).logits.item() / 5

            # calculate fluency reward

            fluency_inputs = fluency_tokenizer(baseline_generated_text, truncation=True, max_length=512, return_tensors="pt")
            fluency_inputs = fluency_inputs.to(device)
            fluency_outputs = fluency_model(**fluency_inputs)
            baseline_fluency_reward = softmax_1(fluency_outputs.logits)[0, 1].item() # [0, 1] fluent score, [0, 0] non-fluent score

            fluency_inputs = fluency_tokenizer(generated_text, truncation=True, max_length=512, return_tensors="pt")
            fluency_inputs = fluency_inputs.to(device)
            fluency_outputs = fluency_model(**fluency_inputs)
            fluency_reward = softmax_1(fluency_outputs.logits)[0, 1].item()


        model.train()

        with accelerator.accumulate(model):

            # top-p sampling generated text loss

            generated_input_ids = tokenizer(generated_text, max_length=max_target_length, truncation=True, return_tensors="pt").input_ids
            generated_input_ids = generated_input_ids.to(device)
            generated_loss = model(input_ids=source_input_ids, labels=generated_input_ids).loss

            # loss for three rewards

            polarity_loss = lambda_polarity * max(0, (polarity_reward - baseline_polarity_reward)) * generated_loss
            similarity_loss = lambda_similarity * max(0, (similarity_reward - baseline_similarity_reward)) * generated_loss
            fluency_loss = lambda_fluency * max(0, (fluency_reward - baseline_fluency_reward)) * generated_loss

            # task loss

            task_loss = lambda_task * max(0, (polarity_reward - baseline_polarity_reward)) * model(input_ids=source_input_ids, labels=target_input_ids).loss

            total_polarity_reward += baseline_polarity_reward
            total_similarity_reward += baseline_similarity_reward
            total_fluency_reward += baseline_fluency_reward
            total_task_loss += task_loss.item()
            num_batch += 1
            total_num_batch += 1

            loss = max(0, (polarity_reward - baseline_polarity_reward) / abs(polarity_reward - baseline_polarity_reward)) * (polarity_loss + similarity_loss + fluency_loss + task_loss) if abs(polarity_reward - baseline_polarity_reward) != 0 else 0 * (polarity_loss + similarity_loss + fluency_loss + task_loss)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


    # valid at the end of each epoch

    elapsed = format_time(time.time() - t0)
    avg_polarity_reward = total_polarity_reward / num_batch if num_batch != 0 else 0
    avg_similarity_reward = total_similarity_reward / num_batch if num_batch != 0 else 0
    avg_fluency_reward = total_fluency_reward / num_batch if num_batch != 0 else 0
    avg_task_loss = total_task_loss / num_batch if num_batch != 0 else 0
    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    polarity reward average: {:.3f}'.format(total_num_batch, len(train_dataloader), elapsed, avg_polarity_reward))
    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    similarity reward average: {:.3f}'.format(total_num_batch, len(train_dataloader), elapsed, avg_similarity_reward))
    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    fluency reward average: {:.3f}'.format(total_num_batch, len(train_dataloader), elapsed, avg_fluency_reward))
    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    task loss average: {:.3f}'.format(total_num_batch, len(train_dataloader), elapsed, avg_task_loss))

    total_polarity_reward = 0
    total_similarity_reward = 0
    total_fluency_reward = 0
    total_task_loss = 0
    num_batch = 0

    rmse, mae, polarity_distance, rouge1, rouge2, rougeL, rougeLsum, rouge, metric = evaluate(model = model, eval_dataloader = dev_dataloader, verbose = 1)

    if polarity_distance < best_polarity_distance:
        torch.save(model.state_dict(), "./saved_models/calibrated_model_best_polarity_distance_NeuS.ckpt")
        best_polarity_distance = polarity_distance

    if metric > best_metric:
        torch.save(model.state_dict(), "./saved_models/calibrated_model_best_metric_NeuS.ckpt")
        best_metric = metric


# test

print("Best metric is: {:}".format(best_metric))
model.load_state_dict(torch.load("./saved_models/calibrated_model_best_metric_NeuS.ckpt", map_location=device))
rmse, mae, polarity_distance, rouge1, rouge2, rougeL, rougeLsum, rouge, metric = evaluate(model = model, eval_dataloader = test_dataloader, verbose = 1)

print("Best polarity distance is: {:}".format(best_polarity_distance))
model.load_state_dict(torch.load("./saved_models/calibrated_model_best_polarity_distance_NeuS.ckpt", map_location=device))
rmse, mae, polarity_distance, rouge1, rouge2, rougeL, rougeLsum, rouge, metric = evaluate(model = model, eval_dataloader = test_dataloader, verbose = 1)






# stop here
