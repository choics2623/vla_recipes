import copy
from datasets import load_dataset
import itertools
import torch

# check system prompt token seq or user prompt token seq is in the current token list
def check_header(targets, seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] in targets:
            return True
    return False

def replace_target(target, seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] == target:
            seq[i], seq[i+1], seq[i+3] = -100, -100, -100
    return seq

def tokenize_dialogs(dialogs, tokenizer):
    input_ids = tokenizer.apply_chat_template(dialogs, padding=True, return_tensors="pt")
    # print("#### dialogs")
    # print(dialogs)
    # print("#### text_prompt")
    # print(text_prompt)
    # print("#### text_prompt decode")
    # print(tokenizer.decode(text_prompt))

    # batch = tokenizer(text_prompt, padding = True, return_tensors="pt")
    labels = input_ids.clone()
    # 모델에 입력할 데이터 딕셔너리 생성
    batch = {
        "input_ids": input_ids,
        "labels": labels
    }
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i,n in enumerate(labels) if n == 128009] # turn 계산을 위해
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007],[128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx:idx+1]
            if check_header(prompt_header_seqs, current_seq):
                # 마스킹 해야하는 prompt header를 발견함
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
            else:
                last_idx = idx+1
        #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch

def get_custom_dataset(dataset_conifg, processor, split, split_ratio=0.9):
    # load_dataset will return DatasetDict that contains all the data in the train set
    dataset_dict = load_dataset("Chang-Su/SFTdata-Plus-v3-Bllossom-nonlogickor")
    dataset = dataset_dict['train']
    # Comment out the following line to use the full dataset, for quick testing only use 2000 samples
    dataset = dataset.select(range(20))
    dataset = dataset.train_test_split(test_size=1-split_ratio, shuffle=True, seed=42)[split]
    return dataset

class MultiturnDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right" # during training, one always uses padding on the right
    def __call__(self, samples):
        dialogs, images = [],[]
        role_mapping = {
            "human": "user",
            "gpt": "assistant"
        }
        for sample in samples:
            sample_list = sample["conversations"]
            dialog = []
            for sample_dict in sample_list:
                role = role_mapping.get(sample_dict["role"], sample_dict["role"])
                dialog += [
                    {"role":role,"content": sample_dict["content"].strip()},
                ]
            dialogs.append(dialog)
        return tokenize_dialogs(dialogs, self.tokenizer)

def get_data_collator(processor):
    return MultiturnDataCollator(processor)
