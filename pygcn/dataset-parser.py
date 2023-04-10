import json
import pandas as pd
import numpy as np
import torch
import sys
#numpy.set_printoptions(threshold=sys.maxsize)
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

dataset = sys.argv[1]
print("selected dataset is: " + sys.argv[1])
current_idx = 1
idx_dictionary = { }
f1 = open("../data/" + dataset + "/data.cites", "w")
f2 = open("../data/" + dataset + "/data.content", "w")
max_sen_length = 53
max_abstract_length = 423
max_title_length = 36
max_seq_length = max_sen_length + max_abstract_length + max_title_length

# Opening JSON files
c = open("../data/" + dataset + "/contexts.json")
p = open("../data/" + dataset + "/papers.json")
context = json.load(c)
papers = json.load(p)

tr = open("../data/" + dataset + "/train.json")
v = open("../data/" + dataset + "/val.json")
te = open("../data/" + dataset + "/test.json")
train = json.load(tr)
val = json.load(v)
test = json.load(te)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("current device is: " + device)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L12-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L12-v2")
model = model.to(device)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
    return sum_embeddings

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()

def convert_list_to_string(org_list, seperator=' '):
    converted_str = ""
    for i in range(0, len(org_list)):
        converted_str = converted_str + str(org_list[i]) + "    "
    #return converted_str[:-1]
    return converted_str

def write_citation(row):
    global current_idx
    global idx_dictionary
    ##unique id generator
    #source_id = row['refid']
    #target_id = row['citing_id']
    source_id = 0
    target_id = 0
    if (row['refid'] in idx_dictionary.keys()):
        source_id  = idx_dictionary[row['refid']]
    else:
        source_id = current_idx
        idx_dictionary[row['refid']] = source_id
        current_idx = current_idx + 1

    if (row['citing_id'] in idx_dictionary.keys()):
        target_id  = idx_dictionary[row['citing_id']]
    else:
        target_id = current_idx
        idx_dictionary[row['citing_id']] = target_id
        current_idx = current_idx + 1

    masked_text = row["masked_text"]
    #masked_text = masked_text.replace("OTHERCIT", "")
    #masked_text = masked_text.replace("()", "")
    masked_text_arr = masked_text.split("TARGETCIT")
    left = masked_text_arr[0]
    right = masked_text_arr[1]
    source_abstract = papers[row['refid']]["abstract"]
    source_title = papers[row['refid']]["title"]

    tokens_source_abstract = tokenizer.tokenize(source_abstract[0:512])
    tokens_source_title = tokenizer.tokenize(source_title[0:512])
    #print(len(tokens_source_title))
    #print(len(tokens_source_title))
    tokens_a = tokenizer.tokenize(left[-512:])
    tokens_b = tokenizer.tokenize(right[0:512])
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_sen_length - 3)
    tokens = []
    tokens.append("[CLS]")
    for token in tokens_a:
        tokens.append(token)
    tokens.append("[SEP]")

    for token in tokens_b:
        tokens.append(token)
    tokens.append("[SEP]")

    if len(tokens_source_abstract) > max_abstract_length - 1:
        tokens_source_abstract = tokens_source_abstract[0:(max_abstract_length - 1)]
    for token in tokens_source_abstract:
        tokens.append(token)
    tokens.append("[SEP]")

    if len(tokens_source_title) > max_title_length - 1:
        tokens_source_title = tokens_source_title[0:(max_title_length - 1)]
    for token in tokens_source_title:
        tokens.append(token)
    tokens.append("[SEP]")

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    input_ids_tensor = torch.from_numpy(np.array([input_ids], dtype=int)).to(device)
    input_mask_tensor = torch.from_numpy(np.array([input_mask], dtype=int)).to(device)

    encoded_input = {
        "input_ids": input_ids_tensor,
        "attention_mask": input_mask_tensor
    }
    with torch.no_grad():
        model_output = model(**encoded_input)

    #Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).tolist()[0]
    f1.write(str(target_id) + " " + str(source_id) + "\n")
    f2.write(str(target_id) + " " + convert_list_to_string(sentence_embeddings) + "  " + str(source_id) + "\n")

train_length = 0
val_length = 0
test_length = 0
print("generating embeddings for train dataset..")
for i in tqdm(range(len(train))):
    d = train[i]
    write_citation(context[d["context_id"]])
    train_length = train_length + 1
print("generating embeddings for validation dataset..")
for i in tqdm(range(len(val))):
    d = val[i]
    write_citation(context[d["context_id"]])
    val_length = val_length + 1
print("generating embeddings for test dataset..")
for i in tqdm(range(len(test))):
    d = test[i]
    write_citation(context[d["context_id"]])
    test_length = test_length + 1
f1.close()
f2.close()

statistics_file = open("../data/" + dataset + "/statistics-new.txt", "w")
s1 = ("train index:0" + " end:" + str(train_length) + "\n")
s2 = ("val index:" + str(train_length) + " end:" + str(train_length + val_length) + "\n")
s3 = ("test index:" + str(train_length + val_length) + " end:" + str(train_length + val_length + test_length) + "\n")
statistics_file.write(s1 + s2 + s3)
statistics_file.close()
print("done!")