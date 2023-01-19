import pandas as pd
import numpy as np
import torch
import sys

#numpy.set_printoptions(threshold=sys.maxsize)
from transformers import AutoTokenizer, AutoModel

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
        converted_str = converted_str + str(org_list[i]) + "	"
    #return converted_str[:-1]
    return converted_str

def cut_off_dataset(df, frequency=5):
    source_cut_data = df[['target_id', 'source_id']].drop_duplicates(subset=['target_id', 'source_id'])
    source_cut = source_cut_data.source_id.value_counts()[(source_cut_data.source_id.value_counts() >= frequency)]
    source_id = np.sort(source_cut.keys())
    df = df.loc[df['source_id'].isin(source_id)]
    return df

f1 = open("../data/custom/PeerRead.cites", "w")
f2 = open("../data/custom/PeerRead.content", "w")
max_sen_length = 50
max_abstract_length = 200
max_seq_length = max_sen_length + max_abstract_length
dictionary = { }
idx_dictionary = { }
current_idx = 1
df = pd.read_csv('../data/custom/full_context_PeerRead.csv')
df = cut_off_dataset(df)
year = 2015
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

#df.sort_values(by='source_id', ascending=False)
for index, row in df.iterrows():
    ##unique id generator
    source_id = 0
    target_id = 0
    if (row['source_id'] in idx_dictionary.keys()):
        source_id  = idx_dictionary[row['source_id']]
    else:
        source_id = current_idx
        idx_dictionary[row['source_id']] = source_id
        current_idx = current_idx + 1

    if (row['target_id'] in idx_dictionary.keys()):
        target_id  = idx_dictionary[row['target_id']]
    else:
        target_id = current_idx
        idx_dictionary[row['target_id']] = target_id
        current_idx = current_idx + 1

    left = row["left_citated_text"]
    right = row["right_citated_text"]

    source_abstract = row["source_abstract"]
    target_abstract = row["target_abstract"]
    tokens_source_abstract = tokenizer.tokenize(source_abstract[0:max_abstract_length - 1])

    tokens_a = tokenizer.tokenize(left[-512:])
    tokens_b = tokenizer.tokenize(right[0:512])
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_sen_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_sen_length - 2:
            tokens_a = tokens_a[0:(max_sen_length - 2)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    if tokens_source_abstract:
        for token in tokens_source_abstract:
            tokens.append(token)
            segment_ids.append(2)
        tokens.append("[SEP]")
        segment_ids.append(2)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    input_ids_tensor = torch.from_numpy(np.array([input_ids], dtype=np.int))
    input_mask_tensor = torch.from_numpy(np.array([input_mask], dtype=np.int))

    encoded_input = {
        "input_ids": input_ids_tensor,
        "attention_mask": input_mask_tensor
    }
    with torch.no_grad():
        model_output = model(**encoded_input)

    #Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).tolist()[0]
    f1.write(str(target_id) + "	" + str(source_id) + "\n")
    f2.write(str(target_id) + " " + convert_list_to_string(sentence_embeddings) + "  " + str(source_id) + "\n")

f1.close()
f2.close()