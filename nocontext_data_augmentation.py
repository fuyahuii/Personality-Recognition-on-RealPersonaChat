# augment the training data
import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import tqdm

# input output folder
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, default='../data/nocontext_data/split1', help='data file')
parser.add_argument('--output_folder', type=str, default='../data/nocontext_data/split3', help='output data file')
parser.add_argument('--data_num', type=int, default=100000, help='number of data to be augmented')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--random_flag', type=int, default=0, help='random flag')
parser.add_argument('--same_person', type=int, default=0, help='only combine data from the same person')
parser.add_argument('--real_time', type=int, default=1, help='generate real time data, where we dont use the full dialouge and use first ith sentences')
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder
data_num = args.data_num
seed = args.seed
random_flag = args.random_flag
same_person = args.same_person
real_time = args.real_time
random.seed(seed)
os.makedirs(output_folder, exist_ok=True)
splits = ['train', 'valid', 'test']

def drop_random_sentences(dialogue):
    sentences = dialogue
    if (sentences.startswith('[CLS]')):
        sentences = sentences[len('[CLS]'):]
    sentences = sentences.split('[SEP]')
    new_dialogue = ''
    for i in range(len(sentences)):
        if random.random() > 0.5:
            new_dialogue += sentences[i] + '[SEP]'
    new_dialogue = '[CLS]' + new_dialogue
    return new_dialogue

def value_merge(value1, value2, num1, num2):
    return (value1 * num1 + value2 * num2) / (num1 + num2)

def dialogue_merge(dialogue1, dialogue2):
    new_dialogue = ''
    sentences1 = dialogue1
    if (sentences1.startswith('[CLS]')):
        sentences1 = sentences1[len('[CLS]'):]
    sentences1 = sentences1.split('[SEP]')
    sentences2 = dialogue2
    if (sentences2.startswith('[CLS]')):
        sentences2 = sentences2[len('[CLS]'):]
    sentences2 = sentences2.split('[SEP]')
    num_from1 = 0
    num_from2 = 0
    ratio=0.5
    if random_flag == 1:
        ratio = random.random()
    if (real_time == 0):
        max_length = max(len(sentences1), len(sentences2))
    else:
        max_length = random.randint(3, max(len(sentences1), len(sentences2)))
    for i in range(max_length):
        if random.random() > ratio:
            if i < len(sentences1):
                new_dialogue += sentences1[i] + '[SEP]'
                num_from1 += 1
        else:
            if i < len(sentences2):
                new_dialogue += sentences2[i] + '[SEP]'
                num_from2 += 1
    if new_dialogue.startswith('[SEP]'):
        new_dialogue = new_dialogue[len('[SEP]'):]
    new_dialogue = '[CLS]' + new_dialogue
    return new_dialogue, num_from1, num_from2

def main():
    train_file = os.path.join(input_folder, 'train.csv')
    valid_file = os.path.join(input_folder, 'valid.csv')
    test_file = os.path.join(input_folder, 'test.csv')

    os.system('cp {} {}'.format(valid_file, output_folder))
    os.system('cp {} {}'.format(test_file, output_folder))

    train_df = pd.read_csv(train_file)
    columns = train_df.columns
    augmented_train_df = pd.DataFrame(columns=columns)
    # add original data
    augmented_train_df = augmented_train_df.append(train_df, ignore_index=True)
    speaker_ids = train_df['speaker_id']
    speaker_dict = {}
    for speaker_id in set(speaker_ids):
        speaker_dict[speaker_id] = train_df[train_df['speaker_id'] == speaker_id].index
    
    new_rows = []
    tot_num = 0
    while (tot_num < data_num):
        if (same_person == 1):
            speaker_id = random.choice(speaker_ids)
            # choose from speaker_dict
            indexes = speaker_dict[speaker_id]
            random_index1 = random.choice(indexes)
            random_index2 = random.choice(indexes)
        else:
            random_index1 = random.randint(0, len(train_df)-1)
            random_index2 = random.randint(0, len(train_df)-1)
        if random_index1 == random_index2:
            continue
        # merge data from two random rows
        new_row = {}
        new_row['dialogue'], num1, num2 = dialogue_merge(train_df.iloc[random_index1]['dialogue'], train_df.iloc[random_index2]['dialogue'])
        for key in ['n', 'e', 'o', 'a', 'c']:
            new_row[key] = value_merge(train_df.iloc[random_index1][key], train_df.iloc[random_index2][key], num1, num2)
        new_rows.append(new_row)
        
        if (tot_num % 1000 == 0):
            print(tot_num, '/', data_num)
        tot_num += 1
    augmented_train_df = augmented_train_df.append(new_rows, ignore_index=True)
    augmented_train_df.to_csv(os.path.join(output_folder, 'train.csv'), index=False)

main()
