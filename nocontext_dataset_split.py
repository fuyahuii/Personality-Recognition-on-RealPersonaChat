# this performs the split of the nocontext data
## data augmentation method is also applied here
import os
import sys
import random
import pandas as pd
import argparse
# deepcopy
import copy

apply_flag=True

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='../data/nocontext_data/nocontext_data_sort.csv', help='data file')
parser.add_argument('--output_folder', type=str, default='../data/nocontext_data/split2', help='output data file')
parser.add_argument('--train_ratio', type=float, default=0.8, help='train data ratio')
parser.add_argument('--valid_ratio', type=float, default=0.1, help='valid data ratio')
parser.add_argument('--test_ratio', type=float, default=0.1, help='test data ratio')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()
input_file = args.input_file
os.makedirs(args.output_folder, exist_ok=True)
output_train = os.path.join(args.output_folder, 'train.csv')
output_valid = os.path.join(args.output_folder, 'valid.csv')
output_test = os.path.join(args.output_folder, 'test.csv')
seed = args.seed
random.seed(seed)

new_col_keys = ['file','speaker_id', 'dialogue', 'n', 'e', 'o', 'a', 'c']
df = pd.read_csv(input_file)
df.columns = new_col_keys
unique_speakers = list(set(df['speaker_id'].tolist()))

def search_best_split():
    mine=1000
    mini=-1
    minids=[]
    for i in range(0,100):
        shuffled_ids = copy.deepcopy(unique_speakers)
        random.seed(i)
        random.shuffle(shuffled_ids)
        minids.append(shuffled_ids)

        # split the file into train, valid, test, according to the ratio of speaker_id
        train_ids = shuffled_ids[:int(len(shuffled_ids)*args.train_ratio)]
        valid_ids = shuffled_ids[int(len(shuffled_ids)*args.train_ratio):int(len(shuffled_ids)*(args.train_ratio+args.valid_ratio))]
        test_ids = shuffled_ids[int(len(shuffled_ids)*(args.train_ratio+args.valid_ratio)):]
        assert len(train_ids)+len(valid_ids)+len(test_ids) == len(shuffled_ids)

        train_data = df[df['speaker_id'].isin(train_ids)]
        valid_data = df[df['speaker_id'].isin(valid_ids)]
        test_data = df[df['speaker_id'].isin(test_ids)]

        # how many data samples in train_data
        train_num = len(train_data)
        valid_num = len(valid_data)
        test_num = len(test_data)
        print (train_num, valid_num, test_num)
        error_abs = abs(train_num-len(df)*0.8)+abs(valid_num-len(df)*0.1)+abs(test_num-len(df)*0.1)
        print ('error from 8:1:1', error_abs)
        if error_abs < mine:
            mine = error_abs
            mini = i

    print ('min error', mine)
    print ('min i', mini)
    return mini, minids


def apply_best_split(mini, minids, apply_flag=False):
    shuffled_ids = minids[mini]
    train_ids = shuffled_ids[:int(len(shuffled_ids)*args.train_ratio)]
    valid_ids = shuffled_ids[int(len(shuffled_ids)*args.train_ratio):int(len(shuffled_ids)*(args.train_ratio+args.valid_ratio))]
    test_ids = shuffled_ids[int(len(shuffled_ids)*(args.train_ratio+args.valid_ratio)):]
    assert len(train_ids)+len(valid_ids)+len(test_ids) == len(shuffled_ids)

    train_data = df[df['speaker_id'].isin(train_ids)]
    valid_data = df[df['speaker_id'].isin(valid_ids)]
    test_data = df[df['speaker_id'].isin(test_ids)]

    # how many data samples in train_data
    train_num = len(train_data)
    valid_num = len(valid_data)
    test_num = len(test_data)
    print (train_num, valid_num, test_num)
    error_abs = abs(train_num-len(df)*0.8)+abs(valid_num-len(df)*0.1)+abs(test_num-len(df)*0.1)
    print ('final error from 8:1:1', error_abs)
    if (apply_flag==True):
        train_data.to_csv(output_train, index=False)
        valid_data.to_csv(output_valid, index=False)
        test_data.to_csv(output_test, index=False)

def main():
    mini, minids = search_best_split()
    apply_best_split(mini, minids, apply_flag=apply_flag)

main()