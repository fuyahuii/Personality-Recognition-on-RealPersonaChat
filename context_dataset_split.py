import os
import sys
import random
import pandas as pd
import argparse
# deepcopy
import copy

apply_flag=True

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='../data/context_data/context_data_raw.csv', help='data file')
parser.add_argument('--reference_folder', type=str, default='../data/nocontext_data/split1', help='reference data file')
parser.add_argument('--output_folder', type=str, default='../data/context_data/split1', help='output data file')
args = parser.parse_args()

input_file = args.input_file
os.makedirs(args.output_folder, exist_ok=True)
output_train = os.path.join(args.output_folder, 'train.csv')
output_valid = os.path.join(args.output_folder, 'valid.csv')
output_test = os.path.join(args.output_folder, 'test.csv')

def check(pd_data):
    for index, row in pd_data.iterrows():
        file_name = row['file']
        if "csv" not in file_name:
            print(index)
            print (row)
            return 0
    return 1

def get_file_speaker_dict(df):
    file_speaker_dict = {}
    for index, row in df.iterrows():
        file_name = row['file']
        speaker_id = row['speaker_id']
        if file_name not in file_speaker_dict:
            file_speaker_dict[file_name] = []
        file_speaker_dict[file_name].append(speaker_id)
    return file_speaker_dict

def reform_df(df):
    new_df = {}
    for index, row in df.iterrows():
        file_name = row['file']
        dialogue = row['dialogue'].strip()
        dialogue = dialogue.replace('\n', '').strip()
        dialogue = dialogue.replace('"', '')
        dialogue = dialogue.replace('\r', '')
        if file_name not in new_df:
            new_df[file_name] = {}
            new_df[file_name]['file'] = row['file']
            new_df[file_name]['speaker_id'] = row['speaker_id']
            new_df[file_name]['dialogue'] = dialogue
            new_df[file_name]['n'] = row['n']
            new_df[file_name]['e'] = row['e']
            new_df[file_name]['o'] = row['o']
            new_df[file_name]['a'] = row['a']
            new_df[file_name]['c'] = row['c']
        else:
            new_df[file_name]['dialogue'] += dialogue
    new_df_list = new_df.values()
    return new_df_list

def main():
    reference_train = os.path.join(args.reference_folder, 'train.csv')
    reference_valid = os.path.join(args.reference_folder, 'valid.csv')
    reference_test = os.path.join(args.reference_folder, 'test.csv')

    reference_df_train = pd.read_csv(reference_train)
    reference_df_valid = pd.read_csv(reference_valid)
    reference_df_test = pd.read_csv(reference_test)

    ref_train_spk_ids = list(set(reference_df_train['speaker_id'].tolist()))
    ref_valid_spk_ids = list(set(reference_df_valid['speaker_id'].tolist()))
    ref_test_spk_ids = list(set(reference_df_test['speaker_id'].tolist()))

    print('reference train spk ids: ', len(ref_train_spk_ids))
    print('reference valid spk ids: ', len(ref_valid_spk_ids))
    print('reference test spk ids: ', len(ref_test_spk_ids))

    new_col_keys = ['file','speaker_id', 'dialogue', 'n', 'e', 'o', 'a', 'c']
    df = pd.read_csv(input_file)
    df.columns = new_col_keys
    file_speaker_dict = get_file_speaker_dict(df)

    train_file_ids = {}
    valid_file_ids = {}
    test_file_ids = {}
    for file_name in file_speaker_dict:
        speaker_ids = file_speaker_dict[file_name]
        first_spk_id = speaker_ids[0]
        second_spk_id = speaker_ids[1]
        if first_spk_id in ref_train_spk_ids:
            train_file_ids[file_name] = 1
        elif first_spk_id in ref_valid_spk_ids:
            valid_file_ids[file_name] = 1
        elif first_spk_id in ref_test_spk_ids:
            test_file_ids[file_name] = 1

    reformed_df_list = reform_df(df)

    train_df_data =[]
    valid_df_data =[]
    test_df_data =[]
    # check each row of df, if file_name in train_file_ids, add to train_df, same for valid and test
    for row in reformed_df_list:
        file_name = row['file']
        if file_name in train_file_ids:
            train_df_data.append(row)
        elif file_name in valid_file_ids:
            valid_df_data.append(row)
        elif file_name in test_file_ids:
            test_df_data.append(row)

    print ("generating pd data")
    train_df = pd.DataFrame(train_df_data, columns=new_col_keys)
    valid_df = pd.DataFrame(valid_df_data, columns=new_col_keys)
    test_df = pd.DataFrame(test_df_data, columns=new_col_keys)
    
    assert check(train_df) and check(valid_df) and check(test_df)
    print ("done")
    # save to csv
    train_df.to_csv(output_train, index=False)
    valid_df.to_csv(output_valid, index=False)
    test_df.to_csv(output_test, index=False)

    train_df = pd.read_csv(output_train)
    valid_df = pd.read_csv(output_valid)
    test_df = pd.read_csv(output_test)
    assert check(train_df) and check(valid_df) and check(test_df)
    print ("done2")




main()