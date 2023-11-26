import os
import sys
import time
import tqdm
import glob
import copy
import logging
import argparse

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GraphConv,GATv2Conv,RGCNConv
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
# folders
parser.add_argument('--data_folder', type=str, default='./data/monologue_split_orig', help='input data folder')
parser.add_argument('--model_folder', type=str, default='./model/exp_name', help='model folder')
parser.add_argument('--log_file', type=str, default='./log/exp_name.log', help='log file')
parser.add_argument('--train_filename', type=str, default='train.csv', help='train data file')
parser.add_argument('--valid_filename', type=str, default='valid.csv', help='valid data file')
parser.add_argument('--test_filename', type=str, default='test.csv', help='test data file')
# flags
parser.add_argument('--train_flag', type=int, default=1, help='whether to train the model')
parser.add_argument('--test_flag', type=int, default=1, help='whether to test the model')
parser.add_argument('--real_time_flag', type=int, default=1, help='whether to test the model in real time setting, which means using only first several turns of a dialogue')

#parser.add_argument('--data_split', type=str, default='split1', help='which data split to use')
parser.add_argument('--max_length', type=int, default=512, help='max length of input sequence')

parser.add_argument('--base_model_name', type=str, default='studio-ousia/luke-japanese-base', help='name of pretrained model in huggingface')
parser.add_argument('--use_dropout', type=int, default=0, help='whether to use dropout after base model')
parser.add_argument('--multilinear', type=int, default=1, help='whether to use multiple linear layers after base model 0 means single linear layer')
parser.add_argument('--hidden_size', type=int, default=16, help='hidden size of the first linear layer if multilinear is 1')

parser.add_argument('--critertion_type', type=str, default='mae', help='criterion type, choose from mae mse')
parser.add_argument('--multitask', type=str, default='11111', help='whether to use multitask learning, 1 means use, 0 means not use, the order is n e o a c')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--warmup_steps', type=int, default=150)
parser.add_argument('--max_epoch', type=int, default=20, help='max epoch for training')
parser.add_argument('--patience', type=int, default=3, help='patience for early stopping')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')

parser.add_argument('--context', type=int, default=0, help='whether to use dialogue scenario. 0 means monologue, 1 means dialogue')
parser.add_argument('--context_model_type', type=str, default='linear', help='context model type, choose from linear gcn-nospk2pred-lastnode')
parser.add_argument('--head_num', type=int, default=3, help='head number of gcn')
parser.add_argument('--model_variant', type=str, default='hcgnn', help='model variant, choose from gcn gatv2 rgcn hcgnn22 hcgnn3rel')

parser.add_argument('--ensemble', type=int, default=0, help='whether to use ensemble model')
parser.add_argument('--ensemble_model_folder', type=str, default='/share03/song/person_rec/model/nocontext_ensemble', help='folder of ensemble models')

args = parser.parse_args()

# variables
input_data_folder = args.data_folder
model_folder = args.model_folder
log_file = args.log_file
train_filename = args.train_filename
valid_filename = args.valid_filename
test_filename = args.test_filename

train_file = os.path.join(input_data_folder, train_filename)
valid_file = os.path.join(input_data_folder, valid_filename)
test_file = os.path.join(input_data_folder, test_filename)
os.makedirs(model_folder, exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

train_flag = args.train_flag
test_flag = args.test_flag
real_time_flag = args.real_time_flag
max_length = args.max_length

# settings for base model
base_model_name = args.base_model_name
dropout = args.use_dropout
multilinear = args.multilinear
hidden_size = args.hidden_size

# settings for training
critertion_type = args.critertion_type
multitask_str = args.multitask
multitask = [int(flag) for flag in multitask_str]
lr = args.lr
warmup_steps = args.warmup_steps
max_epoch = args.max_epoch
patience = args.patience
batch_size = args.batch_size

# settings for dialogue scenario model
context = args.context
context_str = 'context' if context == 1 else 'nocontext'
context_model_type = args.context_model_type
if ('gcn' in context_model_type):
    max_length = 64
head_num = args.head_num
model_variant = args.model_variant
pad_length = -1

# settings for ensemble (not really working well)
ensemble = args.ensemble
ensemble_model_folder = args.ensemble_model_folder

# constants for trail normalization
min_label = 1
max_label = 7

# constants (not used)
#exp_folder = './'
#input_data_folder = os.path.join(exp_folder, 'data', f'{context_str}_data', data_split)
#name = f'{context_str}_{data_split}_contextmodeltype-{context_model_type}_headnum{head_num}_modelvariant-{model_variant}_1'
#name = f'{context_str}_{data_split}_{context_model_type}_{model_variant}'
#model_folder = os.path.join(exp_folder, 'model', name)
#log_file = os.path.join(exp_folder, 'log', f'{name}.log')

def gene_data_from_csv(file_path, tokenizer, max_sentence_num=0):
    csv_data = pd.read_csv(file_path)
    dict_data = []
    for i in range(len(csv_data)):
        data_sample = csv_data.iloc[i]
        text = data_sample['dialogue']
        text = text.replace('\n', '')

        if (context == 0):
            if (max_sentence_num > 0):
                if (text.startswith('[CLS]')):
                    text = text[len('[CLS]'):]
                if (text.startswith('[SEP]')):
                    text = text[len('[SEP]'):]
                sentences = text.split('[SEP]')
                sentences = sentences[:max_sentence_num]
                text = '[CLS]' + '[SEP]'.join(sentences)
            text = text.replace('[CLS]', tokenizer.cls_token)
            text = text.replace('[SEP]', tokenizer.sep_token)
            if (not text.startswith(tokenizer.cls_token)):
                text = tokenizer.cls_token + text
        else:
            if (max_sentence_num > 0):
                if (text.startswith('[CLS]')):
                    text = text[len('[CLS]'):]
                if (text.startswith('[SPK1]')):
                    text = text[len('[SPK1]'):]
                sentences = text.split('[SPK1]')
                sentences = sentences[:max_sentence_num]
                text = '[CLS]' + '[SPK1]'.join(sentences)
            text = text.replace('[CLS]', tokenizer.cls_token)
            if (not text.startswith(tokenizer.cls_token)):
                text = tokenizer.cls_token + text
        assert text.startswith(tokenizer.cls_token)

        labels = [data_sample['n'], data_sample['e'], data_sample['o'], data_sample['a'], data_sample['c']]
        normalized_labels = [(label - min_label) / (max_label - min_label) for label in labels]
        dict_data.append({
            'text': text,
            'labels': normalized_labels,
            })
    return dict_data

def gene_context_data(data, pad_length, tokenizer):
    context_data = []
    for sample in data:
        text = sample['text']
        text = text.replace('[CLS]', tokenizer.cls_token)
        text = text.replace('[SEP]', tokenizer.sep_token)
        text = text.replace(tokenizer.cls_token, '').replace(tokenizer.sep_token, '')

        text_tmp = text.replace('[SPK1]', tokenizer.sep_token).replace('[SPK2]', tokenizer.sep_token)
        if text_tmp.startswith(tokenizer.sep_token):
            text_tmp = text_tmp[len(tokenizer.sep_token):]
        sentences = text_tmp.split(tokenizer.sep_token)
        sentences_with_speaker = []
        for i, sentence in enumerate(sentences):
            if sentence == '':
                continue
            if sentence.startswith(tokenizer.cls_token):
                sentence = sentence[len(tokenizer.cls_token):]
            if i % 2 == 0:
                s = '[SPK1]' + sentence
            else:
                s = '[SPK2]' + sentence
            sentences_with_speaker.append(s)
        if len(sentences_with_speaker) < pad_length:
            sentences_with_speaker += ['<pad>' for _ in range(pad_length - len(sentences_with_speaker))]
        else:
            sentences_with_speaker = sentences_with_speaker[:pad_length]

        context_data.append({
            'text': sentences_with_speaker, # each sentence is <s>[SPK*]text
            'labels': sample['labels'],
            })
    return context_data

class ConversationDataset(Dataset):
	def __init__(self, data, tokenizer,max_length):
		self.data = data
		self.tokenizer = tokenizer
		self.max_length=max_length
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		sample = self.data[idx]
		labels = torch.tensor(sample['labels'], dtype=torch.float)
		input_ids=[]
		attention_mask=[]
		length=0
		for text_row in sample['text']:
			if text_row !='<pad>':
				length+=1
			encoding = self.tokenizer(text_row, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt", add_special_tokens=True)
			input_ids.append(encoding['input_ids'].squeeze())
			attention_mask.append(encoding['attention_mask'].squeeze())
		
		return {
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'labels': torch.tensor(labels),  
			'current_length':length
		}

class MultiTaskDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        encoding = self.tokenizer(sample['text'], max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt', add_special_tokens=False)
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(sample['labels'], dtype=torch.float)
        }

class MultiTaskModel(nn.Module):
    def __init__(self, base_model, dropout_flag, multilinear_flag):
        super(MultiTaskModel, self).__init__()
        self.base_model = base_model
        self.dropout_flag = dropout_flag
        self.multilinear_flag = multilinear_flag

        self.dropout_layer = nn.Dropout(0.1)

        if (self.multilinear_flag == 0):
            self.task_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(base_model.config.hidden_size, 1),
                ) for _ in range(5)
            ])
        else:
            self.task_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(base_model.config.hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1),
                ) for _ in range(5)
            ])
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        if self.dropout_flag == 1:
            pooled_output = self.dropout_layer(pooled_output)
        regression_outputs = []
        for task_head in self.task_heads:
            regression_outputs.append(task_head(pooled_output))
        return torch.cat(regression_outputs, dim=1)

class GCN(nn.Module):
    def __init__(self,input_dim, hidden_dim1, output_dim, relations, heads):
        super(GCN, self).__init__()
        self.relations=1
        self.output_dim=output_dim
        self.conv2 = nn.ModuleList([GraphConv(input_dim, hidden_dim1) for _ in range(self.relations)])
        self.conv3 = nn.ModuleList([GraphConv(hidden_dim1, output_dim) for _ in range(self.relations)])

    def forward(self, x, relationsedge_indices_relations):
        relation_outputs=[]
        for i, conv_layer in enumerate(self.conv2):
            relation_output=(conv_layer(x, relationsedge_indices_relations[-1]))
            relation_output=F.relu(relation_output)
            relation_output=self.conv3[i](relation_output, relationsedge_indices_relations[-1])

            relation_output=relation_output.reshape(-1,1,self.output_dim)
            relation_outputs.append(relation_output)
        x=torch.cat(relation_outputs, dim=1)	
        return x

class GAT(nn.Module):
    def __init__(self,input_dim, hidden_dim1, output_dim, relations, heads):
        super(GAT, self).__init__()
        self.relations=1
        self.heads=heads
        self.output_dim=hidden_dim1*self.heads
        self.conv1 = nn.ModuleList([GATv2Conv(input_dim, hidden_dim1,heads=self.heads) for _ in range(self.relations)])

    def forward(self, x, relationsedge_indices_relations):
        relation_outputs=[]
        for i, conv_layer in enumerate(self.conv1):
            relation_output=(conv_layer(x, relationsedge_indices_relations[-1]))
            relation_output=relation_output.reshape(-1,1,self.output_dim)
            relation_outputs.append(relation_output)
        x=torch.cat(relation_outputs, dim=1)				
        return x

class RGCN (nn.Module):
    def __init__(self,input_dim, hidden_dim1, output_dim, relations, heads):
        super(RGCN, self).__init__()
        self.relations=1
        self.heads=heads
        self.output_dim=output_dim
        self.conv1 = nn.ModuleList([RGCNConv(input_dim, hidden_dim1,2) for _ in range(self.relations)])
        self.conv2 = nn.ModuleList([GraphConv(hidden_dim1, output_dim) for _ in range(self.relations)])

    def forward(self, x, relationsedge_indices_relations,edge_type):
        relation_outputs=[]
        for i, conv_layer in enumerate(self.conv1):
            relation_output=(conv_layer(x, relationsedge_indices_relations[-1],edge_type))
            relation_output=F.relu(relation_output)
            relation_output=self.conv2[i](relation_output, relationsedge_indices_relations[-1])
            relation_output=relation_output.reshape(-1,1,self.output_dim)
            relation_outputs.append(relation_output)
        x=torch.cat(relation_outputs, dim=1)
        return x
    
class GATv2GCN22(nn.Module):
    def __init__(self,input_dim, hidden_dim1, output_dim, relations, heads):
        super(GATv2GCN22, self).__init__()
        self.relations=relations
        self.heads=heads
        self.output_dim=output_dim
        self.conv1 = nn.ModuleList([GATv2Conv(input_dim, hidden_dim1,heads=self.heads) for _ in range(self.relations)])
        self.conv2 = nn.ModuleList([GATv2Conv(hidden_dim1*self.heads, hidden_dim1,heads=self.heads) for _ in range(self.relations)])
        self.conv3 = nn.ModuleList([GraphConv(hidden_dim1*self.heads, hidden_dim1) for _ in range(self.relations)])
        self.conv4 = nn.ModuleList([GraphConv(hidden_dim1, output_dim) for _ in range(self.relations)])

    def forward(self, x, relationsedge_indices_relations):
        relation_outputs=[]
        for i, conv_layer in enumerate(self.conv1):
            relation_output=(conv_layer(x, relationsedge_indices_relations[i]))
            relation_output=F.relu(relation_output)
            relation_output=self.conv2[i](relation_output, relationsedge_indices_relations[i])
            relation_output=F.relu(relation_output)
            relation_output=self.conv3[i](relation_output, relationsedge_indices_relations[i])
            relation_output=F.relu(relation_output)
            relation_output=self.conv4[i](relation_output, relationsedge_indices_relations[i])

            relation_output=relation_output.reshape(-1,1,self.output_dim)
            relation_outputs.append(relation_output)
        x=torch.cat(relation_outputs, dim=1)				
        return x

class GATv2GCN(nn.Module):
    """
        GATv2+GCN model with 4/6 relations (GAT->GCN)
    """
    def __init__(self,input_dim, hidden_dim1, output_dim, relations, heads):
        super(GATv2GCN, self).__init__()
        self.relations=relations
        self.heads=heads
        self.output_dim=output_dim
        self.conv1 = nn.ModuleList([GATv2Conv(input_dim, hidden_dim1,heads=self.heads) for _ in range(self.relations)])
        self.conv2 = nn.ModuleList([GraphConv(hidden_dim1*self.heads, hidden_dim1) for _ in range(self.relations)])
        self.conv3 = nn.ModuleList([GraphConv(hidden_dim1, output_dim) for _ in range(self.relations)])

    def forward(self, x, relationsedge_indices_relations):
        relation_outputs=[]
        for i, conv_layer in enumerate(self.conv1):
            relation_output=(conv_layer(x, relationsedge_indices_relations[i]))
            relation_output=F.relu(relation_output)
            relation_output=self.conv2[i](relation_output, relationsedge_indices_relations[i])
            relation_output=F.relu(relation_output)
            relation_output=self.conv3[i](relation_output, relationsedge_indices_relations[i])

            relation_output=relation_output.reshape(-1,1,self.output_dim)
            relation_outputs.append(relation_output)
        x=torch.cat(relation_outputs, dim=1)				
        return x

class GATv2GCN3REL(nn.Module):
    def __init__(self,input_dim, hidden_dim1, output_dim, relations, heads):
        super(GATv2GCN3REL, self).__init__()
        self.relations=relations
        self.heads=heads
        self.output_dim=output_dim
        self.conv1 = nn.ModuleList([GATv2Conv(input_dim, hidden_dim1,heads=self.heads) for _ in range(self.relations)])
        self.conv2 = nn.ModuleList([GraphConv(hidden_dim1*self.heads, hidden_dim1) for _ in range(self.relations)])
        self.conv3 = nn.ModuleList([GraphConv(hidden_dim1, output_dim) for _ in range(self.relations)])
        self.conv4 = RGCNConv(input_dim, hidden_dim1,2) 
        self.conv5 = GraphConv(hidden_dim1,output_dim)

    def forward(self, x, relationsedge_indices_relations,edge_type):
        relation_outputs=[]
        for i, conv_layer in enumerate(self.conv1):
            relation_output=(conv_layer(x, relationsedge_indices_relations[i]))
            relation_output=F.relu(relation_output)
            relation_output=self.conv2[i](relation_output, relationsedge_indices_relations[i])
            relation_output=F.relu(relation_output)
            relation_output=self.conv3[i](relation_output, relationsedge_indices_relations[i])

            relation_output=relation_output.reshape(-1,1,self.output_dim)
            relation_outputs.append(relation_output)

        relation_output_rgcn=self.conv4(x, relationsedge_indices_relations[-1],edge_type)	
        relation_output_rgcn=self.conv5(relation_output_rgcn,relationsedge_indices_relations[-1])	
        relation_output_rgcn=relation_output_rgcn.reshape(-1,1,self.output_dim)
        relation_outputs.append(relation_output_rgcn)
        x=torch.cat(relation_outputs, dim=1)

        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value,mask=None):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn

def edge_perms(length):
	"""
	Method to construct the edges of a graph (a utterance) considering the all the past utterances.
	return: list of tuples. tuple -> (vertice(int), neighbor(int))
	"""
	user_sys = set()
	sys_user = set()
	user_user = set()
	sys_sys = set()

	eff_array_user=[]
	eff_array_sys=[]
 
	for i in range(length):
		if i%2==0:
			eff_array_user.append(i)
		else:
			eff_array_sys.append(i)

	for i in range (length):
		if i%2==0:
			for j in eff_array_user:
				user_user.add((i,j))  
			for j in eff_array_sys:
				user_sys.add((i,j))
		else:
			for j in eff_array_user:
				sys_user.add((i,j))
			for j in eff_array_sys:
				sys_sys.add((i,j))
	user_sys_user=user_sys.union(user_user)
	sys_user_sys=sys_user.union(sys_sys)

	return [user_user,user_sys, sys_sys, sys_user, sys_user_sys,user_sys_user] 

def batch_graphify_hgcn(features, lengths, device):
    node_features, edge_index1,edge_index2,edge_index3,edge_index4, edge_index5, edge_index6,edge_type= [], [], [],[],[],[],[],[]
    batch_size = features.size(0)
    length_sum = 0
    for j in range(batch_size):
        cur_len = lengths[j].item() 
        node_features.append(features[j, :cur_len, :])
        perms = edge_perms(cur_len) 
		
        perms_rec0 = [(item[0] + length_sum, item[1] + length_sum) for item in perms[0]]
        perms_rec1= [(item[0] + length_sum, item[1] + length_sum) for item in perms[1]]
        perms_rec2 = [(item[0] + length_sum, item[1] + length_sum) for item in perms[2]]
        perms_rec3 = [(item[0] + length_sum, item[1] + length_sum) for item in perms[3]]
        perms_rec4= [(item[0] + length_sum, item[1] + length_sum) for item in perms[4]]
        perms_rec5= [(item[0] + length_sum, item[1] + length_sum) for item in perms[5]]
        length_sum += cur_len

        for item, item_rec in zip(perms[0], perms_rec0):
            edge_index1.append(torch.tensor([item_rec[0], item_rec[1]])) #user_user
            edge_type.append(torch.tensor([0]))
        for item, item_rec in zip(perms[1], perms_rec1):
            edge_index2.append(torch.tensor([item_rec[0], item_rec[1]]))#user_sys
            edge_type.append(torch.tensor([1]))
        for item, item_rec in zip(perms[2], perms_rec2):
            edge_index3.append(torch.tensor([item_rec[0], item_rec[1]])) #sys_sys
        for item, item_rec in zip(perms[3], perms_rec3):
            edge_index4.append(torch.tensor([item_rec[0], item_rec[1]])) #sys_user
        for item, item_rec in zip(perms[4], perms_rec4):
            edge_index5.append(torch.tensor([item_rec[0], item_rec[1]])) #sys_user_sys
        for item, item_rec in zip(perms[5], perms_rec5):
            edge_index6.append(torch.tensor([item_rec[0], item_rec[1]])) #user_sys_user
                    
    node_features = torch.cat(node_features, dim=0).to(device)	# [E, D_g]
    edge_index = [torch.stack(edge_index1).t().contiguous().to(device),torch.stack(edge_index2).t().contiguous().to(device),torch.stack(edge_index3).t().contiguous().to(device),torch.stack(edge_index4).t().contiguous().to(device), torch.stack(edge_index5).t().contiguous().to(device), torch.stack(edge_index6).t().contiguous().to(device)]  # [2, E]
    edge_type = torch.stack(edge_type).to(device)
    edge_type=edge_type.squeeze(-1)
    return node_features, edge_index,edge_type
    
class ContextModel(nn.Module):
    def __init__(self, base_model, batch_size, max_conver_num,device, context_model_type, head_num, model_variant):
        super(ContextModel, self).__init__()
        self.base_model=base_model
        self.max_conver_num = max_conver_num
        self.device = device
        self.heads=head_num
        self.context_model_type = context_model_type
        in_hidden_dim = base_model.config.hidden_size
        gcn_hidden_dim1 = 512
        gcn_out_dim = 256

        if ('nospk2pred' in self.context_model_type):
            self.relations=2
        else:
            self.relations=4

        if (model_variant == 'gcn'):
            self.gcn = GCN(input_dim=in_hidden_dim, hidden_dim1=gcn_hidden_dim1,output_dim=gcn_out_dim,relations=self.relations,heads=self.heads).to(device)
            self.task_heads = nn.ModuleList([nn.Sequential(nn.Linear(gcn_out_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1),) for _ in range(5)])
        elif (model_variant == 'gatv2'):
            self.gcn = GAT(input_dim=in_hidden_dim, hidden_dim1=gcn_hidden_dim1,output_dim=gcn_out_dim,relations=self.relations,heads=self.heads).to(device)
            self.task_heads = nn.ModuleList([nn.Sequential(nn.Linear(self.heads*gcn_hidden_dim1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1),) for _ in range(5)])
        elif (model_variant == 'rgcn'):
            self.gcn = RGCN(input_dim=in_hidden_dim, hidden_dim1=gcn_hidden_dim1,output_dim=gcn_out_dim,relations=self.relations,heads=self.heads).to(device)	
            self.task_heads = nn.ModuleList([nn.Sequential(nn.Linear(gcn_out_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1),) for _ in range(5)])
        elif (model_variant == 'hcgnn22'):
            self.gcn = GATv2GCN22(input_dim=in_hidden_dim, hidden_dim1=gcn_hidden_dim1,output_dim=gcn_out_dim,relations=self.relations,heads=self.heads).to(device)	
            self.task_heads = nn.ModuleList([nn.Sequential(nn.Linear(self.relations*gcn_out_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1),) for _ in range(5)])
        elif (model_variant == 'hcgnn3rel'):
            self.gcn = GATv2GCN3REL(input_dim=in_hidden_dim, hidden_dim1=gcn_hidden_dim1,output_dim=gcn_out_dim,relations=self.relations,heads=self.heads).to(device)	
            self.task_heads = nn.ModuleList([nn.Sequential(nn.Linear((self.relations+1)*gcn_out_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1),) for _ in range(5)])
        else:
            self.gcn = GATv2GCN(input_dim=in_hidden_dim, hidden_dim1=gcn_hidden_dim1,output_dim=gcn_out_dim,relations=self.relations,heads=self.heads).to(device)	
            self.task_heads = nn.ModuleList([nn.Sequential(nn.Linear(self.relations*gcn_out_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1),) for _ in range(5)])

        self.attention=ScaledDotProductAttention(gcn_out_dim)

    def forward(self, input_ids, attention_mask, current_length):
        input_id = input_ids.reshape(-1, input_ids.shape[-1])  #torch.Size([96, 64])
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) 
        
        batch_embeddings=self.base_model(input_ids=input_id, attention_mask=attention_mask).pooler_output #torch.Size([192, 768])
        batch_embeddings = batch_embeddings.reshape(input_ids.shape[0], -1, batch_embeddings.shape[-1]) #torch.Size([4, 48, 768])

        if model_variant == 'rgcn':
            features, edge_index,edge_type = batch_graphify_hgcn(batch_embeddings, current_length, self.device)
            gcn_features = self.gcn(features, edge_index,edge_type)
        elif model_variant == 'hcgnn3rel':
            features, edge_index,edge_type = batch_graphify_hgcn(batch_embeddings, current_length, self.device)
            gcn_features = self.gcn(features, edge_index,edge_type)
            gcn_features, attn = self.attention(gcn_features, gcn_features, gcn_features)
        else:
            features, edge_index, edge_type = batch_graphify_hgcn(batch_embeddings, current_length, self.device)
            gcn_features = self.gcn(features, edge_index)
            gcn_features, attn = self.attention(gcn_features, gcn_features, gcn_features)

        gcn_features = gcn_features.reshape(gcn_features.shape[0], -1) 

        regression_outputs = []
        for task_head in self.task_heads:
            regression_outputs.append(task_head(gcn_features))
        return torch.cat(regression_outputs, dim=1)

def valid(model, valid_dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in valid_dataloader:
            loss, outputs, labels = cal_loss(batch, model, criterion, context_model_type)
            epoch_loss += loss.item()
    return epoch_loss/len(valid_dataloader)

def cal_loss(batch, model, criterion, context_model_type):
    if ('gcn' in context_model_type):
        input_ids = torch.stack(batch['input_ids'],dim=0).to(device).permute(1,0,2) # shape: [batch_size, max_conver_num, max_length]
        attention_mask = torch.stack(batch['attention_mask'],dim=0).to(device).permute(1,0,2)
        current_length=batch['current_length'].to(device)
        labels = batch['labels'].to(device) # [batch_size, 5]
        dialogue_labels = labels.repeat_interleave(current_length, dim=0)
        model_outputs = model(input_ids, attention_mask, current_length)
        if 'lastnode' in context_model_type:
            loss = 0
            outputs = []
            accumulate_num = 0
            for i in range(len(current_length)):
                accumulate_num += current_length[i]
                outputs.append(model_outputs[accumulate_num-1])
                loss += criterion(model_outputs[accumulate_num-1], dialogue_labels[accumulate_num-1])
            outputs = torch.stack(outputs, dim=0).to(device) # torch.stack
        else:
            outputs = []
            accumulate_num = 0
            loss = 0
            for i in range(len(current_length)):
                dialogue_loss = 0
                accumulate_result = torch.zeros_like(model_outputs[0])
                for j in range(current_length[i]):
                    if (j % 2 == 0):
                        accumulate_result += model_outputs[accumulate_num+j]
                        sentence_loss = criterion(model_outputs[i], dialogue_labels[i])
                        dialogue_loss += sentence_loss
                accumulate_result = accumulate_result / (current_length[i]/2)
                #accumulate_result = accumulate_result / (current_length[i])
                outputs.append(accumulate_result)
                accumulate_num += current_length[i]

                dialogue_loss /= (current_length[i]/2)
                loss += dialogue_loss
            outputs = torch.stack(outputs, dim=0).to(device) # torch.stack
    else:
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs[:, [i for i, flag in enumerate(multitask) if flag == 1]], labels[:, [i for i, flag in enumerate(multitask) if flag == 1]])
    return loss, outputs, labels

def train(model, train_dataloader, valid_dataloader, criterion, patience, max_epoch):
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if (warmup_steps > 0):
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_epoch*len(train_dataloader))
    best_loss = float('inf')
    best_epoch = -1
    for epoch in range(max_epoch):
        model.train()
        epoch_loss = 0
        for batch in tqdm.tqdm(train_dataloader):
            loss, outputs, labels = cal_loss(batch, model, criterion, context_model_type)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            if (warmup_steps > 0):
                scheduler.step()
            loss.backward()
            optimizer.step()

        #torch.save(model.state_dict(), os.path.join(model_folder, f'model_{epoch}.pt'))
        print('epoch: {}, train loss: {}'.format(epoch, epoch_loss/len(train_dataloader)))
        logging.info('epoch: {}'.format(epoch))
        logging.info('train loss: {}'.format(epoch_loss/len(train_dataloader)))
        valid_loss = valid(model, valid_dataloader, criterion)
        print('valid loss: {}'.format(valid_loss))
        logging.info('valid loss: {}'.format(valid_loss))
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_folder, 'model.pt'))
        else:
            if epoch - best_epoch >= patience:
                break

def eval_real_time(model, criterion, label_medians, test_file, tokenizer, max_sent, context):
    if ('gcn' in context_model_type):
        tmp_test_data = gene_data_from_csv(test_file, tokenizer, max_sent)
        tmp_test_data = gene_context_data(tmp_test_data, pad_length, tokenizer)
        tmp_test_dataset = ConversationDataset(tmp_test_data, tokenizer, max_length)
        tmp_test_dataloader= DataLoader(tmp_test_dataset, batch_size=batch_size, shuffle=False)
        tmp_res = evaluation(model, tmp_test_dataloader, criterion, label_medians)
    else:
        tmp_test_data = gene_data_from_csv(test_file, tokenizer, max_sent)
        tmp_test_dataset = MultiTaskDataset(tmp_test_data, tokenizer, max_length)
        tmp_test_dataloader= DataLoader(tmp_test_dataset, batch_size=batch_size, shuffle=False)
        tmp_res = evaluation(model, tmp_test_dataloader, criterion, label_medians)
    return tmp_res

def evaluation(model, test_dataloader, criterion, label_medians, ensemble=False):
    if ensemble:
        model_list = model
        for m in model_list:
            m.eval()
    else:
        model.eval()
    epoch_loss = 0
    epoch_labels = []
    epoch_preds = []
    with torch.no_grad():
        for batch in test_dataloader:
            if ensemble:
                output_list = []
                for m in model_list:
                    batch_clone = copy.deepcopy(batch)
                    loss, outputs, labels = cal_loss(batch_clone, m, criterion, context_model_type)
                    output_list.append(outputs.clone())
                outputs = torch.stack(output_list, dim=0)
                outputs = torch.mean(outputs, dim=0)
            else:
                loss, outputs, labels = cal_loss(batch, model, criterion, context_model_type)
            epoch_loss += loss.item()
            epoch_labels.append(labels.cpu().numpy())
            epoch_preds.append(outputs.cpu().numpy())
    avg_loss = epoch_loss/len(test_dataloader)

    labels = np.concatenate(epoch_labels)
    preds = np.concatenate(epoch_preds)

    # pearson correlation
    pearson_corr = []
    pearson_p_values = []
    spearman_corr = []
    spearman_p_values = []

    for i in range(preds.shape[1]):
        labels_i = labels[:, i]
        preds_i = preds[:, i]
        pearson_corr_i, pearson_p_value_i = pearsonr(labels_i, preds_i)
        spearman_corr_i, spearman_p_value_i = spearmanr(labels_i, preds_i)
        pearson_corr.append(pearson_corr_i)
        pearson_p_values.append(pearson_p_value_i)
        spearman_corr.append(spearman_corr_i)
        spearman_p_values.append(spearman_p_value_i)

    # acc for using 0.5 as the threshold
    labels_binary_05 = np.array([[1 if label >= 0.5 else 0 for i, label in enumerate(sample)] for sample in labels])
    preds_binary_05 = np.array([[1 if pred >= 0.5 else 0 for i, pred in enumerate(sample)] for sample in preds])
    acc_list_05 = []
    balanced_acc_list_05 = []
    for i in range(preds_binary_05.shape[1]):
        acc_list_05.append(accuracy_score(labels_binary_05[:, i], preds_binary_05[:, i]))
        balanced_acc_list_05.append(balanced_accuracy_score(labels_binary_05[:, i], preds_binary_05[:, i]))

    # acc
    labels_binary = np.array([[1 if label >= label_medians[i] else 0 for i, label in enumerate(sample)] for sample in labels])
    preds_binary = np.array([[1 if pred >= label_medians[i] else 0 for i, pred in enumerate(sample)] for sample in preds])
    acc_list = []
    balanced_acc_list = []
    for i in range(preds_binary.shape[1]):
        acc_list.append(accuracy_score(labels_binary[:, i], preds_binary[:, i]))
        balanced_acc_list.append(balanced_accuracy_score(labels_binary[:, i], preds_binary[:, i]))

    return {
        'loss': avg_loss,
        'pearson_corr_list': pearson_corr,
        'pearson_p_values': pearson_p_values,
        'spearman_corr_list': spearman_corr,
        'spearman_p_values': spearman_p_values,
        'acc_list_05': acc_list_05,
        'balanced_acc_list_05': balanced_acc_list_05,
        'acc_list': acc_list,
        'balanced_acc_list': balanced_acc_list
    }

def print_res(res, name='test'):
    print (res)
    logging.info(f'-----------{name}---------------------')
    logging.info('{} loss: {:.3f}'.format(name, res['loss']))
    logging.info('pearson correlation: {}'.format([round(corr, 3) for corr in res['pearson_corr_list']]))
    logging.info('pearson p value: {}'.format([round(p_value, 3) for p_value in res['pearson_p_values']]))
    logging.info('spearman correlation: {}'.format([round(corr, 3) for corr in res['spearman_corr_list']]))
    logging.info('spearman p value: {}'.format([round(p_value, 3) for p_value in res['spearman_p_values']]))

    logging.info('acc using 0.5 as threshold: {}'.format([round(acc, 3) for acc in res['acc_list_05']]))
    logging.info('avg acc using 0.5 as threshold: {:.3f}'.format(np.mean(res['acc_list_05'])))
    logging.info('balanced acc using 0.5 as threshold: {}'.format([round(balanced_acc, 3) for balanced_acc in res['balanced_acc_list_05']]))
    logging.info('avg balanced acc using 0.5 as threshold: {:.3f}'.format(np.mean(res['balanced_acc_list_05'])))

    logging.info('acc: {}'.format([round(acc, 3) for acc in res['acc_list']]))
    logging.info('avg acc: {:.3f}'.format(np.mean(res['acc_list'])))
    logging.info('balanced acc: {}'.format([round(balanced_acc, 3) for balanced_acc in res['balanced_acc_list']]))
    logging.info('avg balanced acc: {:.3f}'.format(np.mean(res['balanced_acc_list'])))

def main():
    global pad_length
    logging.info('base model: {}'.format(base_model_name))
    logging.info('batch size: {}'.format(batch_size))
    logging.info('max length: {}'.format(max_length))
    logging.info('max epoch: {}'.format(max_epoch))
    logging.info('patience: {}'.format(patience))
    logging.info('learning rate: {}'.format(lr))
    logging.info('hidden size: {}'.format(hidden_size))
    logging.info('multitask: {}'.format(multitask_str))
    logging.info('dropout: {}'.format(dropout))
    logging.info('multilinear: {}'.format(multilinear))
    logging.info('warmup steps: {}'.format(warmup_steps))
    logging.info('criterion type: {}'.format(critertion_type))
    logging.info('context: {}'.format(context))
    logging.info('context model type: {}'.format(context_model_type))

    logging.info('loading data...')
    tokenizer=AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.add_tokens(['[SPK1]','[SPK2]'])

    train_data = gene_data_from_csv(train_file, tokenizer)
    label_medians = np.median(np.array([sample['labels'] for sample in train_data]), axis=0)
    valid_data = gene_data_from_csv(valid_file, tokenizer)
    test_data = gene_data_from_csv(test_file, tokenizer)

    if ('gcn' in context_model_type):
        pad_length = 0
        for sample in train_data:
            sentences_num = sample['text'].count('[SPK')
            pad_length = max(pad_length, sentences_num)
        train_data = gene_context_data(train_data, pad_length, tokenizer)
        valid_data = gene_context_data(valid_data, pad_length, tokenizer)
        test_data = gene_context_data(test_data, pad_length, tokenizer)
        train_dataset = ConversationDataset(train_data, tokenizer, max_length)
        valid_dataset = ConversationDataset(valid_data, tokenizer, max_length)
        test_dataset = ConversationDataset(test_data, tokenizer, max_length)
    else:
        train_dataset = MultiTaskDataset(train_data, tokenizer, max_length)
        valid_dataset = MultiTaskDataset(valid_data, tokenizer, max_length)
        test_dataset = MultiTaskDataset(test_data, tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logging.info('initializing model...')

    base_model = AutoModel.from_pretrained(base_model_name) 
    base_model.resize_token_embeddings(len(tokenizer))


    if ('gcn' in context_model_type):
        model = ContextModel(base_model, batch_size, pad_length, device, context_model_type, head_num, model_variant)
    else:
        model = MultiTaskModel(base_model=base_model, dropout_flag=dropout, multilinear_flag=multilinear)
    model = DataParallel(model)
    model.to(device)
    if (critertion_type=='mse'):
        criterion = nn.MSELoss()
    elif (critertion_type=='mae'):
        criterion = nn.L1Loss()

    if (train_flag==1):
        logging.info('start training...')
        train(model, train_dataloader, valid_dataloader, criterion, patience, max_epoch)

    if (test_flag==1):
        logging.info('start evaluation...')
        model.load_state_dict(torch.load(os.path.join(model_folder, 'model.pt')))
        if (real_time_flag == 1):
            for max_sent in [2,3,4,5,10]:
                tmp_res = eval_real_time(model, criterion, label_medians, test_file, tokenizer, max_sent, context)
                print_res(tmp_res, name=f'test_max_sent{max_sent}')
        valid_res = evaluation(model, valid_dataloader, criterion, label_medians)
        res = evaluation(model, test_dataloader, criterion, label_medians)
        print_res(valid_res, 'valid')
        print_res(res, 'test')

def eval_ensemble():
    global pad_length
    logging.info('base model: {}'.format(base_model_name))
    logging.info('batch size: {}'.format(batch_size))
    logging.info('max length: {}'.format(max_length))
    logging.info('max epoch: {}'.format(max_epoch))
    logging.info('patience: {}'.format(patience))
    logging.info('learning rate: {}'.format(lr))
    logging.info('hidden size: {}'.format(hidden_size))
    logging.info('multitask: {}'.format(multitask_str))
    logging.info('dropout: {}'.format(dropout))
    logging.info('multilinear: {}'.format(multilinear))
    logging.info('warmup steps: {}'.format(warmup_steps))
    logging.info('criterion type: {}'.format(critertion_type))
    logging.info('context: {}'.format(context))
    logging.info('context model type: {}'.format(context_model_type))

    logging.info('loading data...')
    tokenizer=AutoTokenizer.from_pretrained(base_model_name)
    #tokenizer.add_tokens(['[SPK1]','[SPK2]'])

    train_data = gene_data_from_csv(train_file, tokenizer)
    label_medians = np.median(np.array([sample['labels'] for sample in train_data]), axis=0)

    test_data = gene_data_from_csv(test_file, tokenizer)
    test_dataset = MultiTaskDataset(test_data, tokenizer, max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logging.info('initializing model...')

    base_model = AutoModel.from_pretrained(base_model_name) 
    #base_model.resize_token_embeddings(len(tokenizer))
    logging.info('start evaluation...')
    model_files = glob.glob(os.path.join(ensemble_model_folder, '*.pt'))[:2]
    # sort
    model_files = sorted(model_files)
    print (model_files)
    model_num = len(model_files)
    model_list = []
    for i in range(model_num):
        if ('gcn' in context_model_type):
            model = ContextModel(base_model, batch_size, pad_length, device, context_model_type, head_num, model_variant)
        else:
            model = MultiTaskModel(base_model=base_model, dropout_flag=dropout, multilinear_flag=multilinear)
        model = DataParallel(model)
        model.to(device)
        model_list.append(model)

    if (critertion_type=='mse'):
        criterion = nn.MSELoss()
    elif (critertion_type=='mae'):
        criterion = nn.L1Loss()

    for model, model_name in zip(model_list, model_files):
        try:
            model.load_state_dict(torch.load(model_name))
            print (model_name, 'loaded')
        except:
            print (model_name, 'failed')
    res = evaluation(model_list, test_dataloader, criterion, label_medians, ensemble=True)
    print_res(res, 'test')

if __name__ == '__main__':
    if not ensemble:
        main()
    else:
        # no training only evaluation
        eval_ensemble()