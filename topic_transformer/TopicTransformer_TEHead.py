import os
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class TopicTransformer_TEHead(nn.Module):
    def __init__(self, output_dim, transformer_model=None, transformer_model_name=None, max_length=128):
        super(TopicTransformer_TEHead, self).__init__()
        
        
        if transformer_model==None and transformer_model_name==None:
            print("ERROR : Cannot Load Transformer Model")
            return -1
        if transformer_model != None:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_model.config._name_or_path)
            self.transformer_model = transformer_model
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
            self.transformer_model = AutoModel.from_pretrained(transformer_model_name)
       
        self.hidden_dim = self.transformer_model.config.hidden_size
        self.output_dim = output_dim
        self.max_length = max_length
        
        self.ffl_size = int(self.transformer_model.config.intermediate_size)
        
        self.transformer_encoder_layer1 = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=12, activation='gelu',dim_feedforward=self.ffl_size, batch_first=True)
        self.transformer_encoder_layer2 = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=12, activation='gelu',dim_feedforward=self.ffl_size, batch_first=True)
        self.transformer_encoder_layer3 = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=12, activation='gelu',dim_feedforward=self.ffl_size, batch_first=True)

        self.head_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, input_x, device='cpu', ptm_freeze=False):
        
        # Non-Tokenized Input
        if type(input_x) == list or type(input_x) == tuple :
            tokenized_sentence_list = self.tokenizer(list(input_x), max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt').to(device)
        else:
            tokenized_sentence_list = input_x.to(device)
            
        # Transformer forward
        if ptm_freeze == True:
            with torch.no_grad():
                x = self.transformer_model(**tokenized_sentence_list).last_hidden_state
        else:
            x = self.transformer_model(**tokenized_sentence_list).last_hidden_state
        
        x = self.transformer_encoder_layer1(x)
        x = self.transformer_encoder_layer2(x)
        x = self.transformer_encoder_layer3(x)
        
        # Avg Pooling

        pooling_mask = tokenized_sentence_list.attention_mask.unsqueeze(-1).expand(x.size()).float()
        sum_mask = pooling_mask.sum(1)
        x = (x*pooling_mask).sum(1) / sum_mask
        
        
        # Topic Head
        x = F.relu(x)
        x = self.head_layer(x)
        
        return F.softmax(x, dim=1)
    