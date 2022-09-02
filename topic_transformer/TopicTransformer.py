import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class TopicTransformer(nn.Module):
    def __init__(self,
                 output_dim: int,
                 transformer_model_name: str,
                 transformer_model=None,
                 max_length: int=128):
        super(TopicTransformer, self).__init__()

        if transformer_model == None and transformer_model_name == None:
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

        self.head_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self,
                input_x,
                device='cuda:0'):

        # Non-Tokenized Input
        if type(input_x) == list or type(input_x) == tuple:
            tokenized_sentence_list = self.tokenizer(input_x, max_length=self.max_length, padding='max_length',
                                                     truncation=True, return_tensors='pt').to(device)
        else:
            tokenized_sentence_list = input_x.to(device)

        # Transformer forward
        x = self.transformer_model(**tokenized_sentence_list).last_hidden_state

        # Avg Pooling
        pooling_mask = tokenized_sentence_list.attention_mask.unsqueeze(-1).expand(x.size()).float()
        sum_mask = pooling_mask.sum(1)
        x = (x * pooling_mask).sum(1) / sum_mask

        # Topic Head
        x = F.relu(x)
        x = self.head_layer(x)

        return F.softmax(x, dim=1)

    def loss(self, pred, label):
        return F.mse_loss(pred, label)

