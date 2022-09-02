import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from gensim.models.ldamodel import LdaModel


class SentenceLabelDataset(Dataset):
    def __init__(self,
                 train_docs: List[str],
                 corpus: List[List[Tuple]],
                 lda_model: LdaModel,
                 num_topics: int = 20
                 ):

        self.train_docs = train_docs
        self.num_topics = num_topics

        self.labels = []
        for doc in range(len(train_docs)):
            probabilities = [b for (a, b) in lda_model.get_document_topics(corpus[doc], minimum_probability=1e-5)]
            if len(probabilities) < self.num_topics:
                continue
            self.labels.append(probabilities)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        doc = self.train_docs[idx]
        label = torch.Tensor(self.labels[idx])

        return doc, label
