import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from gensim.models.ldamodel import LdaModel
from konlpy.tag import Mecab


class SentenceLabelDataset(Dataset):
    def __init__(self,
                 dic_path: str,
                 train_docs: List[str],
                 lda_model: LdaModel,
                 num_topics: int = 20
                 ):

        mecab = Mecab(dic_path)
        self.train_docs = train_docs
        self.num_topics = num_topics

        self.labels = []
        for doc in train_docs:
            nouns = [n for n in mecab.nouns(doc) if len(n) > 1]
            content = lda_model.id2word.doc2bow(nouns)
            probabilities = [b for (a, b) in lda_model.get_document_topics(content, minimum_probability=1e-5)]
            if len(probabilities) < self.num_topics:
                continue
            self.labels.append(probabilities)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        doc = self.train_docs[idx]
        label = torch.Tensor(self.labels[idx])

        return doc, label
