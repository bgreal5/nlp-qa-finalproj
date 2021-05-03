import spacy
import torch
import torch.nn as nn
import numpy as np

from utils import Indexer


class NEREncoder():
    def __init__(self, ner_output_dim=100) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.indexer = self.create_indexer()
        self.ner_size = len(self.indexer)

        # self.linear = nn.Linear(self.ner_size, ner_output_dim)

    def forward(self, x):
        max_len = x.shape[1]
        return self.encode(x, max_len)  # [1, max_len_of_input, ner_size]

    def create_indexer(self):
        indexer = Indexer()
        indexer.add_and_get_index("PERSON")
        indexer.add_and_get_index("NORP")
        indexer.add_and_get_index("FAC")
        indexer.add_and_get_index("ORG")
        indexer.add_and_get_index("GPE")
        indexer.add_and_get_index("LOC")
        indexer.add_and_get_index("PRODUCT")
        indexer.add_and_get_index("EVENT")
        indexer.add_and_get_index("WORK_OF_ART")
        indexer.add_and_get_index("LAW")
        indexer.add_and_get_index("LANGUAGE")
        indexer.add_and_get_index("DATE")
        indexer.add_and_get_index("TIME")
        indexer.add_and_get_index("PERCENT")
        indexer.add_and_get_index("MONEY")
        indexer.add_and_get_index("QUANTITY")
        indexer.add_and_get_index("ORDINAL")
        indexer.add_and_get_index("CARDINAL")
        # Manual
        indexer.add_and_get_index("UNKNOWN")
        indexer.add_and_get_index("PADDING")
        return indexer

    def encoding_to_one_hot(self, encoding, max_len):
        encoding_len = len(encoding)
        one_hot = np.zeros((max_len, self.ner_size))
        for i, idx in enumerate(encoding):
            if i >= max_len:
                break
            one_hot[i, idx] = 1
        """
        for i in range(encoding_len, max_len):
            one_hot[i, -1] = 1
        """
        return one_hot

    def encode(self, text, max_len, as_tensor=False):
        doc = self.nlp(text)
        encoding = self.encode_doc(doc)
        one_hot = self.encoding_to_one_hot(encoding, max_len)
        if as_tensor:
            return torch.tensor(one_hot).unsqueeze(0)
        return one_hot

    def encode_doc(self, doc):
        base_vector = np.full(
            len(doc), self.indexer.index_of("UNKNOWN"))
        for ent in doc.ents:
            base_vector[ent.start:ent.end] = self.indexer.index_of(ent.label_)
        return base_vector
