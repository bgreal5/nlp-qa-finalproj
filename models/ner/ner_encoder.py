import spacy
import torch
import torch.nn as nn
import numpy as np

from utils import Indexer


class NEREncoder():
    def __init__(self, ner_output_dim=100) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.ner_indexer = self.create_ner_indexer()
        self.ner_size = len(self.ner_indexer)

        self.pos_indexer = self.create_pos_indexer()
        self.pos_size = len(self.pos_indexer)

        # self.linear = nn.Linear(self.ner_size, ner_output_dim)

    def forward(self, x):
        max_len = x.shape[1]
        return self.encode(x, max_len)  # [1, max_len_of_input, ner_size]

    def create_ner_indexer(self):
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

    def create_pos_indexer(self):
        indexer = Indexer()
        indexer.add_and_get_index("ADJ")
        indexer.add_and_get_index("ADP")
        indexer.add_and_get_index("ADV")
        indexer.add_and_get_index("AUX")
        indexer.add_and_get_index("CONJ")
        indexer.add_and_get_index("CCONJ")
        indexer.add_and_get_index("DET")
        indexer.add_and_get_index("INTJ")
        indexer.add_and_get_index("NOUN")
        indexer.add_and_get_index("NUM")
        indexer.add_and_get_index("PART")
        indexer.add_and_get_index("PRON")
        indexer.add_and_get_index("PROPN")
        indexer.add_and_get_index("PUNCT")
        indexer.add_and_get_index("SCONJ")
        indexer.add_and_get_index("SYM")
        indexer.add_and_get_index("VERB")
        indexer.add_and_get_index("X")
        indexer.add_and_get_index("SPACE")

        # Manual
        indexer.add_and_get_index("UNKNOWN")
        indexer.add_and_get_index("PADDING")
        return indexer

    def encoding_to_one_hot(self, encoding, max_len, dim):
        one_hot = np.zeros((max_len, dim))
        for i, idx in enumerate(encoding):
            if i >= max_len:
                break
            one_hot[i, idx] = 1
        return one_hot

    def encode(self, text, max_len, as_tensor=False, ner=True, pos=True):
        doc = self.nlp(text)
        vectors = []
        if ner:
            encoding = self.encode_doc_ner(doc)
            one_hot = self.encoding_to_one_hot(
                encoding, max_len, self.ner_size)
            vectors.append(torch.tensor(one_hot).unsqueeze(0))
        if pos:
            encoding = self.encode_doc_pos(doc)
            one_hot = self.encoding_to_one_hot(
                encoding, max_len, self.pos_size)
            vectors.append(torch.tensor(one_hot).unsqueeze(0))

        if len(vectors) == 2:
            print(torch.cat(vectors, dim=2).shape)
            return torch.cat(vectors, dim=2)
        elif len(vectors) == 1:
            return vectors[0]

        return None

    def encode_doc_ner(self, doc):
        base_vector = np.full(
            len(doc), self.ner_indexer.index_of("UNKNOWN"))
        for ent in doc.ents:
            base_vector[ent.start:ent.end] = self.ner_indexer.index_of(
                ent.label_)
        return base_vector

    def encode_doc_pos(self, doc):
        base_vector = np.full(
            len(doc), self.pos_indexer.index_of("UNKNOWN"))
        for i, token in enumerate(doc):
            base_vector[i] = self.pos_indexer.index_of(token.pos)
        return base_vector
