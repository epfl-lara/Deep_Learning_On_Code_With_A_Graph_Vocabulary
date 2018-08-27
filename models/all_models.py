# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from models.FITB.CharCNN import FITBCharCNN
from models.FITB.FixedVocab import FITBFixedVocab
from models.FITB.NameGraphVocab import FITBNameGraphVocab
from models.GraphNN.GGNN import GGNN
from models.VarNaming.CharCNN import VarNamingCharCNN
from models.VarNaming.FixedVocab import VarNamingFixedVocab
from models.VarNaming.NameGraphVocab import VarNamingNameGraphVocab


class FITBFixedVocabGGNN(FITBFixedVocab, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBCharCNNGGNN(FITBCharCNN, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBNameGraphVocabGGNN(FITBNameGraphVocab, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingFixedVocabGGNN(VarNamingFixedVocab, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingCharCNNGGNN(VarNamingCharCNN, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingNameGraphVocabGGNN(VarNamingNameGraphVocab, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
