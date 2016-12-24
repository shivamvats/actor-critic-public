"""Wraps Fuel H5PYDataset."""
import glob
from fuel import datasets
import gensim
from keras.preprocessing.text import Tokenizer
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream

all_chars = ([chr(ord('a') + i) for i in range(26)] +
             [chr(ord('0') + i) for i in range(10)] +
             [',', '.', '!', '?', '#'] +
             [' ', '^', '$'])
code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}

def _lower(s):
    return s.lower()

class TextDatasetWord(datasets.TextFile):
    def __init__(self, file_or_path, which_sets, sources, target_source, **kwargs):
        # sources: contains the type of input (text in my case)
        #
        if target_source != 'text':
            raise ValueError
        if sources != ('text',):
            raise ValueError
        unk_token = '<UNK>'

        # Initializing the base class with a dummy dict. Will update it later.
        dummy_token_dict = {'<UNK>': 0}
        self.word2code = dummy_token_dict
        self.code2word = {0: '<UNK>'}
        self.num_labels = len(self.code2word)
        self.word2code['$'] = len(self.word2code)
        self.word2code['^'] = len(self.word2code)

        self.eos_label = self.word2code['$']
        self.bos_label = self.word2code.get('^')
        file_or_path = glob.glob(file_or_path)
        if not file_or_path:
            raise ValueError
        super(TextDatasetWord, self).__init__(
            files=file_or_path,
            dictionary=dummy_token_dict,
            level='word', preprocess=_lower,
            # Bos and Eos token are added at
            # a higher level
            bos_token=None, eos_token=None, unk_token=unk_token,
            **kwargs)
        self.sources = sources
        self.target_source = target_source
        print("Textword initialized")


    def create_dicts(self, data_stream):
        nltk_tokenizer = RegexpTokenizer(r'\w+')
        data = []
        processed_data = []
        for d in data_stream.get_epoch_iterator():
            data.append(d)
        for d in data:
            for sent in d:
                print(sent)
                sent = ' '.join(nltk_tokenizer.tokenize(sent))
                processed_data.append(sent)
        print("Data stream")
        print("$$$$$$$$$$$$$$$$$$$$$$$$")
        print(processed_data[0])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(processed_data)
        word2code_without_spec_tokens = tokenizer.word_index
        word2code_without_spec_tokens[unk_token] = len(word2code_without_spec_tokens)
        self.word2code = tokenizer.word_index.copy()

        self.code2word = {}
        for i, j in word2code_without_spec_tokens.items():
            self.code2word[j] = i
        self.num_labels = len(self.code2word)
        self.word2code['$'] = len(self.word2code)
        self.word2code['^'] = len(self.word2code)

        self.eos_label = self.word2code['$']
        self.bos_label = self.word2code.get('^')
        #self.create_embedding()

    def create_embedding(self):
        self.embedding_matrix = np.zeros((len(self.word2code), self.clip_len))
        for word, i in self.word2code.items():
            if word in self.w2v_model.vocab:
                embedding_vector = np.array(self.w2v_model[word])
            else:
                embedding_vector = np.zeros(TOKEN_REPRESENTATION_SIZE)

            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

    def load_glove_model(self):
        self.w2v_model = gensim.models.Word2Vec.load_word2vec_format(
                "/home/aries/Documents/Learning/DL/autoencoder/data/glove.6B/glove.6B.200d_gensim.txt")


    def token_map(self, source):
        if source != 'text':
            raise ValueError
        return self.word2code

    def dim(self, source):
        raise ValueError

    def decode(self, labels, keep_all=False):
        return [self.code2word[label] for label in labels
                if keep_all or label not in [self.eos_label, self.bos_label]]

    def pretty_print(self, labels, example):
        return "".join(self.decode(labels))

    def monospace_print(self, labels):
        return "".join(self.decode(labels, keep_all=True))


class TextDataset(datasets.TextFile):
    def __init__(self, file_or_path, which_sets, sources, target_source, **kwargs):
        if target_source != 'text':
            raise ValueError
        if sources != ('text',):
            raise ValueError
        # Ignore which_sets and file_or_path
        char2code_without_spec_tokens = dict(char2code)
        del char2code_without_spec_tokens['$']
        del char2code_without_spec_tokens['^']
        file_or_path = glob.glob(file_or_path)
        if not file_or_path:
            raise ValueError
        super(TextDataset, self).__init__(
            files=file_or_path,
            dictionary=char2code_without_spec_tokens,
            level='character', preprocess=_lower,
            # Bos and Eos token are added at
            # a higher level
            bos_token=None, eos_token=None, unk_token='#',
            **kwargs)
        self.sources = sources
        self.target_source = target_source

        self.num2char = dict(enumerate(all_chars))
        self.char2num = {v: k for k, v in code2char.items()}
        self.num_labels = len(self.num2char)
        self.eos_label = self.char2num['$']
        self.bos_label = self.char2num.get('^')

    def token_map(self, source):
        if source != 'text':
            raise ValueError
        return self.char2num

    def dim(self, source):
        raise ValueError

    def decode(self, labels, keep_all=False):
        return [self.num2char[label] for label in labels
                if keep_all or label not in [self.eos_label, self.bos_label]]

    def pretty_print(self, labels, example):
        return "".join(self.decode(labels))

    def monospace_print(self, labels):
        return "".join(self.decode(labels, keep_all=True))
