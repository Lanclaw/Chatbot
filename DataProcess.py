import config
import torch

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

device = "cuda" if torch.cuda.is_available() else "cpu"


class Lang():
    def __init__(self, name):
        self.name = name
        self.word2id = {}
        self.word_count = {}
        self.id2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.n_words
            self.word_count[word] = 1
            self.id2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word_count[word] += 1


def filterPair(p):
    #print(p)
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def readLangs(lang1, lang2):
    # 读取数据放入列表
    lines = open('data/chatdata_all.txt', encoding='utf-8').read().strip().split('\n')
    pairs = [[s for s in line.split('@@')] for line in lines]

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    pairs = filterPairs(pairs)

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    return input_lang, output_lang, pairs





def sen2id(lang, sentence):
    return [lang.word2id[word] for word in sentence.split()]


def sen2tensor(lang, sentence):
    id = sen2id(lang, sentence)
    id.append(EOS_token)
    return torch.tensor(id, device=device).view(-1, 1)


def pair2tensor(input_lang, output_lang, pair):
    input_tensor = sen2tensor(input_lang, pair[0])
    output_tensor = sen2tensor(output_lang, pair[1])
    return (input_tensor, output_tensor)


input_lang, output_lang, pairs = readLangs('human', 'machine')

