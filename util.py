from collections import Counter
import numpy as np
def prepare_vocab(texts, num_words = 5000):
    vocab_list = []
    for text in texts:
        vocab_list.extend(text.split())
    vocab_list = [each[0] for each in Counter(vocab_list).most_common(num_words)]
    return vocab_list

def prepare_word_dict(vocabs, empty_token=''):
    word_to_int_dict = {w:i+1 for i, w in enumerate(vocabs)}
    word_to_int_dict[empty_token] = 0
    return word_to_int_dict

def sent_to_idx(sents, w2id_dict, max_len = 50, pad_token = ''):
    idx_sents = []
    for sent in sents:
        sent = [w2id_dict.get(word, w2id_dict[pad_token]) for word in sent.split()]
        if len(sent) >= max_len:
            idx_sents.append(sent[:max_len])
        else:
            idx_sents.append([w2id_dict[pad_token]]*(max_len-len(sent)) + sent)
    return np.array(idx_sents)