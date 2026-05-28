import re
from collections import Counter, defaultdict
import json

def load_and_tokenize(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    text = re.sub(r"[^a-z0-9\s',.!?\n]", '', text)
    tokens = text.split()
    return tokens

def build_vocab(tokens, min_freq=2):
    counts = Counter(tokens)
    vocab = ['<PAD>', '<UNK>', '<EOS>']  
    vocab += [word for word, cnt in counts.most_common() if cnt >= min_freq]

    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for word, i in word2idx.items()}
    return word2idx, idx2word

def build_bigram_model(tokens, word2idx):
    bigram_counts = defaultdict(Counter)
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i+1]
        if w1 in word2idx and w2 in word2idx:
            bigram_counts[w1][w2] += 1

    # Convert counts to probabilities
    bigram_probs = {}
    for word, next_words in bigram_counts.items():
        total = sum(next_words.values())
        bigram_probs[word] = {nw: cnt/total for nw, cnt in next_words.items()}

    return bigram_probs

def save_vocab(word2idx, idx2word, path='vocab.json'):
    with open(path, 'w') as f:
        json.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)

if __name__ == '__main__':
    tokens = load_and_tokenize('lyrics.txt')
    word2idx, idx2word = build_vocab(tokens)
    bigram_probs = build_bigram_model(tokens, word2idx)
    save_vocab(word2idx, idx2word)
    print(f'Vocabulary size: {len(word2idx)} words')
    print(f'Sample bigrams for "i":', list(bigram_probs.get('i', {}).items())[:5])