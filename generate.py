import torch
import torch.nn.functional as F
import json
import random
from model import LyricsLSTM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 40

def load_model_and_vocab():
    with open('vocab.json', 'r') as f:
        data = json.load(f)
    word2idx = data['word2idx']
    idx2word = {int(k): v for k, v in data['idx2word'].items()}

    model = LyricsLSTM(vocab_size=len(word2idx)).to(DEVICE)
    model.load_state_dict(torch.load('lyrics_model.pt', map_location=DEVICE))
    model.eval()
    return model, word2idx, idx2word

def generate_lyrics(seed_text, num_words=80, temperature=0.8, use_bigram_boost=False, bigram_probs=None):
    """
    Generate lyrics from a seed phrase.
    - seed_text: starting words
    - num_words: how many words to generate
    - temperature: creativity (0.1=safe, 2.0=wild)
    - use_bigram_boost: blend bigram probabilities for more natural flow
    """
    model, word2idx, idx2word = load_model_and_vocab()

    tokens = seed_text.lower().split()
    indices = [word2idx.get(t, word2idx['<UNK>']) for t in tokens]

    generated = list(tokens)
    hidden = None

    with torch.no_grad():
        for _ in range(num_words):
            # Pad or trim to SEQ_LEN
            inp = indices[-SEQ_LEN:]
            inp = [0] * (SEQ_LEN - len(inp)) + inp
            x = torch.tensor([inp]).to(DEVICE)

            logits, hidden = model(x, hidden)
            last_logits = logits[0, -1, :]  # last word prediction

            # Apply temperature
            scaled = last_logits / temperature
            probs = F.softmax(scaled, dim=0).cpu().numpy()

            # Optional: blend with bigram probabilities
            if use_bigram_boost and bigram_probs and generated:
                last_word = generated[-1]
                if last_word in bigram_probs:
                    for word, prob in bigram_probs[last_word].items():
                        if word in word2idx:
                            probs[word2idx[word]] += 0.2 * prob
                    probs /= probs.sum()  # renormalize

            next_idx = int(torch.multinomial(torch.tensor(probs), 1).item())
            next_word = idx2word.get(next_idx, '<UNK>')

            if next_word in ('<UNK>', '<PAD>', '<EOS>'):
                continue

            generated.append(next_word)
            indices.append(next_idx)

            # Add line breaks every 8-10 words for lyric formatting
            if len(generated) % random.randint(8, 10) == 0:
                generated.append('\n')

    return ' '.join(generated).replace(' \n ', '\n')