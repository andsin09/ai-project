import gradio as gr
from generate import generate_lyrics, load_model_and_vocab
from preprocess import load_and_tokenize, build_vocab, build_bigram_model

# Pre-load everything so the UI is fast
print('Loading model...')
model, word2idx, idx2word = load_model_and_vocab()
tokens = load_and_tokenize('lyrics.txt')
_, _ = build_vocab(tokens)
bigram_probs = build_bigram_model(tokens, word2idx)
print('Ready!')

def run_generator(seed, num_words, temperature, use_bigram):
    if not seed.strip():
        return "Please enter a seed phrase to start."
    result = generate_lyrics(
        seed_text=seed,
        num_words=int(num_words),
        temperature=temperature,
        use_bigram_boost=use_bigram,
        bigram_probs=bigram_probs
    )
    return result

with gr.Blocks(title='AI Lyrics Generator') as demo:
    gr.Markdown("# AI Lyrics Generator")
    gr.Markdown("Generate lyrics in the style of your chosen artist using a trained LSTM neural network.")

    with gr.Row():
        seed_input = gr.Textbox(
            label="Seed Phrase",
            placeholder="e.g. i walked through the city",
            lines=1
        )

    with gr.Row():
        num_words = gr.Slider(
            label="Number of Words",
            minimum=20, maximum=150, step=10, value=80
        )
        temperature = gr.Slider(
            label="Temperature (Creativity)",
            minimum=0.1, maximum=2.0, step=0.1, value=0.8
        )

    use_bigram = gr.Checkbox(
        label="Enable Bigram Boost (blends bigram probabilities for more natural flow)",
        value=True
    )

    generate_btn = gr.Button("Generate Lyrics", variant="primary")

    output = gr.Textbox(
        label="Generated Lyrics",
        lines=10
    )

    generate_btn.click(
        fn=run_generator,
        inputs=[seed_input, num_words, temperature, use_bigram],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()