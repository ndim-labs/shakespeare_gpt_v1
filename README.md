# Shakespeare GPT chat

A local Streamlit chat interface for the Tiny Shakespeare character-level GPT
checkpoint in `shakespeare_gpt_v1.pth`. The app recreates the notebook's model
architecture and character vocabulary so the checkpoint can be loaded directly.

A 0.2M-parameter model with architecture inspired by Andrej Karpathy's GPT
video. Trained and built by the nDimensions.ai team. MIT License.

## Run locally

From this folder, create and activate a virtual environment, then install the
dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the interface:

```powershell
streamlit run app.py
```

Keep `shakespeare_gpt_v1.pth` beside `app.py`. The app uses CUDA when available
and otherwise runs on CPU.

This small model was trained to continue Shakespeare-style text, not to follow
chat instructions. Its context window is only 32 characters, and generation is
limited to the model's 65-character training vocabulary.

## What the app shows

**Sidebar:** temperature, top-k, response length, a **seed** for repeatable demos, and a **model card**.

**"How the model made that reply":** four tabs, one concept each:

| Tab | What you see | Concept |
|---|---|---|
| ① What it can see | The whole chat as a play script; only the last 32 characters are highlighted, the rest is greyed out | Context window |
| ② Text → numbers | One tile per character with its ID: all the model receives | Tokenization |
| ③ Next-character guess | Top 10 candidates for the reply's first character | Logits → softmax, temperature |
| ④ One character at a time | Replay of the first 12 steps: top 3 guesses, which one was picked, and 🎲 when it wasn't the favourite | Sampling, autoregression |

The chat is formatted as `YOU:` / `BARD:` dialogue, like the `ROMEO:` lines the model was trained on, and each reply stops at the end of the speech.

**Presenting it?** See **[DEMO.md](DEMO.md)**: a 10-minute opening-demo script (what to click, what to say), common audience questions, and the build-along plan.

Built for *Decoding Language Models* (nDimensions.AI), Phase 1 · session 1.0: [ndim-labs/p1-gpt-from-scratch](https://github.com/ndim-labs/p1-gpt-from-scratch/tree/main/1.0-preparation).
