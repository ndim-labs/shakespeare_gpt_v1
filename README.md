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

| Feature | What it teaches |
|---|---|
| **Temperature** slider | Below 1 the model plays it safe and repeats itself; above 1 it gets creative, then chaotic |
| **Top-k** slider | Sample only from the k most likely next characters |
| **What the model sees** panel | The exact 32 characters fed to the model. Everything earlier in the chat is invisible to it |
| **Next-character chart** | The top 10 candidates for the reply's first character: the softmax output the model samples from |
| **Play-style prompt** | The chat is formatted as `YOU:` / `BARD:` dialogue, like the `ROMEO:` lines the model was trained on, and each reply stops at the end of the speech |

Built for *Decoding Language Models* (nDimensions.AI), Phase 1 · session 1.0: [ndim-labs/p1-gpt-from-scratch](https://github.com/ndim-labs/p1-gpt-from-scratch/tree/main/1.0-preparation).
