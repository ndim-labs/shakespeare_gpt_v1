import html
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F


BLOCK_SIZE = 32
N_EMBD = 64
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.0
VOCAB = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
STOI = {character: index for index, character in enumerate(VOCAB)}
ITOS = {index: character for character, index in STOI.items()}
CHECKPOINT_PATH = Path(__file__).with_name("shakespeare_gpt_v1.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Head(nn.Module):
    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time, channels = x.shape
        key = self.key(x)
        query = self.query(x)
        weights = query @ key.transpose(-2, -1) * channels**-0.5
        weights = weights.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        weights = self.dropout(F.softmax(weights, dim=-1))
        value = self.value(x)
        return weights @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class ShakespeareGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(len(VOCAB), N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)]
        )
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, len(VOCAB))

    def forward(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        _, time = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        position_embeddings = self.position_embedding_table(
            torch.arange(time, device=idx.device)
        )
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        return self.lm_head(self.ln_f(x)), None

    @torch.inference_mode()
    def next_token_probabilities(
        self, idx: torch.Tensor, temperature: float = 1.0, top_k: int | None = None
    ) -> torch.Tensor:
        """Probability of every vocabulary character coming next, after temperature and top-k."""
        logits, _ = self(idx[:, -BLOCK_SIZE:])
        logits = logits[:, -1, :] / temperature  # <1 sharpens, >1 flattens the distribution
        if top_k is not None and top_k < logits.size(-1):
            kth_best = torch.topk(logits, top_k).values[:, [-1]]
            logits = logits.masked_fill(logits < kth_best, float("-inf"))  # keep only the k best
        return F.softmax(logits, dim=-1)

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            probabilities = self.next_token_probabilities(idx, temperature, top_k)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


MODEL_PARAMETERS = sum(p.numel() for p in ShakespeareGPT().parameters())


@st.cache_resource
def load_model() -> ShakespeareGPT:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH.name}. "
            "Place it beside app.py and restart the app."
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location=DEVICE, weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("The model checkpoint must contain a state dictionary.")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("The checkpoint's model_state_dict is not a dictionary.")

    model = ShakespeareGPT().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def encode(text: str) -> list[int]:
    return [STOI[character] for character in text]


def decode(tokens: list[int]) -> str:
    return "".join(ITOS[token] for token in tokens)


# The model learned plays formatted as "NAME:\nline of dialogue\n\n", so we format the chat the same way.
SPEAKERS = {"user": "YOU", "assistant": "BARD"}
TRACE_STEPS = 12  # how many generation steps to show in the replay


def build_script(messages: list[dict[str, str]]) -> str:
    """Turn the chat into play dialogue, ending with a cue for the BARD to speak."""
    script = "".join(
        f"{SPEAKERS[message['role']]}:\n{message['content']}\n\n" for message in messages
    )
    script += f"{SPEAKERS['assistant']}:\n"
    return "".join(character for character in script if character in STOI)


def make_reply(
    model: ShakespeareGPT,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int | None,
) -> tuple[str, dict]:
    """Generate a reply one character at a time, recording what happened for the explainer panel."""
    if seed is not None:
        torch.manual_seed(seed)
    script = build_script(messages)
    context = script[-BLOCK_SIZE:]  # the model can only see this much
    idx = torch.tensor([encode(context)], dtype=torch.long, device=DEVICE)
    first_probabilities, trace = None, []
    for step in range(max_new_tokens):
        probabilities = model.next_token_probabilities(idx, temperature, top_k)
        next_token = torch.multinomial(probabilities, num_samples=1)
        if step == 0:
            first_probabilities = probabilities[0].cpu()
        if step < TRACE_STEPS:
            top = torch.topk(probabilities[0], 3)
            trace.append({
                "window": decode(idx[0, -BLOCK_SIZE:].tolist()),
                "top": [(ITOS[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())],
                "picked": ITOS[next_token.item()],
                "picked_p": probabilities[0, next_token.item()].item(),
            })
        idx = torch.cat((idx, next_token), dim=1)
    reply = decode(idx[0, len(context):].tolist())
    reply = reply.split("\n\n")[0].strip()  # a blank line ends the speech
    internals = {"script": script, "context": context, "probabilities": first_probabilities, "trace": trace}
    return reply, internals


def show_char(character: str) -> str:
    return {"\n": "⏎ newline", " ": "␣ space"}.get(character, character)


def visible(text: str) -> str:
    return "".join({"\n": "⏎", " ": "␣"}.get(character, character) for character in text)


def script_html(script: str, context: str) -> str:
    """The whole conversation: greyed where the model is blind, highlighted where it can see."""
    hidden = script[: len(script) - len(context)]

    def render(text: str) -> str:
        return html.escape(text).replace("\n", "⏎<br>")

    return (
        "<div style='font-family:monospace;font-size:0.9rem;line-height:1.6;padding:10px;"
        "border:1px solid rgba(128,128,128,0.3);border-radius:8px'>"
        f"<span style='opacity:0.35'>{render(hidden)}</span>"
        f"<span style='background:rgba(255,200,0,0.35);border-radius:3px'>{render(context)}</span>"
        "</div>"
    )


def tiles_html(context: str) -> str:
    """One tile per character, with the number the model actually receives."""
    tiles = "".join(
        "<div style='display:inline-flex;flex-direction:column;align-items:center;min-width:30px;"
        "margin:2px;padding:4px 2px;border:1px solid rgba(128,128,128,0.4);border-radius:6px'>"
        f"<span style='font-family:monospace;font-size:1.05rem'>{html.escape(visible(c))}</span>"
        f"<span style='font-size:0.7rem;opacity:0.7'>{STOI[c]}</span></div>"
        for c in context
    )
    return f"<div style='display:flex;flex-wrap:wrap'>{tiles}</div>"


st.set_page_config(page_title="Shakespeare GPT", page_icon="🎭")
st.title("Shakespeare GPT")
st.caption(
    "A 0.2M-parameter model with architecture inspired by Andrej Karpathy's "
    "GPT video. Trained and built by the nDimensions.ai team. MIT License."
)
st.caption(
    "This tiny character-level model continues Shakespeare-style text; it is "
    "not instruction-tuned and can only use the last 32 characters as context."
)

with st.sidebar:
    st.subheader("Generation")
    response_length = st.slider(
        "Maximum response length (characters)",
        min_value=32,
        max_value=512,
        value=192,
        step=32,
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Below 1: safer, more repetitive. Above 1: more creative, more chaotic.",
    )
    top_k_choice = st.slider(
        "Top-k (0 = off)",
        min_value=0,
        max_value=len(VOCAB),
        value=0,
        help="Only sample from the k most likely next characters.",
    )
    top_k = top_k_choice or None
    seed_choice = st.number_input(
        "Seed (0 = random)", min_value=0, value=0, step=1,
        help="Use the same seed to get the same reply every time: handy for live demos.",
    )
    seed = int(seed_choice) or None
    show_internals = st.toggle("Explain how the reply was made", value=True)

    st.subheader("Model card")
    st.markdown(
        f"""
| | |
|---|---|
| Parameters | {MODEL_PARAMETERS:,} |
| Layers × heads | {N_LAYER} × {N_HEAD} |
| Vector size | {N_EMBD} |
| Context window | **{BLOCK_SIZE} characters** |
| Vocabulary | {len(VOCAB)} characters |
| Trained on | Tiny Shakespeare (~1M characters) |
"""
    )
    st.caption(f"Running on {DEVICE}. For scale: GPT-3 has 175 billion parameters.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.pop("internals", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    model = load_model()
except (FileNotFoundError, RuntimeError, KeyError, TypeError) as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Write a prompt or a line of dialogue...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    unsupported = sorted(set(prompt) - STOI.keys())
    if unsupported:
        st.warning(
            "Characters outside the model vocabulary will be omitted from its "
            "prompt: " + ", ".join(repr(character) for character in unsupported)
        )

    with st.chat_message("assistant"):
        with st.spinner("Generating a continuation..."):
            reply, internals = make_reply(
                model, st.session_state.messages, response_length, temperature, top_k, seed
            )
        st.markdown(reply or "_The model returned an empty continuation._")
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.internals = internals

if show_internals and "internals" in st.session_state:
    info = st.session_state.internals
    st.divider()
    st.subheader("🔍 How the model made that reply")
    step1, step2, step3, step4 = st.tabs(
        ["① What it can see", "② Text → numbers", "③ Next-character guess", "④ One character at a time"]
    )

    with step1:
        st.markdown(
            "The chat is rewritten as a **play script** (`YOU:` / `BARD:`), because the model learned from plays. "
            f"But the model can only read the **last {BLOCK_SIZE} characters**, highlighted below. "
            "Everything greyed out is **invisible** to it: it has no memory of the earlier conversation."
        )
        st.markdown(script_html(info["script"], info["context"]), unsafe_allow_html=True)
        used = len(info["context"])
        st.progress(used / BLOCK_SIZE, text=f"Context window: {used} / {BLOCK_SIZE} characters used")
        st.caption("⏎ = new line. For scale, modern chat models read hundreds of thousands of tokens.")

    with step2:
        st.markdown(
            "Neural networks only understand numbers. Each character becomes its **ID** in a "
            f"{len(VOCAB)}-character vocabulary. These {len(info['context'])} numbers are *all* the model receives:"
        )
        st.markdown(tiles_html(info["context"]), unsafe_allow_html=True)
        st.caption("Real models use bigger pieces (tokens): GPT-4's vocabulary has about 100,000.")

    with step3:
        st.markdown(
            f"The model scores all **{len(VOCAB)} possible next characters**, and softmax turns the scores into "
            "probabilities. These are the 10 most likely candidates for the **first** character of the reply:"
        )
        top = torch.topk(info["probabilities"], 10)
        st.bar_chart(
            [{"character": show_char(ITOS[i]), "probability": p}
             for p, i in zip(top.values.tolist(), top.indices.tolist())],
            x="character", y="probability", horizontal=True, sort="-probability",
        )
        st.caption("Change the temperature and send again: low makes the top bar dominate; high flattens the bars.")

    with step4:
        st.markdown(
            "The reply is written **one character at a time**: guess, pick one at random (weighted by probability), "
            "append it, slide the window, repeat. These are the first steps:"
        )
        rows = [
            {
                "step": n,
                "model sees (end of window)": visible(t["window"][-12:]),
                "top 3 guesses": "  ·  ".join(f"{visible(c)} {p:.0%}" for c, p in t["top"]),
                "picked": visible(t["picked"]),
                "chance": f"{t['picked_p']:.0%}",
                "note": "" if t["picked"] == t["top"][0][0] else "🎲 not the top guess",
            }
            for n, t in enumerate(info["trace"], start=1)
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            "The 🎲 rows are why the same prompt gives different replies. Set a seed in the sidebar to make it repeatable."
        )
