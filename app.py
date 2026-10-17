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


def build_context(messages: list[dict[str, str]]) -> str:
    """Turn the chat into play dialogue and keep only what fits in the model's context window."""
    script = "".join(
        f"{SPEAKERS[message['role']]}:\n{message['content']}\n\n" for message in messages
    )
    script += f"{SPEAKERS['assistant']}:\n"  # cue the model to speak next, as the BARD
    script = "".join(character for character in script if character in STOI)
    return script[-BLOCK_SIZE:]


def make_reply(
    model: ShakespeareGPT,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> tuple[str, str, torch.Tensor]:
    """Return (reply, exact context the model saw, probabilities for the first reply character)."""
    context = build_context(messages)
    context_tokens = torch.tensor([encode(context)], dtype=torch.long, device=DEVICE)
    first_probabilities = model.next_token_probabilities(context_tokens, temperature, top_k)[0]
    generated = model.generate(context_tokens, max_new_tokens, temperature, top_k)
    reply = decode(generated[0, context_tokens.shape[1] :].tolist())
    reply = reply.split("\n\n")[0]  # a blank line ends the speech; drop the next speaker's lines
    return reply.strip(), context, first_probabilities.cpu()


def show_char(character: str) -> str:
    return {"\n": "⏎ (newline)", " ": "␣ (space)"}.get(character, character)


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
    show_internals = st.toggle("Show what the model sees", value=True)
    st.caption(f"Running on {DEVICE}.")
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
            reply, context, probabilities = make_reply(
                model, st.session_state.messages, response_length, temperature, top_k
            )
        st.markdown(reply or "_The model returned an empty continuation._")
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.internals = (context, probabilities)

if show_internals and "internals" in st.session_state:
    context, probabilities = st.session_state.internals
    with st.expander("🔍 What the model saw, and what it considered next", expanded=True):
        st.markdown(f"**Context window: the last {BLOCK_SIZE} characters**, exactly as fed to the model:")
        st.code(context.replace("\n", "⏎\n"), language=None)
        st.caption(
            "Everything earlier in the chat is invisible to the model. The chat is formatted "
            "as a play (`YOU:` / `BARD:`) because that's what it learned from."
        )
        top = torch.topk(probabilities, 10)
        st.markdown("**Top 10 candidates for the first character of the reply** (after temperature and top-k):")
        st.bar_chart(
            [{"character": show_char(ITOS[i]), "probability": p}
             for p, i in zip(top.values.tolist(), top.indices.tolist())],
            x="character",
            y="probability",
            horizontal=True,
            sort="-probability",  # most likely first
        )
        st.caption("The model samples one character from this distribution, appends it, and repeats.")
