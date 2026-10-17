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


def make_reply(
    model: ShakespeareGPT, messages: list[dict[str, str]], max_new_tokens: int
) -> str:
    prompt = "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    )
    prompt = "".join(character for character in prompt if character in STOI)
    prompt = prompt[-BLOCK_SIZE:] or "\n"
    prompt_tokens = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    generated = model.generate(prompt_tokens, max_new_tokens)
    return decode(generated[0, prompt_tokens.shape[1] :].tolist()).strip()


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
    st.caption(f"Running on {DEVICE}.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
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
            reply = make_reply(model, st.session_state.messages, response_length)
        st.markdown(reply or "_The model returned an empty continuation._")
    st.session_state.messages.append({"role": "assistant", "content": reply})
