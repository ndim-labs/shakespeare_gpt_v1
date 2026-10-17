# Opening demo: Shakespeare GPT

**Decoding Language Models · session opener** · about 10 minutes

The idea: open the session with a working "chatbot", let the audience play with it, then pull back the curtain step by step. Every tab in the app is one concept the session teaches. The session then **builds this app together**, from the notebook to the UI.

## Before the session

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

- Sidebar: **Seed = 42** (so your demo is repeatable), Temperature 1.0, Top-k 0, "Explain how the reply was made" **on**.
- Have the [1.0 notebook](https://colab.research.google.com/github/ndim-labs/p1-gpt-from-scratch/blob/main/1.0-preparation/gpt_dev.ipynb) open in another tab.

## The demo, beat by beat

| # | Do | Say | Concept |
|---|---|---|---|
| 1 | Point at the title and the **Model card** | "This is a language model we trained ourselves: 0.2 million parameters. GPT-3 has 175 billion. Let's see what a tiny one can do." | Scale |
| 2 | Type `ROMEO:` and send | "It writes Shakespeare-ish English, but it's not *answering* me. It's **continuing a play**." | A language model predicts the next character |
| 3 | Type `hi, how are you?` | "Ask it a question and it still just continues text. Answering questions needs **instruction tuning**, which is Phase 2." | Base model vs chat model |
| 4 | Open tab **① What it can see** | "The chat becomes a play script. Only the **yellow** part reaches the model: 32 characters. The grey part? It can't see it at all. No memory." | Context window |
| 5 | Open tab **② Text → numbers** | "The model never sees letters, only these numbers. One ID per character, 65 in total." | Tokenization |
| 6 | Open tab **③ Next-character guess** | "For each step it scores all 65 characters and turns the scores into probabilities. Here are its top guesses for the first letter." | Logits → softmax |
| 7 | Open tab **④ One character at a time** | "It picks one *at random*, weighted by those probabilities. See the 🎲 rows? It didn't pick the favourite. That's why the same prompt gives different replies." | Sampling, autoregression |
| 8 | Set **Temperature 0.1**, resend `ROMEO:`, then **2.0** | "Low temperature: safe and repetitive. High temperature: creative, then gibberish. Watch tab ③: the bars go from one tall spike to flat." | Temperature |
| 9 | Set **Seed 0** and resend twice | "Different every time. With a seed, the dice are fixed, which is how you make demos repeatable." | Randomness |
| 10 | Close | "By the end of today you'll understand every piece of this, and we'll build it ourselves." | → session |

## Questions the audience usually asks

| Question | Short answer |
|---|---|
| "Why does it talk nonsense?" | It's tiny (0.2M parameters), sees only 32 characters, and trained for a few minutes. ChatGPT-class models are about a million times bigger, read far more context, and are instruction-tuned. |
| "Does it understand me?" | No. It predicts likely next characters based on patterns in Shakespeare. Understanding-like behaviour only appears at much larger scale. |
| "Why YOU: and BARD:?" | The model learned from plays (`ROMEO:` on one line, then his speech). Formatting the chat the same way gives it something familiar to continue. |
| "How is this different from ChatGPT?" | Same core idea (predict the next token with a Transformer), different scale, tokens instead of characters, and extra training (SFT, RLHF) to follow instructions. |

## Build-along: how the session builds this app

After the demo, the audience follows along from the notebook to this app:

| Step | What we build | Where |
|---|---|---|
| 1 | Train the GPT and **save** weights + vocabulary to `shakespeare_gpt_v1.pth` | Notebook, last cells |
| 2 | **Rebuild the same architecture** and load the weights | `app.py`: `Head` → `ShakespeareGPT`, `load_model()` |
| 3 | **Generate**: next-character probabilities, temperature, top-k, sampling loop | `next_token_probabilities()`, `make_reply()` |
| 4 | **Chat UI**: messages, input box, play-style prompt | Streamlit: `st.chat_message`, `st.chat_input`, `build_script()` |
| 5 | **Explain it**: the four tabs | `script_html()`, `tiles_html()`, bar chart, trace table |

Each step runs on its own before moving on, so nobody gets lost.
