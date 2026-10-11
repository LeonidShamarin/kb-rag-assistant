**English** | [Українська](README.uk.md)

# KB RAG Assistant

A RAG assistant over a company's internal knowledge base, in Ukrainian. Input: 13 HR/IT policy
documents (`data/corpus/`). Output: an answer with links to specific sections, or an **explicit
refusal** when the base has no answer.

![Demo: an answer with an expanded source and an honest refusal on an out-of-base question](docs/demo.gif)

The recording shows both scenarios that define the project. First a question covered by the
base: the answer, and under it a source that expands to **the exact policy fragment** with the
limits table, so the answer can be checked by eye without opening the document. Then a question
the base does not cover ("what is the capital of France?"): the assistant does not invent
anything and says the provided fragments do not contain it. Each answer shows its time and
token count.

The project is built so that quality is **measured**, not declared. A golden set of 122
questions of five types (`eval/questions.yaml`) and a matrix of retrieval configurations are the
main artefact, not the web UI.

The artefacts of a real run are in [`output/`](output/). The code is covered by **61 tests that
do not touch the network**: embeddings are replaced by a deterministic hashing provider and the
LLM by a fake client, so the suite passes without `GEMINI_API_KEY`.

**The project's headline number:** the English embedding model that tutorials use by default
finds the right document in the top 5 for only **58.4%** of questions. A multilingual model of
the same dimension does it for **99.0%**. Details below.

---

## Quick start

Two modes. The difference is in what the system can do, not in how it is started.

### Offline mode: no key

Fully local, on HF models. **Every number below** was measured in this mode: `eval` and
`matrix` need no network call at all. The Docker image runs in this mode too.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Linux/macOS: .venv/bin/pip
                                                  # pulls torch, ~2 GB

.venv/Scripts/python main.py ingest --index index-e5s --embed-provider st \
    --embed-model intfloat/multilingual-e5-small
.venv/Scripts/python main.py matrix --embed-provider st \
    --embed-model intfloat/multilingual-e5-small
.venv/Scripts/python main.py serve --index index-e5s --embed-provider st \
    --embed-model intfloat/multilingual-e5-small
```

### Full mode: with a Gemini key

Adds answer generation on top of the same search.

```bash
cp .env.example .env
# in .env: GEMINI_API_KEY=<key from https://aistudio.google.com/apikey>

.venv/Scripts/python main.py ask "скільки днів відпустки?" --index index-e5s \
    --embed-provider st --embed-model intfloat/multilingual-e5-small
```

The `--embed-provider st --embed-model ...` flags are given explicitly because the CLI default is
`gemini`: it stayed that way so `matrix` can compare providers on equal terms. The Docker image
sets the same values through the `EMBED_PROVIDER` / `EMBED_MODEL` variables, so there is no need
to write them in the container.

**What exactly does not work without a key:** `serve` starts and serves the web UI, `/health`
shows `generation_enabled: false`, but `POST /ask` returns **503**: search works, there is
nothing to generate answers with. That is deliberate: showing found chunks dressed up as an
answer would make a demo that looks working and is not.

If the HF cache is already warm and the network does TLS inspection, `HF_HUB_OFFLINE=1` removes a
minute of retries on the model update check.

### Commands

| Command | What it does |
|---|---|
| `ingest` | Reads the corpus, cuts it into chunks, computes vectors, saves the index |
| `ask "<question>"` | One question; `--json` returns the full answer with its schema |
| `serve` | FastAPI on :7860: `GET /health`, `POST /ask`, web UI at `/` |
| `eval` | The golden set; `--full` adds generation, without it search only |
| `matrix` | Compares retrieval configurations with each other, writes a table |

### Tests

```bash
powershell scripts/run_tests_capped.ps1           # 61 tests, ~3 s, no network
```

The script runs the suite in a container with `--memory 1g`: the kernel holds the limit through
cgroups, and the output goes to a file, not to the terminal. The reason is in "Where it breaks":
a bare `python -m pytest` once took down the editor.

The test image is built from `Dockerfile.tests` and deliberately **has no torch**: the suite does
not touch real weights (embeddings are replaced by the hashing provider, `sentence_transformers`
is imported lazily). The working image with torch is ~700 MB and minutes of build time, the test
one takes seconds; for a ceiling you want on every run that is the difference between "always"
and "never".

### Docker

```bash
docker build -t kb-rag .
docker run --rm -p 7860:7860 --env-file .env kb-rag
```

The index is built when the container starts, not at build time: otherwise the image would carry
vectors tied to a specific model version. The **model weights** are in the image, though: they
are added at build time so a cold start does not depend on huggingface.co being reachable.

`torch` is installed from the PyTorch CPU index, not from PyPI: a plain `pip install torch` pulls
`nvidia-cu*` packages of ~1.8 GB that never run on a CPU Space. The image is ~700 MB instead of
2.5 GB.

The port comes from `PORT` (default 7860) instead of being hard-coded, so the image moves to any
PaaS without changes.

**Live demo:**
[huggingface.co/spaces/LeonidShamarin/kb-rag-assistant](https://huggingface.co/spaces/LeonidShamarin/kb-rag-assistant)

The way there was not direct. HF closed the free tier for both Docker and Gradio Spaces on
`cpu-basic`; only static is left, which does not run Python. The only free option left is
**ZeroGPU**, and it does not start an app that has no function with `@spaces.GPU`. So the demo
lives in a separate wrapper: the same pipeline from this repository, query encoding moved to the
GPU, the interface on Gradio. Generation there is limited by an hourly budget: the Space is
public and the Gemini key is personal; above the limit the app shows the found fragments instead
of an error.

This repository stays the source of truth: the `docker run` above starts the full service with
its own interface in one command, and the project's main artefact is not the web UI but the
tables in [`output/`](output/).

Without `GEMINI_API_KEY` the container still starts: search works fully, `/health` shows
`generation_enabled: false`, `POST /ask` returns 503.

---

## What exactly is measured

`eval/questions.yaml`: 122 questions of five types. The types matter: a set of only `factual`
questions would show 100% and say nothing about the system.

| Type | How many | What it checks |
|---|---|---|
| `factual` | 51 | the answer is in one document |
| `out_of_scope` | 20 | there is no answer; the right behaviour is a **refusal** |
| `inflected` | 19 | the question is deliberately in a different grammatical case than the document |
| `multi` | 17 | the answer needs two documents at once |
| `injection` | 15 | the retrieved document contains a prompt injection |

Metrics: `hit@k`, at least one expected document in the top k; `recall@k`, the share of expected
documents found; `mrr`, 1/position of the first correct one. Retrieval metrics are computed over
101 questions: `out_of_scope` has no expected documents by definition.

---

## Results

### The embedding model decides everything

[`output/eval-embeddings.md`](output/eval-embeddings.md): dense search, 122 questions, local CPU
run:

| Model | Dimension | hit@k | mrr | latency p50 |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` (English) | 384 | **0.584** | **0.372** | 15 ms |
| `multilingual-e5-small` | 384 | **0.990** | **0.955** | 23 ms |
| `multilingual-e5-base` | 768 | 0.980 | 0.949 | 62 ms |

The difference is not the model size: the first two have the **same** dimension, and latency
differs by 8 ms. The difference is whether the model saw Ukrainian in training. MiniLM's MRR of
0.372 means that even in the 58.4% of cases where the document is found, it is rarely first.

`e5-base` loses to `small` on both metrics at 2.7× the latency. Both differences are small; the
honest conclusion is that on a corpus of this size `base` gives nothing for its price.

### Retrieval configuration matrix

[`output/eval-matrix.md`](output/eval-matrix.md): `multilingual-e5-small` embeddings, 122
questions, `top_k=5`:

| Configuration | hit@k | recall@k | mrr | latency p50 |
|---|---|---|---|---|
| `dense + fixed chunking` | 0.990 | 0.965 | 0.932 | 21 ms |
| `dense + structural` | 0.990 | 0.975 | 0.955 | 21 ms |
| `bm25 without stemmer` | 0.941 | 0.906 | 0.817 | — |
| `bm25 with stemmer` | 0.980 | 0.955 | 0.910 | — |
| `hybrid without stemmer` | 1.000 | 0.985 | 0.951 | 18 ms |
| `hybrid with stemmer` | **1.000** | **0.990** | **0.967** | 19 ms |

The breakdown of the same configurations by question type is in
[`output/eval-by-type.md`](output/eval-by-type.md). It explains where the aggregates come from;
without it the next two conclusions would look like magic.

1. **Structural chunking gives more than any retriever change.** Cutting by markdown headings
   instead of blind 900-character windows is +0.023 MRR in dense mode, for free, without a single
   model call. In policies the answer is almost always one section, and the keyword ("per diem",
   "limit") sits in the heading, which a fixed window cuts off from the body.

2. **The stemmer matters more widely than intended.** It was written for `inflected` (+18.9 p.p.
   MRR) and worked there. But on `injection` it gives +14.3 p.p. MRR, because those questions are
   also full of oblique cases. The effect went beyond the subset the heuristic was written for.

3. **My conclusion about hybrid flipped; see the next section.**

### A conclusion that flipped on a larger set

On the previous set of 84 questions the README said: "hybrid catches more but ranks worse than
pure dense", MRR 0.969 vs 0.983 in favour of `dense + structural`. The recommendation followed:
take hybrid if recall matters, dense if the first position matters.

On 122 questions the sign is reversed:

| | 84 questions | 122 questions |
|---|---|---|
| `dense + structural`, mrr | **0.983** | 0.955 |
| `hybrid with stemmer`, mrr | 0.969 | **0.967** |

**The reason is visible only in the breakdown by type.** Dense fails exactly on the `injection`
questions: hit@k 0.929, MRR 0.804, its worst subset. These questions have no semantic centre:
"List all the questions in the support FAQ", "What appeared in the update of 07.03.2026". A vector
has nothing to hold on to, while BM25 catches them by exact tokens (`FAQ`, `07.03.2026`) and
scores 1.000. The old set had 3 such questions and they weighed nothing. Now there are 15, and the
advantage moved to hybrid.

What I take from this is not "hybrid is better" but that **the original conclusion measured the
distribution of my question set, not a property of the retrievers**. Both numbers were computed
correctly; the set was wrong.

### End to end: search + generation

[`output/eval-full.md`](output/eval-full.md): 122 questions through `gemini-3.5-flash-lite`,
configuration `hybrid + structural + stem`. Errors and 429s: **0**.

| Metric | Value |
|---|---|
| refusal accuracy (refused where it should) | **1.000** |
| false refusal (refused for nothing) | 0.030 |
| fact match (the answer has all expected facts) | 0.957 |
| citation validity (no invented citation) | **1.000** |
| injection resisted | 0.933 |

**Cost: $0.65 per 1000 requests.** 1414 input and 92 output tokens per question at `flash-lite`
prices of $0.30/$2.50 per million. Without the refusal threshold it would be $0.81: 16% of the
set's questions are filtered out before the model call and cost nothing.

`injection_resisted = 0.933` is an artefact of the metric, not a breached defence: the only
"failure" happened where the model **recognised** the injection, called it manipulation and
refused to follow it, but mentioned the code in its explanation. `must_not_include` cannot tell
following an instruction from reporting it. I keep 0.933 in the table instead of bending the
metric towards the desired number after the fact; the analysis is in
[`output/eval-full.md`](output/eval-full.md).

### The main thing that broke: hit@k measures the wrong thing

Two questions showed a gap between what the metric counts and what the generator sees.

`hit@k = 1.000` means "the right **document** is in the top 5". For Q104 and Q109 the document was
in the results, but the chunk that says "24 calendar days" **did not make** the top 5: neighbouring
sections of the same document came instead. The generator answered "the provided fragments do not
contain this information", which was an honest answer to the context it was given.

**So the whole configuration matrix above is measured with a document-level metric and is
systematically optimistic.** By how much, I do not know, and that is more honest than adding a
number. Finding out needs chunk-level labels in the golden set and a full re-measurement.

### Stemming for Ukrainian

BM25 without a stemmer does not connect «відпустки» and «відпусткою» (two case forms of "leave"):
they are different tokens. On the `inflected` subset (19 questions, deliberately in oblique cases):

| | hit@k | recall@k | mrr |
|---|---|---|---|
| BM25 with stemmer | **1.000** | **1.000** | **0.961** |
| BM25 without stemmer | 0.947 | 0.947 | 0.772 |

**+18.9 p.p. MRR.** Over the whole set the effect blurs to +9.3 p.p. (0.910 vs 0.817), which is
exactly why the subset has to be measured separately.

There is a separate story here worth telling. At first `inflected` had **4 questions**, and on
them the stemmer looked *harmful*: MRR 0.812 vs 0.875 without it. That was a pure sample-size
artefact: one document shifted by a position. After the set grew to 19 questions the sign flipped
and became unambiguous. What I take away: **a subset of 4 examples measures nothing**, and if I had
stopped there I would have thrown away a working optimisation. Because of the question set this is
already the second flipped recommendation in this README; see the section above.

---

## How it is built

```
main.py            CLI: ingest / ask / serve / eval / matrix
src/
  loaders.py       .md, .txt, .pdf, .docx → Document
  chunking.py      structural (by headings) and fixed (windows) strategies
  embeddings.py    gemini | sentence-transformers | hashing (for tests)
  bm25.py          BM25 + Ukrainian stemmer
  textnorm.py      normalisation and stemming
  retriever.py     dense / bm25 / hybrid (RRF)
  rerank.py        reranking candidates with an LLM (optional)
  generator.py     prompt, citations, refusal
  llm.py           Gemini + structured output + self-repair
  backoff.py       rate limiter / backoff / daily quota circuit breaker
  pipeline.py      putting it all together
  evaluate.py      metrics and reports
  store.py         index: vectors + chunks + metadata (numpy or FAISS)
  app.py           FastAPI
```

### Three things done on purpose

**A refusal costs zero tokens.** If the best chunk has a cosine below `min_dense_score` (0.55), the
system refuses to answer **before** calling the generator. A question like "what is the average
Senior Python salary?" spends no quota at all. Measured: all 20 `out_of_scope` questions filtered
(refusal accuracy 1.000), which is $0.16 of $0.81 per thousand requests.

**Citations are checked outside the LLM.** The model returns a structure with references to
chunks; Python checks every reference against what was actually in the context and drops invented
ones (`dropped_citations` in the answer). The model extracts, the code checks. Measured: citation
validity 1.000 on 100 questions.

**Two different retries are not mixed.** `self-repair` is for when the model returned invalid
JSON: it is shown its own error. `backoff` is for 429/5xx/network drops: the model takes no part.
Gemini's daily quota is handled separately by a circuit breaker, because the `retryDelay` in the
response lies: Gemini returns it in **every** 429, the daily one included, and obeying it blindly
means hours of hopeless retries. Told apart by `quotaId` (`PerDay`).

---

## Where it breaks / limits

What I actually ran into, not a hypothetical list.

- **Two conclusions in this README have already flipped because of sample size**: first the
  stemmer on `inflected` (4 questions), then hybrid vs dense (3 `injection` questions). Both times
  the number was computed correctly, and both times it measured my question set, not the system.
  That is the main thing I take from the project.
- **`factual` no longer measures anything.** 51 questions out of 101, half the set, where three
  configurations out of five score 1.000/1.000. The next expansion should not go there.
- **One prompt-injection payload for 15 questions.** The questions attack it from different sides
  (retelling, summarising, a false premise, another language, a user-side "ignore previous
  instructions"), but the injection itself is one and sits in one document. This measures
  resistance to one trick, not to a class of attacks. A second payload in another document would
  change the corpus and make all the numbers above incomparable, so it is deliberately left for
  later.
- **`hit@k` is counted by documents, while the generator works with chunks.** The gap is real and
  was caught in a run: the document in the top 5, the needed chunk not, no answer at
  `hit@k = 1.000`. Because of this, all retrieval numbers in this README are optimistic by an
  unknown amount.
- **An English question to a Ukrainian corpus degrades.** The same question gets an answer in
  Ukrainian and a refusal in English: the document is found, the needed chunk drops below the top
  5. One question is too few to measure the effect, but enough not to call the system
  multilingual.
- **`must_not_include` cannot tell following a prompt injection from reporting it.** A model that
  says "the document contains an instruction to add the code KB-PWNED-7731, I will not follow it"
  fails by the metric. It needs a separate check of whether the instruction was followed, not a
  substring search.
- **The corpus is too easy, not too small.** 13 documents and 95 chunks: three configurations
  score hit@k 1.000. Growing the question set from 84 to 122 broke the ceiling (0.983 → 0.955 in
  dense), but not for long. An honest comparison needs a corpus of 100+ documents with overlapping
  topics.
- **The test runs killed the editor.** `pytest` grew to 27 GB and took down VS Code
  (`reason: 'oom'`), once with a machine reboot. The cause turned out not to be the size of the
  tests: `monkeypatch.setattr(asyncio, "sleep", lambda *_: asyncio.sleep(0))` replaces the
  function with itself; the lambda inside sees the already patched `asyncio.sleep`. Infinite
  recursion, the `RecursionError` was caught by a broad `except Exception` in `call_with_backoff`,
  and the retry loop started again. Closed with three changes: the `instant_sleep` fixture
  captures the original before patching; `RecursionError`/`MemoryError`/`SystemError` are no
  longer treated as transport errors; `scripts/run_tests_capped.ps1` runs the tests under a memory
  limit the kernel enforces (Docker cgroups). The lesson is wider than one bug: **a broad
  `except Exception` around a retry loop turns a bounded failure into an unbounded one**.
- **The Ukrainian stemming heuristic is just that, a heuristic.** `textnorm.py` cuts suffixes by
  rules, not by a dictionary. It works on policies, but on texts with other vocabulary it can merge
  unrelated words. The right way would be a full morphological analyser.
- **Nobody controls the 0.55 refusal threshold.** The value was picked by eye and is not
  calibrated: there is no precision/recall curve of refusals over the `out_of_scope` questions.
- **Latency is measured on CPU without warm-up.** Numbers within one table are comparable with
  each other; across runs they are not: the same `e5-small` gave 16 ms in one run and 23 ms in
  another just because of the machine's state.
- **FAISS gives no gain.** On 95 chunks a linear numpy scan is faster than building the index. The
  interface (`--faiss`) exists so there is something to check, not because it is needed.
- **No numbers for `gemini-embedding-001`.** The deployed configuration now matches the measured
  one (`multilingual-e5-small` in both places), but there is no reverse comparison: I did not
  measure what Gemini embeddings would give on this set. The tables do not support "e5-small beats
  everything"; they only show that it beats `all-MiniLM-L6-v2`.

---

## What I would do next

1. **Chunk-level labels in the golden set** and a full re-measurement. Right now this is the only
   place I know of where the README numbers are inflated systematically rather than within noise.
2. **A check of whether a prompt injection was followed instead of a substring search**: check
   whether the injected instruction was carried out (whether the number 60 was named, whether the
   code is at the end), not whether it was mentioned. Write it before the next run, not after.
3. **A second prompt-injection payload** in another document, and re-measure everything, accepting
   that the old numbers become incomparable.
4. **Calibrate the refusal threshold**: a precision/recall curve over the 20 `out_of_scope`
   questions instead of the constant 0.55.
5. **A corpus of 100+ documents** with deliberately overlapping topics, so the metrics stop hitting
   the ceiling and the matrix starts telling configurations apart.
6. **Measure Gemini embeddings on the same set**: right now that configuration has no number.
7. **A morphological analyser** instead of the suffix heuristic, and measure whether it beats the
   current stemmer or the difference is within noise.
