# Mefarshei HaOtzar

A static study site: pick a pasuk or daf, see the mefarshim, search by meaning.
**Early version.** Exact-word search is on; the AI meaning-search data files are being uploaded and the toggle shows "AI soon" until then.
No server, no build step. Open `index.html` through any static host.

## What it does
- Reader: 39 Tanakh books, 37 Gemara tractates, Hebrew/Aramaic first with English beneath.
- Commentary pane per verse/segment, with Rashi + Tosafot first and kashya / terutz / dispute / cites-Rashi badges and filters. The tags are keyword heuristics, not rulings.
- Tap a word or select a phrase to translate.
- Smart search: type a reference (`Berakhot 2a`, `Genesis 1:3`, `rashi on Genesis 1:1`) or an idea.
  - English idea: MiniLM embeddings of ~104k Tanakh/Talmud segments (int8, shipped as `sem_*.bin`), top 50 reranked in the browser with a cross-encoder.
  - Hebrew idea (experimental): a small hash-ngram model (`he_nn_*`).
  - Exact-word search through Sefaria's search API.
- My sources: save passages with notes, group by topic, copy, download .md, print/PDF, backup/restore.

## Measured quality (held-out, honest)
Realistic topic eval: 85 Sefaria topics, full 104k-segment index, relevant = segments Sefaria tags with the topic (sparse, so absolute precision is a floor). P@10 / nDCG@10 / any-hit@10:
| setup | topic-name queries | topic-description queries |
|---|---|---|
| MiniLM dense (before) | 20.4 / 20.6 / 80% | 20.2 / 21.2 / 68% |
| + ms-marco MiniLM-L-6 rerank top-100 | 24.1 / 25.4 / 86% | 20.5 / 21.7 / 71% |
Tried without a clear win: bge-small/base/large, e5-small/base, gte-small, BM25 hybrid, bge-reranker-base.
Hebrew excerpt to English source (Bava Kamma, Psalms, Sotah held out, pool 4,000), R@1/5/10: shipped 52.7/73.3/79.8; BEREL fine-tune 58.6/79.7/85.5 (not shipped, ~110 MB). Not 99%.

## Sources and licenses
- Texts: Sefaria (https://www.sefaria.org) public API. Each text keeps its own license (many CC-BY, CC-BY-SA, CC0 or public domain). Check the version before reusing.
- Models (Apache-2.0): sentence-transformers/all-MiniLM-L6-v2, cross-encoder/ms-marco-MiniLM-L-6-v2 (ONNX via Xenova), run with transformers.js (Apache-2.0) loaded from jsDelivr / Hugging Face at runtime. BEREL 3.0 (dicta-il, Apache-2.0) was used only for evaluation.

## Not verified
Real-device phone testing; Safari/Firefox; offline use; the keyword tags as a judgment of kashya/terutz.
