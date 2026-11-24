# Atlas

A workbench for the geometry of meaning. This first piece is the **fibre bundle**: a way to see one text through many eyes at once and measure how those views fit together.

## The idea in plain words

- **Base:** the text's aligned units in reading order (verses, sentences, paragraphs, chapters).
- **Fibres:** every view of those units. A view can be a language, a translator, one hop of a translation chain, or an embedding model. Each view gives every unit a point in its own space.
- **Gauge:** a view's axes are arbitrary. You can rotate them without changing a single similarity, so Atlas only reports things that survive any rotation.
- **Connection:** the best rotation that carries one view onto another.
- **Holonomy:** compose the connections around a closed loop of views, for example Chinese to Japanese to French to English and back to Chinese. If every view held the same geometry, you would come back unchanged. What's left over is a rotation, and its angles say how much meaning twists around that loop.
- **Twist:** the share of the text's energy that moves when carried around the loop. It weights each direction by how much of the text lives there, so the poorly determined directions, where flips are just noise, count for little. This is the headline number.
- **Shared geometry:** the single geometry all views agree on best (generalised Procrustes). What each view has left over is its own sub-pattern.

Every result comes with nulls:

- The **shuffled** null destroys the alignment. That is what "no shared meaning" looks like.
- The **noise** (flat) null keeps the shared geometry and how much each view differs, but scatters those differences at random. Loops then nearly close, so twist beyond this null means the views' differences are structured.

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core, runs anywhere, no downloads needed
pip install -e ".[models]"         # adds sentence-transformers and PyTorch for neural models
python -m pytest                   # 14 tests, a few seconds
```

## Run it

**The Bible in many languages** (human translations, verse-aligned). Download the XML files you want from https://github.com/christos-c/bible-corpus (folder `bibles/`) into `data/bible/`, then:

```bash
./experiments/bible_18.sh data/bible reports/bible_18_languages.html
```

**Your own texts.** Put one file per view with one unit per line, so that line *i* is the same unit in every file:

```bash
atlas bundle --files zh=texts/zh.txt ja=texts/ja.txt fr=texts/fr.txt en=texts/en.txt \
  --embed st:all-mpnet-base-v2 st:sentence-transformers/LaBSE \
  --loop zh ja fr en --title "My chain" --out reports/my_chain.html
```

Passing several `--embed` specs turns every language and model pair into its own fibre. The loop then runs through the first model, and the other loops and triangles cover everything else.

Embedder specs:

- `lsa:200`: character-statistics LSA, fitted separately per view, no downloads.
- `st:<model>`: any sentence-transformers model.
- `ollama:<model>`: an embedding model served by a local Ollama.

Embeddings are cached in `.atlas_cache/`, so nothing is embedded twice.

**The scale bundle** takes each chapter apart into words, verses and paragraphs and measures how they reassemble, across languages:

```bash
./experiments/scales_12.sh data/bible reports/scales_12_languages.html
```

It builds one fibre per language and scale (word, verse, paragraph, whole), so the fibres form a grid, and reports:

- **Scale loops:** does reassembly close?
- **Plaquettes:** down a scale then across a language, against across then down, as in lattice gauge theory.
- **Binding:** what the best linear reassembly of a chapter's parts misses of the whole, and whether the same chapters bind in every language.
- **Renormalisation test:** do the top directions survive from words up to the whole?

## In Python

```python
from atlas.loaders import load_bibles
from atlas.embed import embed_corpus, EmbeddingCache
from atlas.bundle import Bundle

corpus = load_bibles("data/bible", ["English", "French", "Chinese", "Japanese"]).aggregate("chapter")
emb = embed_corpus(corpus, "lsa:200", EmbeddingCache(".atlas_cache"))
b = Bundle.from_embeddings(emb, k=32, unit_ids=corpus.unit_ids)

b.consensus().shared_fraction                 # how much geometry all views share
b.holonomy(["Chinese", "Japanese", "French", "English"]).plane_angles
b.twist(["Chinese", "Japanese", "French", "English"], times=2)   # the spinor test
b.unit_twist(["Chinese", "Japanese", "French", "English"])       # where in the text it twists
b.local_holonomy(["Chinese", "Japanese", "French"], window=120, step=10, k_local=8)
```

## Layout

```
src/atlas/corpus.py      aligned units, roll-up to coarser units, segmentation by scale
src/atlas/loaders/       Bible corpus, plain parallel files
src/atlas/embed.py       embedders and the disk cache
src/atlas/bundle.py      connections, holonomy, twist, local holonomy, shared geometry
src/atlas/scales.py      the scale bundle: four scales per unit, plaquettes, binding, persistence
src/atlas/nulls.py       random rotations, shuffled and flat nulls
src/atlas/spectra.py     eigenvalue spectra, log vs power-law decay
src/atlas/report.py      the HTML report
src/atlas/cli.py         the atlas command
experiments/             one script per experiment, kept so every result can be rerun
tests/                   synthetic bundles where the right answer is known
```

## Next

1. **Scale bundle on your own texts,** with real paragraphs (the Bible version uses five-verse blocks).
2. **Several translators per language,** to separate language from translator.
3. **Translation chains** run locally with NLLB-200, plus your War and Peace rerun.
4. **The seven views,** starting with the tensor stack.
