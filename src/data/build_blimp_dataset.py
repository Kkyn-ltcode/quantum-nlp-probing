# “””
BLiMP Dataset Builder

End-to-end pipeline: download → parse → fingerprint → save → validate

What this script does:
1. Downloads 4 BLiMP sub-tasks from HuggingFace
2. Runs BobcatParser on every grammatical sentence
3. Extracts structural fingerprints from parsed diagrams
4. Saves everything in a clean, versioned format
5. Generates statistics and a data/blimp/README.md

Run with:
conda activate qnlp
python build_blimp_dataset.py

Output:
data/blimp/
├── raw/                        # Raw BLiMP jsonl files (one per sub-task)
├── parsed/                     # Serialized diagrams (.pkl per sub-task)
├── embeddings/                 # Structural fingerprints (.npy per sub-task)
├── splits/                     # Train/val/test splits (.pt per sub-task)
├── blimp_master.pt             # All sub-tasks merged, ready for experiments
└── README.md                   # Dataset statistics and provenance
“””

import json
import pickle
import hashlib
import logging
import warnings
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict

warnings.filterwarnings(“ignore”)
logging.basicConfig(
level=logging.INFO,
format=”  %(asctime)s  %(levelname)s  %(message)s”,
datefmt=”%H:%M:%S”,
)
log = logging.getLogger(**name**)

# ──────────────────────────────────────────────

# CONFIG

# ──────────────────────────────────────────────

BLIMP_TASKS = {
“relative_clause”: {
“hf_name”: “relative_clause”,
“description”: “Subject and object relative clauses”,
“relevance”: “Core phenomenon — directly tests relative clause syntax”,
“priority”: 1,
},
“filler_gap_dependency”: {
“hf_name”: “filler_gap_dependency”,
“description”: “Long-distance filler-gap dependencies”,
“relevance”: “Tests wh-movement and long-distance dependencies”,
“priority”: 2,
},
“anaphor_agreement”: {
“hf_name”: “anaphor_agreement”,
“description”: “Reflexive pronoun agreement”,
“relevance”: “Pure syntactic agreement — zero semantic content”,
“priority”: 3,
},
“wh_questions_object_gap”: {
“hf_name”: “wh_questions_object_gap”,
“description”: “Wh-questions with object gap”,
“relevance”: “Complements filler_gap, different gap position”,
“priority”: 4,
},
}

FINGERPRINT_DIM = 16        # Fixed size for all fingerprint vectors
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15      # Must sum to 1.0
RANDOM_SEED     = 42

DATA_DIR = Path(“data/blimp”)

# ──────────────────────────────────────────────

# DATA STRUCTURES

# ──────────────────────────────────────────────

@dataclass
class ParsedSentence:
“”“Single sentence with all representations attached.”””
sentence_id:   str
subtask:       str
sentence:      str
label:         int          # 1 = grammatical, 0 = ungrammatical
parsed:        bool         # Did BobcatParser succeed?
diagram:       object       # lambeq Diagram or None
fingerprint:   Optional[np.ndarray]  # Structural fingerprint or None
parse_error:   str          = “”
split:         str          = “”     # train / val / test

@dataclass
class SubtaskStats:
“”“Statistics for one BLiMP sub-task.”””
subtask:            str
total_sentences:    int = 0
grammatical:        int = 0
ungrammatical:      int = 0
parsed_ok:          int = 0
parse_failed:       int = 0
parse_rate:         float = 0.0
failure_reasons:    dict = field(default_factory=dict)
split_counts:       dict = field(default_factory=dict)

# ──────────────────────────────────────────────

# STEP 1: DOWNLOAD

# ──────────────────────────────────────────────

def download_blimp(tasks: dict, raw_dir: Path) -> dict[str, list[dict]]:
“””
Download BLiMP sub-tasks from HuggingFace.
Falls back to direct GitHub JSONL download if datasets library unavailable.

```
Returns dict: subtask_name → list of raw sentence dicts
"""
raw_dir.mkdir(parents=True, exist_ok=True)
all_raw = {}

log.info("Downloading BLiMP sub-tasks...")

for name, cfg in tasks.items():
    cache_path = raw_dir / f"{name}.jsonl"

    # Use cached file if available
    if cache_path.exists():
        log.info(f"  [CACHED] {name} — loading from {cache_path}")
        rows = [json.loads(l) for l in cache_path.read_text().splitlines() if l.strip()]
        all_raw[name] = rows
        continue

    rows = _download_single_task(cfg["hf_name"], name, cache_path)
    all_raw[name] = rows
    log.info(f"  [OK] {name}: {len(rows)} examples")

return all_raw
```

def _download_single_task(hf_name: str, task_name: str, cache_path: Path) -> list[dict]:
“”“Try HuggingFace datasets first, then direct GitHub download.”””

```
# Method 1: HuggingFace datasets library
try:
    from datasets import load_dataset
    ds = load_dataset("blimp", hf_name, trust_remote_code=True)
    rows = list(ds["train"])
    # Save to JSONL cache
    with open(cache_path, "w") as f:
        for row in rows:
            f.write(json.dumps(dict(row)) + "\n")
    return rows
except Exception as e:
    log.warning(f"  HuggingFace load failed for {task_name}: {e}")

# Method 2: Direct GitHub download
try:
    import urllib.request
    url = (
        f"https://raw.githubusercontent.com/alexwarstadt/blimp/"
        f"master/data/{hf_name}.jsonl"
    )
    log.info(f"  Trying direct download: {url}")
    urllib.request.urlretrieve(url, cache_path)
    rows = [json.loads(l) for l in cache_path.read_text().splitlines() if l.strip()]
    return rows
except Exception as e:
    log.error(f"  Direct download also failed for {task_name}: {e}")
    return []
```

# ──────────────────────────────────────────────

# STEP 2: PARSE

# ──────────────────────────────────────────────

def parse_sentences(
all_raw: dict[str, list[dict]],
tasks: dict,
) -> tuple[dict[str, list[ParsedSentence]], dict[str, SubtaskStats]]:
“””
Run BobcatParser on every grammatical sentence in all sub-tasks.

```
Strategy:
  - Parse only grammatical sentences (label=1). Ungrammatical sentences
    are excluded from CKA analysis because their DisCoCat diagrams are
    often malformed or meaningless — but we record them for reference.
  - Failures are logged with reason for the data README.
  - Sentences that parse successfully are the analysis corpus.
"""
from lambeq import BobcatParser

log.info("Loading BobcatParser...")
parser = BobcatParser(model_name_or_path="bobcat", verbose="suppress")
log.info("BobcatParser loaded.")

all_parsed:  dict[str, list[ParsedSentence]] = {}
all_stats:   dict[str, SubtaskStats]          = {}

for subtask_name, rows in all_raw.items():
    log.info(f"\n  Parsing sub-task: {subtask_name} ({len(rows)} rows)...")
    stats  = SubtaskStats(subtask=subtask_name, total_sentences=len(rows))
    parsed = []

    for idx, row in enumerate(rows):
        # BLiMP format: 'sentence_good' and 'sentence_bad'
        good_sent = row.get("sentence_good", row.get("sentence", ""))
        bad_sent  = row.get("sentence_bad",  "")

        stats.total_sentences += 1  # count both
        stats.grammatical     += 1
        stats.ungrammatical   += 1 if bad_sent else 0

        # --- Parse the grammatical sentence ---
        sent_id = f"{subtask_name}_{idx:05d}_good"
        ps = _parse_one(parser, sent_id, subtask_name, good_sent, label=1)
        parsed.append(ps)

        if ps.parsed:
            stats.parsed_ok += 1
        else:
            stats.parse_failed += 1
            reason = ps.parse_error[:60] if ps.parse_error else "unknown"
            stats.failure_reasons[reason] = stats.failure_reasons.get(reason, 0) + 1

        # Progress log every 100 sentences
        if (idx + 1) % 100 == 0:
            rate = stats.parsed_ok / max(stats.parsed_ok + stats.parse_failed, 1)
            log.info(f"    {idx+1}/{len(rows)}  parse_rate={rate:.1%}")

    # Final parse rate
    attempted = stats.parsed_ok + stats.parse_failed
    stats.parse_rate = stats.parsed_ok / max(attempted, 1)

    all_parsed[subtask_name] = parsed
    all_stats[subtask_name]  = stats

    log.info(
        f"  {subtask_name}: {stats.parsed_ok}/{attempted} parsed "
        f"({stats.parse_rate:.1%})"
    )

return all_parsed, all_stats
```

def _parse_one(
parser,
sent_id: str,
subtask: str,
sentence: str,
label: int,
) -> ParsedSentence:
“”“Parse a single sentence, catching all exceptions.”””
if not sentence.strip():
return ParsedSentence(
sentence_id=sent_id, subtask=subtask, sentence=sentence,
label=label, parsed=False, diagram=None, fingerprint=None,
parse_error=“empty_sentence”,
)
try:
diagram = parser.sentence2diagram(sentence)
if diagram is None:
return ParsedSentence(
sentence_id=sent_id, subtask=subtask, sentence=sentence,
label=label, parsed=False, diagram=None, fingerprint=None,
parse_error=“parser_returned_none”,
)
return ParsedSentence(
sentence_id=sent_id, subtask=subtask, sentence=sentence,
label=label, parsed=True, diagram=diagram, fingerprint=None,
)
except Exception as e:
return ParsedSentence(
sentence_id=sent_id, subtask=subtask, sentence=sentence,
label=label, parsed=False, diagram=None, fingerprint=None,
parse_error=type(e).**name** + “: “ + str(e)[:80],
)

# ──────────────────────────────────────────────

# STEP 3: FINGERPRINT

# ──────────────────────────────────────────────

def extract_fingerprints(
all_parsed: dict[str, list[ParsedSentence]],
) -> dict[str, list[ParsedSentence]]:
“””
Extract structural fingerprints for every successfully parsed sentence.
Fingerprint is a fixed-size (FINGERPRINT_DIM,) numpy vector of
purely structural features — zero semantic content.
“””
log.info(”\n  Extracting structural fingerprints…”)

```
for subtask_name, sentences in all_parsed.items():
    ok_count = 0
    for ps in sentences:
        if ps.parsed and ps.diagram is not None:
            ps.fingerprint = _extract_fingerprint(ps.diagram)
            ok_count += 1
    log.info(f"  {subtask_name}: {ok_count} fingerprints extracted")

return all_parsed
```

def _extract_fingerprint(diagram, dim: int = FINGERPRINT_DIM) -> np.ndarray:
“””
Extract a structural fingerprint vector from a DisCoCat diagram.

```
Features are PURELY structural — identical grammar → identical fingerprint
regardless of word content.

Feature index reference:
    0  n_words              — number of word boxes
    1  n_cups               — syntactic contractions (bonds)
    2  n_caps               — cap morphisms
    3  n_swaps              — crossing wires
    4  n_total_boxes        — all boxes including structural
    5  avg_type_complexity  — avg atomic types per word
    6  max_type_complexity  — max atomic types in any word
    7  n_noun_atoms         — count of 'n' type atoms
    8  n_sent_atoms         — count of 's' type atoms
    9  cups_per_word        — cups / words ratio (syntactic density)
    10 type_spread          — max_complexity - avg_complexity
    11 n_wires_dom          — wires in diagram domain
    12 n_wires_cod          — wires in diagram codomain
    13 n_right_adj          — right adjoint types (.r)
    14 n_left_adj           — left adjoint types (.l)
    15 structural_ratio     — structural boxes / total boxes
"""
boxes = diagram.boxes

n_words, n_cups, n_caps, n_swaps = 0, 0, 0, 0
word_type_complexities = []
full_cod_str_parts = []
n_right_adj, n_left_adj = 0, 0

for box in boxes:
    btype = type(box).__name__
    if btype == "Word":
        n_words += 1
        cod_str = str(box.cod)
        full_cod_str_parts.append(cod_str)
        n_right_adj += cod_str.count(".r")
        n_left_adj  += cod_str.count(".l")
        # Count atomic types (split on tensor product symbol)
        atoms = [
            c.strip() for c in
            cod_str.replace(".r", "").replace(".l", "")
                    .replace("(", "").replace(")", "")
                    .split("@") if c.strip()
        ]
        word_type_complexities.append(len(atoms))
    elif btype == "Cup":
        n_cups += 1
    elif btype == "Cap":
        n_caps += 1
    elif btype == "Swap":
        n_swaps += 1

n_total_boxes  = len(boxes)
n_structural   = n_cups + n_caps + n_swaps
full_cod_str   = " ".join(full_cod_str_parts)
n_noun_atoms   = full_cod_str.count("n")
n_sent_atoms   = full_cod_str.count("s")

avg_complexity  = float(np.mean(word_type_complexities)) if word_type_complexities else 0.0
max_complexity  = float(max(word_type_complexities))     if word_type_complexities else 0.0
cups_per_word   = n_cups / max(n_words, 1)
type_spread     = max_complexity - avg_complexity
struct_ratio    = n_structural / max(n_total_boxes, 1)

try:
    n_wires_dom = len(str(diagram.dom).split("@")) if str(diagram.dom) != "Ty()" else 0
    n_wires_cod = len(str(diagram.cod).split("@")) if str(diagram.cod) != "Ty()" else 0
except Exception:
    n_wires_dom, n_wires_cod = 0, 0

fp = np.array([
    n_words,            # 0
    n_cups,             # 1
    n_caps,             # 2
    n_swaps,            # 3
    n_total_boxes,      # 4
    avg_complexity,     # 5
    max_complexity,     # 6
    n_noun_atoms,       # 7
    n_sent_atoms,       # 8
    cups_per_word,      # 9
    type_spread,        # 10
    n_wires_dom,        # 11
    n_wires_cod,        # 12
    n_right_adj,        # 13
    n_left_adj,         # 14
    struct_ratio,       # 15
], dtype=np.float64)

assert len(fp) == dim, f"Fingerprint dim mismatch: {len(fp)} != {dim}"
return fp
```

# ──────────────────────────────────────────────

# STEP 4: SPLITS

# ──────────────────────────────────────────────

def assign_splits(
all_parsed: dict[str, list[ParsedSentence]],
seed: int = RANDOM_SEED,
) -> dict[str, list[ParsedSentence]]:
“””
Assign train/val/test splits to successfully parsed sentences.
Stratified: each split gets proportional representation from each sub-task.
Unparsed sentences are excluded (split = “excluded”).
“””
rng = np.random.default_rng(seed)
log.info(”\n  Assigning train/val/test splits…”)

```
for subtask_name, sentences in all_parsed.items():
    # Only split successfully parsed sentences
    parseable = [ps for ps in sentences if ps.parsed and ps.fingerprint is not None]
    n = len(parseable)

    if n == 0:
        log.warning(f"  {subtask_name}: no parseable sentences — skipping splits")
        continue

    # Shuffle indices deterministically
    indices = rng.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_idx = set(indices[:n_train].tolist())
    val_idx   = set(indices[n_train:n_train + n_val].tolist())

    for i, ps in enumerate(parseable):
        if i in train_idx:
            ps.split = "train"
        elif i in val_idx:
            ps.split = "val"
        else:
            ps.split = "test"

    # Mark unparsed as excluded
    for ps in sentences:
        if not ps.parsed or ps.fingerprint is None:
            ps.split = "excluded"

    split_counts = defaultdict(int)
    for ps in sentences:
        split_counts[ps.split] += 1

    log.info(
        f"  {subtask_name}: "
        f"train={split_counts['train']}  "
        f"val={split_counts['val']}  "
        f"test={split_counts['test']}  "
        f"excluded={split_counts['excluded']}"
    )

return all_parsed
```

# ──────────────────────────────────────────────

# STEP 5: SAVE

# ──────────────────────────────────────────────

def save_dataset(
all_parsed: dict[str, list[ParsedSentence]],
all_stats:  dict[str, SubtaskStats],
data_dir:   Path,
) -> dict:
“””
Save everything to disk in four formats:

```
1. data/blimp/parsed/{subtask}.pkl     — full ParsedSentence objects (diagrams included)
2. data/blimp/embeddings/{subtask}.npy — fingerprint matrix, parsed-only
3. data/blimp/splits/{subtask}.pt      — split-aware tensors for training
4. data/blimp/blimp_master.pt          — merged, ready for experiments
"""
(data_dir / "parsed").mkdir(parents=True, exist_ok=True)
(data_dir / "embeddings").mkdir(parents=True, exist_ok=True)
(data_dir / "splits").mkdir(parents=True, exist_ok=True)

master_records = []
saved_paths    = {}

for subtask_name, sentences in all_parsed.items():
    parsed_only = [ps for ps in sentences if ps.parsed and ps.fingerprint is not None]

    if not parsed_only:
        log.warning(f"  {subtask_name}: nothing to save (0 parsed sentences)")
        continue

    # 1. Full pickle (includes diagram objects — needed for future fingerprint upgrades)
    pkl_path = data_dir / "parsed" / f"{subtask_name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(sentences, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 2. Fingerprint matrix (parsed only)
    fp_matrix = np.stack([ps.fingerprint for ps in parsed_only])
    npy_path  = data_dir / "embeddings" / f"{subtask_name}.npy"
    np.save(npy_path, fp_matrix)

    # 3. Split-aware tensors
    for split_name in ("train", "val", "test"):
        split_sents = [ps for ps in parsed_only if ps.split == split_name]
        if not split_sents:
            continue
        split_path = data_dir / "splits" / f"{subtask_name}_{split_name}.pt"
        torch.save(
            {
                "sentences":    [ps.sentence    for ps in split_sents],
                "sentence_ids": [ps.sentence_id for ps in split_sents],
                "labels":       torch.tensor([ps.label for ps in split_sents],
                                             dtype=torch.long),
                "fingerprints": torch.tensor(
                                    np.stack([ps.fingerprint for ps in split_sents]),
                                    dtype=torch.float32),
                "subtask":      subtask_name,
                "split":        split_name,
            },
            split_path,
        )

    # 4. Accumulate for master file
    for ps in parsed_only:
        master_records.append(
            {
                "sentence_id": ps.sentence_id,
                "subtask":     ps.subtask,
                "sentence":    ps.sentence,
                "label":       ps.label,
                "split":       ps.split,
                "fingerprint": ps.fingerprint,
            }
        )

    saved_paths[subtask_name] = {
        "pkl":  str(pkl_path),
        "npy":  str(npy_path),
    }
    log.info(f"  Saved {subtask_name}: {len(parsed_only)} sentences")

# Master file — merged across all sub-tasks
master_path = data_dir / "blimp_master.pt"
if master_records:
    torch.save(
        {
            "records":   master_records,
            "subtasks":  list(all_parsed.keys()),
            "fp_dim":    FINGERPRINT_DIM,
            "created":   datetime.now().isoformat(),
            "n_total":   len(master_records),
            "checksum":  _checksum(master_records),
        },
        master_path,
    )
    log.info(f"\n  Master file: {master_path} ({len(master_records)} sentences)")

return saved_paths
```

def _checksum(records: list) -> str:
“”“MD5 of all sentence strings — for reproducibility verification.”””
content = “”.join(r[“sentence”] for r in records).encode()
return hashlib.md5(content).hexdigest()

# ──────────────────────────────────────────────

# STEP 6: VALIDATE

# ──────────────────────────────────────────────

def validate_dataset(data_dir: Path) -> bool:
“””
Sanity checks on the saved dataset.

```
Checks:
  1. Master file exists and loads cleanly
  2. All sub-tasks present
  3. No data leakage between splits (sentence_id uniqueness)
  4. Fingerprints have correct shape and no NaN/Inf
  5. Structural discriminability (fingerprints should cluster by subtask)
"""
log.info("\n  Running dataset validation...")
passed = True

master_path = data_dir / "blimp_master.pt"

# Check 1: Master file loads
if not master_path.exists():
    log.error("  FAIL: blimp_master.pt not found")
    return False

master = torch.load(master_path, weights_only=False)
records = master["records"]
log.info(f"  [OK] Master file loaded: {len(records)} records")

# Check 2: Sub-tasks present
present_subtasks = set(r["subtask"] for r in records)
for task in BLIMP_TASKS:
    if task not in present_subtasks:
        log.warning(f"  [WARN] Sub-task not found in master: {task}")
    else:
        count = sum(1 for r in records if r["subtask"] == task)
        log.info(f"  [OK] {task}: {count} sentences")

# Check 3: No data leakage
ids = [r["sentence_id"] for r in records]
if len(ids) != len(set(ids)):
    log.error(f"  FAIL: Duplicate sentence_ids detected — data leakage risk")
    passed = False
else:
    log.info(f"  [OK] All sentence_ids unique")

# Check 4: Fingerprint integrity
fps = np.stack([r["fingerprint"] for r in records])
if np.any(np.isnan(fps)) or np.any(np.isinf(fps)):
    log.error(f"  FAIL: NaN or Inf in fingerprints")
    passed = False
elif fps.shape[1] != FINGERPRINT_DIM:
    log.error(f"  FAIL: Fingerprint dim {fps.shape[1]} != {FINGERPRINT_DIM}")
    passed = False
else:
    log.info(f"  [OK] Fingerprints: shape={fps.shape}, clean (no NaN/Inf)")

# Check 5: Structural discriminability — do sub-tasks produce different fingerprints?
subtask_means = {}
for task in present_subtasks:
    task_fps = np.stack([r["fingerprint"] for r in records if r["subtask"] == task])
    subtask_means[task] = task_fps.mean(axis=0)

tasks_list = list(subtask_means.keys())
if len(tasks_list) >= 2:
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    mean_matrix = cos_sim(np.stack(list(subtask_means.values())))
    # Off-diagonal entries should be < 1.0 (different constructions differ)
    off_diag = mean_matrix[np.triu_indices(len(tasks_list), k=1)]
    max_sim   = off_diag.max()
    if max_sim > 0.98:
        log.warning(
            f"  [WARN] Sub-task fingerprints are nearly identical (max_cosine={max_sim:.3f}). "
            f"Fingerprints may not discriminate constructions well."
        )
    else:
        log.info(
            f"  [OK] Sub-tasks are fingerprint-distinguishable "
            f"(max inter-task cosine={max_sim:.3f})"
        )

# Check 6: Split balance
split_counts = defaultdict(int)
for r in records:
    split_counts[r["split"]] += 1
log.info(
    f"  [OK] Splits — "
    f"train={split_counts['train']}  "
    f"val={split_counts['val']}  "
    f"test={split_counts['test']}"
)

return passed
```

# ──────────────────────────────────────────────

# STEP 7: README

# ──────────────────────────────────────────────

def write_readme(
all_stats:  dict[str, SubtaskStats],
data_dir:   Path,
master_path: Path,
) -> None:
“”“Write a data/blimp/README.md with full provenance and statistics.”””

```
total_parsed = sum(s.parsed_ok for s in all_stats.values())
total_tried  = sum(s.parsed_ok + s.parse_failed for s in all_stats.values())
overall_rate = total_parsed / max(total_tried, 1)

lines = [
    "# BLiMP Dataset — Build Report",
    "",
    f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"**Total sentences (parsed):** {total_parsed}",
    f"**Overall parse rate:** {overall_rate:.1%}",
    "",
    "## Source",
    "",
    "- **Dataset:** BLiMP (Benchmark of Linguistic Minimal Pairs)",
    "- **Paper:** Warstadt et al. (2020), TACL",
    "- **URL:** https://github.com/alexwarstadt/blimp",
    "- **Parser:** lambeq BobcatParser",
    "",
    "## Sub-tasks",
    "",
    "| Sub-task | Priority | Total | Parsed | Parse Rate | Description |",
    "|:---------|:---------|------:|-------:|-----------:|:------------|",
]

for name, cfg in BLIMP_TASKS.items():
    s = all_stats.get(name)
    if s:
        lines.append(
            f"| {name} | {cfg['priority']} | {s.parsed_ok + s.parse_failed} | "
            f"{s.parsed_ok} | {s.parse_rate:.1%} | {cfg['description']} |"
        )

lines += [
    "",
    "## Splits",
    "",
    f"- **Train:** {TRAIN_RATIO:.0%}",
    f"- **Val:**   {VAL_RATIO:.0%}",
    f"- **Test:**  {TEST_RATIO:.0%}",
    f"- **Seed:**  {RANDOM_SEED}",
    "",
    "## Parse Failure Analysis",
    "",
]

for name, s in all_stats.items():
    if s.failure_reasons:
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Reason | Count |")
        lines.append("|:-------|------:|")
        for reason, count in sorted(s.failure_reasons.items(),
                                    key=lambda x: -x[1])[:10]:
            lines.append(f"| `{reason[:60]}` | {count} |")
        lines.append("")

lines += [
    "## Fingerprint Features",
    "",
    "| Index | Feature | Description |",
    "|------:|:--------|:------------|",
    "| 0 | n_words | Number of word boxes |",
    "| 1 | n_cups | Syntactic contractions |",
    "| 2 | n_caps | Cap morphisms |",
    "| 3 | n_swaps | Wire crossings |",
    "| 4 | n_total_boxes | All boxes |",
    "| 5 | avg_type_complexity | Avg atomic types per word |",
    "| 6 | max_type_complexity | Max atomic types in any word |",
    "| 7 | n_noun_atoms | Count of n-type atoms |",
    "| 8 | n_sent_atoms | Count of s-type atoms |",
    "| 9 | cups_per_word | Syntactic density ratio |",
    "| 10 | type_spread | max_complexity - avg_complexity |",
    "| 11 | n_wires_dom | Domain wire count |",
    "| 12 | n_wires_cod | Codomain wire count |",
    "| 13 | n_right_adj | Right adjoint types (.r) |",
    "| 14 | n_left_adj | Left adjoint types (.l) |",
    "| 15 | structural_ratio | Structural / total boxes |",
    "",
    "## How to Load",
    "",
    "```python",
    "import torch",
    "master = torch.load('data/blimp/blimp_master.pt', weights_only=False)",
    "records = master['records']  # list of dicts",
    "",
    "# Get all sentences for one sub-task",
    "rc_sents = [r for r in records if r['subtask'] == 'relative_clause']",
    "",
    "# Get fingerprint matrix for all training sentences",
    "import numpy as np",
    "train = [r for r in records if r['split'] == 'train']",
    "fp_matrix = np.stack([r['fingerprint'] for r in train])  # (N, 16)",
    "```",
    "",
    "## Reproduction",
    "",
    "```bash",
    "conda activate qnlp",
    "python build_blimp_dataset.py",
    "```",
]

readme_path = data_dir / "README.md"
readme_path.write_text("\n".join(lines))
log.info(f"  README written: {readme_path}")
```

# ──────────────────────────────────────────────

# MAIN

# ──────────────────────────────────────────────

def main():
print(”=” * 62)
print(”  BLiMP Dataset Builder”)
print(”  For: Mechanistic Interpretability of Hybrid QNLP Models”)
print(”=” * 62)

```
DATA_DIR.mkdir(parents=True, exist_ok=True)
raw_dir = DATA_DIR / "raw"

# ── Step 1: Download ──────────────────────────────────────
print("\n── STEP 1: Download ─────────────────────────────────────")
all_raw = download_blimp(BLIMP_TASKS, raw_dir)

total_raw = sum(len(v) for v in all_raw.values())
if total_raw == 0:
    log.error("No data downloaded. Check your internet connection.")
    return
log.info(f"Downloaded {total_raw} total examples across {len(all_raw)} sub-tasks")

# ── Step 2: Parse ─────────────────────────────────────────
print("\n── STEP 2: Parse with BobcatParser ──────────────────────")
all_parsed, all_stats = parse_sentences(all_raw, BLIMP_TASKS)

# ── Step 3: Fingerprint ───────────────────────────────────
print("\n── STEP 3: Extract Structural Fingerprints ───────────────")
all_parsed = extract_fingerprints(all_parsed)

# ── Step 4: Splits ────────────────────────────────────────
print("\n── STEP 4: Assign Splits ────────────────────────────────")
all_parsed = assign_splits(all_parsed)

# ── Step 5: Save ──────────────────────────────────────────
print("\n── STEP 5: Save ─────────────────────────────────────────")
saved_paths = save_dataset(all_parsed, all_stats, DATA_DIR)

# ── Step 6: Validate ──────────────────────────────────────
print("\n── STEP 6: Validate ─────────────────────────────────────")
ok = validate_dataset(DATA_DIR)

# ── Step 7: README ────────────────────────────────────────
print("\n── STEP 7: Write README ─────────────────────────────────")
write_readme(all_stats, DATA_DIR, DATA_DIR / "blimp_master.pt")

# ── Final Summary ─────────────────────────────────────────
print("\n" + "=" * 62)
print("  SUMMARY")
print("=" * 62)

total_parsed = sum(s.parsed_ok for s in all_stats.values())
total_tried  = sum(s.parsed_ok + s.parse_failed for s in all_stats.values())

print(f"\n  Sub-tasks processed: {len(all_stats)}")
print(f"  Total attempted:     {total_tried}")
print(f"  Total parsed:        {total_parsed}")
print(f"  Overall parse rate:  {total_parsed / max(total_tried, 1):.1%}")
print(f"  Validation:          {'PASSED' if ok else 'FAILED — check warnings above'}")
print(f"\n  Output directory:    {DATA_DIR.resolve()}")
print(f"  Master file:         {DATA_DIR / 'blimp_master.pt'}")
print(f"  README:              {DATA_DIR / 'README.md'}")

print("\n  Per sub-task:")
print(f"  {'Sub-task':35s}  {'Parsed':>8}  {'Rate':>7}")
print(f"  {'─'*54}")
for name, s in all_stats.items():
    attempted = s.parsed_ok + s.parse_failed
    print(f"  {name:35s}  {s.parsed_ok:>8}  {s.parse_rate:>6.1%}")

print("\n  Next step: run build_template_dataset.py for Dataset A")
print("=" * 62)
```

if **name** == “**main**”:
main()
