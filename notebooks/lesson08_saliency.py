"""
Lesson 08: Gradient Saliency — Where Does the PQC Attend?
==========================================================
Implements Integrated Gradients to visualize which tokens
the PQC and MLP attend to, then correlates with POS tags.

Run with:
    conda activate qnlp
    cd /Users/nguyen/Documents/Work/Quantum
    python notebooks/lesson08_saliency.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
output_dir = PROJECT_ROOT / "results" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

N_QUBITS = 8
N_LAYERS = 2
N_IG_STEPS = 50  # Integrated Gradients interpolation steps


# ═══════════════════════════════════════════════════════════
# PART 1: Load Models
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  PART 1: Loading SBERT + rebuilding models")
print("=" * 60)

sbert = SentenceTransformer('all-MiniLM-L6-v2')
sbert_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(sbert_model_name)
transformer = AutoModel.from_pretrained(sbert_model_name)
transformer.eval()

try:
    sbert_dim = sbert.get_embedding_dimension()
except AttributeError:
    sbert_dim = sbert.get_sentence_embedding_dimension()

# PQC model
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class PQCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(sbert_dim, N_QUBITS, bias=False)
        self.scale = nn.Parameter(torch.tensor(np.pi / 2.0))
        self.weights = nn.Parameter(0.01 * torch.randn(N_LAYERS, N_QUBITS))

    def forward(self, x):
        h = torch.tanh(self.projection(x)) * self.scale
        out = quantum_circuit(h, self.weights)
        return torch.stack(out).float()

    def forward_from_sbert(self, sbert_vec):
        """Forward pass taking a single SBERT vector (with grad)."""
        h = torch.tanh(self.projection(sbert_vec)) * self.scale
        out = quantum_circuit(h, self.weights)
        return torch.stack(out).float()


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(sbert_dim, N_QUBITS, bias=False)
        self.scale = nn.Parameter(torch.tensor(np.pi / 2.0))
        self.mlp = nn.Sequential(
            nn.Linear(N_QUBITS, 4, bias=False), nn.Tanh(),
            nn.Linear(4, N_QUBITS, bias=False), nn.Tanh(),
        )

    def forward_from_sbert(self, sbert_vec):
        h = torch.tanh(self.projection(sbert_vec)) * self.scale
        return self.mlp(h)


# Train both models quickly on the dataset
print("  Training PQC and MLP on dataset...")

dataset_path = PROJECT_ROOT / "data" / "syntactic_dataset.pt"
if not dataset_path.exists():
    # Fall back to small dataset if Lesson 6 hasn't been run
    print("  ⚠ Full dataset not found — using built-in sentences")
    use_full_dataset = False
else:
    use_full_dataset = True

# Analysis sentences (always use these for visualization)
analysis_sentences = [
    "the tall doctor chased the young cat",
    "the young cat was chased by the tall doctor",
    "the tall doctor that helped the nurse chased the young cat",
    "it was the tall doctor who chased the young cat",
    "the brave farmer praised the clever student",
    "the clever student was praised by the brave farmer",
]

# Encode analysis sentences
analysis_embeds = torch.tensor(
    sbert.encode(analysis_sentences, convert_to_numpy=True),
    dtype=torch.float32
)

# Quick training data
train_pairs = [
    ("the tall doctor chased the young cat",
     "the young cat was chased by the tall doctor", 1.0),
    ("the brave farmer praised the clever student",
     "the clever student was praised by the brave farmer", 1.0),
    ("the kind teacher helped the shy student",
     "the shy student was helped by the kind teacher", 1.0),
    ("the fast driver followed the slow clerk",
     "the slow clerk was followed by the fast driver", 1.0),
    ("the tall doctor chased the young cat",
     "the brave farmer praised the clever student", 0.0),
    ("the kind teacher helped the shy student",
     "the fast driver followed the slow clerk", 0.0),
    ("the young cat was chased by the tall doctor",
     "the clever student was praised by the brave farmer", 0.0),
    ("the shy student was helped by the kind teacher",
     "the slow clerk was followed by the fast driver", 0.0),
]

all_train_sents = list(set([p[0] for p in train_pairs] + [p[1] for p in train_pairs]))
all_train_embeds = sbert.encode(all_train_sents, convert_to_numpy=True)
s2e = {s: torch.tensor(e, dtype=torch.float32)
       for s, e in zip(all_train_sents, all_train_embeds)}

X_a = torch.stack([s2e[p[0]] for p in train_pairs])
X_b = torch.stack([s2e[p[1]] for p in train_pairs])
y = torch.tensor([p[2] for p in train_pairs], dtype=torch.float32)

def train(model, n_epochs=40):
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(n_epochs):
        h_a = torch.stack([model.forward_from_sbert(X_a[i]) for i in range(len(X_a))])
        h_b = torch.stack([model.forward_from_sbert(X_b[i]) for i in range(len(X_b))])
        cs = F.cosine_similarity(h_a, h_b, dim=1)
        pred = ((cs + 1) / 2).float()
        loss = F.binary_cross_entropy(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

pqc_model = PQCModel()
mlp_model = MLPModel()

print("  Training PQC...")
train(pqc_model)
print("  Training MLP...")
train(mlp_model)
print("  Done.\n")


# ═══════════════════════════════════════════════════════════
# PART 2: Integrated Gradients
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  PART 2: Integrated Gradients")
print("=" * 60)


def integrated_gradients(model, sbert_vec, n_steps=N_IG_STEPS):
    """
    Compute Integrated Gradients for a single sentence.

    Args:
        model: PQCModel or MLPModel with forward_from_sbert()
        sbert_vec: (384,) tensor — SBERT embedding of the sentence
        n_steps: number of interpolation steps

    Returns:
        ig: (384,) tensor — attribution scores per SBERT dimension
    """
    baseline = torch.zeros_like(sbert_vec)  # zero baseline
    grads_sum = torch.zeros_like(sbert_vec)

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interp = baseline + alpha * (sbert_vec - baseline)
        interp = interp.clone().detach().requires_grad_(True)

        output = model.forward_from_sbert(interp)
        # Sum all output dimensions as the scalar target
        scalar_out = output.sum()
        scalar_out.backward()
        grads_sum += interp.grad.detach()

    # IG = (input - baseline) * average_gradient
    ig = (sbert_vec - baseline) * grads_sum / (n_steps + 1)
    return ig.detach()


def get_token_saliency(sentence, ig_vector):
    """
    Map 384-dim IG saliency to individual tokens.

    Uses SBERT's tokenizer and transformer to get per-token embeddings,
    then projects the IG vector onto each token.
    """
    # Tokenize
    encoded = tokenizer(sentence, return_tensors='pt', padding=False,
                        truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

    # Get token embeddings from the transformer's embedding layer
    with torch.no_grad():
        token_embeds = transformer.embeddings.word_embeddings(
            encoded['input_ids'][0]
        )  # (n_tokens, 384)

    # Project IG onto each token embedding
    # saliency[i] = |ig · token_embed[i]| / ||token_embed[i]||
    ig_vec = ig_vector.unsqueeze(0)  # (1, 384)
    dots = torch.abs(torch.mm(ig_vec, token_embeds.T)).squeeze()  # (n_tokens,)
    norms = token_embeds.norm(dim=1) + 1e-8
    token_saliency = dots / norms

    # Remove [CLS] and [SEP]
    clean_tokens = []
    clean_saliency = []
    for tok, sal in zip(tokens, token_saliency.numpy()):
        if tok in ['[CLS]', '[SEP]']:
            continue
        # Merge wordpiece tokens (##xyz)
        if tok.startswith('##') and clean_tokens:
            clean_tokens[-1] += tok[2:]
            clean_saliency[-1] = max(clean_saliency[-1], sal)
        else:
            clean_tokens.append(tok)
            clean_saliency.append(sal)

    return clean_tokens, np.array(clean_saliency)


# Compute IG for all analysis sentences
print(f"\n  Computing Integrated Gradients for {len(analysis_sentences)} sentences...")

ig_results_pqc = []
ig_results_mlp = []

for i, (sent, embed) in enumerate(zip(analysis_sentences, analysis_embeds)):
    ig_pqc = integrated_gradients(pqc_model, embed)
    ig_mlp = integrated_gradients(mlp_model, embed)

    tokens_pqc, sal_pqc = get_token_saliency(sent, ig_pqc)
    tokens_mlp, sal_mlp = get_token_saliency(sent, ig_mlp)

    ig_results_pqc.append({'sentence': sent, 'tokens': tokens_pqc,
                            'saliency': sal_pqc})
    ig_results_mlp.append({'sentence': sent, 'tokens': tokens_mlp,
                            'saliency': sal_mlp})

    print(f"\n    [{i}] \"{sent}\"")
    # Normalize for display
    sal_pqc_n = sal_pqc / (sal_pqc.max() + 1e-8)
    sal_mlp_n = sal_mlp / (sal_mlp.max() + 1e-8)
    pqc_str = " | ".join(f"{t}({s:.2f})" for t, s in zip(tokens_pqc, sal_pqc_n))
    mlp_str = " | ".join(f"{t}({s:.2f})" for t, s in zip(tokens_mlp, sal_mlp_n))
    print(f"      PQC: {pqc_str}")
    print(f"      MLP: {mlp_str}")


# ═══════════════════════════════════════════════════════════
# PART 3: POS Tag Correlation
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  PART 3: POS Tag Correlation")
print("=" * 60)

# Simple POS tagger using word lists
FUNCTION_WORDS = {
    'the', 'a', 'an', 'was', 'were', 'is', 'are', 'by', 'that', 'who',
    'which', 'it', 'this', 'those', 'these', 'of', 'in', 'on', 'to',
    'for', 'with', 'from', 'at', 'into',
}

VERBS = {
    'chased', 'helped', 'praised', 'followed', 'examined', 'watched',
    'taught', 'liked', 'visited', 'called', 'pushed', 'pulled',
    'stopped', 'kicked', 'cooked', 'painted', 'cleaned', 'greeted',
    'blamed', 'admired', 'chase', 'help', 'praise', 'follow',
}

def classify_pos(token):
    t = token.lower()
    if t in FUNCTION_WORDS:
        return 'function'
    elif t in VERBS:
        return 'verb'
    else:
        return 'content'  # nouns, adjectives

# Aggregate saliency by POS category
pos_saliency = {'pqc': {'function': [], 'verb': [], 'content': []},
                'mlp': {'function': [], 'verb': [], 'content': []}}

for res_pqc, res_mlp in zip(ig_results_pqc, ig_results_mlp):
    # Normalize per-sentence
    sp = res_pqc['saliency'] / (res_pqc['saliency'].max() + 1e-8)
    sm = res_mlp['saliency'] / (res_mlp['saliency'].max() + 1e-8)

    for tok, s_p in zip(res_pqc['tokens'], sp):
        pos = classify_pos(tok)
        pos_saliency['pqc'][pos].append(s_p)

    for tok, s_m in zip(res_mlp['tokens'], sm):
        pos = classify_pos(tok)
        pos_saliency['mlp'][pos].append(s_m)

print(f"\n  Average normalized saliency by POS category:")
print(f"  {'Category':<12s}  {'PQC':>8s}  {'MLP':>8s}")
print(f"  {'─'*32}")
for pos in ['function', 'verb', 'content']:
    pqc_mean = np.mean(pos_saliency['pqc'][pos]) if pos_saliency['pqc'][pos] else 0
    mlp_mean = np.mean(pos_saliency['mlp'][pos]) if pos_saliency['mlp'][pos] else 0
    print(f"  {pos:<12s}  {pqc_mean:>8.3f}  {mlp_mean:>8.3f}")

# Function word ratio
pqc_func = np.mean(pos_saliency['pqc']['function'])
pqc_cont = np.mean(pos_saliency['pqc']['content'])
mlp_func = np.mean(pos_saliency['mlp']['function'])
mlp_cont = np.mean(pos_saliency['mlp']['content'])

print(f"\n  Function/Content saliency ratio:")
print(f"    PQC: {pqc_func/(pqc_cont+1e-8):.2f}x")
print(f"    MLP: {mlp_func/(mlp_cont+1e-8):.2f}x")
print(f"    (>1 means function words get more attention)")


# ═══════════════════════════════════════════════════════════
# PART 4: Visualization
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'=' * 60}")
print("  PART 4: Generating Saliency Figures")
print("=" * 60)


def plot_saliency_heatmap(results_pqc, results_mlp, save_path):
    """Plot side-by-side saliency heatmaps for PQC and MLP."""
    n_sents = len(results_pqc)
    fig, axes = plt.subplots(n_sents, 2, figsize=(18, 2.5 * n_sents))

    if n_sents == 1:
        axes = axes.reshape(1, 2)

    for i in range(n_sents):
        for j, (res, title) in enumerate([(results_pqc[i], 'PQC'),
                                           (results_mlp[i], 'MLP')]):
            ax = axes[i, j]
            tokens = res['tokens']
            sal = res['saliency'] / (res['saliency'].max() + 1e-8)
            n_tok = len(tokens)

            # Color map
            cmap = plt.cm.YlOrRd
            for k, (tok, s) in enumerate(zip(tokens, sal)):
                bg = cmap(s)
                text_color = 'white' if s > 0.6 else 'black'
                ax.text(k / n_tok + 0.5 / n_tok, 0.5, tok,
                       ha='center', va='center', fontsize=10,
                       fontweight='bold' if s > 0.5 else 'normal',
                       color=text_color,
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor=bg, edgecolor='gray',
                                alpha=0.9))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            if i == 0:
                ax.set_title(title, fontsize=14, fontweight='bold')

    fig.suptitle('Token Saliency: PQC vs MLP\n(darker = higher attention)',
                 fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')

plot_saliency_heatmap(ig_results_pqc, ig_results_mlp,
                      output_dir / "paper_saliency_heatmap.png")
print(f"  Figure 1: {output_dir / 'paper_saliency_heatmap.png'}")

# Figure 2: POS category bar chart
fig2, ax2 = plt.subplots(figsize=(9, 6))
categories = ['Function\nwords', 'Verbs', 'Content\nwords']
pqc_vals = [np.mean(pos_saliency['pqc'][p]) for p in ['function', 'verb', 'content']]
mlp_vals = [np.mean(pos_saliency['mlp'][p]) for p in ['function', 'verb', 'content']]

x = np.arange(len(categories))
w = 0.35
ax2.bar(x - w/2, pqc_vals, w, label='PQC', color='#55A868', edgecolor='black')
ax2.bar(x + w/2, mlp_vals, w, label='MLP', color='#C44E52', edgecolor='black')
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=12)
ax2.set_ylabel('Mean Normalized Saliency', fontsize=13)
ax2.set_title('Saliency by POS Category: PQC vs MLP', fontsize=14, fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(axis='y', alpha=0.3)
fig2.tight_layout()
fig2.savefig(output_dir / "paper_saliency_pos.png", dpi=150)
print(f"  Figure 2: {output_dir / 'paper_saliency_pos.png'}")


# ═══════════════════════════════════════════════════════════
# SAVE & COMPREHENSION QUESTIONS
# ═══════════════════════════════════════════════════════════

save_path = PROJECT_ROOT / "results" / "saliency_analysis.pt"
torch.save({
    'ig_results_pqc': ig_results_pqc,
    'ig_results_mlp': ig_results_mlp,
    'pos_saliency': pos_saliency,
}, save_path)
print(f"\n  Results saved: {save_path}")

print(f"\n\n{'=' * 60}")
print("  COMPREHENSION QUESTIONS")
print("=" * 60)

answers = {
    "Q1: Look at the saliency heatmap. For the active sentence "
    "'the tall doctor chased the young cat', which word gets "
    "the highest saliency in the PQC? In the MLP? Are they "
    "the same word?":
        "...",

    "Q2: Compare the active vs passive pair. Does the PQC "
    "shift its attention to different words when the syntax "
    "changes (even though meaning is the same)?":
        "...",

    "Q3: Look at the POS category bar chart. Does the PQC "
    "attend more to function words than the MLP? What does "
    "this tell us about how each model processes syntax?":
        "...",

    "Q4: Why do we use Integrated Gradients instead of simple "
    "gradients (just one backward pass)? What problem does "
    "the interpolation from baseline solve?":
        "...",

    "Q5: In the paper, this experiment supports which hypothesis? "
    "Write one sentence connecting the saliency results to "
    "our research question about syntax in PQCs.":
        "...",
}

for q, a in answers.items():
    print(f"\n  {q}")
    print(f"  → {a}")

print(f"\n\nDone! This completes all 4 experiments for the paper.")
print(f"Review figures in results/figures/ and share your answers.")
