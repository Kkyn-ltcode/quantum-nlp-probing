# Lesson 1: Understanding DisCoCat — The Grammar as a Wiring Diagram

**Roadmap Phase:** Phase 0 (Week 0)  
**Time estimate:** 3-4 hours  
**Prerequisites:** Python, basic linear algebra (what a tensor is)

---

## Why This Lesson Matters

Everything in this project builds on one foundational idea: **grammar can be represented as a diagram of connected wires**. Not as a tree (like classical parsing), not as a sequence (like Transformers process it), but as a *circuit* where types flow along wires and words are boxes that transform those types.

If you don't understand this, the syntax skeleton, the RSA experiment, the entire paper — none of it will make sense. So we're going to spend real time here.

---

## Part 1: The Big Idea — Meanings Are Compositions

Classical NLP treats words as vectors (Word2Vec, GloVe, BERT). The sentence meaning is some function of the word vectors — usually a pooled average or an attention-weighted sum.

**DisCoCat asks:** *Can we do better? Can the grammar itself tell us how to compose word meanings?*

The answer is yes, and the tool is **category theory**. But don't panic — you don't need to learn category theory. You need to learn **three visual concepts:**

### Concept 1: Types

Every word has a grammatical type. In DisCoCat, types are built from two atoms:
- **`n`** = noun (a thing)
- **`s`** = sentence (a complete meaning)

Complex types are built by composition:
- **`n.r ⊗ s ⊗ n.l`** = "something that takes a noun on the right and a noun on the left to produce a sentence" → this is a **transitive verb**
- **`n.r ⊗ s`** = "something that takes a noun on the right to produce a sentence" → this is an **intransitive verb**

The `.r` and `.l` superscripts mean "I'm looking for a noun to my right/left." Think of them as **plugs** looking for **sockets**.

### Concept 2: Wires and Boxes

A string diagram has:
- **Boxes** = words. Each box has input wires (its type domain) and output wires (its type codomain).
- **Wires** = type connections. A wire carries a type (`n` or `s`).

For the sentence *"dogs chase cats"*:
```
dogs : n          (a box with one output wire of type n)
chase : n.r ⊗ s ⊗ n.l    (a box with one output wire of type s, expecting n on both sides)
cats : n          (a box with one output wire of type n)
```

### Concept 3: Cups — The Grammar Glue

When a noun type (`n`) meets its adjoint (`n.r` or `n.l`), they **annihilate** — like matter and antimatter. This is drawn as a **cup** (a U-shaped curve connecting the two wires).

```
 dogs    chase    cats
  |     / | \      |
  n   n.r  s  n.l  n
  |   /    |    \  |
  ╰──╯    |    ╰──╯    ← these are "cups"
          s              ← the only wire left: a sentence!
```

**This is the key insight:** After all cups resolve, the remaining wires tell you the output type. If only an `s` wire remains, you have a grammatically valid sentence. The *shape* of this wiring — how many cups, where they connect, how deep they nest — **is the syntactic structure**.

**And this is exactly what we'll extract as the "syntax skeleton."**

---

## Part 2: What lambeq Does

`lambeq` automates this process:

1. **Parser** (`BobcatParser`) — Takes an English sentence, produces a syntactic parse.
2. **String Diagram** — Converts the parse into a DisCoCat diagram (boxes + wires + cups).
3. **Ansatz** — Maps the diagram into a quantum circuit or tensor network (we'll cover this in Lesson 3).

For now, we only care about steps 1 and 2. Step 3 is where semantics enter — we explicitly want to study the diagram **before** step 3.

---

## Part 3: What You Need to See With Your Own Eyes

Theory is useless without hands-on inspection. Here's what you need to do:

---

## Homework Assignment

### Step 1: Set Up the Environment

```bash
# Create and activate the conda environment
conda create -n qnlp python=3.10 -y
conda activate qnlp

# Navigate to project
cd /Users/nguyen/Documents/Work/Quantum

# Install dependencies
pip install -r requirements.txt

# Download the spaCy English model (needed by lambeq's parser)
python -m spacy download en_core_web_sm
```

### Step 2: Parse 5 Sentences and Inspect the Diagrams

Create a file `notebooks/lesson01_discocat.py` (a regular Python script, not a notebook — easier for me to review). Write code that does the following:

```python
from lambeq import BobcatParser

parser = BobcatParser(verbose='suppress')

# These 5 sentences are chosen to show different syntactic structures
sentences = [
    "dogs chase cats",             # Simple transitive (SVO)
    "dogs run",                    # Simple intransitive (SV)
    "big dogs chase small cats",   # Transitive with adjectives
    "dogs that chase cats run",    # Relative clause
    "cats are chased by dogs",     # Passive voice
]

for sent in sentences:
    diagram = parser.sentence2diagram(sent)
    
    print(f"\n{'='*60}")
    print(f"Sentence: {sent}")
    print(f"{'='*60}")
    
    # YOUR TASK: For each diagram, print the following information:
    # 1. diagram.boxes — what boxes (words) are in the diagram?
    # 2. diagram.dom — what is the input type of the whole diagram?
    # 3. diagram.cod — what is the output type of the whole diagram?
    # 4. For each box in diagram.boxes:
    #    - box.name (the word)
    #    - box.dom (input type)
    #    - box.cod (output type)
    # 5. Try: diagram.draw() — this renders the diagram as an image.
    #    Save each one: diagram.draw(path=f"results/figures/diagram_{i}.png")
    
    # Write your code here...
```

### Step 3: Answer These Questions

After running the code, write your answers as comments at the bottom of the file:

1. **How many boxes does "dogs chase cats" have?** (It's not 3 — count what `diagram.boxes` actually returns. Some entries are cups/caps, not words.)

2. **What is `diagram.cod` for a valid sentence?** (This should always be the same type. What is it?)

3. **Look at the diagram for "dogs that chase cats run."** How is the relative clause ("that chase cats") visually connected to the subject ("dogs")? Can you see the cup that links them?

4. **Compare the diagrams for "dogs chase cats" and "cats are chased by dogs."** They mean the same thing. Do they have the same diagram structure? (This is critical for understanding what "syntax" means in DisCoCat — it's about *structure*, not meaning.)

5. **Count the number of cups in each diagram.** Do sentences with more complex syntax have more cups?

---

## What I'll Look For When You Come Back

When you share your output with me, I'll check:

- [x] All 5 sentences parsed successfully (some may fail — that's informative too)
- [x] You can identify boxes vs. cups in `diagram.boxes`
- [x] You understand that `diagram.cod` should be type `s` for a sentence
- [x] You noticed the structural differences between active/passive voice
- [x] Your answers to the 5 questions show genuine understanding, not just printout

---

## Reading (Optional but Recommended)

If you want to go deeper, read **Section 2** of this paper — just the diagrams section, skip the category theory:

> Coecke, Sadrzadeh & Clark (2010). *"Mathematical Foundations for a Compositional Distributional Model of Meaning."*  
> [arXiv:1003.4394](https://arxiv.org/abs/1003.4394)

Focus on understanding Figures 1-5. Don't worry about the formal definitions.

---

## Next Lesson Preview

**Lesson 2: PennyLane Fundamentals — Your First Quantum Circuit**

Once you can see and understand diagrams, we'll build the other half of the pipeline: a parameterized quantum circuit in PennyLane. You'll learn what qubits, gates, and measurements are — with code, not math.

---

> [!TIP]
> **Time management:** Step 1 (setup) should take ~30 minutes. Step 2 (code) should take ~1 hour. Step 3 (thinking and answering) should take ~1 hour. If you're spending 3+ hours on setup, something is wrong — come ask me.
