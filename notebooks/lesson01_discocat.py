"""
Lesson 01: DisCoCat Diagram Exploration
========================================
HOMEWORK: Fill in the sections marked with TODO.

Run with:
    conda activate qnlp
    cd /Users/nguyen/Documents/Work/Quantum
    python notebooks/lesson01_discocat.py

What this script does:
    1. Parses 5 English sentences into DisCoCat string diagrams
    2. Inspects diagram structure (boxes, types, cups)
    3. Saves diagram visualizations
    4. Answers comprehension questions

Your goal: Understand what a DisCoCat diagram IS before we build anything on top of it.
"""

from pathlib import Path

# Ensure output directory exists
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════
# STEP 1: Initialize the parser
# ═══════════════════════════════════════════════
from lambeq import BobcatParser

print("Loading BobcatParser (this may take a moment on first run)...")
parser = BobcatParser(verbose='suppress')
print("Parser loaded.\n")

# ═══════════════════════════════════════════════
# STEP 2: Parse sentences and inspect diagrams
# ═══════════════════════════════════════════════

sentences = [
    "dogs chase cats",             # 1. Simple transitive (SVO)
    "dogs run",                    # 2. Simple intransitive (SV)
    "big dogs chase small cats",   # 3. Transitive with adjectives
    "dogs that chase cats run",    # 4. Relative clause
    "cats are chased by dogs",     # 5. Passive voice
]

diagrams = []

for i, sent in enumerate(sentences):
    print(f"\n{'='*60}")
    print(f"  Sentence {i+1}: \"{sent}\"")
    print(f"{'='*60}")

    diagram = parser.sentence2diagram(sent)
    diagrams.append(diagram)

    # ─── Diagram-level info ───
    print(f"\n  diagram.dom (input type):  {diagram.dom}")
    print(f"  diagram.cod (output type): {diagram.cod}")

    # ─── Box-level info ───
    print(f"\n  Boxes ({len(diagram.boxes)} total):")
    for j, box in enumerate(diagram.boxes):
        print(f"    [{j}] name={box.name:20s}  dom={str(box.dom):20s}  cod={str(box.cod):20s}")

    # TODO 1: Count how many boxes are actual WORDS vs structural morphisms (cups/caps/swaps).
    # Hint: Word boxes have a .name that is an actual English word.
    #        Cups typically have names like 'CUP' or are instances of specific classes.
    #        Try: type(box).__name__ to see the class of each box.
    #
    # Write your counting code here:
    # word_count = ...
    # cup_count = ...
    # print(f"\n  Words: {word_count}, Cups: {cup_count}")

    # ─── Save the diagram visualization ───
    try:
        draw_path = output_dir / f"diagram_{i+1}_{sent.replace(' ', '_')}.png"
        diagram.draw(figsize=(12, 4), path=str(draw_path))
        print(f"\n  Diagram saved to: {draw_path}")
    except Exception as e:
        print(f"\n  Could not draw diagram: {e}")

# ═══════════════════════════════════════════════
# STEP 3: Comparative Analysis
# ═══════════════════════════════════════════════

print(f"\n\n{'='*60}")
print("  COMPARATIVE ANALYSIS")
print(f"{'='*60}")

# TODO 2: For each diagram, count the number of boxes and print a summary table.
# Expected output format:
#   Sentence                          | Boxes | Words | Cups | cod
#   dogs chase cats                   |   ?   |   ?   |  ?   |  ?
#   dogs run                          |   ?   |   ?   |  ?   |  ?
#   ...
#
# Write your code here:


# ═══════════════════════════════════════════════
# STEP 4: Answer the comprehension questions
# ═══════════════════════════════════════════════

print(f"\n\n{'='*60}")
print("  COMPREHENSION QUESTIONS")
print(f"{'='*60}")

# TODO 3: Answer these questions by replacing the "..." with your answers.
# Base your answers on what you OBSERVED in the output above, not on guesses.

answers = {
    "Q1: How many boxes does 'dogs chase cats' have? (total, not just words)":
        "...",

    "Q2: What is diagram.cod for a valid sentence? (same for all 5?)":
        "...",

    "Q3: In 'dogs that chase cats run', how is the relative clause connected to the subject?":
        "...",

    "Q4: Do 'dogs chase cats' and 'cats are chased by dogs' have the same diagram structure?":
        "...",

    "Q5: Do more complex sentences have more cups? What pattern do you see?":
        "...",
}

for q, a in answers.items():
    print(f"\n  {q}")
    print(f"  → {a}")

print("\n\nDone! Review the saved diagrams in results/figures/")
print("When you're ready, share this script's output with me for review.")
