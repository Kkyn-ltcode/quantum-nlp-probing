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

Your goal: Understand what a DisCoCat diagram IS before we build anything
on top of it.
"""

from pathlib import Path

# Ensure output directory exists
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════
# STEP 1: Initialize the parser
# ═══════════════════════════════════════════════
from lambeq import BobcatParser

print("Loading BobcatParser (local model)...")
parser = BobcatParser(
    model_name_or_path='bobcat',  # points to ./bobcat/ directory
    verbose='suppress'
)
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

    word_count = 0
    cup_count = 0
    other_count = 0

    for j, box in enumerate(diagram.boxes):
        box_type = type(box).__name__
        print(f"    [{j}] {box_type:15s} name={str(box.name):20s} "
              f"dom={str(box.dom):25s} cod={str(box.cod):25s}")

        if box_type == "Word":
            word_count += 1
        elif box_type == "Cup":
            cup_count += 1
        else:
            other_count += 1

    print(f"\n  Summary: {word_count} words, {cup_count} cups, "
          f"{other_count} other")

    # ─── Save the diagram visualization ───
    try:
        draw_path = output_dir / f"diagram_{i+1}_{sent.replace(' ', '_')}.png"
        diagram.draw(figsize=(14, 5), path=str(draw_path))
        print(f"  Diagram saved to: {draw_path}")
    except Exception as e:
        print(f"  Could not draw diagram: {e}")


# ═══════════════════════════════════════════════
# STEP 3: Comparative Analysis
# ═══════════════════════════════════════════════

print(f"\n\n{'='*60}")
print("  COMPARATIVE ANALYSIS")
print(f"{'='*60}")

# TODO 1: Fill in this table by looking at the output above.
# Replace the ? with actual numbers from your observations.
print("""
  Sentence                       | Total | Words | Cups | cod
  -------------------------------|-------|-------|------|-----
  dogs chase cats                |   ?   |   ?   |   ?  |  ?
  dogs run                       |   ?   |   ?   |   ?  |  ?
  big dogs chase small cats      |   ?   |   ?   |   ?  |  ?
  dogs that chase cats run       |   ?   |   ?   |   ?  |  ?
  cats are chased by dogs        |   ?   |   ?   |   ?  |  ?
""")


# ═══════════════════════════════════════════════
# STEP 4: Deep inspection of word types
# ═══════════════════════════════════════════════

print(f"\n{'='*60}")
print("  WORD TYPE ANALYSIS")
print(f"{'='*60}")
print("\n  Look at the cod (output type) of each Word box.")
print("  This is the GRAMMATICAL TYPE assigned by the CCG parser.\n")

for i, (sent, diagram) in enumerate(zip(sentences, diagrams)):
    print(f"\n  \"{sent}\":")
    for box in diagram.boxes:
        if type(box).__name__ == "Word":
            print(f"    {box.name:20s} → type: {box.cod}")


# ═══════════════════════════════════════════════
# STEP 5: Answer the comprehension questions
# ═══════════════════════════════════════════════

print(f"\n\n{'='*60}")
print("  COMPREHENSION QUESTIONS")
print(f"{'='*60}")

# TODO 2: Answer these questions by replacing "..." with your answers.
# Base your answers on what you OBSERVED in the output above.

answers = {
    "Q1: How many total boxes does 'dogs chase cats' have? "
    "How many are Words vs Cups?":
        "...",

    "Q2: What is diagram.cod for every sentence? Is it always the same?":
        "...",

    "Q3: Look at the type of 'chase' in 'dogs chase cats'. "
    "What does n.r ⊗ s ⊗ n.l mean? Why does it have TWO noun slots?":
        "...",

    "Q4: Compare 'dogs that chase cats run' to 'dogs chase cats'. "
    "What extra boxes/cups does the relative clause add?":
        "...",

    "Q5: Do 'dogs chase cats' and 'cats are chased by dogs' have the "
    "same diagram structure? (same number of boxes, cups, types?)":
        "...",

    "Q6: Look at the word types for 'dogs that chase cats run'. "
    "What type does 'that' have? Why is it different from a noun?":
        "...",
}

for q, a in answers.items():
    print(f"\n  {q}")
    print(f"  → {a}")


print("\n\nDone! Review the saved diagrams in results/figures/")
print("When ready, share this output with me for review.")
