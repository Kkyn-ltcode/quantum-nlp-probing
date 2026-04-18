"""
Lesson 01: DisCoCat Diagram Exploration
========================================
HOMEWORK: Fill in the sections marked with TODO.

Run with:
    conda activate qnlp
    cd /Users/nguyen/Documents/Work/Quantum
    python notebooks/lesson01_discocat.py

NOTE: BobcatParser requires downloading a model from a server that is
currently offline (qnlp.cambridgequantum.com). This lesson uses TWO
approaches:
  Part A: cups_reader (works offline, simpler diagrams)
  Part B: BobcatParser (requires model - we'll fix this separately)

Start with Part A. Come back to Part B once we resolve the model issue.
"""

from pathlib import Path

# Ensure output directory exists
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════
# PART A: Explore diagrams with cups_reader
# (works immediately — no model download needed)
# ═══════════════════════════════════════════════

from lambeq import cups_reader, spiders_reader

sentences = [
    "dogs chase cats",             # 1. Simple transitive (SVO)
    "dogs run",                    # 2. Simple intransitive (SV)
    "big dogs chase small cats",   # 3. Transitive with adjectives
    "dogs that chase cats run",    # 4. Relative clause
    "cats are chased by dogs",     # 5. Passive voice
]

print("=" * 60)
print("  PART A: cups_reader diagrams")
print("  (bag-of-words with cup connections)")
print("=" * 60)

cups_diagrams = []

for i, sent in enumerate(sentences):
    diagram = cups_reader.sentence2diagram(sent)
    cups_diagrams.append(diagram)

    print(f"\n{'─'*60}")
    print(f"  Sentence {i+1}: \"{sent}\"")
    print(f"{'─'*60}")
    print(f"  diagram.dom (input type):  {diagram.dom}")
    print(f"  diagram.cod (output type): {diagram.cod}")
    print(f"\n  Boxes ({len(diagram.boxes)} total):")

    word_count = 0
    cup_count = 0
    other_count = 0

    for j, box in enumerate(diagram.boxes):
        box_type = type(box).__name__
        print(f"    [{j}] {box_type:15s} name={str(box.name):20s} "
              f"dom={str(box.dom):25s} cod={str(box.cod):25s}")

        # Count by type
        if box_type == "Word":
            word_count += 1
        elif box_type == "Cup":
            cup_count += 1
        else:
            other_count += 1

    print(f"\n  Summary: {word_count} words, {cup_count} cups, {other_count} other")

    # Save diagram image
    try:
        draw_path = output_dir / f"cups_diagram_{i+1}.png"
        diagram.draw(figsize=(14, 4), path=str(draw_path))
        print(f"  Diagram saved to: {draw_path}")
    except Exception as e:
        print(f"  Could not draw diagram: {e}")


# ═══════════════════════════════════════════════
# PART A.2: Compare with spiders_reader
# ═══════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  PART A.2: spiders_reader diagrams")
print("  (bag-of-words with spider merging)")
print("=" * 60)

for i, sent in enumerate(sentences):
    diagram = spiders_reader.sentence2diagram(sent)

    print(f"\n  \"{sent}\"")
    print(f"    Boxes: {len(diagram.boxes)}, cod: {diagram.cod}")

    for j, box in enumerate(diagram.boxes):
        box_type = type(box).__name__
        print(f"      [{j}] {box_type:15s} name={str(box.name):15s}")


# ═══════════════════════════════════════════════
# UNDERSTANDING THE DIFFERENCE
# ═══════════════════════════════════════════════

print(f"\n\n{'=' * 60}")
print("  KEY INSIGHT")
print("=" * 60)
print("""
  cups_reader and spiders_reader are SIMPLE readers:
  - They treat every word the same way (no grammar).
  - cups_reader connects words via cups (pair-wise contractions).
  - spiders_reader merges all word meanings via a spider (fan-in).

  Neither of these captures REAL syntax. The sentence
  "dogs chase cats" and "cats chase dogs" produce IDENTICAL diagrams.

  For REAL syntactic diagrams, we need a CCG parser like BobcatParser,
  which understands that "chase" is a transitive verb (type n.r ⊗ s ⊗ n.l)
  and creates cups that connect subject/verb/object properly.

  The BobcatParser model server is currently offline. We will resolve
  this in our next session. For now, study the cups/spiders diagrams
  to understand the MECHANICS of diagrams (boxes, wires, cups, types).
""")


# ═══════════════════════════════════════════════
# TODO: Answer these questions
# ═══════════════════════════════════════════════

print("=" * 60)
print("  COMPREHENSION QUESTIONS")
print("=" * 60)

answers = {
    "Q1: In cups_reader, how many cups does 'dogs chase cats' have? "
    "How many does 'big dogs chase small cats' have?":
        "...",

    "Q2: What is diagram.cod for every sentence? Is it always the same?":
        "...",

    "Q3: In cups_reader, does 'dogs chase cats' have a DIFFERENT diagram "
    "from 'cats chase dogs'? Why is this a problem for syntax?":
        "...",

    "Q4: What is the difference between a Cup and a Spider? "
    "(Look at how they combine word meanings.)":
        "...",

    "Q5: Why do we NEED BobcatParser instead of cups_reader for our "
    "research? (Hint: think about what 'syntax skeleton' means.)":
        "...",
}

for q, a in answers.items():
    print(f"\n  {q}")
    print(f"  → {a}")


print("\n\nDone! Review the saved diagrams in results/figures/")
print("Share this output with me when ready.")
