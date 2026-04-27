"""
Template-based sentence generator for controlled syntactic experiments.

Generates sentences across 5 construction types using a fixed vocabulary
pool, ensuring lexical content is controlled while syntax varies.
"""

import random
from collections import defaultdict


# ── Vocabulary Pools ──

NOUNS = [
    "dog", "cat", "chef", "doctor", "teacher", "student",
    "nurse", "artist", "judge", "farmer", "driver", "pilot",
    "singer", "dancer", "writer", "baker", "lawyer", "sailor",
    "guard", "clerk",
]

VERBS_TRANSITIVE = [
    ("chase", "chased", "chased"),
    ("help", "helped", "helped"),
    ("watch", "watched", "watched"),
    ("praise", "praised", "praised"),
    ("follow", "followed", "followed"),
    ("examine", "examined", "examined"),
    ("teach", "taught", "taught"),
    ("like", "liked", "liked"),
    ("visit", "visited", "visited"),
    ("call", "called", "called"),
    ("push", "pushed", "pushed"),
    ("pull", "pulled", "pulled"),
    ("stop", "stopped", "stopped"),
    ("kick", "kicked", "kicked"),
    ("cook", "cooked", "cooked"),
    ("paint", "painted", "painted"),
    ("clean", "cleaned", "cleaned"),
    ("greet", "greeted", "greeted"),
    ("blame", "blamed", "blamed"),
    ("admire", "admired", "admired"),
]

ADJECTIVES = [
    "tall", "young", "old", "small", "clever", "quiet",
    "brave", "gentle", "proud", "kind", "happy", "wise",
    "strong", "fast", "calm", "shy", "bold", "fair",
    "keen", "warm",
]

CONSTRUCTIONS = ["active", "passive", "relative", "cleft", "ditransitive"]


def generate_active(agent, adj_a, verb_tuple, patient, adj_p):
    _, past, _ = verb_tuple
    return f"the {adj_a} {agent} {past} the {adj_p} {patient}"


def generate_passive(agent, adj_a, verb_tuple, patient, adj_p):
    _, _, pp = verb_tuple
    return f"the {adj_p} {patient} was {pp} by the {adj_a} {agent}"


def generate_relative(agent, adj_a, verb_tuple, patient, adj_p,
                      rel_noun, rel_verb_tuple):
    _, past1, _ = verb_tuple
    _, past2, _ = rel_verb_tuple
    return (f"the {adj_a} {agent} that {past1} the {adj_p} {patient} "
            f"{past2} the {rel_noun}")


def generate_cleft(agent, adj_a, verb_tuple, patient, adj_p):
    _, past, _ = verb_tuple
    return f"it was the {adj_a} {agent} who {past} the {adj_p} {patient}"


def generate_ditransitive(agent, adj_a, verb_tuple, patient, adj_p, obj_noun):
    _, past, _ = verb_tuple
    return f"the {adj_a} {agent} {past} the {adj_p} {patient} the {obj_noun}"


def generate_sentences(n_per_construction=200, seed=42):
    """
    Generate sentences for all 5 construction types.

    Args:
        n_per_construction: how many sentences per type
        seed: random seed

    Returns:
        sentences: list of sentence strings
        metadata: list of dicts with construction type and vocabulary
    """
    rng = random.Random(seed)
    sentences = []
    metadata = []

    for ctype in CONSTRUCTIONS:
        generated = set()
        attempts = 0
        max_attempts = n_per_construction * 20

        while len([m for m in metadata if m['construction'] == ctype]) < n_per_construction:
            attempts += 1
            if attempts > max_attempts:
                break

            agent, patient = rng.sample(NOUNS, 2)
            adj_a, adj_p = rng.sample(ADJECTIVES, 2)
            verb = rng.choice(VERBS_TRANSITIVE)

            if ctype == "active":
                sent = generate_active(agent, adj_a, verb, patient, adj_p)
            elif ctype == "passive":
                sent = generate_passive(agent, adj_a, verb, patient, adj_p)
            elif ctype == "relative":
                rel_noun = rng.choice([n for n in NOUNS
                                       if n not in (agent, patient)])
                rel_verb = rng.choice([v for v in VERBS_TRANSITIVE
                                       if v != verb])
                sent = generate_relative(agent, adj_a, verb, patient, adj_p,
                                        rel_noun, rel_verb)
            elif ctype == "cleft":
                sent = generate_cleft(agent, adj_a, verb, patient, adj_p)
            elif ctype == "ditransitive":
                obj_noun = rng.choice([n for n in NOUNS
                                       if n not in (agent, patient)])
                sent = generate_ditransitive(agent, adj_a, verb, patient,
                                            adj_p, obj_noun)

            if sent not in generated:
                generated.add(sent)
                sentences.append(sent)
                metadata.append({
                    'construction': ctype,
                    'agent': agent,
                    'patient': patient,
                    'verb': verb[0],
                    'source': 'template',
                })

    return sentences, metadata


def generate_paraphrase_pairs(sentences, metadata, seed=42):
    """
    Generate paraphrase pairs from active/passive sentences
    that share the same vocabulary.

    Returns:
        pairs: list of (idx_a, idx_b, label) tuples
               label=1.0 for paraphrase, 0.0 for non-paraphrase
    """
    rng = random.Random(seed)

    # Build lookup by vocabulary key
    active_by_key = {}
    passive_by_key = {}

    for i, m in enumerate(metadata):
        key = (m['agent'], m['patient'], m.get('verb', ''))
        if m['construction'] == 'active':
            active_by_key[key] = i
        elif m['construction'] == 'passive':
            passive_by_key[key] = i

    # Matched pairs (paraphrases)
    pos_pairs = []
    for key in active_by_key:
        if key in passive_by_key:
            pos_pairs.append((active_by_key[key], passive_by_key[key], 1.0))

    # Random mismatched pairs (non-paraphrases)
    active_indices = list(active_by_key.values())
    passive_indices = list(passive_by_key.values())
    neg_pairs = []

    for _ in range(len(pos_pairs)):
        while True:
            ia = rng.choice(active_indices)
            ip = rng.choice(passive_indices)
            ka = (metadata[ia]['agent'], metadata[ia]['patient'],
                  metadata[ia].get('verb', ''))
            kp = (metadata[ip]['agent'], metadata[ip]['patient'],
                  metadata[ip].get('verb', ''))
            if ka != kp:
                neg_pairs.append((ia, ip, 0.0))
                break

    all_pairs = pos_pairs + neg_pairs
    rng.shuffle(all_pairs)
    return all_pairs
