"""
Weisfeiler-Leman (WL) Graph Kernel for DisCoCat diagrams.

Converts lambeq diagrams to labeled graphs, then extracts WL feature
vectors that capture topological structure — not just box counts.

This replaces the count-based fingerprints from Lesson 4.
"""

import numpy as np
from collections import Counter, defaultdict


def diagram_to_graph(diagram):
    """
    Convert a lambeq DisCoCat diagram to a labeled graph.

    Returns:
        nodes: list of node labels (str)
        adjacency: dict mapping node_id -> list of neighbor node_ids
    """
    boxes = diagram.boxes
    nodes = []
    adjacency = defaultdict(list)

    for i, box in enumerate(boxes):
        box_type = type(box).__name__
        if box_type == "Word":
            # Label = "Word:<grammatical_type>"
            cod_str = str(box.cod).replace(' ', '')
            label = f"W:{cod_str}"
        elif box_type == "Cup":
            label = "Cup"
        elif box_type == "Cap":
            label = "Cap"
        elif box_type == "Swap":
            label = "Swap"
        else:
            label = f"Other:{box_type}"
        nodes.append(label)

    # Build edges: sequential adjacency (captures composition order)
    for i in range(len(nodes) - 1):
        adjacency[i].append(i + 1)
        adjacency[i + 1].append(i)

    # Also connect all Word nodes to capture long-range dependencies
    word_indices = [i for i, n in enumerate(nodes) if n.startswith("W:")]
    for i in range(len(word_indices)):
        for j in range(i + 2, len(word_indices)):
            # Connect non-adjacent words (adjacent already connected above)
            adjacency[word_indices[i]].append(word_indices[j])
            adjacency[word_indices[j]].append(word_indices[i])

    return nodes, adjacency


def wl_hash(label, neighbor_labels):
    """
    WL relabeling: hash a node's label with its sorted neighbor labels.
    Returns a deterministic string hash.
    """
    sorted_neighbors = tuple(sorted(neighbor_labels))
    combined = f"{label}|{'|'.join(sorted_neighbors)}"
    return combined


def wl_features(nodes, adjacency, n_iterations=3):
    """
    Run WL subtree kernel and return the label histogram.

    At each iteration, each node's label is updated to include
    its neighbors' labels. The histogram of all labels across
    all iterations is the feature vector.

    Args:
        nodes: list of initial node labels
        adjacency: dict mapping node_id -> list of neighbor ids
        n_iterations: number of WL iterations

    Returns:
        label_counts: Counter of all WL labels across iterations
    """
    current_labels = {i: label for i, label in enumerate(nodes)}
    all_labels = Counter(current_labels.values())  # iteration 0

    for iteration in range(n_iterations):
        new_labels = {}
        for node_id in range(len(nodes)):
            neighbor_labels = [current_labels[n]
                             for n in adjacency.get(node_id, [])]
            new_label = wl_hash(current_labels[node_id], neighbor_labels)
            new_labels[node_id] = new_label

        current_labels = new_labels
        all_labels.update(current_labels.values())

    return all_labels


class WLFingerprint:
    """
    Extracts fixed-size WL fingerprint vectors from DisCoCat diagrams.

    Usage:
        wl = WLFingerprint(n_iterations=3)
        wl.fit(diagrams)          # learn the feature vocabulary
        X = wl.transform(diagrams)  # extract feature vectors
    """

    def __init__(self, n_iterations=3, max_features=256):
        self.n_iterations = n_iterations
        self.max_features = max_features
        self.vocabulary_ = None

    def fit(self, diagrams):
        """
        Learn the WL label vocabulary from a set of diagrams.

        Collects all WL labels that appear, keeps the top max_features
        most frequent ones as the feature dimensions.
        """
        global_counts = Counter()

        for diagram in diagrams:
            nodes, adj = diagram_to_graph(diagram)
            label_counts = wl_features(nodes, adj, self.n_iterations)
            global_counts.update(label_counts.keys())

        # Keep top-k most common labels as features
        most_common = global_counts.most_common(self.max_features)
        self.vocabulary_ = {label: idx for idx, (label, _)
                           in enumerate(most_common)}
        return self

    def transform(self, diagrams):
        """
        Convert diagrams to fixed-size feature vectors.

        Returns:
            X: np.array of shape (n_diagrams, n_features)
        """
        if self.vocabulary_ is None:
            raise RuntimeError("Call fit() before transform()")

        n_features = len(self.vocabulary_)
        X = np.zeros((len(diagrams), n_features), dtype=np.float64)

        for i, diagram in enumerate(diagrams):
            nodes, adj = diagram_to_graph(diagram)
            label_counts = wl_features(nodes, adj, self.n_iterations)

            for label, count in label_counts.items():
                if label in self.vocabulary_:
                    X[i, self.vocabulary_[label]] = count

        return X

    def fit_transform(self, diagrams):
        """Fit and transform in one step."""
        self.fit(diagrams)
        return self.transform(diagrams)

    def feature_names(self):
        """Return the WL label for each feature dimension."""
        if self.vocabulary_ is None:
            return []
        inv = {idx: label for label, idx in self.vocabulary_.items()}
        return [inv[i] for i in range(len(inv))]


def compute_wl_kernel_matrix(diagrams, n_iterations=3):
    """
    Compute the full WL kernel matrix for a set of diagrams.

    K[i,j] = <phi(diagram_i), phi(diagram_j)>

    This can be passed directly to kernel CKA.

    Returns:
        K: np.array of shape (n, n), the kernel matrix
        wl: fitted WLFingerprint object
    """
    wl = WLFingerprint(n_iterations=n_iterations)
    X = wl.fit_transform(diagrams)

    # Normalize features (zero mean, unit variance)
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X - mu) / std

    # Linear kernel
    K = X_norm @ X_norm.T

    return K, wl, X_norm
