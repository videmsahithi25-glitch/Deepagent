import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = SentenceTransformer(
    "all-mpnet-base-v2"
)
intent_embedding_cache = None
def get_embedding(text: str):
    return embedding_model.encode(text)


import os
import json
import hashlib

intent_examples_cache = None

# ---- Disk-backed embedding cache -----------------------------------------
# build_intent_embeddings() used to only cache in-memory (intent_embedding_cache),
# which meant every fresh process (any Streamlit restart, not just reruns
# within one running process) re-encoded all ~12,450 training examples from
# scratch -- the 13-15 min cold start. This persists the computed vectors to
# disk, keyed by a hash of the training data itself, so a fresh process loads
# pre-computed vectors instead of re-encoding. If question_mode.json /
# topic_mode.json (or any file in intent_data/) ever change, the hash changes
# too, so it re-encodes automatically -- no manual invalidation needed.
EMBEDDING_CACHE_DIR = "embedding_cache"


def _examples_hash(examples: dict) -> str:
    """Hash the training data content (not the file paths/mtimes) so the
    cache key only changes when the actual example text changes."""
    hasher = hashlib.sha256()
    for intent in sorted(examples.keys()):
        hasher.update(intent.encode("utf-8"))
        for sentence in examples[intent]:
            hasher.update(sentence.encode("utf-8"))
    return hasher.hexdigest()[:16]


def _cache_path(data_hash: str) -> str:
    return os.path.join(EMBEDDING_CACHE_DIR, f"intent_embeddings_{data_hash}.npz")


def load_intent_examples():

    global intent_examples_cache

    if intent_examples_cache is not None:
        return intent_examples_cache

    intent_examples = {}

    intent_folder = "intent_data"

    for filename in os.listdir(intent_folder):

        if not filename.endswith(".json"):
            continue

        if filename in (
            "topics.json",
            "question_templates.json",
            "topic_templates.json"
        ):
            continue

        intent_name = filename.replace(
            ".json",
            ""
        )

        path = os.path.join(
            intent_folder,
            filename
        )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            intent_examples[intent_name] = json.load(f)

    intent_examples_cache = intent_examples

    return intent_examples_cache

def build_intent_embeddings():

    global intent_embedding_cache

    if intent_embedding_cache is not None:
        return intent_embedding_cache

    examples = load_intent_examples()
    data_hash = _examples_hash(examples)
    cache_path = _cache_path(data_hash)

    # 1. Try the disk cache first -- this is what makes a fresh process
    #    instant instead of a 13-15 min cold start.
    if os.path.exists(cache_path):
        print(f"[Loading cached intent embeddings from {cache_path}]")
        loaded = np.load(cache_path, allow_pickle=False)
        intent_embeddings = {
            intent: loaded[intent] for intent in examples.keys()
        }
        intent_embedding_cache = intent_embeddings
        return intent_embeddings

    # 2. Nothing on disk yet (first run ever, or the training data changed
    #    since the last cached hash) -- encode once, then persist it.
    print(
        f"[No embedding cache found for this dataset (hash={data_hash}) -- "
        f"encoding {sum(len(v) for v in examples.values())} examples once. "
        f"This is the only time this should take a while.]"
    )

    intent_embeddings = {}

    for intent, sentences in examples.items():

        embeddings = embedding_model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        intent_embeddings[intent] = embeddings

    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    np.savez(cache_path, **intent_embeddings)
    print(f"[Saved intent embeddings to {cache_path} for instant future loads]")

    intent_embedding_cache = intent_embeddings

    return intent_embeddings

    
def find_semantic_intent(user_text, top_n_per_class=10, top_k=5):
    """
    For each intent category, computes similarity against ALL of that
    category's examples in one vectorized call (instead of one
    cosine_similarity call per individual example -- with 7719 +
    4731 examples, that was 12,450 separate Python-level calls per
    query). Then takes the average of each category's own top-N
    nearest examples and compares those averages.

    WHY AVERAGE OF TOP-N PER CLASS, NOT A SINGLE NEAREST MATCH:
    The old version trusted whichever single example scored highest
    across the whole pooled set. With a large dataset, one
    ambiguously-phrased generated example being the closest match by
    chance was enough to flip the classification -- that's the
    "detecting topic_mode for a question_mode query" bug. Averaging
    over each class's own top-N smooths out that single-example noise.

    WHY PER-CLASS, NOT A POOLED TOP-K VOTE:
    question_mode.json has 7719 examples, topic_mode.json has 4731 --
    a 62/38 split. A pooled "which class has more examples in the
    global top-k" vote would systematically favor question_mode
    just because it has more examples sitting in embedding space,
    regardless of which class is actually the better semantic match.
    Scoring each class only against its OWN examples removes that
    imbalance entirely -- a class with fewer examples isn't penalized
    for having fewer examples.
    """

    intent_embeddings = build_intent_embeddings()
    intent_examples = load_intent_examples()

    user_embedding = np.array(
        get_embedding(user_text)
    ).reshape(1, -1)

    class_scores = {}
    class_top_examples = {}

    for intent, embeddings in intent_embeddings.items():

        examples = intent_examples[intent]

        # One vectorized call per class instead of one call per example.
        sims = cosine_similarity(
            user_embedding,
            np.array(embeddings)
        )[0]

        n = min(top_n_per_class, len(sims))

        top_indices = np.argsort(sims)[::-1][:n]

        class_scores[intent] = float(np.mean(sims[top_indices]))

        class_top_examples[intent] = [
            {
                "example": examples[i],
                "score": float(sims[i])
            }
            for i in top_indices
        ]

    ranked = sorted(
        class_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    predicted_intent, top_score = ranked[0]
    second_intent, second_score = ranked[1] if len(ranked) > 1 else (None, 0.0)
    margin = top_score - second_score

    # Flatten per-class top examples into one pooled, re-sorted list,
    # for callers that just want "the top_k most similar examples
    # overall" (e.g. for display/debugging) rather than the
    # per-class breakdown.
    pooled = [
        {"intent": intent, **item}
        for intent, items in class_top_examples.items()
        for item in items
    ]
    pooled.sort(key=lambda x: x["score"], reverse=True)

    top_match = {
        "intent": predicted_intent,
        "example": class_top_examples[predicted_intent][0]["example"],
        "score": class_top_examples[predicted_intent][0]["score"],
    }

    return {

        "query": user_text,

        "predicted_intent": predicted_intent,

        "confidence": top_score,

        "margin": margin,

        "second_intent": second_intent,

        "class_scores": class_scores,

        "class_top_examples": class_top_examples,

        # kept for backward compatibility with existing callers
        # (intent_engine.py currently reads top_match / top_k)
        "top_match": top_match,

        "top_k": pooled[:top_k]

    }