import os
import json
import textwrap
import re
import io
import contextlib
import uuid
import hashlib
import threading
import statistics
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, List, Tuple, Dict
from dotenv import load_dotenv
from tavily import TavilyClient
from typing import Union
import matplotlib
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent, FilesystemPermission
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st

import spacy
from nltk.corpus import wordnet
import nltk

# Guarded, same pattern meaning_resolver.py uses for KeyBERT -- PDF/DOCX
# support is optional. A missing package degrades that one file format
# to "unsupported" (caught and reported per-upload) instead of crashing
# the whole app at import time.
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

# New, for table + image extraction (docx tables/images, pdf tables/images).
# Same guarded pattern as PdfReader/DocxDocument above -- a missing package
# degrades that one capability (table extraction, or OCR) to "skipped" for
# that file type instead of crashing the whole app at import time.
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import pytesseract
    from PIL import Image as PILImage
except Exception:
    pytesseract = None
    PILImage = None

# ---- New intent-engine pipeline (replaces the old rule-engine/LogisticRegression
# hybrid pipeline that lived in embedcopy.py) ----
from conversation_resolver import DialogueState, update_entity, resolve_context
from intent_engine import decide_intent, update_dialogue_state

# Needed by the embedding-based bucket_concept_history() further down --
# same function semantic_retriever.py already uses for intent scoring.
from sklearn.metrics.pairwise import cosine_similarity

# Reuse the ONE SentenceTransformer already loaded in semantic_retriever.py
# instead of letting Chroma construct its own separate copy of the same
# model. Was previously 3 independent loads of all-mpnet-base-v2 across
# semantic_retriever.py / intent_engine.py / here -- now just 1.
#
# chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction
# already keeps a class-level cache of loaded models keyed by model_name
# (`models: Dict[str, Any] = {}`) -- if model_name is already a key in that
# dict when the class is instantiated, it reuses the cached model instead
# of loading a new one. Pre-seeding it here means we keep using chromadb's
# real class (so name()/get_config()/build_from_config() all still work
# correctly for validating against the already-persisted collection)
# while still avoiding a second load of the same weights.
from semantic_retriever import embedding_model as _shared_embedding_model

embedding_functions.SentenceTransformerEmbeddingFunction.models[
    "all-mpnet-base-v2"
] = _shared_embedding_model

try:
    wordnet.synsets("test")
except Exception:
    # Broadened from `except LookupError` -- a corrupted/incomplete cache
    # (wordnet.zip present but unreadable) throws something else entirely
    # from deep inside nltk's zip reader, not LookupError, and was crashing
    # the whole app at import time instead of triggering a redownload.
    # force=True re-fetches even if nltk *thinks* it already has the data,
    # which is exactly the case a corrupted-but-present cache needs.
    nltk.download("wordnet", force=True)
    nltk.download("omw-1.4", force=True)


load_dotenv(override=True)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")
os.environ["LANGSMITH_TRACING"] = "True"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = "DeepAgent"

# Belt-and-suspenders: also set the OLDER pre-rename variable names.
# Traces were confirmed working under the old LangChain naming scheme
# (visible in LangSmith history) and stopped after this code moved to the
# newer LANGSMITH_* names -- if the installed langsmith/langchain-core
# version predates full support for the new names, it may only read these.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def init_chroma():
    print("[Setting up preference memory...]")
    chroma_client = chromadb.PersistentClient(path="./chroma_preferences")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-mpnet-base-v2"
    )
    pref_collection = chroma_client.get_or_create_collection(
        name="user_preferences",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"[Preference memory ready — {pref_collection.count()} preferences stored so far]\n")
    return pref_collection


pref_collection = init_chroma()


# multi-qa-mpnet-base-cos-v1 is used ONLY for the document-RAG collection
# below -- everywhere else in the app (preferences, intent classification)
# keeps all-mpnet-base-v2 unchanged. all-mpnet-base-v2 is trained for
# SYMMETRIC similarity (sentence vs sentence of similar length/style);
# document retrieval is ASYMMETRIC (short question vs longer passage), a
# different task the model was never optimized for -- confirmed as the
# likely ceiling on a real case: a query that's a near-verbatim match to
# a cleanly-isolated chunk's own heading still only scored 0.4336, below
# the 0.5 routing threshold. multi-qa-mpnet-base-cos-v1 is purpose-built
# for question->passage retrieval and (unlike its dot-product sibling
# multi-qa-mpnet-base-dot-v1) is trained for cosine similarity specifically
# -- matches this collection's existing "hnsw:space": "cosine" metadata,
# so nothing else about the collection setup needs to change.
#
# IMPORTANT one-time migration note: any chunks already embedded under
# all-mpnet-base-v2 are in a different vector space than this model
# produces. They are NOT comparable to new queries embedded with this
# model and must be wiped and re-ingested once after this change lands --
# see the reset instructions provided alongside this edit.
DOC_EMBEDDING_MODEL = "multi-qa-mpnet-base-cos-v1"


@st.cache_resource
def init_doc_chroma():
    print("[Setting up document memory...]")
    chroma_client = chromadb.PersistentClient(path="./chroma_documents")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=DOC_EMBEDDING_MODEL
    )
    doc_collection = chroma_client.get_or_create_collection(
        name="user_documents",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"[Document memory ready — {doc_collection.count()} chunks stored so far]\n")
    return doc_collection


doc_collection = init_doc_chroma()


def generate_training_examples(model, category: str, description: str, n: int = 40) -> list:
    """One-time generation — run manually, not on every app start."""
    prompt = (
        f"Generate {n} short, varied, natural phrases a real user might type that mean: "
        f"{description}\n"
        "Vary the phrasing, vocabulary, and sentence structure widely — avoid repeating "
        "similar wording. Output ONE phrase per line, nothing else."
    )
    result = model.invoke(prompt)
    text = result.content if hasattr(result, "content") else str(result)
    return [line.strip() for line in text.split("\n") if line.strip()]


def save_preference(text: str, dedup_threshold: float = 0.6, pref_type: str = "domain"):
    """Add a preference string to the Chroma collection, skipping it if a near-duplicate
    already exists. pref_type is stored as metadata ("style" or "domain") so retrieval
    can treat the two differently: a style preference ("explain in bullet points")
    describes HOW to answer and should apply regardless of what the current query is
    about; a domain preference ("frequently asks about RAG") describes WHAT the user
    is interested in and should only apply when the current query is actually about
    that domain. Dedup check still runs against the whole collection, not just same-type
    entries -- a style line and a domain line are never going to look like near-duplicates
    of each other anyway, so no need to scope that check by type.

    NOTE: preferences saved before this change have no "type" metadata and won't match
    either type filter used in retrieval below -- they'll go silent rather than error.
    Re-save them (or backfill metadata directly in Chroma) if you want them picked up
    again.
    """
    if pref_collection.count() > 0:
        existing = pref_collection.query(query_texts=[text], n_results=1)
        existing_docs = existing["documents"][0]
        existing_dists = existing["distances"][0]
        if existing_docs:
            closest_similarity = 1 - existing_dists[0]
            if closest_similarity >= dedup_threshold:
                print(f"[Preference already known — skipped duplicate: {text}]")
                return

    new_id = str(uuid.uuid4())
    metadata = {"type": pref_type}
    pref_collection.add(documents=[text], ids=[new_id], metadatas=[metadata])
    print(f"[Memory updated ({pref_type}): {text}]")


DOMAIN_PREF_WHERE = {"type": "domain"}  # filter for fetching domain-type
                                         # preferences directly (no similarity
                                         # search involved).


def get_style_preferences() -> list:
    """Style preferences ('explain in bullet points', 'keep it concise') describe
    HOW to answer, independent of what the current query is about -- there's no
    query-relevance question to ask here, so no embedding similarity search and no
    threshold. Just fetch everything tagged type='style' and return it as-is."""
    if pref_collection.count() == 0:
        return []
    try:
        results = pref_collection.get(where={"type": "style"})
        return results.get("documents", []) or []
    except Exception as e:
        print(f"[get_style_preferences failed: {e}]")
        return []


def retrieve_preferences(
    query: str,
    current_entity: Optional[str] = None,
    k: int = 5,
):
    """DOMAIN preferences only surface when the current turn's own topic
    (current_entity, from DialogueState) is itself a recognized recurring
    interest -- checked DIRECTLY against each stored domain via embedding
    similarity, using the same validated EMBEDDING_BUCKET_THRESHOLD=0.65
    bar bucket_concept_history uses for "same underlying concept,
    different wording". Without current_entity there's nothing to check
    against, so nothing is returned.

    REMOVED (was here before): a first-pass cosine search of the raw
    query text against every stored domain, which surfaced anything
    above a low threshold as a "related topic" even when the domain had
    nothing to do with what was actually asked -- e.g. a "meditation"
    query surfacing a stored "gradient descent" preference purely
    because the two vectors sat close enough by chance. There is no
    "related topic" concept anymore, no broad search, and no fallback --
    only a direct own-domain check.

    Returns a list of dicts: {"domain": str, "similarity": float}.
    """
    if not current_entity:
        return []
    if pref_collection.count() == 0:
        return []

    try:
        results = pref_collection.get(where=DOMAIN_PREF_WHERE)
    except Exception as e:
        print(f"[retrieve_preferences fetch failed: {e}]")
        return []

    documents = results.get("documents", []) or []
    if not documents:
        return []

    own_vec = _shared_embedding_model.encode([current_entity], convert_to_numpy=True)
    domain_vecs = _shared_embedding_model.encode(documents, convert_to_numpy=True)
    sims = cosine_similarity(own_vec, domain_vecs)[0]

    matches = []
    for doc, sim in zip(documents, sims):
        if sim >= EMBEDDING_BUCKET_THRESHOLD:
            print(f"[User domain preference found: {doc} (similarity={sim:.3f})]")
            matches.append({"domain": doc, "similarity": float(sim)})

    matches.sort(key=lambda m: m["similarity"], reverse=True)
    return matches[:k]


def retrieve_user_preferences(query: str, current_entity: Optional[str] = None) -> str:
    """Search stored preferences relevant to this turn. Style preferences (bullet
    points, detail level, tone, etc.) describe HOW to answer and always apply, so
    they're included unconditionally via get_style_preferences(). Domain
    preferences (which subjects the user keeps returning to) only apply when the
    current turn's own topic (current_entity) is itself a recognized recurring
    interest, checked directly via retrieve_preferences() -- no separate
    "related topic" concept anymore.

    Returns a plain-text block of concrete instructions for the agent -- not
    just a list of facts -- so the instruction to "apply preferences" (in
    deepagent_system_prompt) has something actionable to work with: an own-
    domain hit should make the answer noticeably more thorough and briefly
    acknowledge prior engagement.
    """
    style_prefs = get_style_preferences()
    own_matches = retrieve_preferences(query, current_entity=current_entity)

    if style_prefs:
        print(f"[Style preferences applied: {'; '.join(style_prefs)}]")

    instruction_lines = []

    if style_prefs:
        instruction_lines.append(
            "STYLE — always apply: " + "; ".join(style_prefs)
        )

    if own_matches:
        own_names = ", ".join(f"'{m['domain']}'" for m in own_matches)
        instruction_lines.append(
            f"OWN TOPIC — the user has engaged with {own_names} before. "
            "Briefly acknowledge that (e.g. \"you've engaged with this before, "
            "so here's a more detailed answer\") and give a noticeably more "
            "thorough answer than you would by default -- more depth, not just "
            "a longer restatement."
        )

    if not instruction_lines:
        return "No relevant preferences found."
    return "\n".join(instruction_lines)


# ---------------------------------------------------------------------------
# Document RAG: upload a file -> chunk -> embed into doc_collection ->
# TRADITIONAL, deterministic retrieval, question_mode turns ONLY (never
# topic_mode). search_document() runs in Python for every question_mode
# turn where something is indexed, and its top_similarity return value
# decides which of two branches this turn takes:
#   - similarity >= DOC_SCOPED_THRESHOLD: this question IS about the
#     document. Preference retrieval is skipped entirely for this turn
#     (explicit requirement -- domain/style preferences must not apply when
#     the answer is supposed to come from the document, not from general
#     "how this user likes to be helped" guidance), and the model is told
#     to answer ONLY from the retrieved chunks -- no own knowledge, no
#     web_search.
#   - otherwise: this question is NOT about the document. Nothing
#     document-related is injected at all, and the turn proceeds through
#     the exact same intent-classifier + preference-retrieval + RULE 2 path
#     that existed before any of this document work started.
# ---------------------------------------------------------------------------

DOC_SCOPED_THRESHOLD = 0.46  # cosine+BM25 hybrid similarity.
                             # FIX: was 0.5, based on no real measurement.
                             # Now re-measured against real queries logged
                             # against the live resume.pdf: four genuine,
                             # unambiguous, on-document questions produced
                             # hybrid scores of 0.496, 0.504, 0.539, 0.588 --
                             # ALL real matches, confirmed by their own
                             # comfortable per-source margins (0.116-0.319,
                             # all clearing DOC_SCOPED_MARGIN). The lowest of
                             # these (0.496, "what percentage did Sahithi
                             # score in her B.Tech" -- literally asking for
                             # the "78%" sitting in the document) missed the
                             # old 0.5 cutoff by 0.004 despite being a
                             # completely valid, answerable, on-topic
                             # question -- confirming 0.5 sat almost exactly
                             # on top of where real valid queries land for
                             # this embedding model, rejecting good matches
                             # essentially at random. 0.46 gives real
                             # matches like that comfortable headroom while
                             # still sitting well above scores seen for
                             # genuinely off-document queries elsewhere in
                             # this file (0.29-0.39 range). Re-tune again if
                             # a genuinely irrelevant query is ever observed
                             # scoring above this.
                             # Because this now gates a real behavior change
                             # (whether preferences apply, and whether the
                             # model is restricted to document-only
                             # answering) rather than just what's shown to
                             # the model to judge itself, a misfire here is
                             # more consequential than the old always-
                             # inject-and-let-the-model-judge approach was.
                             # NOTE: as of the hybrid-scoring rework below,
                             # this now applies to the HYBRID (cosine + BM25)
                             # score, not raw cosine alone -- see
                             # search_document().

DOC_BM25_WEIGHT = 0.35  # how much weight the lexical (BM25) score gets in
                         # the hybrid combination; cosine keeps the majority
                         # share since it still carries the paraphrase/
                         # conceptual-match signal BM25 can't. Added because
                         # a real test against this exact pipeline showed
                         # cosine alone underscoring a near-verbatim keyword
                         # match ("emergency procedures" vs a chunk headed
                         # "Emergency Procedures" that talks about skin/eye
                         # contact and exposure) at 0.392 -- BM25 catches
                         # that kind of lexical overlap cosine similarity
                         # routinely misses. Starting point, not tuned.

BM25_SATURATION_SCALE = 5.0  # calibration constant for the BM25 normalization
                              # curve: bm25_norm = raw / (raw + SCALE). Replaces
                              # dividing by max(raw_bm25) within the search
                              # (per-query self-normalization), which forced the
                              # single top-ranked chunk to always read as a
                              # "perfect" 1.000 lexical match no matter how weak
                              # its real overlap was -- one incidental shared
                              # phrase was enough to fake a 1.000 and drag an
                              # unrelated query over DOC_SCOPED_THRESHOLD via
                              # DOC_BM25_WEIGHT. This formula instead saturates
                              # toward 1.0 only as the RAW score itself grows,
                              # on a scale that doesn't move query to query --
                              # raw==SCALE gives 0.5, raw>>SCALE approaches 1.0,
                              # a weak raw score (e.g. one short shared phrase)
                              # stays low regardless of what else was searched.
                              # 5.0 is a starting point pending calibration
                              # against real indexed documents (see
                              # calibrate_bm25_saturation.py) -- same open-
                              # loop status as DOC_BM25_WEIGHT above until
                              # that's run.

# FIX: raised from 0.08. This margin now compares each SOURCE DOCUMENT's own
# best chunk against the next-best source's own best chunk (see the
# per-source-scoped rewrite of search_document below) instead of comparing
# against a single BM25 index shared across the whole permanent, ever-growing
# collection. Because search_document now keeps every document's lexical
# stats separate, this margin is the ONLY thing standing between "confident
# match" and "confused across two plausible documents" once dozens of
# documents accumulate in doc_collection over many sessions -- 0.08 was
# calibrated back when corpora were small (1-2 test docs); a wider margin is
# the deliberate tradeoff requested for staying permanent without cross-doc
# confusion. Starting point, not tuned against real multi-doc traffic yet --
# recommend logging real margin values for a few weeks and adjusting down if
# genuine document-scoped turns are being rejected too often, or up if two
# different real documents are ever both being accepted for the same query.
# FIX: flat margin replaced with population-relative confidence (z-score)
# as the primary check once there are enough OTHER source documents to form
# a real distribution. Confirmed real repro of the flat margin's failure
# mode: "what is the summary of the results in credit risk score card"
# against a 231-chunk collection scored top_hybrid=0.572
# (CREDIT RISK SCORECARD.docx) with only margin_over_other_source=0.117 --
# below the flat DOC_SCOPED_MARGIN=0.15 -- purely because one OTHER
# unrelated document happened to land close (0.455). The flat margin can't
# tell "one close competitor out of 30 unrelated docs" apart from "two
# genuinely tied candidates" -- it only ever compares top vs #2. Under
# DOC_SCOPED_MARGIN=0.15 alone this correct match was rejected and fell
# through to the wrong branch (preference retrieval instead of the
# document). Population-relative confidence instead asks whether the top
# score is a statistical outlier against the OTHER sources' own best-chunk
# scores for this specific query (mean/stdev of that population), so one
# incidentally-close unrelated document among many doesn't sink a
# genuinely confident match, while two real candidates landing close
# together (a small population, or one that's itself tightly clustered
# near the top) still correctly fails to clear the bar. See
# DOC_SCOPED_Z_THRESHOLD below and the population-relative branch in
# search_document. DOC_SCOPED_MARGIN itself is KEPT, not removed -- it's
# still the fallback comparison used when there are too few other
# documents (0 or 1) to compute a meaningful mean/stdev, which is exactly
# the small-corpus case this constant was originally calibrated for.
DOC_SCOPED_MARGIN = 0.08  # how much clearer the best-matching chunk's hybrid
                           # score must be over the best chunk from a
                           # DIFFERENT source document in the corpus, on top
                           # of clearing DOC_SCOPED_THRESHOLD itself, before a
                           # turn counts as document-scoped. Deliberately
                           # cross-SOURCE, not just rank-2 overall -- an
                           # early version compared against whatever ranked
                           # #2 regardless of source, which wrongly rejected
                           # "what are the things out of scope in this doc"
                           # (0.607 hybrid, clearly about the document) just
                           # because a SECOND chunk from that same correct
                           # document also scored close -- that's not
                           # ambiguity, that's a broad query legitimately
                           # matching two spots in the right file. What this
                           # is actually meant to catch is the real failure
                           # case: a top match barely edging out a chunk from
                           # an unrelated document (0.299 genuine match vs a
                           # 0.296 match against the wrong doc entirely, both
                           # seen for real against this pipeline). With only
                           # one document indexed, there's nothing else to be
                           # confused with, so this imposes no penalty at
                           # all.
                           #
                           # LOWERED from 0.10 to 0.08: the 0.10 comment
                           # itself named the exact trigger for this --
                           # "if a future real case lands between 0.10 and
                           # 0.109 and turns out to be a genuine match,
                           # that's the signal to lower further". Confirmed
                           # real repro: "what are verdict bands" (0.533
                           # hybrid, within this file's own calibrated
                           # genuine-match range 0.496-0.588 -- and "verdict"
                           # verified absent from every other indexed
                           # document, so there's no real ambiguity) scored
                           # margin=0.098 against CREDIT RISK SCORECARD.docx
                           # (which shares real "score band" language but not
                           # the word "verdict"), rejected by 0.10 with no
                           # escape hatch (0.533 < DOC_SCOPED_HIGH_CONFIDENCE=
                           # 0.60). 0.08 clears this case while staying above
                           # the one confirmed genuine-ambiguity margin on
                           # record (0.003) and the flat-margin regime's own
                           # originally-calibrated value (this constant's
                           # very first tuned value, before 0.15 and 0.10
                           # were both tried and found too strict in turn).

DOC_SCOPED_HIGH_CONFIDENCE = 0.60  # absolute-score escape hatch for the
                                     # flat-margin fallback (n<DOC_SCOPED_
                                     # MIN_POPULATION other documents) -- see
                                     # its use in search_document. Set just
                                     # above 0.588, the highest real
                                     # genuine-match score recorded in
                                     # DOC_SCOPED_THRESHOLD's own calibration
                                     # notes, so this only rescues a match
                                     # that's already stronger than every
                                     # confirmed genuine match on record, not
                                     # a borderline one. Starting point --
                                     # like DOC_SCOPED_MARGIN itself, not yet
                                     # tuned against a large volume of real
                                     # multi-document traffic; revisit if a
                                     # genuinely wrong document ever clears
                                     # 0.60 for real.


DOC_SCOPED_RESCUE_FLOOR = 0.35  # lower bound for the below-threshold margin
                                  # rescue (see its use in search_document,
                                  # right after is_confident is first
                                  # computed). A top_score under
                                  # DOC_SCOPED_THRESHOLD=0.46 was previously
                                  # always "not document-scoped," full stop
                                  # -- even when every other source scored
                                  # far worse. That's real signal (nothing
                                  # else in the corpus is even close), not
                                  # noise, and it's what distinguishes a
                                  # weakly-WORDED real match from a
                                  # genuinely unrelated query: an unrelated
                                  # query tends to have BOTH a low top_score
                                  # AND a narrow margin, since nothing in
                                  # ANY document matches it well.
                                  # STARTING POINT ONLY -- same open-loop
                                  # status DOC_BM25_WEIGHT/
                                  # BM25_SATURATION_SCALE started in. Not
                                  # yet calibrated against real
                                  # margin_over_other_source values for the
                                  # actual blocked queries (document
                                  # revision, Dice, C value, LIKELY verdict
                                  # -- top_hybrid 0.418/0.444/0.458/0.570)
                                  # -- calibrate this the same way
                                  # DOC_SCOPED_Z_THRESHOLD was: pull the
                                  # real margin lines from debug_log.txt for
                                  # those specific blocked queries and set
                                  # the floor/margin pair from where they
                                  # actually cluster, not a guess.
DOC_SCOPED_RESCUE_MARGIN = 0.30  # margin over the next-best source required
                                  # to rescue a top_score that's below
                                  # DOC_SCOPED_THRESHOLD but above
                                  # DOC_SCOPED_RESCUE_FLOOR. Deliberately
                                  # wider than DOC_SCOPED_MARGIN=0.08 (which
                                  # governs confidence ABOVE threshold) --
                                  # rescuing a WEAKER absolute score should
                                  # demand a STRONGER relative signal to
                                  # compensate, not a looser one.
                                  # STARTING POINT ONLY, same as
                                  # DOC_SCOPED_RESCUE_FLOOR above -- needs
                                  # the same real-data calibration pass
                                  # before this is trustworthy.

DOC_SCOPED_MIN_POPULATION = 3  # minimum number of OTHER source documents
                                 # (besides the top one) required before
                                 # population-relative (z-score) confidence
                                 # is used at all. Below this, mean/stdev
                                 # over the population isn't meaningful (you
                                 # can't tell an outlier from a small sample
                                 # of 0-2 points), so search_document falls
                                 # back to the flat DOC_SCOPED_MARGIN check
                                 # instead -- which is exactly the small-doc-
                                 # count regime DOC_SCOPED_MARGIN was
                                 # originally calibrated for.
                                 #
                                 # RAISED from 2 to 3: DOC_SCOPED_Z_THRESHOLD
                                 # itself (see its own comment) was
                                 # calibrated against a population of 3 OTHER
                                 # documents (the 4-document/231-chunk
                                 # calibration corpus), not 2. At n=2 the
                                 # z-score is being applied outside the
                                 # regime it was actually validated against
                                 # -- confirmed real repro: "what are verdict
                                 # bands" (0.533 hybrid, a confirmed genuine
                                 # match -- "verdict" verified absent from
                                 # every other indexed document) scored
                                 # z_score=1.170 at n=2, BELOW even the
                                 # "ambiguous query" range (1.241-1.340) the
                                 # 1.7 threshold's own calibration notes
                                 # recorded -- a real match reading as more
                                 # ambiguous than genuine ambiguity itself is
                                 # the small-sample-size artifact this raise
                                 # is meant to prevent (2 population points
                                 # give a noisy, unstable stdev estimate).
                                 # Below n=3, the flat-margin fallback (also
                                 # just recalibrated, see DOC_SCOPED_MARGIN)
                                 # is used instead.
DOC_SCOPED_Z_THRESHOLD = 1.7  # how many standard deviations above the OTHER
                               # source documents' own mean best-chunk score
                               # the top source's best-chunk score must be,
                               # once DOC_SCOPED_MIN_POPULATION is met, for a
                               # turn to count as document-scoped.
                               #
                               # CALIBRATED against real data via
                               # calibrate_z_threshold.py, run against the
                               # real 231-chunk/4-document corpus (credit
                               # scorecard, resume, DevOps lifecycle, Belarc
                               # system-profile report). Only rows that also
                               # clear DOC_SCOPED_THRESHOLD=0.46 matter here
                               # (is_confident requires BOTH checks, so any
                               # row below 0.46 is rejected regardless of its
                               # z-score -- confirmed this alone already
                               # rejects every genuinely unrelated test
                               # query, even ones with surprisingly high raw
                               # z-scores). Among rows that DID clear 0.46:
                               # real "ambiguous" queries (generic
                               # "summary"/"score" wording shared by two
                               # real documents, naming neither by domain)
                               # scored z=1.241-1.340; the real bug-repro
                               # query itself ("what is the summary of the
                               # results in credit risk score card") scored
                               # z=2.092; every other real "own" query
                               # scored z=3.9-23.0. 1.7 sits in the resulting
                               # 1.340-2.092 gap. Narrower than the gaps
                               # PREFERENCE_RELEVANCE_THRESHOLD/
                               # EMBEDDING_BUCKET_THRESHOLD were calibrated
                               # against, so revisit if a real production
                               # query lands close to this boundary.
                               # If the population's stdev is 0 (every other
                               # source tied exactly), z-score is undefined
                               # -- search_document falls back to a plain
                               # top_score > population mean check in that
                               # case, not to DOC_SCOPED_MARGIN.

DOC_CHUNK_SIZE = 900     # characters per chunk. Character-based rather than
                          # token-based -- simple, and consistent with how
                          # everything else in this file already treats text.
DOC_CHUNK_OVERLAP = 150  # characters of overlap between consecutive chunks,
                          # so a fact sitting right on a chunk boundary isn't
                          # split in a way that loses its context entirely.

ASSET_DIR = "./extracted_assets"  # where extracted images are saved to disk
                                   # so they can be re-displayed in the UI
                                   # later, exactly like graph_path already
                                   # does for generated charts -- but this is
                                   # a SEPARATE folder/mechanism, not reusing
                                   # graph_path or anything topic-mode
                                   # touches. Tables don't need this: a
                                   # table's cell text IS its own content, so
                                   # it's stored as structured data directly
                                   # in chunk metadata, no file needed.
os.makedirs(ASSET_DIR, exist_ok=True)


def _iter_docx_block_items(doc):
    """Yields each paragraph and table in a docx in true document order.
    Standard python-docx recipe -- doc.paragraphs and doc.tables (used
    separately below) each only return their own type in isolation, with NO
    information about how they were actually interleaved in the real
    document. Walking doc.element.body's raw XML children directly is the
    only way to know "paragraph, paragraph, TABLE, paragraph" is the real
    order, not "all paragraphs first, then all tables" -- necessary so an
    extracted table can be inserted as its own chunk in roughly the right
    place relative to the surrounding text, instead of getting appended
    only at the very end of the document."""
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield "paragraph", Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield "table", Table(child, doc)


def _strip_markdown_tables(text: str) -> str:
    """Remove markdown-table blocks (consecutive '|...|' lines) from LLM
    answer text. Used for document-scoped turns whose winning chunk is a
    table: doc_assets already renders the exact grid deterministically via
    st.table() (see ingest_document's docstring), but the raw markdown
    table also sits in the model's own context (doc_result), so a prompt
    instruction alone only reduces -- doesn't guarantee -- the model
    re-printing it in prose too. Confirmed real repro: a "model accuracy
    comparison" answer showed the table twice, back to back (the model's
    own markdown table via st.markdown, then the same grid again via
    st.table()). This strips any markdown table the model included
    regardless of whether it followed the prompt instruction, so the UI
    only ever shows the one deterministic table. Leaves any prose before/
    after the table block untouched; collapses the blank lines left behind
    so removal doesn't leave a visible gap."""
    # A markdown table row starts and ends with '|' (ignoring surrounding
    # whitespace); a real table is 2+ such consecutive lines (header +
    # separator, at minimum).
    row_re = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        if row_re.match(lines[i]):
            j = i
            while j < len(lines) and row_re.match(lines[j]):
                j += 1
            if j - i >= 2:  # genuine table block, not one stray '|...|' line
                i = j
                continue
        out.append(lines[i])
        i += 1
    # Collapse 3+ blank lines left behind by a removed block down to a
    # single blank line, same as a normal paragraph break.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _looks_like_real_table(grid: list) -> bool:
    """Validates a pdfplumber 'text'-strategy table candidate before it's
    accepted -- guards specifically against the confirmed real failure mode
    of that strategy: word-shredded prose, not a real table (e.g. a title-
    page line "Steel Surface Defect Detection..." coming back as cells
    "Steel S" | "urf" | "ace De" | "fect D"). 'text' strategy infers column
    boundaries purely from horizontal whitespace gaps between words -- on
    ordinary prose those gaps are just normal word-spacing that happens to
    look aligned, and it slices individual WORDS apart mid-character. A
    real table cell is a complete token (a number, a short label, a whole
    word); a fabricated one is frequently a fragment that isn't a real word
    at all.

    Reuses the wordnet lookup already loaded elsewhere in this file (see
    the wordnet.synsets probe near the top of the file) instead of adding a
    new dependency: any alphabetic token of 3+ letters that ISN'T a
    recognized word/lemma counts as a fragment. Numeric/code-heavy tables
    (few real words) skip the word check entirely -- it has nothing to
    judge there and shouldn't reject a legitimate numbers-only table.

    This is deliberately conservative: reject on any doubt rather than risk
    indexing garbage as a "table" (a real table missed here still gets
    indexed as ordinary text -- nothing is lost outright, just not given
    the two-representation table treatment)."""
    if not grid or len(grid) < 2 or not grid[0] or len(grid[0]) < 2:
        return False

    total = non_empty = alpha_tokens = fragment_tokens = 0
    for row in grid:
        for cell in row:
            total += 1
            cell = (cell or "").strip()
            if cell:
                non_empty += 1
            for tok in re.findall(r"[A-Za-z]+", cell):
                if len(tok) < 3:
                    continue
                alpha_tokens += 1
                if not wordnet.synsets(tok.lower()):
                    fragment_tokens += 1

    if total == 0 or non_empty / total < 0.5:
        return False  # too sparse to be a real table

    if alpha_tokens >= 4:  # enough real words present to judge; a
                           # numbers/codes-only table has nothing here to
                           # check and shouldn't be penalized for it
        if fragment_tokens / alpha_tokens > 0.35:
            return False
    for row in grid:
        for c1, c2 in zip(row, row[1:]):
            c1, c2 = (c1 or "").strip(), (c2 or "").strip()
            if not c1 or not c2:
                continue
            last_tok = re.findall(r"[A-Za-z]+", c1)
            first_tok = re.findall(r"[A-Za-z]+", c2)
            if not last_tok or not first_tok:
                continue
            last_tok, first_tok = last_tok[-1], first_tok[0]
            joined = last_tok + first_tok
            if (len(joined) > max(len(last_tok), len(first_tok))
                    and wordnet.synsets(joined.lower())
                    and not wordnet.synsets(last_tok.lower())):
                return False  # adjacent cells reconstruct a word when
                              # merged -- mid-word shred across a column
                              # boundary (e.g. "F" | "actor" -> "Factor")

    return True
    


def _table_to_grid(table) -> list:
    """Extract a docx or pdfplumber-style table into a plain list-of-lists
    of cell text (a 'grid'). This is the ONE shared representation used for
    both searchability (formatted to markdown for embedding/BM25) and exact
    UI re-display (passed to st.table/st.dataframe as-is) -- one extraction,
    two uses, so the table shown in the UI is guaranteed to match what was
    actually searched, not a second independent re-parse."""
    grid = []
    for row in table.rows:
        grid.append([cell.text.strip() for cell in row.cells])
    return grid


def _grid_to_markdown(grid: list) -> str:
    """Format a table grid as markdown table text -- searchable (embedded +
    BM25-scored like any other chunk) AND, if the LLM includes it verbatim
    in its answer, Streamlit's st.markdown renders it as a real table
    automatically. Falls back to plain pipe-joined rows if the grid is
    ragged (unequal column counts across rows -- real-world docx tables with
    merged cells sometimes produce this)."""
    if not grid or not grid[0]:
        return ""
    header = grid[0]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _find_pdf_table_caption(page_text: str, header_row: list, start_line: int = 0):
    """Locate the heading/label that precedes a pdfplumber-extracted table on
    its page, so the table chunk can carry it (same purpose as docx's
    recent_captions for images -- see _extract_document_content).

    EARLIER, DISPROVEN APPROACH (kept here as a warning, not code): matching
    a single header-row CELL (e.g. the longest one) as a substring anchor in
    the flattened page text, then reading the "line" immediately before that
    substring's position. Verified broken against a real doc: pypdf flattens
    a table's own header row into ONE plain-text line (e.g. "Band Score
    Meaning"), so searching for just "Meaning" and backing up one line lands
    back inside that SAME row's own text ("Band Score"), not the real
    caption above the table ("Verdict Bands") -- confirmed by hand-tracing
    real output where this returned "Band Score" and "Model" instead of the
    real captions. A single cell is never long/unique enough to anchor past
    its own row.

    FIX: anchor on the WHOLE header row (all cells joined and whitespace-
    normalized), matched against whole LINES of page_text -- not a substring
    search. That guarantees a match lands on the row's own full line, so
    "the line before it" is genuinely a different line, not a fragment of
    the row itself.

    Returns (caption_or_None, next_start_line) -- next_start_line lets the
    caller advance a per-page cursor across multiple tables on one page
    without re-matching an earlier table's row.
    """
    cells = [c for c in header_row if c]
    if not cells or not page_text:
        return None, start_line
    header_norm = re.sub(r"\s+", " ", " ".join(cells)).strip().lower()
    if len(header_norm) < 3:
        return None, start_line
    lines = page_text.split("\n")
    for idx in range(start_line, len(lines)):
        line_norm = re.sub(r"\s+", " ", lines[idx]).strip().lower()
        if not line_norm:
            continue
        # Exact match is the common case (pypdf flattens the row cleanly).
        # Containment is a fallback for a row that shares its line with
        # something else, but require most of the line to be the header so
        # an unrelated line that happens to contain one shared word doesn't
        # false-positive.
        if line_norm == header_norm or (
            header_norm in line_norm and len(header_norm) >= 0.6 * len(line_norm)
        ):
            # Walk back through non-empty lines looking for a heading-shaped
            # one, accumulating the character length of skipped body text as
            # a budget instead of counting a fixed number of LINES. A fixed
            # 2-line window (matching docx's recent_captions) isn't enough
            # here: PDF text extraction preserves visual line WRAPS, so one
            # intro sentence between a heading and its table can itself span
            # 2+ physical lines (confirmed real case: "Production Deployment
            # Specifications" / "Below is a realistic deployment setup
            # showing how this solution would work in a typical steel
            # rolling" / "mill:" / <table> -- the intro alone is 2 lines,
            # so a 2-line cap lands on the intro's own wrap-continuation and
            # never reaches the real heading one line further back). A
            # ~200-character budget covers a wrapped sentence or two without
            # letting the search wander arbitrarily far up the page into an
            # unrelated section.
            lookback_budget = 200
            used = 0
            back = idx - 1
            while back >= 0 and used < lookback_budget:
                prev = lines[back].strip()
                back -= 1
                if not prev:
                    continue
                if _HEURISTIC_HEADING_RE.match(prev) and len(prev) < 80:
                    return prev, idx + 1
                used += len(prev)
            return None, idx + 1
    return None, start_line


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Extract any text printed inside an image via OCR (pytesseract).
    Returns "" if OCR isn't installed or finds nothing -- an image with no
    machine-readable text still gets indexed (see _extract_images_and_ocr
    callers), just with an empty OCR field, so it isn't silently dropped."""
    if pytesseract is None or PILImage is None:
        return ""
    try:
        img = PILImage.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(img).strip()
    except Exception as e:
        print(f"[OCR failed on an extracted image: {e}]")
        return ""


def _save_image_asset(image_bytes: bytes, file_hash: str, index: int, ext: str = "png") -> str:
    """Save extracted image bytes to disk once at ingest time, return the
    path. Mirrors how a generated chart already gets saved and later
    re-displayed via graph_path -- same idea (save once, re-render by path
    later), but a completely separate folder/field so this never touches
    graph_path or anything topic-mode reads."""
    ext = (ext or "png").lower().lstrip(".")
    if ext not in ("png", "jpg", "jpeg", "gif", "bmp", "webp"):
        ext = "png"
    path = os.path.join(ASSET_DIR, f"{file_hash}_{index}.{ext}")
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path


_SECTION_BREAK = "\x00SECTION\x00"  # sentinel marking a real heading/title
_SECTION_BREAK_HEURISTIC = "\x00SECTIONH\x00"  # sentinel marking a heading
                          # detected only via _HEURISTIC_HEADING_RE (no real
                          # style backing it) -- kept separate from
                          # _SECTION_BREAK so _chunk_text can drop a
                          # heuristic heading that turns out to have no body
                          # (a bare list item) instead of keeping it as a
                          # standalone hollow chunk. See docx branch of
                          # _extract_text_from_upload for why this exists.
                                     # paragraph boundary in extracted docx
                                     # text -- never appears in real document
                                     # content, so splitting on it is safe.


# pypdf's extract_text() (and plain .txt) return flat text with no
# structural info at all -- unlike docx, there's no style/heading metadata
# to read. Numbered/lettered section headings ("1. Purpose", "5. Emergency
# Procedures") are common enough in guideline/policy/spec documents that a
# line-level pattern match is a reasonable heuristic (not as reliable as
# docx's real style info, but far better than none). Without this,
# _chunk_text's section-aware packing had NO signal at all for pdf/txt/md
# uploads -- the whole document was always treated as a single "section"
# (see _chunk_text's own docstring), so unrelated sections got packed into
# the same chunk purely by character count. Confirmed as the root cause of
# a real false-negative: "what are emergency procedures" scored only 0.374
# against a PPE guideline PDF whose own section 5 is literally titled
# "Emergency Procedures", because that heading and its content were never
# isolated into their own chunk.
#
# Originally PDF-only (_mark_pdf_section_breaks) -- generalized here to
# _mark_heuristic_section_breaks so the same numbered/title-case heading
# detection also applies to .txt uploads, which carry exactly the same
# "no structural metadata" problem PDFs do.
_HEURISTIC_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+\S"
    r"|[A-Z][A-Za-z ]{2,40}:?\s*$"
    # NEW: identifier/snake_case single-token labels ("application_train:",
    # "POS_CASH_balance:") -- confirmed missing in CREDIT RISK SCORECARD.docx:
    # 7 real sub-headings under Section 2 fell through this regex (no space
    # allowed for underscores, and lowercase-first tokens failed the
    # uppercase-start branch), so all ~40 paragraphs of their content silently
    # glued onto whatever heading was still open ("Project Flowchart:")
    # instead of getting their own section break.
    r"|[A-Za-z][A-Za-z0-9_]{1,40}:\s*$)"
)
# .md files DO carry real structural info -- ATX-style "#" headings -- so
# they get a dedicated, authoritative check first (checked before falling
# back to the same heuristic .txt/.pdf use, in case a markdown file also
# has numbered plain-text lines that aren't "#" headings).
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
# A paragraph that names its OWN topic and already has real content of its
# own, e.g. "KS Statistic: KS Statistic captures the maximum vertical
# separation..." -- as opposed to a bare label ("ROC Curve:") or a plain
# continuation sentence with no leading label ("The ROC curve plots...").
_SELF_NAMED_CAPTION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 /&\-]{1,40}:\s+\S.{10,}")


def _resolve_image_caption(recent_captions: list) -> str:
    """Decide what should caption the image about to be found, from the
    rolling window of up to the last 2 non-empty paragraphs seen.

    Two real-world patterns this tells apart:
    1. Label+body split across two paragraphs ("ROC Curve:" then "The ROC
       curve plots...") -- neither paragraph names its own topic AND has
       its own content, so they belong together. Combine both.
    2. Two adjacent, fully self-contained metric write-ups ("Gini
       Coefficient: Gini is derived..." then "KS Statistic: KS Statistic
       captures...") -- BOTH paragraphs independently name their own topic
       and have their own content, confirming they're two distinct units,
       not one continuing thought. Confirmed real repro: an image between
       these got a caption blending Gini and KS explanations together,
       risking the wrong chart surfacing for a query about either metric.
       Use only the closer paragraph.
    Anything else (plain body text with no colon-led label at all, e.g. the
    score-band explanation split across two sentences) stays combined,
    same as before -- that's one continuing thought, not two topics.
    """
    if not recent_captions:
        return ""
    last = recent_captions[-1]
    if len(recent_captions) >= 2:
        prev = recent_captions[-2]
        if _SELF_NAMED_CAPTION_RE.match(last) and _SELF_NAMED_CAPTION_RE.match(prev):
            return last
    return " ".join(recent_captions).strip()


def _mark_heuristic_section_breaks(text: str, is_markdown: bool = False) -> str:
    """Marks probable heading lines with a chunk-boundary sentinel.

    FIX: a markdown '#' heading is unambiguous (real syntax), so it keeps
    the REAL sentinel (_SECTION_BREAK) -- _chunk_text trusts it and never
    drops an orphan '#'-heading-with-no-body.

    Every OTHER heading here is a guess from _HEURISTIC_HEADING_RE (a
    short title-case-looking line) -- for pdf/txt this is the ONLY
    signal available (no real style metadata), and it WILL misfire on
    things that merely look heading-shaped but aren't, most importantly
    a flattened table's own header row (e.g. a "Band | Score | Meaning"
    table rendered by pypdf as the plain text line "Band Score
    Meaning" -- title-case, short, matches the regex perfectly).

    Confirmed real failure mode from this exact misfire: it splits a
    table's title ("Verdict Bands") away from its own body into an
    orphan heading-only chunk, AND strips that title out of the actual
    table-content chunk entirely -- so the chunk with the real data no
    longer contains the word "Verdict" at all, and a short, low-content
    orphan fragment consisting of just the (mis-detected) title
    out-competes the real content on lexical (BM25) score for exactly
    the queries a user would ask (verified: "verdict bands" scored the
    orphan fragment at bm25_norm=0.653 vs the actual table chunk not
    even placing in the top 5).

    Previously this ALWAYS tagged with _SECTION_BREAK (the "trusted,
    confirmed-real" sentinel) regardless of which branch fired -- so
    _chunk_text's existing orphan-heading-with-no-body protection
    (`if is_heuristic_section: continue` -- built for exactly this
    class of bug) never actually applied to pdf/txt/heuristic-detected
    headings, only to docx's separate heuristic path. Tagging the
    regex-guessed case with _SECTION_BREAK_HEURISTIC instead lets that
    existing protection do its job here too: _chunk_text now carries an
    orphan heuristic heading FORWARD into the next section instead of
    emitting it standalone, so "Verdict Bands" ends up merged onto the
    front of the real table chunk ("Verdict Bands\nBand Score
    Meaning\nCONFIRMED ...") -- confirmed via direct BM25 re-test: the
    merged table chunk now wins first place for "verdict bands",
    "what are the verdict bands", and "what is the score for each
    verdict band", beating the summary sentence that was winning
    before."""
    lines = text.split("\n")
    marked = []
    for line in lines:
        stripped = line.strip()
        is_heading = False
        is_confirmed_real = False
        if stripped:
            if is_markdown and _MD_HEADING_RE.match(stripped):
                is_heading = True
                is_confirmed_real = True
            elif _HEURISTIC_HEADING_RE.match(stripped) and len(stripped) < 80:
                is_heading = True
                is_confirmed_real = False
        if is_heading:
            sentinel = _SECTION_BREAK if is_confirmed_real else _SECTION_BREAK_HEURISTIC
            marked.append(sentinel + stripped)
        else:
            marked.append(line)
    return "\n".join(marked)


def _extract_document_content(filename: str, file_bytes: bytes):
    """Extract text, tables, and images from an uploaded file's bytes.
    Returns (text, tables, images):
      - text: same as before -- plain paragraph text, with section-break
        sentinels for _chunk_text.
      - tables: list of (grid, caption) tuples, one per table found, in
        document order. grid is a list-of-lists of cell strings; caption is
        the heading/label text found immediately before the table in the
        source doc, or None if none was found.
      - images: list of (image_bytes, file_extension) tuples, one per
        embedded image found.
    Returns (None, [], []) for an unsupported or unparseable file --
    callers must check `text is None` before chunking/embedding it, same
    as the old _extract_text_from_upload contract."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext in ("txt", "md"):
        try:
            raw = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw = file_bytes.decode("latin-1", errors="ignore")
        return _mark_heuristic_section_breaks(raw, is_markdown=(ext == "md")), [], []

    if ext == "pdf":
        if PdfReader is None:
            print(f"[Document upload skipped: pypdf not installed, can't read {filename}]")
            return None, [], []
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = _mark_heuristic_section_breaks("\n".join(pages))

            images = []
            for page in reader.pages:
                try:
                    for img in page.images:
                        img_ext = img.name.rsplit(".", 1)[-1] if "." in img.name else "png"
                        # Third element is caption context (see docx branch
                        # below for why) -- pypdf doesn't give us adjacent
                        # paragraph text, so this stays empty here; PDF
                        # images fall back to OCR-only findability exactly
                        # as before, unchanged.
                        images.append((img.data, img_ext, "", None))
                except Exception as e:
                    print(f"[Image extraction skipped for one PDF page: {e}]")

            tables = []
            if pdfplumber is not None:
                try:
                    # Per-page line cursor for _find_pdf_table_caption, so a
                    # second table further down the same page doesn't get
                    # matched against the first table's row (or re-consume
                    # the first table's caption).
                    page_caption_cursor = {}
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        for pno, page in enumerate(pdf.pages):
                            # Default extract_tables() uses "lines" strategy
                            # (needs ruling/border lines). An earlier version
                            # of this code assumed that fails on borderless,
                            # whitespace-aligned tables (Word/Markdown ->
                            # PDF exports) and added an UNGUARDED "text"
                            # strategy fallback for pages where "lines" found
                            # nothing. DISPROVEN against a real doc at the
                            # time: "lines" alone found all 3 real tables
                            # there, and the unguarded "text" fallback word-
                            # shredded ordinary prose into a fake table on
                            # pages with no real table at all (see
                            # _looks_like_real_table's docstring for the
                            # exact repro) -- so it was removed outright.
                            #
                            # FIX: re-added as a GUARDED fallback -- it now
                            # only runs on a page where "lines" found ZERO
                            # tables (never competes with or interferes with
                            # an already-found ruled table), and every
                            # candidate it produces must pass
                            # _looks_like_real_table before being accepted.
                            # This is what actually closes the borderless-
                            # table gap without reopening the word-shredding
                            # regression that got this fallback removed the
                            # first time.
                            raw_tables = page.extract_tables()
                            found_on_page = False
                            for raw_table in raw_tables:
                                grid = [
                                    [(cell or "").strip() for cell in row]
                                    for row in raw_table
                                ]
                                if grid and len(grid) > 1 and len(grid[0]) > 1:
                                    found_on_page = True
                                    page_text = pages[pno] if pno < len(pages) else ""
                                    start_line = page_caption_cursor.get(pno, 0)
                                    caption, next_line = _find_pdf_table_caption(
                                        page_text, grid[0], start_line
                                    )
                                    page_caption_cursor[pno] = next_line
                                    tables.append((grid, caption, caption))

                            if not found_on_page:
                                try:
                                    text_tables = page.extract_tables(
                                        table_settings={
                                            "vertical_strategy": "text",
                                            "horizontal_strategy": "text",
                                        }
                                    )
                                except Exception as e:
                                    text_tables = []
                                    print(f"[Borderless-table extraction failed on page {pno} of {filename}: {e}]")
                                for raw_table in text_tables:
                                    grid = [
                                        [(cell or "").strip() for cell in row]
                                        for row in raw_table
                                    ]
                                    if not (grid and len(grid) > 1 and len(grid[0]) > 1):
                                        continue
                                    if not _looks_like_real_table(grid):
                                        print(
                                            f"[Borderless-table candidate rejected on page {pno} of {filename} "
                                            f"-- looked like word-shredded prose, not a real table]"
                                        )
                                        continue
                                    page_text = pages[pno] if pno < len(pages) else ""
                                    start_line = page_caption_cursor.get(pno, 0)
                                    caption, next_line = _find_pdf_table_caption(
                                        page_text, grid[0], start_line
                                    )
                                    page_caption_cursor[pno] = next_line
                                    tables.append((grid, caption, caption)) 
                                    print(f"[Borderless table accepted on page {pno} of {filename}]")
                except Exception as e:
                    print(f"[Table extraction failed for {filename}: {e}]")
            else:
                print(f"[Table extraction skipped: pdfplumber not installed -- {filename}'s text still indexed normally]")

            return text, tables, images
        except Exception as e:
            print(f"[Document upload failed: couldn't parse {filename} as PDF ({e})]")
            return None, [], []

    if ext == "docx":
        if DocxDocument is None:
            print(f"[Document upload skipped: python-docx not installed, can't read {filename}]")
            return None, [], []
        try:
            from docx.oxml.ns import qn

            doc = DocxDocument(io.BytesIO(file_bytes))
            parts = []
            recent_captions = []
            images_since_caption_refresh = 0
            tables = []
            images = []

            found_rids = set()
            # Rolling window of the last couple of non-empty paragraphs seen
            # so far -- used as "caption context" for any image found in the
            # NEXT paragraph. Real-world pattern confirmed in this exact
            # document: a short label paragraph ("ROC Curve: ") immediately
            # followed by a descriptive paragraph, immediately followed by
            # the image itself. Keeping the last 2 non-empty paragraphs
            # captures both without dragging in the whole document.
            recent_captions = []
            current_section = None

            for kind, item in _iter_docx_block_items(doc):
                if kind == "table":
                    grid = _table_to_grid(item)
                    if grid:
                        # Same recent_captions rolling window already
                        # maintained for images below (last couple of
                        # non-empty paragraphs seen so far) -- a table
                        # sitting right after its own label paragraph is the
                        # same real-world pattern as an image after a
                        # caption, so this reuses that existing, already-
                        # verified mechanism rather than a new one.
                        caption = " ".join(recent_captions).strip() or None
                        tables.append((grid, caption, current_section))
                        n_caption_paragraphs = len(recent_captions)
                        if n_caption_paragraphs:
                            del parts[len(parts) - n_caption_paragraphs:]
                        # FIX: recent_captions was never cleared after being
                        # consumed here. Left stateful across tables, a
                        # SECOND table with fewer than 2 new real paragraphs
                        # since the last one would still hold some of that
                        # PREVIOUS table's already-deleted paragraph(s) --
                        # n_caption_paragraphs on this table then overcounts
                        # and deletes that many items from parts' tail again,
                        # except those specific paragraphs are already gone,
                        # so it silently deletes whatever unrelated content
                        # (our own marker, or real prior-section text) is
                        # sitting there instead. Confirmed as the likely
                        # cause of the real regression: fewer total chunks
                        # than expected, and the correct chunk scoring worse
                        # than before this fix, not just failing to improve.
                        recent_captions.clear()
                        parts.append(_SECTION_BREAK_HEURISTIC)
                    continue
                    
                   

                p = item

                # Images anchored to THIS paragraph (w:drawing/a:blip runs),
                # found via the SAME true-document-order walk already used
                # for text/table order above. Unlike the old doc.part.rels
                # sweep (still kept below as a fallback), this lets each
                # image carry the caption text that appeared just before it
                # -- which is what actually makes the image findable by a
                # query like "ROC curve" when OCR finds nothing in the
                # picture itself (the common case: OCR unavailable/failing
                # on this machine per the debug log). Runs before the
                # `if not p.text.strip()` skip below on purpose -- an image
                # frequently sits alone in a paragraph with no text of its
                # own.
                for run in p.runs:
                    for blip in run._element.findall(".//" + qn("a:blip")):
                        rId = blip.get(qn("r:embed"))
                        if not rId or rId in found_rids:
                            continue
                        try:
                            rel = doc.part.rels[rId]
                            blob = rel.target_part.blob
                            content_type = rel.target_part.content_type or "image/png"
                            # FIX: docx often stores a vector rendition (SVG,
                            # or EMF/WMF) alongside the raster PNG fallback for
                            # the same drawing, sharing the same "image"
                            # reltype. Without this check, the non-raster blob
                            # gets saved to disk and indexed exactly like a
                            # normal image; if it later wins a query, PIL's
                            # st.image() call at render time raises
                            # UnidentifiedImageError since it can't decode raw
                            # SVG/EMF/WMF bytes. Skip those here instead, at
                            # extraction time, so nothing non-raster ever
                            # reaches doc_assets in the first place.
                            if not content_type.startswith("image/") or content_type in (
                                "image/svg+xml", "image/x-emf", "image/x-wmf"
                            ):
                                continue
                            img_ext = content_type.split("/")[-1]
                            base_caption = _resolve_image_caption(recent_captions)                        
                            # FIX: images clustered with no text paragraph between
                            # them (e.g. a histogram immediately followed by a bar
                            # chart) all read the SAME recent_captions window, since
                            # only a new non-empty paragraph advances it -- an
                            # image-only paragraph doesn't. Confirmed real repro:
                            # two score-band charts got byte-identical captions,
                            # making them indistinguishable to retrieval. Tagging
                            # every image after the first one in the same window
                            # keeps embeddings from colliding; single, non-clustered
                            # images are unaffected (still get the plain caption).
                            if images_since_caption_refresh:
                                caption = f"{base_caption} (image {images_since_caption_refresh + 1} in this group)".strip()
                            else:
                                caption = base_caption
                            images.append((blob, img_ext, caption, current_section))
                            found_rids.add(rId)
                            images_since_caption_refresh += 1
                        except Exception as e:
                            print(f"[Image extraction skipped for one embedded image in {filename}: {e}]")

                if not p.text.strip():
                    continue
                # Mark real heading/title paragraphs (docx carries this as
                # actual style info, not just bold text) so _chunk_text can
                # force a chunk boundary there instead of blending this
                # section's content with whatever heading comes next purely
                # because both fit under chunk_size together.
                style_name = getattr(getattr(p, "style", None), "name", "") or ""
                is_real_heading = style_name.startswith("Heading") or style_name == "Title"
                # FIX: the comment above assumes docx always carries real
                # heading style info -- confirmed false for a real uploaded
                # doc (DevOps_Lifecycle.docx has zero <w:pStyle> tags at all;
                # every "heading" is just manually bolded text). Without a
                # fallback, this branch marks NO section breaks for such a
                # doc, so _chunk_text's boundary logic never fires and a
                # heading gets severed from its own content by blind
                # size-based packing. Confirmed real repro: "2. Continuous
                # Integration (CI)" ended up in a different chunk than its
                # own "Tools: Jenkins, GitHub, Maven, SonarQube" line,
                # causing that chunk to lose retrieval to an unrelated
                # phase's chunk for a query asking about CI tools. Reuses
                # the SAME heuristic already trusted for pdf/txt/md
                # (_HEURISTIC_HEADING_RE) as a fallback -- marked with a
                # SEPARATE sentinel (not real headings) so _chunk_text can
                # tell the two apart and drop a heuristic heading-only
                # section (a bare bulleted list item with no real content,
                # e.g. a TOC-style list) instead of emitting it as a hollow
                # fragment that could out-compete the real content chunk.
                # Docs that DO have real Heading styles are unaffected --
                # is_real_heading always takes priority and behaves exactly
                # as before.
                stripped = p.text.strip()
                is_heuristic_heading = (
                    not is_real_heading
                    and bool(_HEURISTIC_HEADING_RE.match(stripped))
                    and len(stripped) < 80
                )
                if is_real_heading:
                    parts.append(_SECTION_BREAK + p.text)
                    current_section = stripped 
                elif is_heuristic_heading:
                    parts.append(_SECTION_BREAK_HEURISTIC + p.text)
                    current_section = stripped 
                else:
                    parts.append(p.text)

                recent_captions.append(stripped)
                if len(recent_captions) > 2:
                    recent_captions.pop(0)
                images_since_caption_refresh = 0
            text = "\n".join(parts)

            # Fallback sweep: catch any embedded image the paragraph walk
            # above didn't anchor (header/footer images, or an anchoring
            # pattern outside w:drawing/a:blip in a run). Skips anything
            # already captured via found_rids so nothing is ever double-
            # counted -- these just get no caption context, same
            # OCR-only findability as before this fix.
            for rel in doc.part.rels.values():
                if rel.rId in found_rids:
                    continue
                if "image" in rel.reltype:
                    try:
                        blob = rel.target_part.blob
                        content_type = rel.target_part.content_type or "image/png"
                        # FIX: same non-raster guard as the per-paragraph walk
                        # above (SVG/EMF/WMF renditions aren't PIL-openable and
                        # must never reach doc_assets). Applied here too since
                        # this fallback sweep is a separate code path that
                        # appends to the same `images` list.
                        if not content_type.startswith("image/") or content_type in (
                            "image/svg+xml", "image/x-emf", "image/x-wmf"
                        ):
                            found_rids.add(rel.rId)
                            continue
                        img_ext = content_type.split("/")[-1]
                        images.append((blob, img_ext, "", None))
                        found_rids.add(rel.rId)
                    except Exception as e:
                        print(f"[Image extraction skipped for one embedded image in {filename}: {e}]")

            return text, tables, images
        except Exception as e:
            print(f"[Document upload failed: couldn't parse {filename} as DOCX ({e})]")
            return None, [], []

    print(f"[Document upload skipped: unsupported file type '.{ext}' for {filename}]")
    return None, [], []


def _extract_text_from_upload(filename: str, file_bytes: bytes) -> Optional[str]:
    """Backward-compatible text-only wrapper around _extract_document_content,
    kept in case anything else in the file still calls it directly."""
    text, _tables, _images = _extract_document_content(filename, file_bytes)
    return text


SECTION_PACK_SIZE = 350  # chars. Deliberately smaller than DOC_CHUNK_SIZE --
                          # packing every paragraph under a heading together up
                          # to the full 900-char budget still let one section's
                          # 6-item capabilities bullet list swallow a 2-line
                          # scope declaration sitting right after it in the same
                          # section. Confirmed empirically (proxy similarity):
                          # isolating that declaration from the bullet list
                          # raised query-relevance from 0.105 to 0.141. A
                          # tighter sub-budget keeps genuinely distinct
                          # paragraph groups apart even when they share a
                          # heading, while still packing short related
                          # paragraphs together for context.


def _chunk_text(text: str, chunk_size: int = DOC_CHUNK_SIZE, overlap: int = DOC_CHUNK_OVERLAP) -> list:
    """Section-aware, then paragraph-aware, chunking. _extract_text_from_upload
    marks every real heading/title paragraph in a docx with _SECTION_BREAK --
    a chunk boundary is forced there first, so two different sections (e.g.
    "1.2 Scope" and "1.3 Definitions") never get blended into one chunk purely
    because both fit under chunk_size together.

    Within a section, paragraphs are then packed together only up to
    SECTION_PACK_SIZE (not the full chunk_size) -- confirmed necessary because
    a section can itself contain multiple distinct ideas (a capabilities list,
    then a separate scope declaration) that dilute each other's embedding when
    forced into one chunk just because they share a heading. Each packed piece
    carries that section's heading along with it, so a chunk of just the scope
    declaration still reads as "1.2 Scope\\n...declaration", not floating
    context-free text.

    This directly fixes a confirmed false negative: the real "Out of scope"
    sentence in a test doc's 1.2 originally scored only 0.299 against a clean
    "what's out of scope" query, because the chunk holding it also carried a
    6-item capabilities list AND three unrelated section headers (1.3, 1.4, 2,
    2.1) that followed it in the same 900-char window.

    pdf/txt/md uploads carry no *real* structural metadata the way docx
    style info does, so _extract_text_from_upload marks headings in those
    via a numbered/title-case heuristic (md also gets a real "#" check
    first). Xlsx sheets are marked one section per sheet. If none of a
    document's lines match, the whole text still falls back to being one
    "section" and this still applies the paragraph-packing step within it.

    A single paragraph that alone exceeds chunk_size still falls back to a
    sliding window within that paragraph, breaking on the nearest whitespace
    rather than mid-word where possible.
    """
    text = text.strip()
    if not text:
        return []

    def _sliding_window(t: str) -> list:
        pieces = []
        start = 0
        n = len(t)
        while start < n:
            end = min(start + chunk_size, n)
            if end < n:
                last_space = t.rfind(" ", start, end)
                if last_space > start:
                    end = last_space
            piece = t[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= n:
                break
            start = max(end - overlap, start + 1)  # +1 guards against a zero-progress
                                                     # loop when overlap >= a very short piece
        return pieces

    # FIX: was `text.split(_SECTION_BREAK)`, which only ever split on the
    # real-heading sentinel -- the new heuristic-heading sentinel (docx
    # fallback, see _extract_text_from_upload) needs to split the text
    # too, while still recording WHICH kind of heading started each
    # section, so a heuristic heading with no body can be dropped below
    # without changing anything about how a real heading is handled.
    _section_pattern = f"({re.escape(_SECTION_BREAK)}|{re.escape(_SECTION_BREAK_HEURISTIC)})"
    _raw_parts = re.split(_section_pattern, text)
    sections = []
    if _raw_parts and _raw_parts[0].strip():
        sections.append((False, _raw_parts[0]))  # preamble before any heading
    _i = 1
    while _i < len(_raw_parts):
        _marker = _raw_parts[_i]
        _body = _raw_parts[_i + 1] if _i + 1 < len(_raw_parts) else ""
        sections.append((_marker == _SECTION_BREAK_HEURISTIC, _body))
        _i += 2

    chunks = []
    pending_heading = None  # FIX: an orphan heuristic heading (no body of
                             # its own) is carried forward into the NEXT
                             # section's heading instead of being dropped
                             # outright. Confirmed real case this fixes: a
                             # table's title line ("Verdict Bands") getting
                             # heuristically split from its own table body
                             # because the table's flattened header row
                             # ("Band Score Meaning") ALSO matches the
                             # heading-shaped regex and starts a new
                             # section right after it. Dropping the title
                             # silently (old behavior) meant the real
                             # content chunk lost its only reference to
                             # "Verdict" -- carrying it forward merges them
                             # back into one coherent, correctly-labeled
                             # chunk instead.
    for is_heuristic_section, section in sections:
        section = section.strip()
        if not section:
            continue

        paragraphs = [p.strip() for p in section.split("\n") if p.strip()]
        if not paragraphs:
            continue

        # The first paragraph of a section that came from a real heading
        # break IS that heading -- carry it along with every packed piece
        # from this section, rather than only keeping it once. A section
        # with no heading (txt/pdf, or the pre-first-heading preamble of a
        # docx) just packs its own paragraphs the same way, with no
        # artificial heading text to prepend.
        heading = paragraphs[0]
        body = paragraphs[1:]

        if pending_heading is not None:
            heading = pending_heading + "\n" + heading
            pending_heading = None

        if not body:
            # A heuristic-only heading with no body at all is either a bare
            # list item (e.g. "Continuous Integration" from a TOC-style
            # enumeration) OR a title mis-split from a body that starts on
            # the very next heuristically-detected line (the table-title
            # case above) -- can't tell which from here, so don't emit it
            # as its own hollow chunk (confirmed via direct BM25 test: such
            # a fragment can out-score the chunk that actually has the real
            # content) and don't discard it either -- carry it forward so
            # whichever the next section turns out to be, it gets the
            # heading text prepended instead of losing it. A REAL
            # style-based heading with no body is unchanged -- still kept
            # standalone, exactly as before this fix.
            if is_heuristic_section:
                pending_heading = heading
                continue
            chunks.append(heading)
            continue

        current = [heading]
        current_len = len(heading)
        for para in body:
            if len(para) > chunk_size:
                if len(current) > 1:
                    chunks.append("\n".join(current))
                    current, current_len = [heading], len(heading)
                chunks.extend(_sliding_window(para))
                continue
            if current_len + len(para) + 1 > SECTION_PACK_SIZE and len(current) > 1:
                chunks.append("\n".join(current))
                current, current_len = [heading], len(heading)
            current.append(para)
            current_len += len(para) + 1
        if len(current) > 1:
            chunks.append("\n".join(current))

    return chunks



def _validate_extraction(filename: str, all_chunks: list, all_metadatas: list,
                          tables: list, images: list) -> list:
    """Automatic post-extraction sanity checks -- runs on every ingest, no
    manual per-document testing needed across hundreds of uploads. Never
    raises: a bad check should show up as a logged warning (visible in
    debug_log.txt the same way every other diagnostic in this file already
    is), not block ingestion of the chunks that DID extract fine. Returns
    the list of warning strings (empty list = clean)."""
    warnings = []

    if not all_chunks:
        warnings.append("no text/table/image chunks produced at all")

    seen_asset_ids = set()
    for idx, (chunk, meta) in enumerate(zip(all_chunks, all_metadatas)):
        if not chunk or not chunk.strip():
            warnings.append(f"empty chunk body (chunk_index={meta.get('chunk_index')})")
        for required_key in ("source", "chunk_index", "file_hash", "asset_type", "asset_id"):
            if not meta.get(required_key) and meta.get(required_key) != 0:
                warnings.append(f"chunk {idx} (chunk_index={meta.get('chunk_index')}) missing metadata field '{required_key}'")
        asset_id = meta.get("asset_id")
        if asset_id:
            if asset_id in seen_asset_ids:
                warnings.append(f"duplicate asset_id: {asset_id}")
            seen_asset_ids.add(asset_id)

        asset_type = meta.get("asset_type")
        if asset_type == "table":
            if not meta.get("asset_grid"):
                warnings.append(f"table chunk {meta.get('chunk_index')} has no stored grid (asset reference invalid)")
        elif asset_type == "image":
            path = meta.get("asset_path")
            if not path or not os.path.exists(path):
                warnings.append(f"image chunk {meta.get('chunk_index')} asset file missing on disk: {path}")

    n_table_chunks = sum(1 for m in all_metadatas if m.get("asset_type") == "table")
    n_image_chunks = sum(1 for m in all_metadatas if m.get("asset_type") == "image")
    if tables and n_table_chunks == 0:
        warnings.append(f"{len(tables)} table(s) were extracted from the source but produced zero indexable chunks")
    if images and n_image_chunks == 0:
        warnings.append(f"{len(images)} image(s) were extracted from the source but produced zero indexable chunks")
    if not any(m.get("asset_type") == "text" for m in all_metadatas) and not tables and not images:
        warnings.append("no text content extracted from document at all")

    if warnings:
        print(f"[VALIDATION WARNING: {filename} -- {len(warnings)} issue(s) found]")
        for w in warnings:
            print(f"  - {w}")
    else:
        print(f"[VALIDATION OK: {filename} -- {len(all_chunks)} chunk(s), "
              f"{n_table_chunks} table(s), {n_image_chunks} image(s)]")
    return warnings


def ingest_document(filename: str, file_bytes: bytes) -> int:
    """Extract, chunk, and embed one uploaded file into doc_collection.
    Skips re-ingesting a file whose exact bytes were already stored (checked
    via a content hash in chunk metadata, not just the filename -- so
    re-uploading the same file under a different name doesn't duplicate it,
    and uploading a genuinely edited file under the same name does re-embed
    it). Returns the number of chunks stored (0 if skipped or unparseable).

    Three KINDS of chunk can now be produced, all landing in the SAME
    doc_collection, all tagged with the same "source"/"file_hash" as before
    -- this is deliberate: search_document's existing per-source hybrid
    scoring, z-score population check, etc. all keep working completely
    unchanged, because as far as that scoring logic is concerned a table or
    image chunk is just another chunk from this source competing on its own
    searchable text. Each chunk's metadata["asset_type"] is what lets the
    caller later tell "this winning chunk is actually a table/image, go
    render it in the UI too" apart from a plain text chunk:
      - "text":  metadata unchanged from before.
      - "table": chunk TEXT is the table formatted as markdown (searchable,
                 and Streamlit renders markdown tables as real grids on its
                 own if the LLM echoes it back). metadata["asset_grid"] is
                 the SAME table as a JSON-encoded list-of-lists, for exact
                 re-display via st.table/st.dataframe regardless of what the
                 LLM chooses to say.
      - "image": chunk TEXT is a small header plus whatever OCR found (may
                 be empty -- an image with no OCR text is still indexed and
                 still displayable, just not findable by its own content
                 unless a nearby text chunk mentions it).
                 metadata["asset_path"] points at the saved image file on
                 disk for exact re-display via st.image.
    """
    file_hash = hashlib.md5(file_bytes).hexdigest()

    existing = doc_collection.get(where={"file_hash": file_hash}, limit=1)
    if existing.get("ids"):
        print(f"[Document already indexed, skipping: {filename}]")
        return 0

    text, tables, images = _extract_document_content(filename, file_bytes)
    if not text and not tables and not images:
        return 0

    all_chunks = []
    all_metadatas = []

    # doc_id: short stable prefix for human-readable asset IDs
    # (doc_id_table_001, doc_id_image_001, ...) -- derived from file_hash
    # (already the thing that makes re-ingestion idempotent, see the
    # existing-file check above) rather than filename, so two different
    # uploads that happen to share a filename never collide, and the SAME
    # file re-uploaded always gets the SAME asset IDs.
    doc_id = file_hash[:12]

    text_chunks = _chunk_text(text) if text else []
    for i, chunk in enumerate(text_chunks):
        all_chunks.append(chunk)
        all_metadatas.append({
            "source": filename, "chunk_index": i, "file_hash": file_hash,
            "asset_type": "text", "asset_id": f"{doc_id}_text_{i:03d}",
        })

    for t_idx, (grid, table_caption, section) in enumerate(tables):
        md = _grid_to_markdown(grid)
        if not md:
            continue
        # Caption goes in BEFORE the markdown grid, same ordering as the
        # image chunks below (context, then content) -- this is what makes
        # the table findable by a query built from its own title (e.g.
        # "verdict bands", "model accuracy comparison") instead of only by
        # words that literally appear inside the grid cells. Confirmed real
        # failure mode without this: the table's title line got split into
        # its own near-empty chunk (or, for docx, wasn't attached to the
        # table at all), and a plain-text sentence that merely MENTIONED the
        # topic out-scored the actual table data for that query.
        context_bits = [f"[Table from {filename}]"]
        if table_caption:
            context_bits.append(table_caption)
        context_bits.append(md)
        all_chunks.append("\n".join(context_bits))
        all_metadatas.append({
            "source": filename, "chunk_index": f"table-{t_idx}", "file_hash": file_hash,
            "asset_section": section or "",
            "asset_type": "table", "asset_grid": json.dumps(grid),
            "asset_caption": table_caption or "",
            "asset_id": f"{doc_id}_table_{t_idx:03d}",
        })
    if tables:
        print(f"[{len(tables)} table(s) extracted from {filename}]")

    for i_idx, (img_bytes, img_ext, img_caption, section) in enumerate(images):
        img_path = _save_image_asset(img_bytes, file_hash, i_idx, img_ext)
        ocr_text = _ocr_image_bytes(img_bytes)
        # Caption context (the paragraph(s) immediately preceding this image
        # in the source document -- e.g. "ROC Curve: The ROC curve plots the
        # True Positive Rate against...") comes first, OCR text second. This
        # is what actually makes the image findable by a query like "ROC
        # curve" even when OCR finds nothing inside the picture itself (the
        # common case: OCR unavailable/failing). Falls back to the same
        # generic header-only text as before when no caption was available
        # (images caught only by the doc.part.rels fallback sweep, or PDF
        # images, which don't carry caption context).
        context_bits = [f"[Image from {filename}]"]
        if img_caption:
            context_bits.append(img_caption)
        if ocr_text:
            context_bits.append(ocr_text)
        chunk_text = "\n".join(context_bits)
        all_chunks.append(chunk_text)
        all_metadatas.append({
            "source": filename, "chunk_index": f"image-{i_idx}", "file_hash": file_hash,
            "asset_section": section or "",
            "asset_type": "image", "asset_path": img_path,
            "asset_id": f"{doc_id}_image_{i_idx:03d}",
        })
    if images:
        ocr_status = "OCR applied" if pytesseract is not None else "OCR unavailable -- image indexed without extracted text"
        print(f"[{len(images)} image(s) extracted from {filename} ({ocr_status})]")

    if not all_chunks:
        print(f"[Document produced no usable text, tables, or images: {filename}]")
        return 0

    # Requirement: validate every upload automatically instead of relying on
    # manual per-document testing (hundreds of documents may be uploaded).
    # A warning here doesn't block ingestion -- it just gets logged so a
    # broken table/image extraction is visible in debug_log.txt instead of
    # silently missing at query time.
    _validate_extraction(filename, all_chunks, all_metadatas, tables, images)

    ids = [str(uuid.uuid4()) for _ in all_chunks]
    doc_collection.add(documents=all_chunks, ids=ids, metadatas=all_metadatas)
    print(f"[Document indexed: {filename} ({len(all_chunks)} chunks total)]")
    return len(all_chunks)


MAX_RECENT = 6  # kept for recent_user_messages tracking below — the OLD
                 # repeated-pattern extract_preference() that used to read
                 # this list is gone; replaced by extract_domain_engagement_
                 # preference / extract_holistic_preference further down,
                 # which use st.session_state.concept_history and the full
                 # chat_history instead. See that section for why: the old
                 # mechanism required the user to literally retype a style
                 # request 2+ times, which this rebuild explicitly avoids.


def extract_text(content):
    """Normalize .content across providers into a clean string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)
# ============================================================================
# NEW INTENT ROUTING PIPELINE
# conversation_resolver.resolve_context -> intent_engine.decide_intent
# (gate stack: dimension override -> pronoun gate -> WH-question gate ->
# scope tiebreak -> semantic classifier fallback). Replaces the old
# rule-engine + LogisticRegression pipeline entirely. See
# classify_intent()'s docstring below for the exact mapping and the
# assumptions made where the old and new pipelines didn't line up 1:1.
# ============================================================================

MAX_CLARIFY_ROUNDS = 2  # after this many follow-ups with no clear signal, stop
                         # asking and just decide — carried over unchanged from
                         # the old pipeline; decide_intent has no equivalent
                         # forced-decision safety net on its own, so it's
                         # re-implemented in classify_intent() below.


def check_verb_fusion_ambiguity(resolved_text: str) -> Optional[List[str]]:
    """
    Extracted (not just copied) from the old resolve_query_context's
    WordNet polysemy check on the head verb — kept running because it
    catches a DIFFERENT kind of ambiguity than evidence_evaluator does.
    evidence_evaluator asks "did context resolve at all"; this asks "even
    once resolved, is the resulting sentence itself genuinely ambiguous"
    (e.g. "update" could mean a software update or a news update). No
    equivalent exists anywhere in the new pipeline, so this is preserved
    as its own standalone check — ASSUMPTION: you didn't say to drop it,
    so it's kept; tell me if you'd rather remove it.
    """
    doc = load_nlp()(resolved_text)

    # AUX excluded on purpose: auxiliaries like "do"/"be"/"have" ("does",
    # "is", "has"...) are near-meaninglessly polysemous in WordNet (e.g.
    # "do" has 13 verb senses) regardless of the sentence they're in, so
    # including them let a query like "how does it form" get flagged as
    # ambiguous on "do" (13 senses) instead of the actually-relevant
    # content verb "form" -- producing a clarify question about generic
    # "do" readings ("engage in" vs "be sufficient") that had nothing to
    # do with what was asked. Only real content verbs (main VERB tokens)
    # should be checked here.
    #
    # FIX: the AUX exclusion above solved this for auxiliaries but not for
    # ordinary "light" verbs used as main verbs -- confirmed via real
    # production logs that even direct, unambiguous WH-questions like
    # "Which university did Sahithi get her B.Tech from" were getting
    # forced through 2 forced clarify rounds every time, because the
    # threshold below (>=5 senses) is cleared by nearly EVERY common
    # English verb regardless of context: WordNet lists 'make'=49 senses,
    # 'get'=36, 'take'=42, 'run'=41, 'give'=44, 'go'=30, 'call'=28,
    # 'work'=27, 'set'=25, 'see'=24, 'come'=21, 'find'=16, 'show'=12,
    # 'know'=11, 'say'=11, 'read'=11, 'build'=10, 'apply'=10, 'write'=10,
    # 'put'=9, 'tell'=8, 'score'=7, 'form'=7, 'ask'=7, 'use'=6, 'create'=6,
    # 'learn'=6, 'study'=6 -- dictionary polysemy is the NORM for common
    # verbs, not a signal that THIS sentence is unclear; sentence structure
    # (subject/object/prepositional phrase) already disambiguates "get her
    # B.Tech from Anurag University" with zero real ambiguity to a human
    # reader. Same underlying flaw as the AUX case, same fix: exclude verbs
    # whose high sense-count is a property of the word itself, not of this
    # sentence. A real content verb with genuine contextual ambiguity (e.g.
    # "update" -- software update vs news update, both plausible in many
    # sentences) still gets caught normally; this stoplist only removes
    # verbs common enough that flagging them adds noise, not signal.
    _LIGHT_VERB_STOPLIST = {
        "get", "make", "take", "run", "give", "go", "call", "work", "set",
        "see", "come", "find", "show", "know", "say", "read", "build",
        "apply", "write", "put", "tell", "score", "form", "ask", "use",
        "create", "learn", "study",
    }
    head_verbs = [
        tok.lemma_.lower() for tok in doc
        if tok.pos_ == "VERB" and tok.lemma_.lower() not in _LIGHT_VERB_STOPLIST
    ]
    max_senses = 0
    ambiguous_verb = None
    for v in head_verbs:
        synsets = wordnet.synsets(v, pos=wordnet.VERB)
        if len(synsets) > max_senses:
            max_senses = len(synsets)
            ambiguous_verb = v

    if max_senses >= 5 and ambiguous_verb:
        synsets = wordnet.synsets(ambiguous_verb, pos=wordnet.VERB)
        def_a = synsets[0].definition()
        def_b = synsets[len(synsets) // 2].definition()
        readings = [def_a, def_b]
        print(f"[Context: ambiguous fusion (verb='{ambiguous_verb}', senses={max_senses}) -> {readings}]")
        return readings

    return None


def classify_intent(user_text: str, model, round_index: int = 0, latest_reply: str = None) -> dict:
    """
    SAME RETURN CONTRACT as the old classify_intent: {"type": "topic" /
    "question" / "clarify", ...}. Both UI call sites and _apply_intent
    need zero changes — only this function's body was replaced.

    Internals fully replaced: conversation_resolver.resolve_context +
    intent_engine.decide_intent's gate stack, instead of the old rule
    engine + LogisticRegression classifier.

    ASSUMPTIONS made here, flagged for review rather than silently baked
    in:
    - The `topic` field (used by _apply_intent to trigger the timeline-
      graph agent flow) is set to meaning.resolved_text — matching the
      old code's OWN behavior in its confident-ML path
      (`resolved_query.strip()`), so this preserves existing behavior
      rather than introducing something new.
    - Forced-decision safety net: if state.clarification_count reaches
      MAX_CLARIFY_ROUNDS and the gate stack still wants to clarify, this
      forces through the raw semantic classifier's vote instead of
      asking indefinitely — same safety property MAX_CLARIFY_ROUNDS had
      in the old code, since decide_intent alone has no such cutoff.
    - The WordNet fusion-ambiguity check (see check_verb_fusion_ambiguity
      above) is kept, but now runs AFTER decide_intent's gate stack, only
      as a fallback when no deterministic gate fired. It used to run
      first (matching the old pipeline's position) and could intercept
      direct WH-questions before the gate stack ever ran -- e.g. "fight"
      alone has enough WordNet senses to trip the ambiguity threshold,
      so "How does the immune system fight viruses?" was being sent to
      clarify regardless of what wh_question_gate would have said.
    - add_learned_intent_example (the old classifier's online-learning
      loop, retraining LogisticRegression on every resolved
      clarification) has NO equivalent here — the new classifier is
      static. This is a real capability gap, not silently dropped: flag
      if/when you want a replacement built.
    """
    # BUG FIX: this used to be `latest_reply if latest_reply else user_text`,
    # which discarded the caller's `combined` text (original question +
    # "(Reply: ...)", already built correctly by the UI call sites) and
    # routed on the bare reply alone (e.g. "of devops") on every clarify
    # round, with no trace of what question the reply was even answering.
    # Likely inherited from the old classifier's design, where context was
    # carried purely via a persistent entity string rather than resent
    # text -- that assumption no longer holds reliably under the new
    # resolve_context/meaning_resolver pipeline. `user_text` is already the
    # full combined text by the time it reaches here on a clarify round.
    query_for_routing = user_text

    state = st.session_state.dialogue_state

    meaning = resolve_context(query_for_routing, state)

    # Gate stack runs FIRST now. Previously check_verb_fusion_ambiguity was
    # checked before decide_intent was even called, so a direct WH-question
    # like "How does the immune system fight viruses?" got intercepted here
    # -- "fight" alone has enough WordNet verb senses to trip max_senses >= 5
    # -- and returned clarify immediately, before wh_question_gate ever ran.
    # That meant no fix inside decide_intent's gate stack could help, because
    # the gate stack was never reached. Fusion-ambiguity is a softer signal
    # than a fired structural gate (dimension/pronoun/WH/topic-keyword), so
    # it should only be consulted as a fallback when no gate settled things,
    # same principle as the needs_clarification fix inside decide_intent.
    result = decide_intent(meaning)
    # hard_gate_fired excludes the WH gate on purpose -- see
    # intent_engine.py's matching comment. Recognizing a sentence as a
    # WH-question only settles question_mode vs topic_mode; it should
    # not, by itself, block the fusion-ambiguity clarify check below.
    hard_gate_fired = result["reasoning"].get("hard_gate_fired", False)

    if not hard_gate_fired:
        ambiguous_readings = check_verb_fusion_ambiguity(meaning.resolved_text)
        if ambiguous_readings and round_index < MAX_CLARIFY_ROUNDS:
            # BUG FIX: this branch used to return without ever updating the
            # "sticky note" (dialogue state), unlike every other clarify-
            # triggering branch below. That left it stale for the NEXT
            # round to read from -- update it here too, same as the others.
            state = update_dialogue_state(state, result, meaning)
            st.session_state.dialogue_state = state
            followup = generate_clarify_question(meaning.resolved_text, meaning.entity, round_index)
            print("[Intent: fusion ambiguous -> asking follow-up]")
            return {"type": "clarify", "followup": followup, "resolved_query": meaning.resolved_text}

    forced = round_index >= MAX_CLARIFY_ROUNDS and result["needs_clarification"]

    if result["needs_clarification"] and not forced:
        state = update_dialogue_state(state, result, meaning)
        st.session_state.dialogue_state = state
        followup = generate_clarify_question(meaning.resolved_text, meaning.entity, round_index)
        print(f"[Intent: gate stack inconclusive (reason={result['reasoning']['reason']}) -> asking follow-up]")
        return {"type": "clarify", "followup": followup, "resolved_query": meaning.resolved_text}

    if forced:
        # BUG FIX: this used to read result["reasoning"]["semantic_intent"]
        # -- the raw semantic classifier's vote, completely bypassing the
        # gate stack that decide_intent already ran (dimension/pronoun/WH/
        # topic-keyword gates + scope tiebreak). decide_intent computes ALL
        # of that into reasoning["resolved_intent"] BEFORE it gets stomped
        # back to "unclear" by the evidence-based clarification check (see
        # intent_engine.py's "THE ACTUAL BUG" comment block) -- reasoning
        # still carries the gate-resolved answer even when result["intent"]
        # itself is "unclear". Confirmed real repro: reply "who" to a
        # Cassandra clarify question -- wh_question_gate correctly resolves
        # this to "question_mode" in reasoning["resolved_intent"], but the
        # old code here ignored that and fell back to the raw semantic vote
        # (topic_mode, on a near-tied 0.767 vs 0.764 margin), sending a
        # plain "who created X" question into the full topic-mode
        # research+timeline-graph flow. Falls back to semantic_intent only
        # if resolved_intent is somehow unset, so behavior is unchanged for
        # any case where no gate fired at all.
        final_intent = result["reasoning"].get("resolved_intent") or result["reasoning"]["semantic_intent"]
        print(f"[Intent: {MAX_CLARIFY_ROUNDS} clarify rounds used, forcing '{final_intent}']")
        # BUG FIX: without this, update_dialogue_state below still sees
        # needs_clarification=True (unchanged) and marks the sticky note
        # "still waiting for a reply" even though we just gave up asking
        # and answered anyway -- leaving it stuck for whatever unrelated
        # question comes next. A forced decision is a finished decision.
        result = {**result, "needs_clarification": False}
    else:
        final_intent = result["intent"]

    state = update_dialogue_state(state, result, meaning)
    st.session_state.dialogue_state = state

    # Domain/concept-engagement tracking for the new preference-extraction
    # mechanism — see extract_domain_engagement_preference below.
    tracked_string = meaning.entity or meaning.focus or meaning.resolved_text
    st.session_state.concept_history.append(tracked_string)
    print(
        f"[concept_history <- entity={meaning.entity!r} focus={meaning.focus!r} "
        f"resolved_text={meaning.resolved_text!r} | TRACKED: {tracked_string!r}]"
    )

    if final_intent == "topic_mode":
        print("[Intent: gate stack -> topic_mode]")
        return {"type": "topic", "topic": meaning.resolved_text, "resolved_query": meaning.resolved_text}

    print("[Intent: gate stack -> question_mode]")
    return {"type": "question", "resolved_query": meaning.resolved_text}


# ============================================================================
# NEW RAG PREFERENCE EXTRACTION
# Retrieval/injection (retrieve_preferences / retrieve_user_preferences,
# above) is UNCHANGED — still retrieved and injected into the prompt
# before generation, on the same on-disk Chroma collection. Only the
# EXTRACTION side (deciding what's worth storing) is new: no explicit
# typed preference required, no reactive/corrective feedback required —
# topic/domain engagement (which subjects the user keeps returning to)
# plus a periodic holistic read over the whole transcript.
# ============================================================================

DOMAIN_ENGAGEMENT_THRESHOLD = 3  # how many times a domain must recur in-session
                                  # before it's stored as a preference.
                                  # ASSUMPTION / starting guess, not measured —
                                  # tune once you have real session volume, same
                                  # as scope_analyzer.py's other threshold
                                  # constants are flagged as starting guesses.

HOLISTIC_READ_EVERY_N_TURNS = 8  # cadence for full-transcript holistic reads.
                                  # ASSUMPTION / starting guess — same caveat.


# ---- OLD: string-overlap bucketing (kept for reference / rollback) --------
# Replaced because it never generalized across true synonyms/rephrasings
# that share no words at all (e.g. "similarity search" vs "vector
# databases" for the same underlying question) -- Jaccard word-overlap and
# substring checks are fundamentally lexical, not semantic, so two
# same-topic phrases with zero shared words could never bucket together
# no matter how BUCKET_OVERLAP_THRESHOLD was tuned. See embedding-based
# replacement below, validated against real concept_history-style entity
# extractions (real spaCy noun-phrase extractions run through
# meaning_resolver.py, not full sentences) for gradient
# descent / vector databases / backpropagation test sets on the SAME
# model this app already loads (all-mpnet-base-v2, via
# _shared_embedding_model from semantic_retriever.py) -- threshold=0.65
# chosen because it's the largest gap found between the highest
# legitimate cross-concept score (0.626, gradient descent vs
# backpropagation -- related but must stay separate per explicit user
# decision) and the lowest well-formed same-concept score (0.706).
#
# BUCKET_STOPWORDS = {
#     "how", "does", "do", "is", "are", "what", "why", "when", "where",
#     "which", "who", "the", "a", "an", "works", "work", "working",
#     "about", "on", "in", "of", "to", "for", "and", "its", "it", "me",
#     "explain", "tell", "understand", "learn", "know",
# }
#
# BUCKET_OVERLAP_THRESHOLD = 0.5  # Jaccard on significant words, used by
#                                  # _bucket_match below. High enough that
#                                  # phrasing variants of the SAME concept
#                                  # ("vector search" / "how vector search
#                                  # works" / "ANN vector search") land in
#                                  # one bucket, but not so loose that
#                                  # adjacent-but-distinct concepts ("vector
#                                  # search" vs "vector databases") get fused
#                                  # just because they share one word
#                                  # ("vector") -- per explicit user decision
#                                  # that those two stay separate domains.
#                                  # STARTING GUESS, not measured -- same
#                                  # caveat as DOMAIN_ENGAGEMENT_THRESHOLD and
#                                  # the other threshold constants in this
#                                  # file; tune against real logged
#                                  # concept_history once you have volume.
#
#
# def _clean_phrase(text: str) -> str:
#     """Lowercase, strip punctuation, drop stopwords/question-words, and
#     singularize trailing 's' (so 'databases' and 'database' count as the
#     same word) -- while preserving word order, so substring checks in
#     _bucket_match still mean something."""
#     words = re.findall(r"[a-z0-9]+", text.lower())
#     cleaned = []
#     for w in words:
#         if w in BUCKET_STOPWORDS:
#             continue
#         if len(w) > 3 and w.endswith("s"):
#             w = w[:-1]
#         cleaned.append(w)
#     return " ".join(cleaned)
#
#
# def _bucket_match(new_clean: str, bucket_clean: str) -> bool:
#     """True if new_clean belongs in the same domain bucket as bucket_clean
#     -- either one's cleaned phrase contains the other (catches 'vector
#     search' inside 'advanced vector search techniques'), or their
#     significant-word overlap (Jaccard) clears BUCKET_OVERLAP_THRESHOLD."""
#     if not new_clean or not bucket_clean:
#         return new_clean == bucket_clean
#     if new_clean in bucket_clean or bucket_clean in new_clean:
#         return True
#     new_words = set(new_clean.split())
#     bucket_words = set(bucket_clean.split())
#     intersection = len(new_words & bucket_words)
#     union = len(new_words | bucket_words)
#     return union > 0 and (intersection / union) >= BUCKET_OVERLAP_THRESHOLD
#
#
# def bucket_concept_history(history: list) -> dict:
#     """Group raw concept_history strings (varying phrasings appended once
#     per resolved turn) into domain buckets, so 'vector search' / 'how
#     vector search works' / 'ANN vector search' count as ONE recurring
#     domain, while 'vector search' and 'vector databases' stay separate --
#     per explicit user decision: distinct domains even though topically
#     adjacent. Replaces the old exact-string Counter, which never let
#     phrasing variants accumulate toward DOMAIN_ENGAGEMENT_THRESHOLD.
#     Returns {representative_raw_string: occurrence_count}, using the
#     first-seen raw string in each bucket as the representative so
#     save_preference gets a real phrase, not a normalized token soup.
#     Deterministic: processed in history order every call, so a bucket's
#     representative never changes once set, keeping stored_domains'
#     dedup key stable across repeated calls."""
#     buckets = []  # [{"clean": str, "rep": str, "count": int}]
#     for raw in history:
#         clean = _clean_phrase(raw)
#         matched = None
#         for b in buckets:
#             if _bucket_match(clean, b["clean"]):
#                 matched = b
#                 break
#         if matched is None:
#             buckets.append({"clean": clean, "rep": raw, "count": 1})
#         else:
#             matched["count"] += 1
#     return {b["rep"]: b["count"] for b in buckets}


# ---- NEW: embedding-similarity bucketing -----------------------------------
# Uses the SAME SentenceTransformer already loaded once in
# semantic_retriever.py (_shared_embedding_model, imported at the top of
# this file) -- no new model, no new dependency.
#
# Clustering rule: "max similarity to any existing member" (single-linkage),
# NOT centroid/running-average. Centroid-averaging was tested first and
# rejected: a single off-topic turn (e.g. "compare vector databases to
# traditional relational databases") pulls the shared average vector
# permanently toward itself, degrading match quality for every later turn
# in that bucket even after the outlier is long past. Max-similarity-to-
# any-member can't drift that way -- each member vector stays exactly what
# it was extracted as, so one odd turn just sits there as one point, it
# never contaminates the others.
EMBEDDING_BUCKET_THRESHOLD = 0.65  # cosine similarity. Measured, not a
                                    # starting guess -- see comment block
                                    # above old bucketing code for the
                                    # real numbers this was picked from
                                    # (0.626 highest real cross-concept
                                    # score, 0.706 lowest real well-formed
                                    # same-concept score; 0.65 sits in that
                                    # gap). Re-validate if you swap
                                    # embedding models (all-mpnet-base-v2
                                    # assumed here) or if real production
                                    # concept_history data starts landing
                                    # outside this range.


def bucket_concept_history(history: list) -> dict:
    """Group raw concept_history strings (varying phrasings appended once
    per resolved turn) into domain buckets using embedding similarity
    instead of word overlap -- so 'gradient descent' / 'GD' / 'similarity
    search' phrasing-variants of the SAME concept can bucket together even
    when they share zero literal words, while topically-adjacent-but-
    distinct concepts ('gradient descent' vs 'backpropagation') stay
    separate, per explicit user decision. A new phrase joins the first
    existing bucket where it scores >= EMBEDDING_BUCKET_THRESHOLD against
    ANY member already in that bucket (see module-level comment above for
    why max-similarity, not centroid). Returns {representative_raw_string:
    occurrence_count}, using the first-seen raw string in each bucket as
    the representative so save_preference gets a real phrase. Deterministic:
    processed in history order every call, so a bucket's representative
    never changes once set, keeping stored_domains' dedup key stable
    across repeated calls."""
    if not history:
        return {}

    embeddings = _shared_embedding_model.encode(history, convert_to_numpy=True)

    buckets = []  # [{"rep": str, "count": int, "member_vecs": [np.ndarray, ...]}]
    for raw, vec in zip(history, embeddings):
        matched = None
        for b in buckets:
            sims = cosine_similarity([vec], b["member_vecs"])[0]
            if sims.max() >= EMBEDDING_BUCKET_THRESHOLD:
                matched = b
                break
        if matched is None:
            buckets.append({"rep": raw, "count": 1, "member_vecs": [vec]})
        else:
            matched["count"] += 1
            matched["member_vecs"].append(vec)

    return {b["rep"]: b["count"] for b in buckets}


DOMAIN_ALREADY_STORED_THRESHOLD = 0.7  # same bar as save_preference's own
                                        # dedup_threshold -- reused deliberately,
                                        # since this is asking the same question
                                        # save_preference asks at save time ("is
                                        # this a near-duplicate of something
                                        # already in Chroma?"), just asked EARLIER
                                        # and against a different generated
                                        # sentence than PREFERENCE_RELEVANCE_
                                        # THRESHOLD, which is about whether a
                                        # stored preference is relevant to THIS
                                        # turn's query -- a completely different
                                        # question, kept as a separate constant
                                        # on purpose.


def _is_domain_preference_already_stored(domain: str) -> Tuple[bool, Optional[float]]:
    """Check Chroma directly for an existing domain preference before
    st.session_state.stored_domains (session-local, reset to empty every
    session) is trusted as the source of truth. Without this,
    extract_domain_engagement_preference() has no memory across sessions
    that a domain was already saved last time -- it silently re-runs the
    whole 1/3 -> 2/3 -> 3/3 in-session recount every session, and the only
    thing that ever catches the resulting re-save attempt is
    save_preference()'s own dedup check at count==3, which is too late to
    explain to yourself (or see in the log) WHY nothing changed. Checking
    here, once, up front, means the rejection happens at the point where
    the decision is actually being made, not three turns later as a side
    effect of a different function's dedup logic."""
    if pref_collection.count() == 0:
        return False, None
    # Compare the bare domain name only -- NOT wrapped in a templated sentence.
    # Wrapping it (e.g. "User frequently asks about {domain}") made every
    # stored domain share the same boilerplate text, which inflated similarity
    # between UNRELATED domains (e.g. "supervised learning" vs "gradient
    # descent" scored 0.725 purely from the shared wrapper, well above
    # DOMAIN_ALREADY_STORED_THRESHOLD, even though bucket_concept_history --
    # the validated same-topic check -- never grouped them together).
    candidate_text = domain
    try:
        results = pref_collection.query(
            query_texts=[candidate_text], n_results=1, where={"type": "domain"}
        )
    except Exception as e:
        print(f"[_is_domain_preference_already_stored query failed: {e}]")
        return False, None
    docs = results["documents"][0]
    dists = results["distances"][0]
    if not docs:
        return False, None
    closest_similarity = 1 - dists[0]
    return closest_similarity >= DOMAIN_ALREADY_STORED_THRESHOLD, closest_similarity


def extract_domain_engagement_preference() -> None:
    """
    Buckets st.session_state.concept_history (appended once per resolved
    turn inside classify_intent) into domains via bucket_concept_history()
    -- phrasing variants of the same concept collapse into one bucket,
    while topically-adjacent-but-distinct concepts stay separate buckets.
    Once a bucket crosses DOMAIN_ENGAGEMENT_THRESHOLD occurrences and
    hasn't already been stored this session, saves a preference for it.
    No explicit statement or feedback needed — purely which subjects
    recur.

    Before counting a domain that isn't in st.session_state.stored_domains
    yet, checks Chroma directly (once, via
    _is_domain_preference_already_stored) in case it was already saved in
    a PRIOR session -- stored_domains itself has no memory of that, since
    it's reset empty every session. A hit is cached straight into
    stored_domains so this only ever queries Chroma once per domain per
    session, not on every turn.
    """
    counts = bucket_concept_history(st.session_state.concept_history)
    print(f"[Domain buckets this session ({len(counts)} total):]")
    skipped_existing = []  # domains confirmed already in Chroma this call --
                            # collected here so the wrapper below can print
                            # one explicit "skipped saving" confirmation line
                            # per domain, instead of that fact only being
                            # inferable from the per-domain status text above.
    for domain, count in counts.items():
        if domain in st.session_state.stored_domains:
            status = "already stored earlier — skipped"
        else:
            already_stored, similarity = _is_domain_preference_already_stored(domain)
            if already_stored:
                st.session_state.stored_domains.add(domain)
                skipped_existing.append(domain)
                # Dedicated line, separate from the bucket-status line below,
                # so build_status_lines() can pick it out and surface it in
                # the chat UI the same way it already does for the "[User
                # domain preference found: ...]" line from retrieve_preferences()
                # -- this is the domain-bucket equivalent of that "found"
                # moment, just triggered from the pre-save check instead of
                # retrieval.
                print(f"[Domain preference already stored: {domain} (similarity={similarity:.3f})]")
                status = "domain already detected, no counting"
            elif count >= DOMAIN_ENGAGEMENT_THRESHOLD:
                status = f"crossed threshold ({DOMAIN_ENGAGEMENT_THRESHOLD}) — SAVING NOW"
            else:
                status = f"below threshold ({count}/{DOMAIN_ENGAGEMENT_THRESHOLD}) — not saved yet"
        print(f"    {domain!r}: count={count} -> {status}")

        if count >= DOMAIN_ENGAGEMENT_THRESHOLD and domain not in st.session_state.stored_domains:
            # Store the bare domain name -- NOT wrapped in a templated
            # sentence. The wrapper is reconstructed only where it's actually
            # needed for display (build_status_lines) or for the LLM prompt
            # (retrieve_preferences), never persisted as the document itself.
            #
            # Definition generated ONCE, here, at save time -- not at query
            # time, which would mean an LLM call on every retrieval instead
            # of once per newly-discovered domain. Used only by
            # retrieve_preferences()'s cross-encoder rerank stage. A failed
            # generation (returns None) still saves the domain -- retrieval
            # just falls back to the bare name for cross-encoder scoring on
            # that one, same as it does for pre-existing rows with no
            # definition at all.
            save_preference(domain, pref_type="domain")
            st.session_state.stored_domains.add(domain)

    for domain in skipped_existing:
        print(f"[Same domain exists — skipped saving: {domain!r}]")


def extract_holistic_preference(chat_history: list, model) -> None:
    """
    Every HOLISTIC_READ_EVERY_N_TURNS user turns: reads the WHOLE
    transcript once and asks the LLM to infer a general pattern about
    how this user likes to be helped — not scanning for any single
    repeated phrase, reasoning over the conversation's gestalt instead.
    Catches things too diffuse for turn-by-turn counting to notice.
    Writes through save_preference, same Chroma collection and same
    dedup-on-similarity protection as everything else here.

    BUG FIXED HERE: this used to ask the model to "infer any general
    pattern" with NO visibility into what had already been saved on a
    previous holistic read. Every run re-derives its impression of the
    user from scratch, so the same underlying preference kept getting
    re-phrased slightly differently each time ("User prefers
    tradeoffs." one run, "User prefers inclusion of trade-off analysis
    in technical answers." the next) -- and save_preference's dedup is
    a single-nearest-neighbor embedding check at a fixed threshold,
    which two differently-worded paraphrases of the same idea often
    don't clear (documented elsewhere in this file: a genuinely
    on-topic query scored as low as 0.299 against its own source
    chunk). The model has to be TOLD what's already known so it can
    skip it, not left to be caught by embedding similarity after the
    fact -- that's a backstop, not the primary defense.

    This also matters beyond noise: get_style_preferences() returns
    every stored style line, uncapped, into every RULE 2 message --
    each undetected duplicate permanently grows every future prompt,
    which was measurably contributing to the low-TPM Groq models'
    413 "Request too large" failures (confirmed: request size climbed
    ~8000 -> ~13000 tokens over 8 turns in one session, tracking the
    duplicate accumulation turn for turn).
    """
    if len(chat_history) < 4:
        print("[Not enough conversation yet for a holistic read]")
        return

    transcript = "\n".join(
        f"{'User' if e['role'] == 'user' else 'Agent'}: {e['text']}"
        for e in chat_history
    )

    known_style = get_style_preferences()
    known_domain = []
    if pref_collection.count() > 0:
        try:
            known_domain = pref_collection.get(where={"type": "domain"}).get("documents", []) or []
        except Exception as e:
            print(f"[extract_holistic_preference: fetching known domains failed: {e}]")

    known_block = ""
    if known_style or known_domain:
        known_lines = "\n".join(f"- {p}" for p in (known_style + known_domain))
        known_block = (
            "\nAlready-known preferences for this user (do NOT repeat these, and do NOT "
            "restate the same underlying idea in different words -- only report something "
            "genuinely NEW that isn't already covered by this list):\n"
            f"{known_lines}\n"
        )

    prompt = (
        "Below is a conversation between a user and an AI assistant. Read "
        "the whole thing and infer any GENERAL pattern about how this user "
        "likes to be helped — their typical level of detail, tone, or "
        "recurring interests — even if no single phrase repeats literally. "
        "Do NOT invent a preference that isn't actually supported by the "
        "conversation.\n"
        f"{known_block}\n"
        "STRICT RULES:\n"
        "1. Each line must describe exactly ONE atomic preference.\n"
        "2. Each line must be a single short sentence, third person, "
        "starting with 'User prefers' or 'User is interested in'.\n"
        "3. Output ONE such sentence per line, nothing else — no "
        "numbering, no bullets, no extra commentary.\n"
        "4. If nothing genuinely NEW stands out (beyond what's already known "
        "above), respond with exactly: NONE\n\n"
        f"Conversation:\n{transcript}\n\n"
        "Your answer:"
    )
    result = model.invoke(prompt)
    text = result.content.strip() if hasattr(result, "content") else str(result).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text or text.strip().upper() == "NONE":
        print("[Holistic read: no new pattern found]")
        return
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        # The prompt above generates two different KINDS of line, per its
        # own rule 2: "User prefers..." (HOW to answer -- manner, tone,
        # detail level; should apply regardless of topic) vs "User is
        # interested in..." (WHAT the user cares about -- a domain/interest
        # fact; should only apply when the current query is actually about
        # that thing). Tagging both as "style" would apply interest-type
        # lines to every unrelated query too -- the same universal-
        # injection problem this whole change was meant to fix, just moved
        # to a different source. Split on the line's own generated prefix
        # instead of assuming.
        pref_type = "domain" if line.lower().startswith("user is interested in") else "style"
        save_preference(line, pref_type=pref_type)


# ---- Follow-up question wording (LLM — wording ONLY, zero say in the
# topic/question decision, which already happened in classify_intent
# above) — kept verbatim from the old pipeline; this is UI-text
# generation, independent of which classifier decided clarification
# was needed. ----


def generate_clarify_question(context: str, entity: Optional[str], round_index: int) -> str:
    """LLM's ONLY role: phrase a natural follow-up question about the CONCEPT
    itself. It has zero say in topic/question routing — that decision already
    happened via the gate stack in classify_intent/decide_intent before
    this is ever called. Falls back to a plain generic line only if the API
    call itself fails (rate limit, network) — not as the normal path."""
    # Previously this was `entity or context`, which meant that once an entity
    # was set, EVERY clarify question collapsed to just the bare entity name —
    # ignoring whatever specific word/phrase the user actually just said. That's
    # why the question felt identical and generic every round. Prefer the fuller
    # resolved context when it exists and actually differs from the bare entity.
    if context and entity and entity.lower() not in context.lower():
        subject = context
    else:
        subject = context or entity
    prompt = (
        f"Someone is asking about \"{subject}\" but hasn't said enough for you to "
        "know what specifically they want to know. Write ONE short, natural "
        "follow-up question that invites them to say more about the SUBJECT "
        "ITSELF — what aspect, detail, or part of it they're curious about.\n\n"
        "RULES:\n"
        "1. NEVER ask them to define the term or explain what it is.\n"
        "2. NEVER phrase it as yes/no, or as a choice between two named options.\n"
        "3. NEVER mention 'topic', 'question', 'timeline', 'mode', or similar "
        "category words — this is not about how the answer will be delivered.\n"
        "4. Ask about the concept/subject matter itself, open-ended, inviting "
        "a free-form answer in their own words.\n"
        "5. Make it feel fresh and specific to this subject — do not reuse "
        "generic phrasing you might have used before for other subjects.\n\n"
        f"Subject: \"{subject}\"\n\n"
        "Your follow-up question:"
    )
    try:
        result = extraction_model.invoke(prompt)
        # BUG FIX: result.content is not always a plain string -- some
        # models return a list of pieces instead. Convert to a string
        # first, whichever shape it comes back as, before reading it.
        raw_content = result.content if hasattr(result, "content") else result
        if isinstance(raw_content, list):
            raw_content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw_content
            )
        text = str(raw_content).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if text:
            print(f"[Clarify question generated: \"{text}\"]")
            return text
    except Exception as e:
        print(f"[generate_clarify_question failed: {e}]")
    return f"Could you tell me a bit more about {subject}?"
def provider_label(agent_name: str) -> str:
    """Map an internal agent/model name to its provider for user-facing display."""
    name = agent_name.lower()
    if "claude" in name or "anthropic" in name:
        return "Claude"
    if "gemini" in name or "google" in name:
        return "Gemini"
    if "groq" in name or "oss" in name or "llama" in name or "qwen" in name:
        return "Groq"
    if "openai" in name or "gpt" in name:
        return "OpenAI"
    return agent_name


def format_error(provider: str, error: Exception) -> str:
    """Turn a raw provider exception into a single clean, human-readable line."""
    msg = str(error)

    if "insufficient_quota" in msg or "exceeded your current quota" in msg:
        return f"{provider}: quota exceeded, check your plan/billing"

    if "rate_limit_exceeded" in msg or "429" in msg:
        if "tokens per day" in msg.lower() or "tpd" in msg.lower():
            wait_match = re.search(r"try again in ([\d.]+m?[\d.]*s)", msg.lower())
            wait_text = f", try again in {wait_match.group(1)}" if wait_match else ""
            return f"{provider}: daily token limit reached{wait_text}"
        if "tokens per minute" in msg.lower() or "tpm" in msg.lower():
            return f"{provider}: rate limit error, too many tokens per minute, try again after some time"
        return f"{provider}: rate limit error 429, try again after some time"

    if "413" in msg or "Request too large" in msg:
        return f"{provider}: request too large, try again after some time"

    if "404" in msg or "not_found_error" in msg:
        return f"{provider}: model not found"

    if isinstance(error, json.JSONDecodeError) or "Extra data" in msg or "Expecting value" in msg:
        return f"{provider}: returned malformed data, try again"

    first_line = msg.strip().split("\n")[0][:120]
    return f"{provider}: failed — {first_line}"


def build_status_lines(debug_text: str) -> list:
    """Extract RAG/preference status lines from captured stdout for clean UI display."""
    lines = []
    seen = set()
    for raw_line in debug_text.split("\n"):
        line = raw_line.strip()
        rendered = None
        if line.startswith("[User domain preference found:"):
            content = line[len("[User domain preference found:"):].rstrip("]").strip()
            domain_text = content.split("(similarity=")[0].strip()
            rendered = f"🔎 User domain preference found: {domain_text}"
        elif line.startswith("[Domain preference already stored:"):
            content = line[len("[Domain preference already stored:"):].rstrip("]").strip()
            domain_text = content.split("(similarity=")[0].strip()
            rendered = f"🔎 User preference domain detected: {domain_text}"
        elif line == "[No relevant preferences found]":
            rendered = "🔎 No preference detected yet"
        elif re.match(r"^\[Memory updated \((domain|style)\):", line):
            match = re.match(r"^\[Memory updated \((domain|style)\):\s*(.*)\]$", line)
            pref_type, pref_text = match.group(1), match.group(2).strip()
            rendered = f"💾 Preference updated ({pref_type}): {pref_text}"
        elif line.startswith("[Preference already known"):
            rendered = "💾 No new preference to remember"
        elif line.startswith("[Document context found:"):
            content = line[len("[Document context found:"):].rstrip("]").strip()
            doc_text = content.split("(similarity=")[0].strip()
            rendered = f"📄 Document context found: {doc_text}"
        elif line.startswith("[Document indexed:"):
            content = line[len("[Document indexed:"):].rstrip("]").strip()
            rendered = f"📄 Document indexed: {content}"
        elif line.startswith("[No document context above threshold"):
            content = line[len("[No document context above threshold"):].rstrip("]").strip()
            rendered = f"📄 No document match ({content})"
        elif line == "[Not enough turns yet to detect a pattern]":
            rendered = "🧩 Not enough patterns detected yet to store"
        if rendered and rendered not in seen:
            seen.add(rendered)
            lines.append(rendered)
    return lines


class LiveStatusCallback(BaseCallbackHandler):
    """Streams live status text into a Streamlit placeholder from any thread,
    including deepagents' internal ThreadPoolExecutor workers."""

    def __init__(self, status_placeholder, provider):
        self.status_placeholder = status_placeholder
        self.provider = provider
        self.ctx = get_script_run_ctx()

    def _update(self, text):
        current_thread = threading.current_thread()
        if current_thread is not threading.main_thread():
            add_script_run_ctx(current_thread, self.ctx)
        try:
            self.status_placeholder.markdown(text)
        except Exception:
            pass

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._update(f"⏳ {self.provider} working...")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "tool")
        if tool_name == "task":
            subagent = input_str.get("subagent_type") if isinstance(input_str, dict) else None
            self._update(f"⏳ {subagent or 'Subagent'} running...")
        elif tool_name == "web_search":
            self._update("⏳ Web search running...")
        elif tool_name == "create_graph":
            self._update("⏳ Creating graph...")
        else:
            self._update(f"⏳ Calling {tool_name}...")

    def on_tool_end(self, output, **kwargs):
        pass


def web_search(query: str, max_results: Union[int, str] = 1,
               include_raw_content: Union[bool, str] = False):
    """Run a web search"""
    max_results = int(max_results)
    max_results = min(max_results, 3)
    if isinstance(include_raw_content, str):
        include_raw_content = include_raw_content.lower() == "true"
    return tavily_client.search(query,
        max_results=max_results, include_raw_content=include_raw_content, topic="general")


try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())


def search_document(query: str, max_results: Union[int, str] = 2):
    """Plain internal helper (NOT an agent tool -- not passed in any tools=
    list) -- searches the user's uploaded document(s) for content relevant
    to a query. Called only for question_mode turns, via the traditional-RAG
    block below, to decide whether THIS question is actually about the
    document (via the returned top_similarity) before anything else about
    the turn is decided -- that decision has to happen before preference
    retrieval, not after, since a document-scoped turn skips preferences
    entirely.

    Hybrid scoring (dense + sparse), no LLM call:
      - Semantic similarity: existing Chroma cosine search.
      - Lexical similarity: BM25 (rank_bm25, same library already used
        elsewhere in this project's document-management-mcp stack).
      Combined via DOC_BM25_WEIGHT.

    FIX: BM25 is now computed PER SOURCE DOCUMENT, not once over the whole
    collection. doc_collection is a permanent chromadb.PersistentClient --
    by design, documents accumulate across every session forever (that's
    the intended behavior, not a bug). But a single shared BM25 index over
    an ever-growing multi-document corpus means one document's IDF
    statistics keep shifting as unrelated documents get added, and a
    genuinely-relevant chunk in a small/plainly-worded document can get
    outscored by an unrelated chunk elsewhere just because of how terms
    happen to distribute across the whole pile. Confirmed real repro: a
    resume's own "EDUCATION / B.Tech .../ Anurag University 2020-2024"
    chunk -- correctly chunked, exactly the right content -- scored only
    semantic=0.310 against a 194-chunk collection built from many
    unrelated prior test documents, landing the query as "not
    document-scoped" and silently skipping the resume entirely. Building a
    separate BM25 index per source keeps each document's own lexical
    statistics independent of what else has ever been uploaded, so the
    collection can grow indefinitely without diluting any one document's
    own scoring.

    Relative-confidence gate (margin), not just an absolute cutoff:
      A turn only counts as a genuine match if the best DOCUMENT's own best
      chunk BOTH clears DOC_SCOPED_THRESHOLD and beats the next-best
      DOCUMENT's own best chunk by DOC_SCOPED_MARGIN (now compared
      per-source-best-vs-per-source-best, not chunk-vs-chunk -- see FIX
      above). This is the primary safeguard against cross-document
      confusion in a permanent, ever-growing collection: two candidate
      documents landing close together on their own best evidence is
      exactly the "actually ambiguous which doc this is about" case this
      is meant to catch, so DOC_SCOPED_MARGIN was raised (see its
      definition) to require clearer separation now that dozens of
      documents can realistically coexist.

    Returns (top_similarity, formatted_text):
      - top_similarity: the hybrid score of the best chunk if it passed
        BOTH the threshold and margin checks; otherwise a value just under
        DOC_SCOPED_THRESHOLD, so the caller's existing
        `top_similarity >= DOC_SCOPED_THRESHOLD` check still reflects the
        real (margin-checked) decision without changing the caller's
        code. None if nothing is indexed at all.
      - formatted_text: the closest-matching chunks, each labeled with its
        source filename and score breakdown, OR a plain "nothing indexed"
        string when top_similarity is None. Only meaningful/used by the
        caller when top_similarity clears the threshold.
    """
    # Capped at 4, not the previous 6 -- confirmed in testing that a 4-chunk
    # result (each chunk up to DOC_CHUNK_SIZE=900 chars) pushed one Groq
    # free-tier model's total request 204 tokens over its 8000 TPM cap on
    # the very next completion (successful search_document call, then a 413
    # "Request too large" killing that agent entirely, no wasted call but a
    # lost turn). This is a tight margin, not a guarantee -- Groq's free
    # tier is small enough that a long query/system-prompt combo can still
    # blow it even at 3 chunks; that's a quota problem to solve separately
    # (smaller/summarized system prompt, or dropping the free-tier models
    # from the fallback chain), not something this cap alone fixes.
    max_results = int(max_results)
    max_results = min(max(max_results, 1), 4)

    total_chunks = doc_collection.count()
    if total_chunks == 0:
        return None, "No document has been uploaded/indexed this session.", None, []

    # n_results=total_chunks -- pull the FULL corpus's cosine scores, not
    # just a small top-k, so BM25 and the margin check below both have the
    # complete picture to compare against (a handful of chunks per doc, so
    # this is cheap).
    try:
        cosine_results = doc_collection.query(query_texts=[query], n_results=total_chunks)
    except Exception as e:
        return None, f"ERROR: document search failed: {e}", None, []

    documents = cosine_results["documents"][0]
    distances = cosine_results["distances"][0]
    metadatas = cosine_results["metadatas"][0]

    if not documents:
        return None, "No document content found for this query.", None, []

    cosine_scores = [1 - d for d in distances]

    # FIX: group chunk indices by source document BEFORE building any BM25
    # index -- each document gets its own BM25Okapi built only from its own
    # chunks, so its lexical scores can never be diluted or skewed by the
    # vocabulary of whatever else has ever been uploaded to this permanent
    # collection. source_groups preserves the original index into
    # documents/distances/metadatas so cosine scores (already computed
    # against the full corpus, which is fine -- Chroma's ANN cosine search
    # doesn't have the IDF-dilution problem BM25 does) line up correctly.
    source_groups = defaultdict(list)
    for i in range(len(documents)):
        source_groups[metadatas[i].get("source", "unknown")].append(i)

    bm25_norm = [0.0] * len(documents)
    if BM25Okapi is not None:
        for source, idxs in source_groups.items():
            tokenized_corpus = [_tokenize(documents[i]) for i in idxs]
            bm25 = BM25Okapi(tokenized_corpus)
            raw_bm25 = bm25.get_scores(_tokenize(query))
            # Fixed saturation curve, NOT per-query self-normalization.
            # Previously this divided every score by max(raw_bm25) -- the
            # best score found in THIS search -- which guaranteed the
            # top-ranked chunk always read as a "perfect" 1.000 lexical
            # match, even when its only real overlap was one incidental
            # shared phrase. raw / (raw + SCALE) instead only approaches
            # 1.0 as the RAW score itself grows on an absolute,
            # query-independent scale.
            for local_i, global_i in enumerate(idxs):
                score = raw_bm25[local_i]
                bm25_norm[global_i] = score / (score + BM25_SATURATION_SCALE) if score > 0 else 0.0
    else:
        print("[search_document: rank_bm25 not installed -- falling back to cosine-only scoring]")

    hybrid_scores = [
        (1 - DOC_BM25_WEIGHT) * cos + DOC_BM25_WEIGHT * bm
        for cos, bm in zip(cosine_scores, bm25_norm)
    ]

    # FIX: rank SOURCE DOCUMENTS by each one's own best chunk, not raw
    # chunks pooled together -- this is what makes the margin check below
    # a genuine per-document comparison instead of a chunk-vs-chunk one
    # that a broad query matching two spots in the same correct document
    # could accidentally fail.
    source_best = {}
    for source, idxs in source_groups.items():
        best_i = max(idxs, key=lambda i: hybrid_scores[i])
        source_best[source] = best_i
    ranked_sources = sorted(source_best.items(), key=lambda kv: hybrid_scores[kv[1]], reverse=True)

    top_source, top_idx = ranked_sources[0]
    top_score = hybrid_scores[top_idx]

    # Population-relative confidence: compare the top source's best score
    # against the DISTRIBUTION of the other sources' own best scores for
    # THIS query, not just against whatever ranked #2. See
    # DOC_SCOPED_Z_THRESHOLD/DOC_SCOPED_MIN_POPULATION above for why. Falls
    # back to the old flat-margin check when there aren't enough other
    # documents to form a meaningful population.
    population_scores = [hybrid_scores[i] for _, i in ranked_sources[1:]]
    second_score = population_scores[0] if population_scores else 0.0
    margin = top_score - second_score

    if len(population_scores) >= DOC_SCOPED_MIN_POPULATION:
        pop_mean = statistics.mean(population_scores)
        pop_stdev = statistics.stdev(population_scores)
        if pop_stdev > 0:
            z_score = (top_score - pop_mean) / pop_stdev
            # FIX: DOC_SCOPED_HIGH_CONFIDENCE was only ever wired into the
            # flat-margin fallback below, not here -- an inconsistency with
            # no principled reason behind it. A top_score strong enough to
            # bypass population comparison entirely shouldn't stop being
            # strong enough to do that just because enough OTHER documents
            # happened to get indexed to cross DOC_SCOPED_MIN_POPULATION and
            # switch which branch runs. Confirmed real case this fixes: the
            # "what are verdict bands" query used to hit this exact
            # inconsistency the moment a 3rd document got indexed -- same
            # 0.533 top_score, but the z-score branch (now active at n=2)
            # had no escape hatch at all, unlike the branch it replaced.
            margin_confident = z_score >= DOC_SCOPED_Z_THRESHOLD or top_score >= DOC_SCOPED_HIGH_CONFIDENCE
            confidence_desc = f"z_score={z_score:.3f} (pop_mean={pop_mean:.3f}, pop_stdev={pop_stdev:.3f}, n={len(population_scores)})"
        else:
            # Every other source tied exactly -- z-score is undefined.
            # A real gap above that tied cluster is still meaningful even
            # without a spread to measure it against. Same escape-hatch
            # consistency fix as the z-score branch above.
            margin_confident = top_score > pop_mean or top_score >= DOC_SCOPED_HIGH_CONFIDENCE
            confidence_desc = f"pop_stdev=0, top_vs_mean={top_score - pop_mean:.3f} (pop_mean={pop_mean:.3f}, n={len(population_scores)})"
    else:
        # Too few other documents to trust a mean/stdev -- fall back to the
        # flat margin this constant was originally calibrated for.
        #
        # FIX: the flat margin alone reintroduces the exact failure mode the
        # z-score path above was built to fix (see DOC_SCOPED_MARGIN's own
        # comment), just in the n=1 case the z-score path doesn't cover.
        # Confirmed real repro: "what is the model accuracy comparison"
        # scored top_hybrid=0.638 against steel_defect_project_
        # documentation.pdf -- ABOVE the top of this file's own calibrated
        # genuine-match range (0.496-0.588, see DOC_SCOPED_THRESHOLD) -- but
        # was rejected because the only OTHER indexed document (a leftover
        # from an earlier test session, not this query's real source)
        # happened to also score 0.529, giving margin=0.109 < 0.15 with
        # n=1. A single stale/incidental other document shouldn't get to
        # veto a match that's already stronger than every genuine match
        # this threshold was calibrated against. DOC_SCOPED_HIGH_CONFIDENCE
        # gives an absolute-score escape: when top_score alone is at or
        # above the genuine-range ceiling, trust it regardless of what one
        # other document happens to score. Below that ceiling, the margin
        # still has to clear normally -- this only rescues scores that were
        # already strong on their own.
        margin_confident = margin >= DOC_SCOPED_MARGIN or top_score >= DOC_SCOPED_HIGH_CONFIDENCE
        confidence_desc = f"margin_over_other_source={margin:.3f} (flat-margin fallback, n={len(population_scores)})"

    is_confident = top_score >= DOC_SCOPED_THRESHOLD and margin_confident

    # FIX: below-threshold margin rescue. `margin` (top source's best score
    # minus the next-best source's own best score) is already computed
    # above, unconditionally -- reused here, not recomputed. Only fires when
    # the normal path above rejected the turn on the THRESHOLD check
    # specifically (top_score < DOC_SCOPED_THRESHOLD), not on the margin
    # check -- a low top_score with a wide margin is exactly the
    # weakly-worded-but-real-match case DOC_SCOPED_RESCUE_FLOOR/
    # DOC_SCOPED_RESCUE_MARGIN target (see their own comments); a query that
    # already cleared 0.46 but failed on margin_confident is the
    # cross-document-ambiguity case and must NOT be rescued here.
    rescued = False
    if not is_confident and DOC_SCOPED_RESCUE_FLOOR <= top_score < DOC_SCOPED_THRESHOLD and margin >= DOC_SCOPED_RESCUE_MARGIN:
        is_confident = True
        rescued = True
     # NEW -- coverage rescue. A different failure than the score-based
    # rescue above: a query's content words can land in DIFFERENT chunks
    # of the SAME source (e.g. a resume's name lives only in its header
    # chunk, while a different chunk holds "certifications"), so no
    # single chunk ever scores well on both -- top_score can fall below
    # both DOC_SCOPED_THRESHOLD and DOC_SCOPED_RESCUE_FLOOR even though
    # the source, as a whole, plainly contains everything asked.
    # Confirmed real case: "what are sahithi certifications" against
    # ss_resume.pdf -- top_score=0.310, chunk 0 has "sahithi" (no
    # "certifications"), chunk 14 has "certifications" (no "sahithi").
    #
    # Content words are POS-filtered (NOUN/PROPN only, via load_nlp() --
    # already used elsewhere in this file, no new dependency) rather than
    # a fixed stopword list or a document-frequency cutoff -- tested
    # against 5 real queries from this project and correctly separated
    # content words from filler in every case, with no score or
    # percentage to calibrate.
    covered = False
    if not is_confident:
        content_words = {
            t.text.lower()
            for t in load_nlp()(query)
            if t.pos_ in ("NOUN", "PROPN")
        }
        if content_words:
            source_tokens = set()
            for i in source_groups[top_source]:
                source_tokens.update(_tokenize(documents[i]))
            if content_words.issubset(source_tokens):
                is_confident = True
                covered = True


    breakdown = (
        f"top_hybrid={top_score:.3f} "
        f"(semantic={cosine_scores[top_idx]:.3f}, lexical={bm25_norm[top_idx]:.3f}, "
        f"source={top_source}), "
        f"{confidence_desc}, margin_over_other_source={margin:.3f} -- "
        f"{'COVERAGE RESCUED (content words split across chunks)' if covered else ('RESCUED (below threshold, wide margin)' if rescued else ('CONFIDENT' if is_confident else 'not confident'))}"
    )
    print(f"[search_document: {breakdown}]")

    # FIX: per-source breakdown logging, rejected turns only. Real repro
    # that motivated this: "What score threshold defines the LIKELY verdict
    # band" against CREDIT RISK SCORECARD.docx scored top_hybrid=0.570,
    # margin_over_other_source=0.008 (flat-margin fallback, n=2) -- an
    # almost-exact tie with whatever ranked #2. The old single-line
    # breakdown only ever showed the WINNING source's own semantic/lexical
    # split; there was no way to tell from the log alone whether that
    # near-tie was genuine cross-document ambiguity (two docs plausibly
    # both about this) or a lexical-overlap artifact (e.g. a generic shared
    # word like "threshold" inflating an unrelated doc's BM25 score the
    # same way the earlier one-shared-phrase "AI Agent" bug did before
    # BM25_SATURATION_SCALE). Guessing which one it is from a single number
    # would repeat the exact mistake this file's own calibration notes
    # warn against -- so log every ranked source's own score breakdown
    # instead, only when the turn was rejected (a confident turn doesn't
    # need this; the ambiguity is exactly what "not confident" means).
    if not is_confident:
        for source, idx in ranked_sources:
            print(
                f"[search_document:   source breakdown -- {source}: "
                f"hybrid={hybrid_scores[idx]:.3f} "
                f"(semantic={cosine_scores[idx]:.3f}, lexical={bm25_norm[idx]:.3f})]"
            )

    # Report the real hybrid score when confident; otherwise report
    # something just under threshold so the caller's existing
    # `top_similarity >= DOC_SCOPED_THRESHOLD` check reflects this
    # function's full decision (threshold AND margin) without needing to
    # change what the caller checks.
    top_similarity = max(top_score, DOC_SCOPED_THRESHOLD) if is_confident else min(top_score, DOC_SCOPED_THRESHOLD - 0.001)
    # FIX: once a winning source is chosen, only return context chunks
    # FROM that source -- previously this could mix in a chunk from an
    # entirely different (losing) document just because it ranked highly
    # in the raw pooled list, which makes no sense once documents are
    # compared and chosen at the source level.
    #
    # FIX (retrieval separation): the pooled top-N sort below used to rank
    # text/table/image chunks from the winning source together on raw
    # hybrid score alone. That means a real, relevant table or image can
    # lose its spot in the returned context just because a handful of
    # ordinary text chunks from the same document happened to score a few
    # hundredths higher -- e.g. "what is the accuracy of Model B" pulling
    # in 4 text chunks that each mention "Model B" in passing, while the
    # one table chunk that actually HAS the accuracy number scores 5th and
    # never gets returned. Retrieve text, table, and image independently
    # within the winning source instead: reserve a slot for the winning
    # source's best table and best image (if either clears ASSET_MIN_SCORE,
    # so an unrelated table/image in the same doc isn't force-included) and
    # fill the rest with the best-scoring text chunks. This does not change
    # anything about which DOCUMENT wins (source-level ranking above is
    # untouched) or how tables/images are embedded/scored -- only which of
    # the winning document's own chunks make it into the returned context.
    ASSET_MIN_SCORE = 0.15
    by_type = defaultdict(list)
    for i in source_groups[top_source]:
        by_type[metadatas[i].get("asset_type", "text")].append(i)
    for asset_type in by_type:
        by_type[asset_type].sort(key=lambda i: hybrid_scores[i], reverse=True)
    # NEW -- section gate. Runs AFTER sorting, BEFORE best_table/best_image are
# picked and BEFORE the ASSET_MIN_SCORE check below: this narrows the
# candidate POOL by position in the document, then score decides among
# whatever survives. Empty asset_section ("" -- PDF images, fallback-sweep
# images with no position info) is never filtered out, since we have no
# section to compare.
    winning_section = metadatas[top_idx].get("asset_section", "")

    def _same_section(i):
      s = metadatas[i].get("asset_section", "")
      return not s or not winning_section or s == winning_section
    excluded_table = [i for i in by_type.get("table", []) if not _same_section(i)]
    excluded_image = [i for i in by_type.get("image", []) if not _same_section(i)]
    by_type["table"] = [i for i in by_type.get("table", []) if _same_section(i)]
    by_type["image"] = [i for i in by_type.get("image", []) if _same_section(i)]
    if excluded_table or excluded_image:
        print(f"[ASSET_CALIBRATION: section gate -- winning_section={winning_section!r}]")
    for i in excluded_table:
        print(f"[ASSET_CALIBRATION:   table excluded (different section) -- "
              f"score={hybrid_scores[i]:.3f}, section={metadatas[i].get('asset_section','')!r}]")
    for i in excluded_image:
        print(f"[ASSET_CALIBRATION:   image excluded (different section) -- "
              f"score={hybrid_scores[i]:.3f}, section={metadatas[i].get('asset_section','')!r}]")
    text_idxs = by_type.get("text", [])
    best_table = by_type.get("table", [None])[0] if by_type.get("table") else None
    best_image = by_type.get("image", [None])[0] if by_type.get("image") else None
    if best_table is not None and hybrid_scores[best_table] < ASSET_MIN_SCORE:
        best_table = None
    if best_image is not None and hybrid_scores[best_image] < ASSET_MIN_SCORE:
        best_image = None

    # FIX: winner-take-all between table and image. Previously best_table and
    # best_image were independent -- each only needed to clear its own
    # section gate + ASSET_MIN_SCORE, so a table-only question (e.g. "which
    # is the home credit default risk dataset") could still drag in an
    # unrelated same-section image that happened to independently clear the
    # score floor (confirmed real repro: table correctly won at score=0.614,
    # but image-7 also cleared ASSET_MIN_SCORE at score=0.392 and got
    # attached anyway). User wants exclusivity: never show both a table and
    # an image on the same turn, irrespective of section or topical
    # relation -- only the higher-scoring of the two survives. This is a
    # deliberate trade-off: a genuinely mixed question ("show the ROC curve
    # and the accuracy table") will now only surface whichever of the two
    # scores higher, not both.
    if best_table is not None and best_image is not None:
        if hybrid_scores[best_table] >= hybrid_scores[best_image]:
            print(f"[ASSET_CALIBRATION: image dropped (table/image exclusivity) -- "
                  f"image_score={hybrid_scores[best_image]:.3f} < table_score={hybrid_scores[best_table]:.3f}]")
            best_image = None
        else:
            print(f"[ASSET_CALIBRATION: table dropped (table/image exclusivity) -- "
                  f"table_score={hybrid_scores[best_table]:.3f} < image_score={hybrid_scores[best_image]:.3f}]")
            best_table = None

    # CALIBRATION LOGGING (temporary): collecting real reserved-vs-relevant
    # numbers to set ASSET_RELATIVE_FLOOR from data, not a guess -- same
    # method used for PREFERENCE_RELEVANCE_THRESHOLD/EMBEDDING_BUCKET_THRESHOLD
    # earlier in this project. For each reserved asset, log its own score
    # against top_score (the winning source's best chunk overall) so the
    # ratio between them can be read off later. Remove once
    # ASSET_RELATIVE_FLOOR is set.
    if best_table is not None:
        print(f"[ASSET_CALIBRATION: table reserved -- score={hybrid_scores[best_table]:.3f}, "
              f"top_score={top_score:.3f}, ratio={hybrid_scores[best_table] / top_score:.3f}, "
              f"query={query!r}]")
    if best_image is not None:
        print(f"[ASSET_CALIBRATION: image reserved -- score={hybrid_scores[best_image]:.3f}, "
              f"top_score={top_score:.3f}, ratio={hybrid_scores[best_image] / top_score:.3f}, "
              f"query={query!r}]")

    reserved = sum(x is not None for x in (best_table, best_image))
    n_text = max(max_results - reserved, 1)
    selected = list(text_idxs[:n_text])
    if best_table is not None and best_table not in selected:
        selected.append(best_table)
    if best_image is not None and best_image not in selected:
        selected.append(best_image)
    # Presentation order only (highest score first in what the LLM reads) --
    # doesn't change which chunks were selected above.
    winning_idxs = sorted(selected, key=lambda i: hybrid_scores[i], reverse=True)
    n = len(winning_idxs)
    chunks = []
    # doc_assets: the exact image/table content behind any winning chunk
    # that IS an image or table (asset_type set by ingest_document), kept
    # completely separate from `chunks` (the LLM's text context) -- this is
    # for the UI to render as-is later, not for the model to read. A plain
    # text chunk contributes nothing here.
    doc_assets = []
    for i in winning_idxs[:n]:
        meta = metadatas[i]
        chunks.append(
            f"[Source: {meta.get('source', 'unknown')} "
            f"(hybrid={hybrid_scores[i]:.3f}, semantic={cosine_scores[i]:.3f}, "
            f"lexical={bm25_norm[i]:.3f})]\n{documents[i]}"
        )
        asset_type = meta.get("asset_type", "text")
        if asset_type == "image" and meta.get("asset_path"):
            doc_assets.append({"type": "image", "path": meta["asset_path"]})
        elif asset_type == "table" and meta.get("asset_grid"):
            try:
                doc_assets.append({"type": "table", "grid": json.loads(meta["asset_grid"])})
            except Exception as e:
                print(f"[Couldn't parse stored table grid for display: {e}]")

    # Diagnostic only -- lists exactly which chunk(s) won and what type each
    # one is, so "why didn't the table show up" is answerable straight from
    # debug_log.txt (chunk_index + asset_type + score) instead of needing a
    # screen recording to infer it after the fact. Doesn't affect ranking,
    # max_results, or which chunks get selected -- purely reports what the
    # selection above already decided.
    winners_desc = ", ".join(
        f"{metadatas[i].get('chunk_index', '?')}"
        f"[{metadatas[i].get('asset_type', 'text')}]"
        f"(hybrid={hybrid_scores[i]:.3f})"
        for i in winning_idxs[:n]
    )
    print(f"[search_document: winning chunks -> {winners_desc}]")

    return top_similarity, "\n\n".join(chunks), breakdown, doc_assets


def create_graph(milestones, topic: str) -> str:
    """Create a timeline graph from a topic's milestones."""
    if isinstance(milestones, str):
        cleaned = milestones.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        start = min((cleaned.find(c) for c in "{[" if c in cleaned), default=-1)
        end = max((cleaned.rfind(c) for c in "}]" if c in cleaned), default=-1)
        if start != -1 and end != -1 and end > start:
           cleaned = cleaned[start:end + 1]

        # Was json.loads(cleaned), which requires the ENTIRE string to be one
        # valid JSON value with nothing after it. That's what was actually
        # throwing the "Extra data: line 1 column N" errors surfacing as
        # "Gemini: returned malformed data" -- Gemini's tool-call arguments
        # sometimes include a complete, valid JSON blob followed by trailing
        # content (extra commentary, a stray second fragment) that the
        # fence-stripping/brace-slicing above doesn't always fully remove.
        # raw_decode parses only the FIRST complete JSON value at the start
        # of the string and explicitly returns wherever it stopped, ignoring
        # anything after -- which is exactly the "extra data" case this was
        # failing on. Traced via debug_log.txt: the crash consistently landed
        # right as the visualiser subagent (create_graph's only caller) ran,
        # not inside deepagents' PatchToolCallsMiddleware, which doesn't call
        # json.loads at all.
        try:
            milestones, _ = json.JSONDecoder().raw_decode(cleaned)
        except json.JSONDecodeError as e:
            # Was `raise ValueError(...)`. This function's own signature
            # promises `-> str` -- every other path returns a string, only
            # the failure paths broke that contract by raising instead.
            # Depending on exactly where in the deepagents call stack this
            # fires (this runs inside the 'visualiser' subagent's own tool
            # call, nested inside the coordinator's task() call), a raised
            # exception can propagate past whatever layer normally converts
            # a tool error into model-visible output -- which is what
            # actually happened: the whole agent run died instead of the
            # model seeing this and retrying, the way RULE 4 in
            # deepagent_system_prompt already assumes it can. Returning a
            # string here doesn't loosen validation at all -- same failure,
            # same condition -- it just guarantees the model sees it.
            return (
                f"ERROR: create_graph could not parse the milestones JSON: {e}\n"
                f"Cleaned input was: {cleaned!r}\n"
                f"Call create_graph again with milestones as a JSON list of "
                f"objects, each with a 4-digit year and a description, e.g. "
                f'[{{"year": 1939, "description": "..."}}, ...], with at '
                f"least 2 valid entries."
            )

    # Diagnostic: always show exactly what create_graph received, before any
    # validation can fail. This is the single piece of information every past
    # "no usable milestones" debugging session was missing -- without it there
    # was no way to tell "researcher genuinely found nothing" apart from
    # "researcher found plenty but the parser didn't recognize the key names."
    print(f"[create_graph received type={type(milestones).__name__}]: {milestones!r}")

    # Case-insensitive, alias-tolerant key lookup. Different models (and even
    # the same model across calls) label these fields inconsistently -- Gemini
    # in particular has been seen using "date"/"event" instead of "year"/
    # "description". Matching only exact-cased "year"/"Year" silently dropped
    # every entry that used any other spelling, which is what caused entries
    # to vanish between the researcher's output and create_graph's input.
    YEAR_KEYS = ("year", "date", "years", "when")
    DESC_KEYS = ("description", "milestone", "event", "title", "label", "summary", "detail", "fact")

    def _get_ci(d: dict, keys):
        """Case-insensitive lookup across a list of candidate key names."""
        lower_map = {str(k).lower(): v for k, v in d.items()}
        for k in keys:
            if k in lower_map and lower_map[k] not in (None, ""):
                return lower_map[k]
        return None

    if isinstance(milestones, list):
        parsed = {}
        dropped = []
        for item in milestones:
            if isinstance(item, dict):
                year = str(_get_ci(item, YEAR_KEYS) or "").strip()
                desc = str(_get_ci(item, DESC_KEYS) or "").strip()
                if year and desc:
                    parsed[year] = desc
                else:
                    dropped.append(item)
            elif isinstance(item, str):
                year, _, desc = item.partition(":")
                year, desc = year.strip(), desc.strip()
                if year and desc:
                    parsed[year] = desc
                else:
                    dropped.append(item)
        if dropped:
            print(f"[create_graph: {len(dropped)} item(s) dropped during parsing — "
                  f"no recognized year/description key found]: {dropped!r}")
        milestones = parsed

    if not isinstance(milestones, dict) or len(milestones) < 2:
        # Was `raise ValueError(...)` -- same reasoning as the JSON-parse
        # failure above. This is the exact path that produced your "Parsed
        # result: 1942" error: Gemini's tool call passed a bare int, not a
        # dict/list at all, so it failed this check and the raise took the
        # whole agent run down with it instead of letting Gemini see the
        # problem and retry with the correct shape.
        return (
            f"ERROR: create_graph received no usable milestones. "
            f"Parsed result: {milestones!r}. "
            f"milestones must be a JSON list of objects, each with a "
            f"4-digit year and a description, e.g. "
            f'[{{"year": 1939, "description": "..."}}, {{"year": 1940, '
            f'"description": "..."}}], with at least 2 valid entries. '
            f"Call create_graph again with milestones in that exact shape."
        )

    items = sorted(milestones.items())
    years = []
    labels = []
    for y, desc in items:
        try:
            years.append(int(y))
            labels.append(desc.strip())
        except ValueError:
            continue

    if len(years) < 2:
        # Same reasoning as the two return-instead-of-raise fixes above.
        return (
            "ERROR: create_graph received fewer than 2 usable numeric year "
            "milestones. Each entry needs a plain 4-digit year (not a date "
            "range, decade, or century) and a non-empty description. Call "
            "create_graph again with at least 2 such entries, or if this "
            "topic genuinely has no traceable milestones in the given "
            "range, say so instead of retrying."
        )

    fig, ax = plt.subplots(figsize=(max(14, len(years) * 2), 8))
    ax.plot(years, [0] * len(years), marker="o", markersize=8, linestyle="-", color="steelblue", zorder=2)

    for x, label in zip(years, labels):
        wrapped = "\n".join(textwrap.wrap(label, width=22))
        ax.annotate(
            wrapped, (x, 0), xytext=(0, -15), textcoords="offset points",
            rotation=0, ha="center", va="top", fontsize=8
        )

    ax.set_yticks([])
    ax.set_ylim(-6, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Year")
    ax.set_title(f"{topic} — Timeline", pad=20)
    ax.margins(x=0.08)

    plt.tight_layout()

    safe_topic = topic.strip()
    safe_topic = re.sub(r'[\\/*?:"<>|\n\r\t]', "", safe_topic)
    safe_topic = safe_topic.replace(" ", "_")
    filename = f"{safe_topic}_timeline.png"

    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Graph saved as {filename}]")
    return f"Graph saved as {filename}"


researcher_subagent = {
    "name": "researcher",
    "description": "Gathers facts on a topic via web search, for whatever 4-year range is given in the task description.",
    "system_prompt": (
        "Search and report facts only, focused on key dates and milestones in chronological order, "
        "strictly within the 4-year range given to you in your task description. Ignore anything outside that range.\n"
        "You MUST attempt to find and report at least one milestone for EVERY year in that 4-year range, inclusive.\n"
        "Run multiple separate web searches if needed — one broad search is often not enough to cover "
        "all 4 years for a large topic. Try different search queries per year if the topic is broad.\n"
        "Never call web_search with include_raw_content=True — always use the default (False). "
        "Titles and snippets are sufficient; full page content is not needed and wastes tokens.\n"
        "If, after genuinely trying, you cannot find a real milestone for a specific year, explicitly state: "
        "'No milestone found for [year]' — do not simply omit the year silently.\n"
        "No writing or formatting beyond the year-by-year facts."
    ),
    "tools": [web_search],
}

visual_subagent = {
    "name": "visualiser",
    "description": "Takes milestones gathered by the researcher and creates a graph from them.",
    "system_prompt": (
        "You will receive milestone text containing years and descriptions for a specific 4-year range. "
        "Your ONLY job is to immediately call the create_graph tool with this data — do not explain, "
        "plan, write a todo list, or reason step by step in plain text. "
        "Do not output any markdown, headers, or numbered steps. "
        "Extract the milestones into the required format and call create_graph directly, in your very "
        "first response, with no intermediate text."
    ),
    "tools": [create_graph],
}

ALL_SUBAGENTS = [researcher_subagent, visual_subagent]

deepagent_system_prompt = (
    "Coordinator. You have exactly two subagents: 'researcher' and 'visualiser'. "
    "Never use 'general-purpose' or any other subagent, and never call glob, ls, read_file, write_file, "
    "or edit_file, under any circumstances.\n\n"
    "You also have ONE direct tool, usable only under RULE 2: 'web_search'.\n\n"
    "The 'write_todos' tool (and any other planning/scratchpad tool available to you) is for your own "
    "internal tracking only — the user never sees your tool calls, only your final answer text. NEVER "
    "reproduce, summarize, or narrate your todo list in that final answer: no 'Status Update:' section, "
    "no checklist, no strikethrough-marked completed items, no restating of your own plan or progress. "
    "If you catch yourself about to write something like '(Completed)' next to a task, stop — that's "
    "internal bookkeeping leaking into the response, not something the user asked for. Answer directly, "
    "in the format the user's preferences call for, with nothing about your own process included.\n\n"
    "Every message you receive starts with an explicit flow tag, either '[FLOW: QUESTION_MODE]' or "
    "'[FLOW: TOPIC_MODE]', put there by Python BEFORE the message reached you. This tag is ground truth "
    "and OVERRIDES any inference you might otherwise make from the rest of the message's wording or "
    "structure. If the tag says QUESTION_MODE, RULE 1 does not apply this turn, full stop — do not run "
    "any part of RULE 1's procedure, even if something later in the message superficially resembles a "
    "topic-research request. Go straight to RULE 2. Only a '[FLOW: TOPIC_MODE]' tag means RULE 1 "
    "applies.\n\n"
    "RULE 1 — TOPIC RESEARCH: only when the message is tagged '[FLOW: TOPIC_MODE]'. When it applies, "
    "the message will also contain the literal text 'Research and create a timeline graph for the "
    "topic:'. Do this exactly:\n"
    "1. First, decide which 4 consecutive years are most likely to contain real milestones for this "
    "topic, based on your own knowledge of when it was most historically active or significant. Do not "
    "default to any fixed range. If you have no reliable knowledge of this topic at all, or you know it "
    "hasn't existed long enough for any 4-year window to contain real milestones, skip straight to step 4 "
    "and explain that instead of researching.\n"
    "2. Call task with subagent_type='researcher', including in the description the topic name and the "
    "exact year range you chose, e.g. 'Gather milestone facts strictly between {start_year} and "
    "{start_year + 3} for topic: {topic}'.\n"
    "3. Paste the researcher's FULL result verbatim into the visualiser's task call description — never "
    "summarize or shorten it. Verify it includes a milestone (or 'no milestone found') for every year "
    "before calling the visualiser, then call task with subagent_type='visualiser'.\n"
    "4. If the visualiser's create_graph call fails because there are fewer than 2 usable milestones "
    "(this happens when the subject is too recent, too new, or you have no real knowledge of it), do NOT "
    "treat this as a generic error. Instead, explicitly tell the user, plainly, that you can't build a "
    "timeline for this subject because it doesn't have enough known history yet (briefly say why), and "
    "then answer their underlying question directly using your own knowledge or web_search instead, in "
    "the same response.\n"
    "5. Regardless of whether step 3 or step 4 applies: if preference guidance is present AFTER the topic "
    "instruction, under '[Known user preferences: ...]' near the end of this message, use it in your "
    "accompanying text (the text alongside the graph, or alongside your step-4 explanation). If it says "
    "OWN TOPIC, briefly acknowledge prior engagement (e.g. \"you've engaged with this before\") and add a "
    "bit more explanatory depth than you would by default, alongside the graph — something like \"here are "
    "some points\" plus the extra depth, not just the graph on its own. Never mention that you checked, "
    "unless asked.\n\n"
    "RULE 2 — ALL OTHER MESSAGES (i.e. '[FLOW: QUESTION_MODE]'): Never call researcher or visualiser "
    "here — relevant user preferences, "
    "if any, are already provided to you at the start of the message under '[Known user preferences: "
    "...]'. Apply each one that genuinely fits this question — let it actually shape the answer, not "
    "just get mentioned. Apply it even if it takes more effort, as long as it's relevant; only leave one "
    "out if applying it would mean forcing something irrelevant in (e.g. tradeoffs for 'what is a while "
    "loop'). Never mention that you checked, unless asked.\n"
    "If this message starts with '[This question has been identified as being about the user's "
    "uploaded document', this turn was pre-identified (in Python, before you ever saw it) as being "
    "about the document — answer ONLY using the document context provided in this message. Do NOT use "
    "your own general knowledge, do NOT call web_search, and do NOT apply any style or domain "
    "preference guidance — none is provided for this turn on purpose, so skip the preference-"
    "application guidance above entirely for this case; there's nothing to apply. If the provided "
    "context genuinely doesn't answer the specific question asked, say so plainly rather than "
    "guessing — do NOT invent what the document might contain. Stop there; do not fall through to "
    "conversation history, your own knowledge, or web_search for this case.\n"
    "If this message does NOT start with that prefix, proceed as normal:\n"
    "Then check conversation history first — answer from it directly if possible, no web_search.\n"
    "If not there, answer from your own knowledge if confident. If not confident, or the topic is "
    "recent/current/unfamiliar, call web_search yourself with a concise query and answer only from "
    "what it returns.\n"
    "Never call glob, ls, read_file, write_file, edit_file, researcher, or visualiser for these messages."
)

# The prompt above already tells the model never to call glob/ls/read_file/
# write_file/edit_file -- but those are deepagents' own built-in tools,
# auto-attached to every agent (and every subagent) regardless of what's
# passed via tools=, and a prompt-only instruction is not reliable enough:
# a weaker/smaller model (confirmed: gemini-3.1-flash-lite, after the
# primary agents failed over) ignored it and called ls/grep repeatedly
# instead of search_document, then answered from whatever it found that
# way -- content never verified against the actual uploaded document.
# permissions=DENY_ALL_FILESYSTEM makes every read/write filesystem tool
# call return a permission-denied error at the middleware level instead,
# which the model can't just choose to ignore. Subagents inherit this
# automatically (neither researcher nor visualiser use filesystem tools).
DENY_ALL_FILESYSTEM = [
    FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
]


@st.cache_resource
def init_agents():

    groq_model_1 = init_chat_model("groq:openai/gpt-oss-120b", timeout=20, max_retries=1)
    groq_model_2 = init_chat_model("groq:openai/gpt-oss-20b", timeout=20, max_retries=1)
    gemini_model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", timeout=20, max_retries=1)
    openai_model = init_chat_model("openai:gpt-4.1-nano", timeout=20, max_retries=1)
    Anthropic_model = ChatAnthropic(model="claude-sonnet5", timeout=20, max_retries=1)
    checkpointer = InMemorySaver()

    groq_agent_1 = create_deep_agent(
        model=groq_model_1,
        system_prompt=deepagent_system_prompt,
        subagents=ALL_SUBAGENTS,
        tools=[web_search],
        permissions=DENY_ALL_FILESYSTEM,
        checkpointer=checkpointer,
    )
    groq_agent_2 = create_deep_agent(
        model=groq_model_2,
        system_prompt=deepagent_system_prompt,
        subagents=ALL_SUBAGENTS,
        tools=[web_search],
        permissions=DENY_ALL_FILESYSTEM,
        checkpointer=checkpointer,
    )
    gemini_agent = create_deep_agent(
        model=gemini_model,
        system_prompt=deepagent_system_prompt,
        subagents=ALL_SUBAGENTS,
        tools=[web_search],
        permissions=DENY_ALL_FILESYSTEM,
        checkpointer=checkpointer,
    )
    openai_agent = create_deep_agent(
        model=openai_model,
        system_prompt=deepagent_system_prompt,
        subagents=ALL_SUBAGENTS,
        tools=[web_search],
        permissions=DENY_ALL_FILESYSTEM,
        checkpointer=checkpointer,
    )
    Anthropic_agent = create_deep_agent(
        model=Anthropic_model,
        system_prompt=deepagent_system_prompt,
        subagents=ALL_SUBAGENTS,
        tools=[web_search],
        permissions=DENY_ALL_FILESYSTEM,
        checkpointer=checkpointer,
    )

    agents_to_try = [
        ("openai/gpt-oss-120b", groq_agent_1),
        ("openai/gpt-oss-20b", groq_agent_2),
        ("gemini-3.1-flash-lite", gemini_agent),
        ("openai/gpt-4.1-nano", openai_agent),
        ("claude-sonnet5", Anthropic_agent),
    ]
    return agents_to_try, groq_model_1, checkpointer


agents_to_try, extraction_model, checkpointer = init_agents()
def run_agent_stream(agent, config, user_message, provider="Agent", status_placeholder=None):
    final_message = None
    run_config = dict(config)
    callbacks = []
    if status_placeholder is not None:
        callbacks.append(LiveStatusCallback(status_placeholder, provider))

    # Explicit tracer, rather than relying on implicit env-var auto-detection.
    # Streamlit runs the script inside its own ScriptRunner worker thread, not
    # the OS's true main thread (the same reason add_script_run_ctx is needed
    # above for the status callback) -- LangChain's implicit env-var-only
    # tracer attachment can silently fail to trigger in that context with no
    # error raised, which matches "clean terminal, zero traces" exactly.
    # Instantiating LangChainTracer directly and passing it in as a callback
    # bypasses whatever detection step was failing.
    try:
        from langchain_core.tracers.langchain import LangChainTracer
        callbacks.append(LangChainTracer(project_name=os.environ.get("LANGSMITH_PROJECT", "DeepAgent")))
    except Exception as e:
        print(f"[LangChainTracer setup failed: {e}]")

    if callbacks:
        run_config["callbacks"] = callbacks

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        config=run_config,
        stream_mode="updates",
    ):
        for node_name, node_output in chunk.items():
            print(f"[{node_name} running...]")
            if node_output and "messages" in node_output:
                msg = node_output["messages"][-1]
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.get("name", "")
                        args = tc.get("args", {})
                        if tool_name == "task":
                            subagent = args.get("subagent_type", "unknown")
                            print(f"    >>> SUBAGENT STARTED: {subagent}")
                        else:
                            print(f"    >>> TOOL CALLED: {tool_name}")
                final_message = msg
    return final_message


st.set_page_config(page_title="DeepAgent", layout="centered")
st.title("DeepAgent")

with st.expander("📄 Upload a document"):
    uploaded_file = st.file_uploader(
        "Drag and drop a file here, or browse",
        type=["txt", "md", "pdf", "docx"],
    )
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        if file_hash in st.session_state.ingested_doc_hashes:
            st.info(f"{uploaded_file.name} is already indexed.")
        else:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                ingest_debug = io.StringIO()
                with contextlib.redirect_stdout(ingest_debug):
                    n_chunks = ingest_document(uploaded_file.name, file_bytes)
                with open("debug_log.txt", "a", encoding="utf-8") as f:
                    f.write(ingest_debug.getvalue())
            st.session_state.ingested_doc_hashes.add(file_hash)
            if n_chunks > 0:
                st.success(f"{uploaded_file.name} indexed — {n_chunks} chunks ready to answer from.")
            else:
                st.warning(f"Couldn't index {uploaded_file.name} — see debug_log.txt for why.")
    if doc_collection.count() > 0:
        st.caption(f"{doc_collection.count()} document chunk(s) currently indexed.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "conversation-1"
if "doc_thread_id" not in st.session_state:
    # Document-scoped turns get their OWN LangGraph checkpointer thread,
    # separate from the main conversation thread above. Without this, every
    # document-scoped turn's message -- which includes the full raw chunk
    # text, not just the user's question -- gets permanently written into
    # the SAME thread's persisted history via InMemorySaver. That history
    # is replayed to the LLM on every later turn regardless of that turn's
    # own document-scoped decision, so a document's content, once injected
    # once, stays visible to the model forever afterward. Confirmed as a
    # real bug: "what is M1" was correctly judged NOT document-scoped this
    # turn (hybrid score below threshold, nothing new injected) but still
    # answered with exact facts (table names, agent names) from an earlier
    # turn's document injection, because that earlier turn's full document
    # dump was still sitting in "conversation-1"'s shared history. Isolating
    # document-scoped turns into their own thread keeps them free to have
    # their own multi-turn continuity (follow-up document questions can
    # still reference each other) WITHOUT that content ever reaching the
    # general conversation thread non-document turns read from.
    st.session_state.doc_thread_id = "document-qa-1"
if "recent_user_messages" not in st.session_state:
    st.session_state.recent_user_messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {role, text, graph_path?}

if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False
if "clarification_question" not in st.session_state:
    st.session_state.clarification_question = ""
if "pending_original_message" not in st.session_state:
    st.session_state.pending_original_message = ""
if "clarify_round" not in st.session_state:
    st.session_state.clarify_round = 0
if "original_message" not in st.session_state:
    st.session_state.original_message = ""
if "dialogue_state" not in st.session_state:
    # Replaces the old bare-string `active_entity` — the new pipeline
    # expects a full DialogueState (current_entity, pending_entity,
    # pending_clarification, clarification_count), not a loose string.
    st.session_state.dialogue_state = DialogueState()
if "concept_history" not in st.session_state:
    st.session_state.concept_history = []
if "stored_domains" not in st.session_state:
    st.session_state.stored_domains = set()
if "turns_since_holistic_read" not in st.session_state:
    st.session_state.turns_since_holistic_read = 0
if "ingested_doc_hashes" not in st.session_state:
    # Session-local cache of file hashes already sent to ingest_document()
    # this run -- avoids re-hitting Chroma's get(where=...) check (and
    # re-reading the file) on every Streamlit rerun for the same uploaded
    # file, since Streamlit reruns the whole script on every interaction
    # and st.file_uploader keeps returning the same file until it's cleared
    # or replaced. ingest_document() itself still has the real, persistent
    # dedup check against Chroma (by content hash), so this is purely a
    # same-session speed shortcut, not the source of truth for "already
    # indexed" -- that's still Chroma, checked fresh every new session.
    st.session_state.ingested_doc_hashes = set()

config = {"configurable": {"thread_id": st.session_state.thread_id}}
doc_config = {"configurable": {"thread_id": st.session_state.doc_thread_id}}


user_message = None
raw_user_message = None
intent_debug_text = ""


def _apply_intent(intent, raw_text):
    if intent["type"] == "topic":
        # BUG FIX: this returned immediately, before the "(Reply: ...)"
        # strip below ever ran -- so a topic_mode turn that went through a
        # clarify round (intent['topic'] built from meaning.resolved_text,
        # same value as resolved_query) leaked the internal bookkeeping
        # bracket straight into the "**You:**" chat bubble AND into the
        # "Research and create a timeline graph for the topic: ..." text
        # sent to the agent. Confirmed real repro: "...Cassandra creator
        # (Reply: who)" and "...lab work (Reply: What PPE should I
        # generally wear when doing any kind of lab work)" both shown
        # verbatim in the UI. question_mode turns never had this problem
        # because they fall through to the strip below -- topic_mode just
        # never reached it. Reuses the exact same strip, at the exact same
        # point (text becomes outward-facing), only extended to cover this
        # branch too -- classification itself (meaning.resolved_text) is
        # still completely untouched.
        topic_text = re.sub(r"\n\(Reply:\s*(.*?)\)", r" \1", intent["topic"]).strip()
        return f"Research and create a timeline graph for the topic: {topic_text}"
    text = intent.get("resolved_query") or raw_text
    # BUG FIX: resolved_query now correctly carries the full clarify-round
    # context (see the query_for_routing fix above) -- but this same value
    # is also reused as the outward-facing text: shown in the "**You:**"
    # chat bubble AND sent to the agent as the literal user message. The
    # internal "(Reply: ...)" bookkeeping label was never meant to be
    # user-facing or agent-facing -- strip it here, at the point this text
    # becomes outward-facing, leaving classification itself untouched.
    text = re.sub(r"\n\(Reply:\s*(.*?)\)", r" \1", text).strip()
    return text


if st.session_state.awaiting_clarification:
    st.markdown(f"**Agent:** {st.session_state.clarification_question}")
    followup_text = st.text_input("Your answer", key=f"clarify_input_{st.session_state.clarify_round}")
    if st.button("Continue") and followup_text:
        combined = f"{st.session_state.original_message}\n(Reply: {followup_text})"
        st.session_state.clarify_round += 1
        intent_debug = io.StringIO()
        with contextlib.redirect_stdout(intent_debug):
            intent = classify_intent(combined, extraction_model, round_index=st.session_state.clarify_round, latest_reply=followup_text)
        intent_debug_text = intent_debug.getvalue()

        with open("debug_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n===== CLARIFY ROUND {st.session_state.clarify_round}: reply = \"{followup_text}\" =====\n")
            f.write(intent_debug_text)

        if intent["type"] == "clarify":
            st.session_state.clarification_question = intent["followup"]
            st.session_state.pending_original_message = combined
            st.rerun()
        else:
            # NOTE: the old add_learned_intent_example() call that used to
            # sit here retrained the old LogisticRegression classifier live
            # on every resolved clarification. No equivalent exists for the
            # new static classifier — see classify_intent()'s docstring for
            # this gap, flagged there rather than silently dropped here.
            st.session_state.awaiting_clarification = False
            st.session_state.clarify_round = 0
            raw_user_message = _apply_intent(intent, combined)
            user_message = raw_user_message
else:
    user_text = st.text_input("What would you like to know?")
    if st.button("Send") and user_text:
        intent_debug = io.StringIO()
        with contextlib.redirect_stdout(intent_debug):
            intent = classify_intent(user_text, extraction_model, round_index=0)
        intent_debug_text = intent_debug.getvalue()
        with open("debug_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n===== INITIAL: {user_text} =====\n")
            f.write(intent_debug_text)
        if intent["type"] == "clarify":
            st.session_state.awaiting_clarification = True
            st.session_state.clarification_question = intent["followup"]
            st.session_state.pending_original_message = user_text
            st.session_state.original_message = user_text
            st.session_state.clarify_round = 1
            st.rerun()
        else:
            raw_user_message = _apply_intent(intent, user_text)
            user_message = raw_user_message

is_question_flow = user_message is not None and not user_message.startswith(
    "Research and create a timeline graph for the topic:"
)

if user_message:
    debug_output = io.StringIO()
    final_message = None
    used_agent = None
    error_summary = []
    status_placeholder = st.empty()

    status_placeholder.markdown("⏳ Starting...")
    with contextlib.redirect_stdout(debug_output):
        # Document RAG (question_mode ONLY): decide relevance BEFORE
        # preferences, since a document-scoped turn must skip preference
        # retrieval entirely -- can't know whether to skip it until this
        # runs. Topic-mode turns never touch this at all.
        is_document_scoped_turn = False
        doc_chunk_count = doc_collection.count()
        doc_assets = []  # only ever populated inside the document-scoped
                          # branch below; stays empty for every other kind
                          # of turn (no document indexed, not question_flow,
                          # or a question_flow turn that wasn't confidently
                          # document-scoped).
        if is_question_flow and doc_chunk_count > 0:
            top_similarity, doc_result, doc_breakdown, doc_assets = search_document(raw_user_message)
            if top_similarity is not None and top_similarity >= DOC_SCOPED_THRESHOLD:
                is_document_scoped_turn = True
            else:
                # Previously silent on a miss -- printed nothing at all, so a
                # too-low similarity and an empty collection looked identical
                # in the log. Always show the number that was actually
                # compared against DOC_SCOPED_THRESHOLD.
                print(
                    f"[Document check: {doc_chunk_count} chunk(s) indexed, "
                    f"top_similarity={top_similarity if top_similarity is not None else 'None'} "
                    f"< {DOC_SCOPED_THRESHOLD} -- NOT document-scoped, falling through to preferences]"
                )
        elif is_question_flow:
            print(f"[Document check: doc_collection empty ({doc_chunk_count} chunks) -- skipping document check entirely]")

        if is_document_scoped_turn:
            # This question is about the document -- per requirement, no
            # domain/style preference guidance for this turn (that's "how
            # this user generally likes to be helped"; irrelevant when the
            # answer has to come only from the document), and the model is
            # restricted to document-only answering, not blended with its
            # own knowledge or web_search.
            print(f"[Document-scoped turn ({doc_breakdown}) -- skipping preference retrieval]")

            # classify_intent() already appended this turn's entity/focus into
            # concept_history unconditionally, BEFORE this document-scoped
            # check ran (search_document/is_document_scoped_turn don't exist
            # yet at that point in the flow). Skipping
            # extract_domain_engagement_preference() later (see is_question_flow
            # block below) only skips BUCKETING/PRINTING that turn -- it does
            # NOT undo the append, so a document lookup was still silently
            # sitting in concept_history and would get counted (and eventually
            # saved as a "recurring domain interest") the next time a
            # non-document-scoped turn ran the extractor. Confirmed via a real
            # debug_log.txt: a document-scoped "corrosive reagents" turn's
            # entry showed up bucketed under a LATER "emergency procedures"
            # turn's printout. Since is_document_scoped_turn can only be True
            # on a question-mode turn, and classify_intent() appends exactly
            # one entry per turn, the last item in concept_history at this
            # point is guaranteed to be this turn's own entry -- pop it back
            # out so a document lookup never counts as a domain-engagement
            # signal, matching the original intent of the skip above.
            if st.session_state.concept_history:
                removed = st.session_state.concept_history.pop()
                print(f"[Removing {removed!r} from concept_history -- document lookup, not a domain-engagement signal]")

            # If the winning chunk includes a table, its exact grid is ALSO
            # rendered separately and automatically below the reply (see
            # doc_assets handling near the bottom of this file) -- that's
            # what guarantees the exact table always displays correctly
            # regardless of what the model says (see ingest_document's
            # docstring). But the RAW markdown table is sitting right in
            # doc_result below, in the model's own context, so without this
            # note the model reasonably reproduces it in its own answer
            # text too -- st.markdown renders THAT as a real table, then
            # doc_assets renders the SAME table again via st.table() right
            # after it. Confirmed real repro: a "model accuracy comparison"
            # answer showed the table twice, back to back. Telling the
            # model the table is already shown stops it from re-printing
            # the grid itself while still letting it discuss/reference the
            # numbers in prose.
            table_note = ""
            if any(a.get("type") == "table" for a in doc_assets):
                table_note = (
                    " The exact table from the document is already displayed "
                    "to the user automatically, separately from your reply -- "
                    "do NOT reproduce it (as a markdown table or any other "
                    "grid/table format) in your own answer text. Refer to or "
                    "summarize specific values in prose instead."
                )

            user_message = (
                f"[This question has been identified as being about the user's uploaded document "
                f"(similarity={top_similarity:.3f}). Answer ONLY using the document context below -- "
                f"do NOT use your own general knowledge, do NOT call web_search, and do NOT apply any "
                f"style or domain preference guidance for this turn. If the retrieved context "
                f"genuinely doesn't answer the specific question asked, say so plainly rather than "
                f"guessing -- do NOT invent what the document might contain.{table_note}]\n\n"
                f"{doc_result}\n\n{raw_user_message}"
            )
        else:
            # Not document-scoped (or topic-mode, or nothing indexed) --
            # exactly the original flow, untouched. Runs for BOTH question
            # and topic flow, same as before this document work started.
            print("[RAG subagent: checking memory for relevant preferences...]")
            current_entity = st.session_state.dialogue_state.current_entity
            pref_text = retrieve_user_preferences(raw_user_message, current_entity=current_entity)
            if pref_text and "no relevant preferences found" not in pref_text.lower():
                if is_question_flow:
                    user_message = f"[Known user preferences: {pref_text}]\n\n{raw_user_message}"
                else:
                    # Topic-mode messages MUST start with exactly "Research and
                    # create a timeline graph for the topic:" -- RULE 1 in
                    # deepagent_system_prompt detects this flow via user_message.
                    # startswith(...). Prepending here would break that detection
                    # entirely, so the preference block is appended AFTER the
                    # required prefix instead.
                    user_message = f"{raw_user_message}\n\n[Known user preferences: {pref_text}]"

        # Explicit, deterministic flow marker -- Python already knows the flow
        # type with certainty (is_question_flow, computed above). Previously
        # that certainty was thrown away -- the model had to re-derive it by
        # pattern-matching the message text against RULE 1's literal trigger
        # phrase, and a weak fallback model (confirmed: gemini-3.1-flash-lite,
        # reached only after groq rate-limits) got distracted by RULE 1's
        # detailed step-by-step procedure and applied it anyway, even on a
        # question-mode message that never contained the trigger phrase
        # ("[Known user preferences: ...]\n\nWho created Cassandra?"). Tagging
        # the flow explicitly removes the need for the model to infer it.
        # deepagent_system_prompt above is updated to treat this tag as
        # authoritative over its own pattern-matching.
        flow_tag = "[FLOW: QUESTION_MODE]" if is_question_flow else "[FLOW: TOPIC_MODE]"
        user_message = f"{flow_tag}\n\n{user_message}"

        # Use the isolated document-QA thread for document-scoped turns, so
        # the full document content injected into user_message never gets
        # permanently written into the main conversation thread that later,
        # non-document turns' agent calls read their history from.
        active_config = doc_config if is_document_scoped_turn else config

        for agent_name, agent in agents_to_try:
            provider = provider_label(agent_name)
            try:
                print(f"--- Trying agent: {agent_name} ---")
                final_message = run_agent_stream(agent, active_config, user_message, provider, status_placeholder)
                used_agent = agent_name
                break
            except Exception as e:
                   print(f"[{agent_name} failed: {e}]")   
                   clean_error = format_error(provider, e)  
                   error_summary.append(clean_error)
                   continue

    status_placeholder.empty()

    debug_text = intent_debug_text + debug_output.getvalue()
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n\n===== {raw_user_message} =====\n")
        f.write(debug_text)

    # is_document_scoped_turn was already set above, before the agent loop --
    # True only when this was a question-mode turn AND search_document's top
    # similarity cleared DOC_SCOPED_THRESHOLD. Used below to skip domain-
    # engagement-preference saving for this turn (a document lookup isn't a
    # recurring topic interest the way a repeated general question is).

    status_lines = build_status_lines(debug_text)
    graph_match = re.search(r"\[Graph saved as (.+?)\]", debug_text)

    st.session_state.chat_history.append({"role": "user", "text": raw_user_message})

    if final_message:
        answer = extract_text(final_message.content)

        # Deterministic backstop for the prompt instruction added to
        # user_message above (search for table_note): that instruction
        # only reduces the chance the model re-prints the table, it
        # doesn't guarantee it won't. doc_assets already renders the exact
        # grid via st.table() further down for ANY document-scoped turn
        # whose winning chunk is a table, so strip any markdown table the
        # model included in its own prose -- the UI must only ever show
        # the one deterministic table, not whatever the model chose to do.
        if is_document_scoped_turn and any(a.get("type") == "table" for a in doc_assets):
            answer = _strip_markdown_tables(answer)

        # Domain-engagement tracking runs on every OTHER turn, question-mode
        # or topic-mode -- concept_history is appended for both inside
        # classify_intent(), so the checker must see both too, or an exact
        # threshold crossing that lands on a topic-mode turn gets silently
        # skipped (the count moves past the threshold before this ever runs
        # again on a later question-mode turn). Document-scoped turns are
        # the one exception -- see is_document_scoped_turn above.
        if not is_document_scoped_turn:
            domain_pref_debug = io.StringIO()
            with contextlib.redirect_stdout(domain_pref_debug):
                try:
                    extract_domain_engagement_preference()
                except Exception as e:
                    print(format_error("Domain preference extraction", e))
            with open("debug_log.txt", "a", encoding="utf-8") as f:
                f.write(domain_pref_debug.getvalue())

        if is_question_flow:
            st.session_state.recent_user_messages.append(raw_user_message)
            if len(st.session_state.recent_user_messages) > MAX_RECENT:
                st.session_state.recent_user_messages.pop(0)
            pref_debug = io.StringIO()
            with contextlib.redirect_stdout(pref_debug):
                try:
                    st.session_state.turns_since_holistic_read += 1
                    if st.session_state.turns_since_holistic_read >= HOLISTIC_READ_EVERY_N_TURNS:
                        extract_holistic_preference(st.session_state.chat_history, extraction_model)
                        st.session_state.turns_since_holistic_read = 0
                except Exception as e:
                    print(format_error("Preference extraction", e))
            pref_debug_text = pref_debug.getvalue()
            with open("debug_log.txt", "a", encoding="utf-8") as f:
                f.write(pref_debug_text)
            status_lines += build_status_lines(pref_debug_text)

        response_parts = []
        if status_lines:
            response_parts.append("  \n".join(status_lines))
        response_parts.append(answer)

        entry = {"role": "agent", "text": "\n\n".join(response_parts)}
        if graph_match:
            entry["graph_path"] = graph_match.group(1)
            entry["text"] += "\n\n✅ Graph successfully created"

        # Independent of graph_path above -- no shared key, no shared
        # condition, no interaction with topic-mode's graph detection at
        # all. Only ever set when THIS turn was confidently document-scoped
        # (is_document_scoped_turn), so a non-document turn (even one that
        # briefly touched search_document without clearing the confidence
        # bar) never carries stale assets into the UI.
        if is_document_scoped_turn and doc_assets:
            entry["doc_assets"] = doc_assets

        st.session_state.chat_history.append(entry)
    else:
        error_text = "\n".join(f"- {e}" for e in error_summary) if error_summary else "Unknown error occurred."
        st.session_state.chat_history.append({
            "role": "agent",
            "text": f"⚠️ All agents failed to respond.\n\n{error_text}",
        })

st.divider()

for entry in reversed(st.session_state.chat_history):
    if entry["role"] == "user":
        st.markdown(f"**You:** {entry['text']}")
    else:
        st.markdown(f"**Agent:** {entry['text']}")
        if entry.get("graph_path") and os.path.exists(entry["graph_path"]):
            st.image(entry["graph_path"])
        # Separate block, separate key -- doesn't read or touch graph_path,
        # doesn't run any topic-mode/graph logic, doesn't care how this
        # entry's text was produced. Only exists if is_document_scoped_turn
        # set entry["doc_assets"] earlier for this specific turn.
        for asset in entry.get("doc_assets", []):
            if asset["type"] == "image" and os.path.exists(asset["path"]):
                st.image(asset["path"])
            elif asset["type"] == "table" and asset.get("grid"):
                st.table(asset["grid"])
    st.markdown("---")