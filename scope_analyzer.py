import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import spacy

from semantic_retriever import get_embedding


nlp = spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# Reference phrasings for the "broad exploration" vs "focused ask" signal.
#
# ASSUMPTION: the topic word is replaced with a neutral placeholder
# ("the subject") on purpose. This is still semantic comparison, not
# keyword matching -- we are comparing sentence *shape and intent*
# ("everything about X" vs "what is X"), not checking for literal
# words like "history" or "future" in the user's text.
# ---------------------------------------------------------------------------

BROAD_EXEMPLARS = [
    "everything about the subject",
    "complete guide to the subject",
    "comprehensive overview of the subject",
    "in-depth study of the subject",
    "all aspects of the subject",
    "the subject from basics to advanced",
    "full explanation of the subject",
    "complete understanding of the subject",
]

FOCUSED_QUESTION_EXEMPLARS = [
    "what is the subject",
    "how does the subject work",
    "explain the subject",
    "why is the subject used",
    "how efficient is the subject",
]

FOCUSED_ASPECT_EXEMPLARS = [
    "the history of the subject",
    "the advantages of the subject",
    "the limitations of the subject",
    "the architecture of the subject",
    "the theories behind the subject",
    "the future of the subject",
    "the applications of the subject",
    "the components of the subject",
]

# HISTORY (kept as a note, not as live code -- see intent_engine.py's
# scope-gate comment block for the full story): this signal was
# originally computed three ways -- a single blended centroid, a
# split two-centroid version, and a top-3 mean -- before landing here
# on top-1 (single closest exemplar, no averaging). Each version was
# checked against test_scope.py and test_exemplar.py; averaging of
# any kind consistently diluted the match for real focused questions,
# and top-1 was the version that actually distinguished them best
# (still imperfect -- see intent_engine.py's SEMANTIC_MARGIN_THRESHOLD
# comment for why this score is only ever used as a weak tie-break,
# never a primary gate). The centroid and top-3 code paths were
# removed once top-1 was confirmed as the only one actually consulted
# downstream -- no point computing two extra embeddings comparisons
# per call that nothing reads.
#
# ASSUMPTION: two clauses are treated as "the same dimension" if their
# embedding similarity is >= this threshold. 0.55 is a starting guess,
# not a measured value -- tune it against test_scope.py once you can
# see where real examples land.
DIMENSION_SIMILARITY_THRESHOLD = 0.55
# Minimum shared trailing words for two clauses to be treated as the same
# dimension regardless of embedding similarity. "AUC and Gini coefficient of
# this model" splits (via split_clauses) into "AUC of this model" / "Gini
# coefficient of this model" -- sharing the trailing "of this model" (3
# words) -- but "AUC" and "Gini coefficient" alone don't embed close enough
# to clear DIMENSION_SIMILARITY_THRESHOLD, forcing topic_mode on a query
# that's genuinely one coherent question about one model. >=2 words (not 1)
# avoids merging clauses that only share a trailing stopword by coincidence.
# Verified no-op against "history and future of Linux": that case already
# clusters to 1 via embeddings alone, and separately shares "of Linux" (2
# words) -- this reaches the same answer by a second route, doesn't change it.
MIN_SHARED_TRAILING_WORDS = 2


def _shared_trailing_words(a: str, b: str) -> int:
    """Word-count of the common trailing subsequence, comparing from the
    end backward. Case-insensitive."""
    count = 0
    for wa, wb in zip(reversed(a.split()), reversed(b.split())):
        if wa.lower() != wb.lower():
            break
        count += 1
    return count

# ---------------------------------------------------------------------------
# Lexical (not semantic) totality-marker check.
#
# WHY THIS EXISTS: test_exemplar.py showed four rounds of fixes to the
# BROAD_EXEMPLARS/FOCUSED_*_EXEMPLARS embedding comparison above --
# centroid, top-3, top-1, single-exemplar removal -- plateauing at
# roughly 4/7 known cases, with errors migrating between exemplars
# rather than resolving. That's the signature of a structural limit:
# "complete guide to Linux" and "history and recent developments of
# Linux" are semantically almost the same thing (both are about
# comprehensive Linux knowledge), so a sentence-embedding model
# clusters them close together even though the distinction that
# actually matters -- generic totality claim vs specific named
# aspect(s) -- is a lexical/structural one, not a semantic one.
#
# This is a closed-list substring check, deliberately not "smart" --
# it is meant to catch a specific class of generic totality wording,
# not to generalize via embeddings. Expand the list as real misses
# show up.
# ---------------------------------------------------------------------------

TOTALITY_MARKERS = [
    "complete", "everything", "full", "comprehensive",
    "basics to advanced", "teach me", "overview",
    "in-depth", "in depth", "all about", "as a whole",
    "fully", "in general", "all aspects",
    "start to finish", "end to end", "top to bottom",
    "walk me through", "a to z",
]


def has_totality_marker(text: str) -> bool:
    """
    Lexical check, not semantic. See the module-level comment above
    TOTALITY_MARKERS for why this exists as a separate signal from
    the embedding-based broad_exploration_score below.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in TOTALITY_MARKERS)


# ---------------------------------------------------------------------------
# COMPARISON_MARKERS: closes a gap this file's own split_clauses() docstring
# already documented as "known, not fixed" -- coordinated ENTITIES ("a list
# and a tuple", "Python and Java") parse identically to coordinated ASPECTS
# of one topic ("history and future of Linux"), so split_clauses() and
# count_distinct_dimensions() can't structurally tell them apart. Confirmed
# as a real miss: "the difference between a list and a tuple in Python"
# scored distinct_dimension_count=2 and got forced into topic_mode by
# intent_engine.py's dimension gate, even though comparing two things is
# one coherent question, not two independent topics to explore separately.
# Same closed-list, deliberately-not-"smart" design as TOTALITY_MARKERS
# above -- meant to catch this specific, common phrasing of a comparison
# request, not to generalize via embeddings. Expand as real misses show up.
# ---------------------------------------------------------------------------

COMPARISON_MARKERS = [
    "difference between", "differences between", "compare",
    "comparison between", "comparison of", "which is better",
    "pros and cons of", "better than",
]

# "vs"/"versus" need word-boundary matching, not substring -- unlike the
# phrases above, "vs" as a bare substring could false-positive inside
# unrelated words.
_COMPARISON_TOKEN_RE = re.compile(r"\b(vs\.?|versus)\b", re.IGNORECASE)


def has_comparison_marker(text: str) -> bool:
    """
    Lexical check, not semantic -- same design as has_totality_marker.
    See the COMPARISON_MARKERS comment above for why this exists.
    """
    lowered = text.lower()
    if any(marker in lowered for marker in COMPARISON_MARKERS):
        return True
    return bool(_COMPARISON_TOKEN_RE.search(text))


_broad_exemplar_embeddings = None
_focused_question_exemplar_embeddings = None
_focused_aspect_exemplar_embeddings = None


def _get_exemplar_embeddings():
    """
    Computes (once, cached) the individual embedding of every exemplar
    in each group -- no centroid, no averaging. See the HISTORY note
    above BROAD_EXEMPLARS for why top-1 (single closest exemplar) is
    the only version kept.
    """

    global _broad_exemplar_embeddings
    global _focused_question_exemplar_embeddings
    global _focused_aspect_exemplar_embeddings

    if _broad_exemplar_embeddings is None:
        _broad_exemplar_embeddings = np.array(
            [get_embedding(t) for t in BROAD_EXEMPLARS]
        )

    if _focused_question_exemplar_embeddings is None:
        _focused_question_exemplar_embeddings = np.array(
            [get_embedding(t) for t in FOCUSED_QUESTION_EXEMPLARS]
        )

    if _focused_aspect_exemplar_embeddings is None:
        _focused_aspect_exemplar_embeddings = np.array(
            [get_embedding(t) for t in FOCUSED_ASPECT_EXEMPLARS]
        )

    return (
        _broad_exemplar_embeddings,
        _focused_question_exemplar_embeddings,
        _focused_aspect_exemplar_embeddings
    )


def _top1_similarity(query_embedding, exemplar_embeddings) -> float:
    """
    Similarity of the query against its single closest exemplar in
    the group -- no averaging at all.
    """

    sims = cosine_similarity(
        query_embedding,
        exemplar_embeddings
    )[0]

    return float(np.max(sims))


def _split_on_repeated_trailing_phrase(doc) -> List[str]:
    """
    Handles the 'ASPECT1 of X and ASPECT2 of X' surface shape --
    e.g. "benefits of yoga and risks of yoga", "pros of remote work
    and cons of remote work". This is a DIFFERENT shape from what
    the conjunct-based logic in split_clauses() below was built for.

    WHY A SEPARATE CHECK, NOT A FIX INSIDE THE EXISTING LOGIC:
    verified directly against spaCy's real parse (not assumed) that
    for "benefits of yoga and risks of yoga", the parser attaches the
    conjunction as yoga<-conj-risks (the OBJECT of the first
    preposition, not the aspect noun "benefits"), and attaches the
    second "of yoga" as an independent second prep child directly on
    "benefits" -- structurally disconnected from the "risks" conjunct
    entirely. That means the existing children-of-conjunct search for
    a shared trailing modifier can never find it (it isn't a child of
    either conjunct), and the existing leading-context logic has no
    principled way to know "benefits" is item-1-specific while "What
    are the" is universal. This isn't a small tweak away from working
    -- the dependency tree just doesn't expose the right structure for
    this shape. A surface/text-level check sidesteps that.

    HOW IT WORKS: split the sentence on its single "and"/"or", then
    find the longest matching sequence of trailing words shared by
    both sides (word-for-word, not embedding similarity -- this is a
    literal repeated phrase, e.g. "of yoga" appearing verbatim on
    both sides). If found, the left side is already a complete clause
    as-is; the right side becomes (shared leading frame from the left
    side) + (right side's own aspect word) + (the repeated trailing
    phrase).

    GUARDS (each verified against a real test case, not just reasoned
    about -- see test_split_clauses_regression.py):
    - Requires exactly one comma-free "and"/"or" in the sentence, so
      it never fires on 3+-way coordination ("before, during, and
      after the exam") -- that shape falls through to the existing
      conjunct-chain logic below. NOTE: verified that 3+-way
      coordination is a PRE-EXISTING separate bug in that logic, not
      something this change touches or fixes -- flagged, not fixed.
    - Requires >= 2 shared trailing words, so a single incidentally-
      matching word doesn't cause a false positive.
    - Requires the shared span not consume the entire right side, so
      there's an actual aspect word left over to build clause 2 from.
    - Requires >= 2 tokens on each side of "and", so short two-word
      coordinations like "Compare Python and Java" fall through to
      the existing logic untouched (verified unaffected either way --
      see that same regression file for a note on this case's
      pre-existing, unrelated docstring inaccuracy).

    Returns None (does not fire) when the shape doesn't match --
    caller falls through to the existing conjunct-based logic.
    """

    text = doc.text

    if "," in text:
        return None

    cc_tokens = [
        t for t in doc
        if t.dep_ == "cc" and t.text.lower() in ("and", "or")
    ]

    if len(cc_tokens) != 1:
        return None

    cc = cc_tokens[0]

    left_tokens = list(doc[:cc.i])
    right_tokens = list(doc[cc.i + 1:])

    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return None

    left_texts = [t.text for t in left_tokens]
    right_texts = [t.text for t in right_tokens]

    common_len = 0
    while (
        common_len < len(left_texts)
        and common_len < len(right_texts)
        and left_texts[-1 - common_len].lower() == right_texts[-1 - common_len].lower()
    ):
        common_len += 1

    if common_len < 2 or common_len >= len(right_texts):
        return None

    shared_span = right_tokens[len(right_tokens) - common_len:]
    shared_text = " ".join(t.text for t in shared_span)

    aspect2_tokens = right_tokens[:len(right_tokens) - common_len]
    aspect2_text = " ".join(t.text for t in aspect2_tokens).strip()

    if not aspect2_text:
        return None

    left_before_shared = left_tokens[:len(left_tokens) - common_len]

    if len(left_before_shared) < 1:
        return None

    clause1 = " ".join(t.text for t in left_tokens).strip()

    leading_frame_tokens = left_before_shared[:-1]
    leading_frame = " ".join(t.text for t in leading_frame_tokens).strip()

    clause2 = (
        f"{leading_frame} {aspect2_text} {shared_text}".strip()
        if leading_frame
        else f"{aspect2_text} {shared_text}".strip()
    )

    return [clause1, clause2]


def _split_on_repeated_prep_object(doc) -> Optional[List[str]]:
    """
    Handles the '[SHARED CONTEXT] PREP1 X, PREP2 X, ..., and PREPn X'
    shape -- e.g. "How should I prepare before the exam, during the
    exam, and after the exam". This is the PRE-EXISTING 3-way-
    coordination bug flagged (not fixed) in split_clauses' docstring
    and in _split_on_repeated_trailing_phrase's guard comment.

    WHY A SEPARATE CHECK: verified directly against spaCy's real parse
    (not assumed) that for this sentence, spaCy does NOT attach
    "before"/"during"/"after" to each other as a symmetric conj
    cluster the way "history"/"future" are for noun coordination.
    Only the LAST preposition ("after") gets dep_="conj" at all, and
    its head is the main verb ("prepare"), not the first preposition
    -- "before" and "during" are just plain dep_="prep" siblings of
    the verb, structurally invisible to token.conjuncts entirely. That
    is exactly why the existing anchor-via-.conjuncts logic below only
    ever saw a 2-item cluster (the verb + "after") and mis-split
    "before"/"during" into the wrong clause. This isn't a small tweak
    to that logic -- it needs a different detection strategy that
    doesn't depend on .conjuncts for this shape.

    HOW IT WORKS: group every ADP (preposition) token in the sentence
    by its head token (the word it attaches to -- e.g. all of
    "before"/"during"/"after" attach to "prepare"), regardless of
    whether spaCy separately marked any of them dep_="conj". If a
    group has 3+ prepositions sharing one head, and each carries its
    own prepositional object (pobj), and those objects are all
    word-for-word identical (e.g. "the exam" repeated after each
    preposition), this is confirmed as the repeated-object shape.
    Each output clause = (shared leading context before the first
    preposition) + (that preposition) + (the shared object).

    GUARDS:
    - Requires a comma in the text, so this never fires on the
      already-handled 2-way comma-free case (e.g. "before a workout
      and after a workout"), which the general conjunct-chain logic
      below already handles correctly -- kept fully untouched.
    - Requires >= 3 prepositions sharing one head, so it only targets
      the 3+-way shape this function exists for.
    - Requires every preposition to have its own pobj object, and all
      objects to match exactly (case-insensitive) -- if the objects
      differ (e.g. "risks of smoking and benefits of quitting
      smoking"), this declines and falls through, same as the
      existing repeated-trailing-phrase special case does for its
      own non-matching shape.

    Returns None (does not fire) when the shape doesn't match --
    caller falls through to the existing conjunct-based logic,
    unchanged.
    """

    text = doc.text

    if "," not in text:
        return None

    groups: dict = {}
    for token in doc:
        if token.pos_ == "ADP" and token.dep_ in ("prep", "conj"):
            groups.setdefault(token.head.i, []).append(token)

    items = None
    for head_i, preps in groups.items():
        if len(preps) >= 3:
            items = sorted(preps, key=lambda t: t.i)
            break

    if items is None:
        return None

    object_spans = []
    object_texts = []

    for item in items:

        pobj_children = [c for c in item.children if c.dep_ == "pobj"]

        if not pobj_children:
            return None

        span = list(pobj_children[0].subtree)
        object_spans.append(span)
        object_texts.append(
            " ".join(t.text for t in span).strip().lower()
        )

    if len(set(object_texts)) != 1:
        return None

    shared_object_text = " ".join(t.text for t in object_spans[0]).strip()

    leading_tokens = list(doc[:items[0].i])
    leading_text = " ".join(t.text for t in leading_tokens).strip()

    clauses = [
        f"{leading_text} {item.text} {shared_object_text}".strip()
        if leading_text
        else f"{item.text} {shared_object_text}".strip()
        for item in items
    ]

    return clauses


def _split_on_misattached_conjunct(doc) -> Optional[List[str]]:
    """
    Handles 'ASPECT1 of X and ASPECT2 [of/for] Y' where X != Y -- e.g.
    "risks of smoking and benefits of quitting smoking", "causes of
    insomnia and treatments for managing insomnia", "symptoms of
    anxiety and coping strategies for anxiety". This is the other
    PRE-EXISTING gap flagged (not fixed) alongside the 3-way exam bug:
    _split_on_repeated_trailing_phrase correctly declines on all of
    these (the trailing words differ across the two sides), so they
    fell through to the general conjunct-chain logic below, which
    mishandled each of them.

    WHY THIS NEEDS THREE SUB-CASES, NOT ONE: verified against spaCy's
    real parse (not assumed) that all three example sentences share
    the same top-level coordination shape -- an anchor noun that is
    itself the object of "of" (e.g. "smoking"/"insomnia"/"anxiety"),
    coordinated with a second conjunct (e.g.
    "benefits"/"treatments"/"coping") -- but spaCy attaches each
    sentence's trailing modifier ("of quitting smoking" / "for
    managing insomnia" / "for anxiety") to a DIFFERENT token in each
    case:
      (a) to the FIRST/anchor conjunct itself (smoking case) -- the
          same shape as the already-correct Linux case, distinguished
          from it only by a lexical self-reference signal (see below)
      (b) to N1, the noun governing the "of" preposition (insomnia
          case) -- a sibling prep attached one level up, not to
          either conjunct at all
      (c) to the SECOND conjunct itself, already correctly attached
          (anxiety case) -- here the only bug is that the general
          logic's range-based own-text slicing (item.i-bounded)
          drops the second conjunct's own already-correctly-attached
          children instead of using them, and also wrongly treats
          them as a "shared" trailing phrase to smear onto clause 1
          too
    A single fixed attachment point can't cover all three, so this
    checks each in turn and uses whichever is actually present.

    HOW EACH SUB-CASE IS DETECTED:
      (a) self-reference: a trailing prep/pobj child of the FIRST
          conjunct, positioned after the second conjunct, whose own
          subtree contains the first conjunct's own word again (e.g.
          "of quitting SMOKING" contains "smoking") -- this recurrence
          is the tell that the phrase is misattached and actually
          describes the second conjunct, not the first. Without this
          check, the already-correct Linux case ("of Linux" attached
          to "history", but "history" never recurs inside "of Linux")
          would be wrongly rewritten too.
      (b) sibling-of-N1: a trailing prep/pobj child of N1 itself
          (excluding the "of" prep already used for clause 1),
          positioned after the second conjunct -- this only fires
          when (a) found nothing, so it never double-fires on the
          smoking-style case.
      (c) second's own subtree: if the second conjunct's dependency
          subtree already contains more than just itself (i.e. it has
          its own children, like "coping" having "strategies" and
          "for anxiety"), that subtree IS the correct clause-2
          content already -- checked FIRST, before (a) or (b), since
          it's the cheapest and most direct signal and takes priority
          whenever the parser got the attachment right on its own.

    Returns None (does not fire) when none of the three signals
    match -- caller falls through to the existing conjunct-based
    logic, unchanged.
    """

    anchor = None

    for token in doc:
        if list(token.conjuncts):
            anchor = token
            break

    if anchor is None:
        return None

    conjuncts = list(anchor.conjuncts)

    if len(conjuncts) != 1:
        return None

    first, second = sorted([anchor, conjuncts[0]], key=lambda t: t.i)

    if first.dep_ != "pobj":
        return None

    prep = first.head

    if prep.dep_ != "prep":
        return None

    n1 = prep.head

    if n1.i >= prep.i:
        return None

    leading_tokens = list(doc[:n1.i])
    leading_text = " ".join(t.text for t in leading_tokens).strip()

    clause1 = " ".join(t.text for t in doc[:first.i + 1]).strip()

    second_own_text = None

    # Sub-case (c): second conjunct already has its own attached
    # modifier(s) in its dependency subtree -- use it directly.
    second_subtree = sorted(list(second.subtree), key=lambda t: t.i)

    if len(second_subtree) > 1:
        second_own_text = " ".join(t.text for t in second_subtree).strip()

    # Sub-case (a): trailing modifier misattached to the first
    # conjunct, distinguished from the Linux shape by self-reference.
    if second_own_text is None:

        first_trailing = [
            c for c in first.children
            if c.dep_ in ("prep", "pobj") and c.i > second.i
        ]

        if first_trailing:
            candidate = sorted(first_trailing[0].subtree, key=lambda t: t.i)
            candidate_words = [t.text.lower() for t in candidate]

            if first.text.lower() in candidate_words:
                second_own_text = (
                    second.text + " "
                    + " ".join(t.text for t in candidate)
                ).strip()

    # Sub-case (b): trailing modifier attached as a sibling prep of
    # N1 itself, not to either conjunct.
    if second_own_text is None:

        n1_trailing = [
            c for c in n1.children
            if c.dep_ in ("prep", "pobj")
            and c.i > second.i
            and c.i != prep.i
        ]

        if n1_trailing:
            candidate = sorted(n1_trailing[0].subtree, key=lambda t: t.i)
            second_own_text = (
                second.text + " "
                + " ".join(t.text for t in candidate)
            ).strip()

    if second_own_text is None:
        return None

    clause2 = (
        f"{leading_text} {second_own_text}".strip()
        if leading_text
        else second_own_text
    )

    return [clause1, clause2]


def split_clauses(text: str) -> List[str]:
    """
    Splits a query into its coordinated clauses, re-attaching any
    trailing modifier shared across the whole coordination. Example:

        "history and future of Linux"
            -> ["history of Linux", "future of Linux"]

        "history, architecture, security, and future of Linux"
            -> ["history of Linux", "architecture of Linux",
                "security of Linux", "future of Linux"]

    VERIFIED AGAINST ACTUAL PARSES (not assumed): in spaCy's parse of
    "history and future of Linux", the trailing "of Linux" attaches
    only to the anchor token ("history"), not to "future" -- even
    though "future" sits right next to it in the sentence. A naive
    token-position split (the earlier version of this function) kept
    that phrase on whichever clause it happened to be adjacent to in
    the text, which meant only ONE clause out of the coordination
    ever carried the shared subject. This version detects that
    shared trailing phrase explicitly and copies it onto every
    clause, using each conjunct's own token span (not its dependency
    subtree, which -- also verified against the actual parse -- turned
    out to nest later conjuncts inside earlier ones for coordination
    chains of 3+ items).

    KNOWN LIMITATION (verified, not fixed): this only makes sense for
    coordinated ASPECTS of one topic. For coordinated ENTITIES, e.g.
    "Compare Python and Java", it returns ["Compare Python", "Java"]
    -- which is correct in that they really are two different things,
    but this analyzer has no way to know that's a comparison request
    rather than a scope question. That distinction should already be
    resolved by semantic intent classification (the "comparison"
    category) before scope's dimension count is used for anything.

    ALSO NOTE (observed in real logged output, not yet fixed): this
    is verified only against noun-phrase coordination sharing an
    anchor ("history and future of Linux"). Full-clause coordination
    like "when it start and whats future updates" is a different
    shape -- two independent clauses joined by "and", not two
    conjuncts of one noun phrase -- and hasn't been verified against
    this function. A real run classified that example's dimension
    count as 2 (not 1, as the "history and future" pattern would
    suggest it should be, since both are timeline sub-points of one
    coherent need). Flagged as a known gap, not yet fixed.
    """

    doc = nlp(text)

    special_case = _split_on_repeated_trailing_phrase(doc)
    if special_case is not None:
        return special_case

    prep_special_case = _split_on_repeated_prep_object(doc)
    if prep_special_case is not None:
        return prep_special_case

    misattached_case = _split_on_misattached_conjunct(doc)
    if misattached_case is not None:
        return misattached_case

    anchor = None

    for token in doc:

        if list(token.conjuncts):
            anchor = token
            break

    if anchor is None:
        return [text.strip()]

    items = sorted(
        [anchor] + list(anchor.conjuncts),
        key=lambda t: t.i
    )

    last_pos = items[-1].i

    # Find a modifier phrase (e.g. "of Linux") that trails the whole
    # coordination and applies to every item in it, regardless of
    # which single item it's attached to in the parse tree.
    shared_tokens = []

    for item in items:

        for child in item.children:

            # "prep" covers noun coordination sharing a trailing PP
            # ("history and future OF LINUX" -- "of Linux" attaches to
            # the anchor noun as a prep child). "pobj" covers the
            # mirror case: preposition coordination sharing a trailing
            # object ("before and after A WORKOUT" -- here "before"/
            # "after" ARE the prepositions, so the shared object
            # attaches directly to the last conjunct as its pobj, with
            # no intermediate prep token to catch). Confirmed via
            # actual spaCy parse, not assumed: "workout" parses as
            # dep_="pobj", head="after" -- the old prep-only check
            # silently dropped it from both clauses, producing a
            # near-empty second clause ("after") that embedded nothing
            # like the first, forcing a false 2-dimension split.
            if child.dep_ in ("prep", "pobj") and child.i > last_pos:
                shared_tokens = list(child.subtree)
                break

        if shared_tokens:
            break

    shared_ids = set(t.i for t in shared_tokens)

    shared_text = " ".join(
        t.text for t in sorted(shared_tokens, key=lambda t: t.i)
    )

    # Leading shared context -- whatever precedes the first conjunct
    # (the anchor) belongs to every clause, not just the one that
    # naturally contains it. "of Linux"-style trailing modifiers were
    # already handled above; this is the mirror case for context that
    # comes BEFORE the coordination instead of after it. Example:
    # "What should I eat before and after a workout" -- "What should I
    # eat" sits entirely before the anchor "before", so only the first
    # conjunct's own token range naturally includes it. Without this,
    # later conjuncts come out as bare fragments ("after a workout",
    # no subject/verb) that don't embed anything like the first
    # clause, forcing a false extra dimension in
    # count_distinct_dimensions(). Confirmed: "history and future of
    # Linux" has no leading context (the anchor IS the first token),
    # so this is a no-op there -- verified unchanged against the
    # already-verified noun-coordination case.
    leading_tokens = [doc[i] for i in range(0, items[0].i)]
    leading_text = " ".join(t.text for t in leading_tokens)

    clauses = []
    prev_boundary = -1

    for item in items:

        own_tokens = [
            doc[i]
            for i in range(prev_boundary + 1, item.i + 1)
            if doc[i].i not in shared_ids
            and doc[i].dep_ != "cc"
            and not (doc[i].dep_ == "punct" and doc[i].text == ",")
        ]

        own_text = " ".join(t.text for t in own_tokens)

        if leading_text and leading_text not in own_text:
            own_text = f"{leading_text} {own_text}".strip()

        if shared_text and shared_text not in own_text:
            own_text = f"{own_text} {shared_text}".strip()

        clauses.append(own_text)

        prev_boundary = item.i

    return clauses if len(clauses) > 1 else [text.strip()]


def count_distinct_dimensions(clauses: List[str]) -> int:
    """
    Greedily clusters clauses by embedding similarity. Clauses in the
    same cluster count as one dimension (e.g. "history" and "recent
    developments" both land in a timeline-ish cluster and count as 1,
    matching your Linux example). Clauses in different clusters count
    as separate dimensions.
    """

    if len(clauses) == 1:
        return 1

    embeddings = [get_embedding(c) for c in clauses]

    clusters: List[List[np.ndarray]] = []
    cluster_texts: List[List[str]] = []


    for clause, emb in zip(clauses, embeddings):

        placed = False

        for cluster, texts in zip(clusters, cluster_texts): 

            sim = cosine_similarity(
                np.array(emb).reshape(1, -1),
                np.array(cluster[0]).reshape(1, -1)
            )[0][0]
            shares_subject = (
                _shared_trailing_words(clause, texts[0]) >= MIN_SHARED_TRAILING_WORDS
            ) 
            if sim >= DIMENSION_SIMILARITY_THRESHOLD or shares_subject:

                cluster.append(emb)
                texts.append(clause)
                placed = True
                break

        if not placed:
            clusters.append([emb])
            cluster_texts.append([clause])
    return len(clusters)


@dataclass
class Scope:

    clause_count: int

    distinct_dimension_count: int

    is_coherent: bool

    # Top-1 (single closest exemplar) similarity scores -- see the
    # HISTORY note above BROAD_EXEMPLARS. Used only as a weak,
    # last-resort tie-break in intent_engine.py, never as a primary
    # gate.
    broad_exploration_score: float

    focused_score: float

    focused_question_score: float

    focused_aspect_score: float


def analyze_scope(text: str) -> Scope:
    """
    Analyzer only -- produces metadata about the request's scope.
    Does NOT decide the final intent. The decision engine is
    responsible for combining this with semantic score, evidence,
    and dialogue state.
    """

    text = text.strip()

    clauses = split_clauses(text)

    distinct_dimension_count = count_distinct_dimensions(clauses)

    query_embedding = np.array(
        get_embedding(text)
    ).reshape(1, -1)

    broad_exemplar_embeddings, focused_question_exemplar_embeddings, focused_aspect_exemplar_embeddings = (
        _get_exemplar_embeddings()
    )

    broad_exploration_score = _top1_similarity(
        query_embedding, broad_exemplar_embeddings
    )

    focused_question_score = _top1_similarity(
        query_embedding, focused_question_exemplar_embeddings
    )

    focused_aspect_score = _top1_similarity(
        query_embedding, focused_aspect_exemplar_embeddings
    )

    # Whichever focused shape (direct question or aspect phrase) the
    # query resembles more closely wins -- a query only needs to
    # match ONE focused shape well to count as focused.
    focused_score = max(focused_question_score, focused_aspect_score)

    is_coherent = distinct_dimension_count == 1

    return Scope(

        clause_count=len(clauses),

        distinct_dimension_count=distinct_dimension_count,

        is_coherent=is_coherent,

        broad_exploration_score=broad_exploration_score,

        focused_score=focused_score,

        focused_question_score=focused_question_score,

        focused_aspect_score=focused_aspect_score

    )