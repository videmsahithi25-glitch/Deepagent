from dataclasses import dataclass
from typing import Optional
import spacy
from semantic_retriever import find_semantic_intent
from evidence_evaluator import evaluate_evidence
from decision_engine import decide_from_evidence
from scope_analyzer import analyze_scope, has_totality_marker, has_comparison_marker

nlp = spacy.load("en_core_web_sm")

def load_nlp():
    return nlp


@dataclass
class IntentResult:
    intent: str
    confidence: float
    reason: str




def is_complete_request(user_text: str, state):

    doc = load_nlp()(user_text)

    has_verb = any(
        token.pos_ in ("VERB", "AUX")
        for token in doc
    )

    has_subject = any(
        token.dep_ in ("nsubj", "nsubjpass")
        for token in doc
    )

    has_question = any(
        token.tag_ in ("WP", "WRB")
        for token in doc
    )

    root = doc[:].root

    if root.pos_ == "VERB":
        return True

    if has_subject and has_verb:
        return True

    if has_question:
        return True

    return False
def extract_entities(user_text: str):

    doc = load_nlp()(user_text)

    entities = [
        ent.text
        for ent in doc.ents
    ]

    noun_chunks = [
        chunk.text
        for chunk in doc.noun_chunks
    ]

    # Same fix applied to meaning_resolver.py's self-extraction (see
    # that file's comments for the full "world war 2" repro): don't
    # let a non-empty `entities` list win outright just because it's
    # non-empty. spaCy's NER is case-sensitive and can tag only a
    # fragment (e.g. "2" as CARDINAL from lowercase "world war 2")
    # while the much better candidate ("world war") never enters
    # doc.ents at all -- pool both lists and let the strongest
    # (longest) candidate win instead of ents unconditionally
    # shadowing noun_chunks.
    #
    # _fold_trailing_numeral covers a second gap: spaCy's noun_chunk
    # boundary stops at the head noun and excludes a nummod numeral
    # one token to its right (e.g. "world war" | "2"), even though
    # it's grammatically attached to that exact root. Dropping it
    # isn't just imprecise -- it's actively ambiguous (there's more
    # than one "world war") -- so it's folded back in when the next
    # token is a NUM whose head sits inside the span.
    def _fold_trailing_numeral(span):
        end = span.end
        if end < len(doc):
            nxt = doc[end]
            if nxt.pos_ == "NUM" and span.start <= nxt.head.i < span.end:
                return doc[span.start:end + 1].text
        return span.text

    entity_candidates = (
        [_fold_trailing_numeral(ent) for ent in doc.ents]
        + [_fold_trailing_numeral(chunk) for chunk in doc.noun_chunks]
    )

    if entity_candidates:

        entity = max(
            entity_candidates,
            key=len
        )

    else:

        entity = user_text.strip()

    return {

        "entity": entity,

        "entities": entities,

        "noun_chunks": noun_chunks

    }

def is_ambiguous(user_text: str, doc) -> bool:

    has_verb = any(
        token.pos_ in ("VERB", "AUX")
        for token in doc
    )

    has_subject = any(
        token.dep_ in ("nsubj", "nsubjpass")
        for token in doc
    )

    has_entity = len(doc.ents) > 0

    noun_chunks = list(doc.noun_chunks)

    # A complete sentence is not ambiguous
    if has_subject and has_verb:
        return False

    # If the root itself is a verb, the user is expressing an action
    if doc[:].root.pos_ == "VERB":
        return False

    # Only a short noun/entity reference
    if (
        len(noun_chunks) == 1
        and len(doc) <= 4
        and not has_verb
    ):
        return True

    # Named entity without any action
    if has_entity and not has_verb:
        return True

    return False
def pronoun_structure_gate(meaning) -> Optional[str]:
    """
    Rung two of the gate stack (see the SCOPE-BASED GATE comment
    block in decide_intent for rung one). Fires only on contextual
    follow-ups -- i.e. meaning.has_context is True because there's a
    live entity from a prior turn. A pronoun-referenced follow-up
    ("its pricing", "why it lags") is almost always a specific-aspect
    ask -> question_mode, UNLESS it also carries a totality marker
    ("explain it fully", "it as a whole") -> topic_mode.

    This is what generalizes past template vocabulary: it doesn't
    care WHICH aspect word appears, only whether the follow-up is
    pronoun-referenced and whether it's a totality claim. Templates
    can't close that gap because the aspect vocabulary is infinite;
    this rule doesn't need to have seen the phrasing before.

    Returns None (does not fire) when there's no context or no
    pronoun -- callers must fall through to the next signal in that
    case. Intentionally ADDITIVE: it only narrows cases the dimension
    gate doesn't already catch, it never overrides
    scope_dimension_override.
    """

    if not meaning.has_context:
        return None

    doc = load_nlp()(meaning.follow_up)

    has_pronoun = any(
        token.pos_ == "PRON"
        for token in doc
    )

    if not has_pronoun:
        return None

    if has_totality_marker(meaning.follow_up):
        return "topic_mode"

    return "question_mode"
def has_wh_question(text: str) -> bool:
    """
    Same WH-tag check is_complete_request already does internally
    (tag_ in ("WP", "WRB")) but exposed as its own function so it can
    be used as a decision signal, not just buried inside a boolean
    return. is_complete_request computes this and throws it away --
    that's the gap this closes.

    Also treats a literal "?" as an instant match -- a question mark
    is an even harder structural signal than a WH-tag (someone typed
    a question, full stop), no reason to make it depend on spaCy's
    parse of the rest of the sentence.
    """
    if "?" in text:
        return True
    doc = load_nlp()(text)
    return any(
        # WDT added: catches wh-determiner phrasing like "Which tools
        # are used in X" / "What features does X have" (no "?", "Which"/
        # "What" modifying a noun rather than standing alone as WP).
        token.tag_ in ("WP", "WRB", "WDT")
        for token in doc
    )


def wh_question_gate(meaning) -> Optional[str]:
    """
    Rung 2.5 of the gate stack -- consulted only when the dimension
    gate and pronoun gate (rungs one and two) didn't already settle
    it. A grammatical WH-question ("how efficient is X", "why does X
    happen") is a hard structural fact, not a similarity guess -- it
    should not be left to broad_exploration_score vs focused_score,
    which test_exemplar.py measured at only ~4/7 (~57%) reliable.

    Unlike pronoun_structure_gate, this does NOT require has_context
    -- it applies to fresh messages too (e.g. a first message that's
    already a full WH-question with its entity named directly, not
    via a pronoun), which is exactly the case that had no gate
    covering it before this.

    Returns None (does not fire) when there's no WH-word -- callers
    fall through to the semantic tiebreak as before.
    """

    if has_wh_question(meaning.resolved_text):
        return "question_mode"

    return None


# Words/phrases that structurally signal "explore this subject broadly"
# on their own, independent of any pronoun or prior context -- e.g. a
# first message like "History of Rome" or "Evolution of jazz" should go
# straight to topic_mode without a clarify round, the same way a WH-
# question goes straight to question_mode. Starting list, not exhaustive
# -- extend as real queries surface ones this misses.
TOPIC_KEYWORDS = {
    "history", "evolution", "revolution", "origins", "origin",
    "development of", "future of", "overview of", "guide to",
    "complete guide", "everything about", "advantages and disadvantages",
    "pros and cons", "in detail", "from scratch", "deep dive",
    "timeline of", "roadmap of",
}


def has_topic_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in TOPIC_KEYWORDS)


def topic_keyword_gate(meaning) -> Optional[str]:
    """
    Rung 2.6 -- same structural-signal idea as wh_question_gate, mirror
    image for topic_mode: certain vocabulary ("history of X", "evolution
    of X", "everything about X") signals a broad-exploration ask on its
    own, hard enough that it shouldn't be left to the semantic
    classifier or the evidence-based clarify check either. Consulted
    only if none of the earlier rungs (dimension gate, pronoun gate, WH
    gate) already resolved it -- WH-questions take priority since a
    literal question mark or WH-word is a harder signal than vocabulary
    overlap (e.g. "How did X evolve?" is a question despite containing
    "evolve").

    NOTE: this function existed in the previous version of this file
    too, but was never actually called from decide_intent -- it was
    dead code. It's wired into the gate stack below now.
    """
    if has_topic_keyword(meaning.resolved_text):
        return "topic_mode"

    return None
def decide_intent(meaning):

    doc = load_nlp()(meaning.resolved_text)

    semantic_result = find_semantic_intent(
        meaning.resolved_text
    )
    print("\n===== SEMANTIC RESULT =====")
    print(semantic_result)
    best = semantic_result["top_match"]

    confidence = best["score"]

    evidence = evaluate_evidence(
        meaning
    )

    print("\n===== EVIDENCE =====")
    print(evidence)
    decision = decide_from_evidence(
        evidence,
        meaning
    )
    print("\n===== DECISION =====")
    print(decision)
    is_complete = is_complete_request(
        meaning.resolved_text,
        None
    )

    ambiguous = is_ambiguous(
        meaning.resolved_text,
        doc
    )

    scope = analyze_scope(
        meaning.resolved_text
    )

    print("\n===== SCOPE =====")
    print(scope)

    # -----------------------------------------------------------------
    # SCOPE-BASED GATE (fix b).
    #
    # scope.distinct_dimension_count is the ONE scope signal verified
    # correct against test_scope.py (2-item and 4-item Linux
    # coordinations both correctly cluster to 1 dimension; only known
    # miss is true entity comparisons like "Compare Python and Java",
    # which is a documented, separate limitation). When it reports
    # more than one independent dimension, that alone is a reliable
    # topic_mode signal -- e.g. "history, architecture, security, and
    # future of Linux" -- and overrides the semantic classifier's vote
    # outright, regardless of what the embedding-based intent
    # classifier says.
    #
    # scope.broad_exploration_score / scope.focused_score are NOT
    # gated on. test_exemplar.py showed that even after four rounds
    # of fixes (centroid -> split centroid -> top-3 -> top-1 ->
    # exemplar removal) they still only separate real broad queries
    # from focused questions on roughly 4/7 known cases, with the
    # errors migrating between exemplars rather than resolving --
    # evidence this is a structural limitation of the embedding
    # comparison, not a fixable wording bug. They're used only as a
    # last-resort tie-break, and only when BOTH:
    #   (a) distinct_dimension_count == 1, so dimension count can't
    #       distinguish a single focused question from a single broad
    #       ask (e.g. "Docker" vs "explain Docker" vs "complete guide
    #       to Docker" are all 1 dimension), AND
    #   (b) the semantic classifier's own margin is thin, meaning it
    #       wasn't confident either.
    # If either the dimension gate or a confident semantic vote
    # already settled it, the broad/focused scores are not consulted
    # at all -- they can only ever break a tie, never override a
    # signal that already spoke.
    #
    # SEMANTIC_MARGIN_THRESHOLD / SCOPE_TIEBREAK_MARGIN are starting
    # guesses, not measured values -- tune against real logged
    # queries once you have volume, the same way
    # DIMENSION_SIMILARITY_THRESHOLD in scope_analyzer.py is flagged
    # as a starting guess.
    # -----------------------------------------------------------------

    SEMANTIC_MARGIN_THRESHOLD = 0.05
    SCOPE_TIEBREAK_MARGIN = 0.03

    # has_comparison_marker guards against the exact gap this file's own
    # comment above documented as accepted-but-unfixed: coordinated
    # ENTITIES ("a list and a tuple") parse identically to coordinated
    # ASPECTS of one topic ("history and future of Linux"), so
    # distinct_dimension_count alone can't tell "compare these two things"
    # (one coherent question) from "explore these two independent topics"
    # (genuinely two dimensions). Confirmed as a real miss: "the difference
    # between a list and a tuple in Python" scored distinct_dimension_count
    # =2 and was forced into topic_mode before this check existed.
    scope_dimension_override = (
        "topic_mode"
        if scope.distinct_dimension_count > 1
        and not has_comparison_marker(meaning.resolved_text)
        else None
    )

    # Rung two: pronoun-structure gate. Only consulted if the
    # dimension gate (rung one) didn't already settle it. See
    # pronoun_structure_gate's docstring for why this generalizes
    # past template vocabulary where the totality-marker check alone
    # wouldn't.
    pronoun_gate_intent = (
        pronoun_structure_gate(meaning)
        if scope_dimension_override is None
        else None
    )

    # Rung 2.5: WH-question gate. Only consulted if neither the
    # dimension gate nor the pronoun gate already settled it. See
    # wh_question_gate's docstring for why this covers a case the
    # pronoun gate structurally can't (fresh messages with no
    # pronoun, e.g. "How efficient is Quantum Computing").
    wh_gate_intent = (
        wh_question_gate(meaning)
        if scope_dimension_override is None and pronoun_gate_intent is None
        else None
    )

    # Rung 2.6: topic-keyword gate. Only consulted if none of the
    # dimension/pronoun/WH gates already settled it. This gate was
    # DEFINED in the previous file version but never actually called
    # here -- that's why "history of X" / "evolution of X"-style
    # direct topics were still falling through to the clarify check
    # even after topic_keyword_gate was written.
    topic_gate_intent = (
        topic_keyword_gate(meaning)
        if scope_dimension_override is None
        and pronoun_gate_intent is None
        and wh_gate_intent is None
        else None
    )

    scope_tiebreak_used = False
    scope_tiebreak_intent = None

    if (
        scope_dimension_override is None
        and pronoun_gate_intent is None
        and wh_gate_intent is None
        and topic_gate_intent is None
        and semantic_result["margin"] < SEMANTIC_MARGIN_THRESHOLD
    ):

        score_gap = scope.broad_exploration_score - scope.focused_score

        if score_gap > SCOPE_TIEBREAK_MARGIN:
            scope_tiebreak_intent = "topic_mode"
            scope_tiebreak_used = True
        elif -score_gap > SCOPE_TIEBREAK_MARGIN:
            scope_tiebreak_intent = "question_mode"
            scope_tiebreak_used = True

    resolved_intent = (
        scope_dimension_override
        or pronoun_gate_intent
        or wh_gate_intent
        or topic_gate_intent
        or scope_tiebreak_intent
        or best["intent"]
    )

    # -----------------------------------------------------------------
    # THE ACTUAL BUG (this is what was still causing follow-ups on
    # direct questions/topics after the gates were added).
    #
    # decision.needs_clarification comes from evaluate_evidence /
    # decide_from_evidence -- a completely separate evidence-based
    # check that has no visibility into whether a deterministic gate
    # above already fired. Previously, final_intent unconditionally
    # deferred to decision.needs_clarification:
    #
    #     final_intent = "unclear" if decision.needs_clarification
    #                    else resolved_intent
    #
    # So even when wh_gate_intent (or topic_gate_intent, or
    # scope_dimension_override) correctly resolved the intent on
    # structural grounds, the evidence evaluator could independently
    # say "not enough context" and stomp it back to "unclear" --
    # triggering a follow-up anyway. The gate's decision was being
    # computed and then silently discarded.
    #
    # Fix: a deterministic gate firing means the structural evidence
    # IS the evidence -- it should not be second-guessed by the
    # separate evidence/clarification check. scope_tiebreak_intent is
    # deliberately NOT included in gate_fired: it's a soft, score-
    # based tiebreak (see the tiebreak block above), not a hard
    # structural signal, so it stays subject to the clarify check like
    # the plain semantic fallback does.
    # -----------------------------------------------------------------

    gate_fired = (
        scope_dimension_override is not None
        or pronoun_gate_intent is not None
        or wh_gate_intent is not None
        or topic_gate_intent is not None
    )

    # hard_gate_fired deliberately EXCLUDES the WH gate BY DEFAULT. The
    # dimension/pronoun/topic-keyword gates are structural signals about
    # the sentence's SHAPE -- firing means the shape itself already
    # answers "was this understood well enough", so it's fair to
    # suppress the evidence-based clarification check. The WH gate is
    # different: it only tells you this is a question rather than a
    # topic-exploration request -- it says nothing on its own about
    # whether the question was understood well enough to answer well.
    # A WH-phrased question (e.g. "Which tools are used in X") should
    # still be able to trigger a clarify round if something in it is
    # genuinely ambiguous.
    #
    # BUG FIX: "genuinely ambiguous" was supposed to be judged by
    # is_complete / is_ambiguous (computed above) -- but neither was
    # ever actually consulted anywhere in this function; both were
    # computed and only written into `reasoning` for logging. The
    # ONLY thing standing between a WH-gated question and a forced
    # clarify round was evaluate_evidence's `has_context` check, which
    # tests something unrelated to ambiguity: "was there a live entity
    # carried over from a PRIOR turn". That's true on nearly every
    # first message of a conversation by construction, regardless of
    # how complete the question is -- so a fully-formed, unambiguous
    # question like "Which university did Sahithi get her B.Tech
    # from" was still being sent to clarify on turn one. Confirmed
    # repro in debug_log.txt.
    #
    # Fix: a WH-gated question now also bypasses the evidence-based
    # clarification check when it's structurally complete
    # (is_complete_request) AND not flagged ambiguous
    # (is_ambiguous) -- i.e. when there's an actual structural signal
    # of clarity, not just the absence of prior-turn context. A
    # WH-question that IS incomplete or genuinely ambiguous still
    # falls through to the evidence check (and to
    # check_verb_fusion_ambiguity in fix.py) exactly as before --
    # this only stops well-formed questions from being clarified for
    # no reason other than being the first message in the
    # conversation.
    wh_gate_confident = (
        wh_gate_intent is not None
        and is_complete
        and not ambiguous
    )

    hard_gate_fired = (
        scope_dimension_override is not None
        or pronoun_gate_intent is not None
        or topic_gate_intent is not None
        or wh_gate_confident
    )

    reasoning = {

        "semantic_intent": best["intent"],

        "semantic_confidence": confidence,

        "semantic_margin": semantic_result["margin"],

        "is_complete": is_complete,

        "is_ambiguous": ambiguous,

        "has_context": meaning.has_context,

        "clarification_depth": meaning.clarification_depth,

        "evidence": evidence.reason,

        "needs_clarification": decision.needs_clarification,

        "reason": decision.reason,

        "scope_clause_count": scope.clause_count,

        "scope_distinct_dimension_count": scope.distinct_dimension_count,

        "scope_is_coherent": scope.is_coherent,

        "scope_broad_exploration_score": scope.broad_exploration_score,

        "scope_focused_score": scope.focused_score,

        "scope_focused_question_score": scope.focused_question_score,

        "scope_focused_aspect_score": scope.focused_aspect_score,

        "scope_dimension_override": scope_dimension_override,

        "pronoun_gate_intent": pronoun_gate_intent,

        "wh_gate_intent": wh_gate_intent,

        "wh_gate_confident": wh_gate_confident,

        "topic_gate_intent": topic_gate_intent,

        "gate_fired": gate_fired,

        "hard_gate_fired": hard_gate_fired,

        "scope_tiebreak_used": scope_tiebreak_used,

        "scope_tiebreak_intent": scope_tiebreak_intent,

        "resolved_intent": resolved_intent
    }


    final_intent = (
        resolved_intent
        if hard_gate_fired
        else (
            "unclear"
            if decision.needs_clarification
            else resolved_intent
        )
    )


    return {

        "intent": final_intent,

        "confidence": confidence,

        "needs_clarification": decision.needs_clarification and not hard_gate_fired,

        "reasoning": reasoning,

        "semantic_result": semantic_result,

        "scope": scope

    }
def update_dialogue_state(
    state,
    intent_result,
    meaning
):

    state.last_intent = intent_result["intent"]

    state.last_user_message = meaning.follow_up

    if meaning.entity is not None:

        state.current_entity = meaning.entity

    if intent_result["needs_clarification"]:

        state.pending_clarification = True

        state.pending_entity = state.current_entity

        state.clarification_count += 1

    else:

        state.pending_clarification = False

        state.pending_entity = None

        state.clarification_count = 0

    return state






