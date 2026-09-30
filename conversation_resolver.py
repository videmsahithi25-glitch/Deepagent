import re
from dataclasses import dataclass
from typing import Optional

import spacy

from meaning_resolver import resolve_meaning


# Load spaCy once
nlp = spacy.load("en_core_web_sm")


# fix.py's classify_intent() always passes the FULL combined text
# (original question + "(Reply: ...)") into resolve_context on every
# clarify round, not just the bare new reply -- a deliberate earlier
# fix so decide_intent's semantic classifier keeps full-question
# context (see fix.py's classify_intent docstring, the
# `query_for_routing` bug fix).
#
# But can_resolve_with_context / is_pronoun_followup below use
# doc.ents / has_new_subject to decide "did the user say something
# NEW, or are they just continuing the clarification". Since the
# ORIGINAL question is always still sitting in that combined text,
# its named entities (e.g. "Sahithi") and its subject kept tripping
# has_named_entity / has_new_subject on every round, regardless of
# what the actual reply said -- so can_resolve_with_context returned
# False for every clarify reply to a question that happened to
# contain a named entity. That fed back into resolve_meaning as
# entity=None on round 2 exactly like round 1, so the clarify never
# resolved (confirmed repro: "Which university did Sahithi get her
# B.Tech from" -> reply "university name" -> still needs_more_context,
# forced after MAX_CLARIFY_ROUNDS).
#
# Extract just the "(Reply: ...)" segment for these structural checks
# when present, so they inspect what the user actually just typed,
# not the original question riding along with it. Falls back to the
# whole text when there's no "(Reply: ...)" suffix (a fresh,
# non-clarify message, e.g. round 1) -- unchanged behavior for that
# case. Does NOT touch resolve_meaning()'s own `follow_up` argument
# anywhere below -- that still gets the full combined text, same as
# before, so decide_intent's semantic classification keeps the full
# context it was fixed to rely on.
_REPLY_SEGMENT_RE = re.compile(r"\(Reply:\s*(.*?)\s*\)\s*$", re.DOTALL)


def _extract_latest_segment(text: str) -> str:
    match = _REPLY_SEGMENT_RE.search(text)
    return match.group(1).strip() if match else text


@dataclass
class DialogueState:

    current_entity: Optional[str] = None

    last_intent: Optional[str] = None

    user_goal: Optional[str] = None

    last_agent_action: Optional[str] = None

    pending_clarification: bool = False

    pending_entity: Optional[str] = None

    last_user_message: Optional[str] = None

    clarification_count: int = 0


def update_entity(state: DialogueState, entity: str) -> DialogueState:

    state.current_entity = entity
    state.pending_entity = entity

    return state

# spaCy tags interrogative words ("what", "who", "which", "whom") as
# pos_ == "PRON" too (fine-grained tag WP/WP$/WDT). They are NOT
# anaphoric back-references -- "what are emergency procedures" contains
# no reference to any prior entity, it IS the question. Treating a
# wh-word as evidence of "this continues the prior entity" was the root
# cause of a real bug: after a turn about "Redis", a fresh unrelated
# question like "what are emergency procedures" was misread as a
# pronoun follow-up and got the stale entity spliced in, producing the
# nonsense resolved text "Redis are emergency procedures". Excluded
# here, and must also be excluded anywhere else pos_ == "PRON" is used
# for this same "is this a back-reference" purpose (see
# meaning_resolver._substitute_pronoun, fixed the same way).
WH_TAGS = {"WP", "WP$", "WDT"}


def _has_anaphoric_pronoun(doc) -> bool:
    return any(
        token.pos_ == "PRON" and token.tag_ not in WH_TAGS
        for token in doc
    )


def can_resolve_with_context(
    user_text: str,
    state: DialogueState
) -> bool:

    # If we are not waiting for clarification,
    # there is nothing to resolve.
    if not state.pending_clarification:
        return False

    # Check only the latest reply segment, not the original question
    # that's riding along in the same combined string -- see
    # _extract_latest_segment's docstring above for why.
    doc = nlp(_extract_latest_segment(user_text))

    has_named_entity = len(doc.ents) > 0

    has_pronoun = _has_anaphoric_pronoun(doc)

    # A real subject excludes pronouns like it/this/that
    has_new_subject = any(
        token.dep_ in ("nsubj", "nsubjpass")
        and token.pos_ != "PRON"
        for token in doc
    )

    # User mentioned a completely new entity.
    if has_named_entity:
        return False

    # User introduced a brand-new subject.
    if has_new_subject:
        return False

    # Everything else is assumed to continue the clarification.
    return state.current_entity is not None

def resolve_context(
    user_text,
    state
):

    if can_resolve_with_context(
        user_text,
        state
    ):

        return resolve_meaning(

            entity=state.current_entity,

            follow_up=user_text,

            clarification_depth=state.clarification_count

        )

    # Not a clarification reply. Only inherit state.pending_entity when
    # this message is itself a genuine pronoun-style continuation ("how
    # does it work", "what about that") -- previously this branch handed
    # pending_entity through UNCONDITIONALLY, so any fresh, self-
    # contained message (no pronoun at all, e.g. a brand-new question
    # about an uploaded document) could still get a stale entity from
    # several turns ago spliced into its resolved text. A message with
    # no anaphoric pronoun stands on its own; meaning_resolver.
    # resolve_meaning already extracts its own entity from the text
    # when entity=None (see its "if not entity:" branch), so this is
    # safe to leave unset here.
    # Same reasoning as can_resolve_with_context above -- check only
    # the latest segment, not the original question also present in
    # the combined text.
    doc = nlp(_extract_latest_segment(user_text))
    is_pronoun_followup = (
        _has_anaphoric_pronoun(doc)
        and len(doc.ents) == 0
        and not any(
            token.dep_ in ("nsubj", "nsubjpass") and token.pos_ != "PRON"
            for token in doc
        )
    )

    return resolve_meaning(

    entity=state.pending_entity if is_pronoun_followup else None,

    follow_up=user_text,

    clarification_depth=state.clarification_count

)