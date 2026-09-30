from dataclasses import dataclass
from typing import Optional

import spacy
from keybert import KeyBERT

from scope_analyzer import has_comparison_marker


# Load spaCy once
nlp = spacy.load("en_core_web_sm")

# Load KeyBERT once. spaCy still generates entity CANDIDATES below
# (doc.ents, _adjacent_noun_runs) and still does everything else in this
# file -- pronoun substitution, predicate/subject/focus parsing. KeyBERT
# only replaces the final "which candidate wins" decision, which used to
# be a grammar-role tier + character-length tiebreak. That tiebreak is
# what caused "similarity search" vs "vector databases" to flip winners
# across near-identical phrasings of the same question -- it had no
# connection to meaning, only to sentence structure. KeyBERT scores each
# candidate by semantic closeness to the whole sentence instead, which
# doesn't depend on which grammatical role the phrase happened to land in.
#
# KeyBERT() loads its embedding model the moment it's constructed (not
# lazily on first use) -- so if that load fails (no network, model not
# cached), it fails HERE, at import time, and takes down every import of
# this module with it, before the fallback logic inside resolve_meaning
# ever gets a chance to run. Guarded so a KeyBERT/model outage degrades
# to the grammar-role heuristic instead of crashing the whole file.
try:
    kw_model = KeyBERT()
except Exception as _keybert_load_error:
    print(f"[KeyBERT model load failed, will fall back to grammar-role heuristic: {_keybert_load_error}]")
    kw_model = None


from dataclasses import dataclass
from typing import Optional


@dataclass
class Meaning:

 entity: Optional[str]

 follow_up: Optional[str]

 resolved_text: str

 has_context: bool

 complete: bool

 clarification_depth: int

 focus: Optional[str] = None

 predicate: Optional[str] = None

 subject: Optional[str] = None


POSSESSIVE_PRONOUNS = {"its", "his", "her", "their", "your", "my", "our"}


# spaCy tags interrogative words ("what", "who", "which", "whom") as
# pos_ == "PRON" too (fine-grained tag WP/WP$/WDT). These are NOT
# anaphoric back-references to a prior entity -- they're the question's
# own grammatical core. Without this exclusion, a fresh, self-contained
# question like "what are emergency procedures" would have its "what"
# swapped straight out for a stale carried-over entity (e.g. "Redis"),
# producing nonsense like "Redis are emergency procedures". See the
# matching exclusion in conversation_resolver.py's has_pronoun check --
# both must stay in sync since they guard the same failure mode.
_WH_TAGS = {"WP", "WP$", "WDT"}


def _substitute_pronoun(follow_up: str, doc, entity: str):
    """Replace the first anaphoric pronoun in follow_up with entity,
    instead of just prepending entity in front of the whole sentence.
    Prepending produced text like "eclipse its formation" for
    follow_up="its formation" -- grammatically broken (a dangling,
    still-unresolved "its" sitting right next to the entity), which
    downstream even confused the clarify-question LLM into asking what
    "its" refers to, despite "eclipse" technically being present in the
    string. Possessive pronouns ("its", "their"...) become "entity's"
    ("eclipse's formation"); all other anaphoric pronouns ("it", "this",
    "that"...) are replaced outright ("it form" -> "eclipse form").
    Wh-words ("what", "who", "which") are deliberately skipped -- they
    ask the question, they don't refer back to anything. Returns
    (text, True) if a pronoun was found and replaced, else
    (follow_up, False) so the caller can fall back to prepending for
    pronoun-free follow-ups (e.g. "explain more"), or (in
    resolve_meaning's caller) avoid touching the text at all when no
    entity should have been carried over in the first place.
    """
    pieces = []
    replaced = False

    for token in doc:
        text = token.text
        if not replaced and token.pos_ == "PRON" and token.tag_ not in _WH_TAGS:
            lower = text.lower()
            text = f"{entity}'s" if lower in POSSESSIVE_PRONOUNS else entity
            replaced = True
        pieces.append(text + token.whitespace_)

    return "".join(pieces).strip(), replaced


def resolve_meaning(
    entity: Optional[str],
    follow_up: str,
    clarification_depth: int = 0
) -> Meaning:

    follow_up = follow_up.strip()

    doc = nlp(follow_up)

    predicate = None
    focus = None
    subject = None

    for token in doc:

        if token.dep_ == "ROOT":
            
         if token.pos_ in ("VERB", "AUX"):

             predicate = token.lemma_
        if token.dep_ in ("nsubj", "nsubjpass"):

         subject = token.text
    focus_candidates = []

    for chunk in doc.noun_chunks:
        

        focus_candidates.append(chunk.text)

    for token in doc:

        if (
            token.pos_ in ("NOUN", "PROPN", "ADJ")
            and token.text not in focus_candidates
        ):

            focus_candidates.append(token.text)

    focus = focus_candidates[0] if focus_candidates else None

    self_extracted = False

    if not entity:
        # No entity carried over from DialogueState (fresh turn, or
        # context was never established -- e.g. the very first message
        # of a session). Previously this left `entity` as None
        # unconditionally, which meant state.current_entity never got
        # seeded on turn 1, and every later pronoun ("it") had nothing
        # to resolve against -- resolved_text stayed exactly equal to
        # follow_up forever. Extract straight from THIS text instead,
        # same priority intent_engine.extract_entities() already uses:
        # named entities first, then noun chunks.
        # Pool BOTH sources instead of letting doc.ents short-circuit
        # noun_chunks. spaCy's NER is case-sensitive -- "World War 2"
        # tags as one EVENT entity, but "world war 2" (lowercase) only
        # tags the trailing "2" as a CARDINAL, with "world war" never
        # entering doc.ents at all. Under the old "ents, else
        # noun_chunks" logic, a non-empty-but-low-quality ents list
        # (e.g. just "2") permanently won over a much better noun
        # chunk ("world war"), corrupting state.current_entity for
        # every later pronoun-resolved turn in the conversation.
        #
        # _fold_trailing_numeral handles a second, separate gap: even
        # the noun chunk "world war" drops the "2" entirely, because
        # spaCy's noun_chunk boundary stops at the head noun ("war")
        # and doesn't include a nummod modifier one token to its
        # right -- even though "2" is grammatically attached to that
        # exact root. Dropping it isn't just imprecise, it's actively
        # ambiguous (there is more than one "world war"), so it's
        # folded back in whenever the token right after the span is a
        # NUM whose head is inside the span.
        def _fold_trailing_numeral(span):
            end = span.end
            if end < len(doc):
                nxt = doc[end]
                if nxt.pos_ == "NUM" and span.start <= nxt.head.i < span.end:
                    return doc[span.start:end + 1]
            return span

        # doc.noun_chunks (spaCy's dependency-based noun phrases) turned
        # out to fragment unpredictably on short/terse queries: the exact
        # two-word phrase "vector databases" forms as ONE noun_chunk in
        # some sentences but splits into two separate one-word chunks in
        # others, purely because of how the surrounding words happen to
        # attach in that sentence's parse tree -- confirmed by inspecting
        # real parses of "what are vector databases" (one chunk) vs
        # "what are the uses of vector databases" (splits into 'vector'
        # and 'databases' separately). That fragmentation was silently
        # corrupting concept_history/domain-preference tracking upstream
        # in intent_rag.py: the same real concept kept producing
        # different extracted strings turn to turn.
        #
        # _adjacent_noun_runs replaces noun_chunks as a candidate source:
        # it merges consecutive NOUN/PROPN/ADJ *tokens* by raw position,
        # ignoring dependency labels entirely. This is more stable
        # because it only depends on POS tags (which stay consistent
        # across these sentences) rather than the dependency tree (which
        # doesn't).
        def _adjacent_noun_runs():
            runs = []
            start = None
            for i, tok in enumerate(doc):
                if tok.pos_ in ("NOUN", "PROPN", "ADJ"):
                    if start is None:
                        start = i
                else:
                    if start is not None:
                        runs.append(doc[start:i])
                        start = None
            if start is not None:
                runs.append(doc[start:len(doc)])
            return runs

        # _ENTITY_ROLE_TIER / _entity_candidate_score are kept as a
        # FALLBACK ONLY now -- used solely if the KeyBERT call below
        # raises (model unreachable, runtime error, etc.), so a KeyBERT
        # outage degrades to the previous behavior instead of crashing.
        # The primary selection now uses KeyBERT: instead of ranking
        # candidates by grammar role + character length (which had no
        # relationship to meaning -- 'similarity search' (17 chars) beat
        # 'vector databases' (16 chars) purely by being one character
        # longer, in a sentence that's actually about vector databases),
        # KeyBERT scores each candidate by semantic closeness to the
        # FULL sentence, using the same embedding-based comparison
        # already used elsewhere in this pipeline for Chroma. It doesn't
        # depend on which grammatical role a candidate happened to land
        # in, so it isn't sensitive to the same sentence-structure
        # differences that made the old rule flip winners across
        # near-identical phrasings of the same question.
        _ENTITY_ROLE_TIER = {"pobj": 0, "ROOT": 1, "appos": 2}

        def _entity_candidate_score(span):
            tier = _ENTITY_ROLE_TIER.get(span.root.dep_, 3)
            return (tier, -len(span.text))

        entity_candidates = (
            [_fold_trailing_numeral(ent) for ent in doc.ents]
            + [_fold_trailing_numeral(run) for run in _adjacent_noun_runs()]
        )

        # doc.ents and _adjacent_noun_runs can each surface a span that's
        # fully CONTAINED inside a span the other one produced -- e.g. for
        # "AI Agent vs Agentic AI", doc.ents tags a bare "AI" (GPE) at
        # tokens [5:6), while _adjacent_noun_runs already produced the
        # larger, more correct "AI Agent" at [5:7) covering it. Confirmed
        # via direct spaCy inspection (doc.ents start/end vs noun-run
        # start/end). Left unresolved, this silently inflated the
        # candidate count by counting the same real entity twice under
        # different strings, which is what made the len(...) == 2 check
        # below (an earlier version of this fix) never fire for that
        # exact query -- it saw 3 raw candidates ('AI', 'AI Agent',
        # 'Agentic AI'), not the real 2. Keep only the longer/outer span
        # whenever one candidate's token range is entirely inside
        # another's.
        def _dedupe_overlapping(spans):
            ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
            kept = []
            for s in ordered:
                if any(k.start <= s.start and s.end <= k.end for k in kept):
                    continue
                kept.append(s)
            return kept

        entity_candidates = _dedupe_overlapping(entity_candidates)

        # Comparison-style queries ("what is the difference between X and
        # Y", "how does X compare to Y") always surface the comparison
        # word itself ("difference", "comparison") as a candidate
        # alongside the real entities being compared. Nothing scored this
        # candidate pool by semantic role, so KeyBERT was free to rank
        # the generic anchor word above either actual entity -- it's a
        # single word describing the whole sentence's intent, while the
        # two real entities split relevance between them. Confirmed
        # repro: "what is difference between AI agents and agentic AI"
        # resolved entity='difference' instead of either real term.
        #
        # Also covers evaluative-angle nouns ("pros", "cons", "benefits",
        # "drawbacks"...) for the SAME reason as the comparison words
        # above: "compare the pros and cons of remote work" is not a
        # two-entity comparison -- it's one entity ("remote work")
        # examined from two angles. Confirmed via real parse: "pros" and
        # "cons" are conjuncts of EACH OTHER (dep_="conj"), while "remote
        # work" sits in its own, unrelated prepositional phrase -- the
        # exact same sentence shape as "tuple and list in a python" vs
        # "python" below, but here the conjunct pair is the thing to
        # DISCARD, not preserve, since neither word names a real entity.
        # Filtering them out here (same as "difference") means this
        # query never even reaches the comparison-join logic below --
        # it resolves through the ordinary single-candidate path once
        # "remote work" is the only thing left.
        #
        # Filtered out before scoring, not after -- if it were only
        # deprioritized post-hoc, it could still win on a tiebreak.
        # Only drops it if OTHER candidates remain; a bare single-word
        # message like "difference" alone (no other candidate) still
        # needs to resolve to something, so the filter never empties the
        # pool completely.
        _GENERIC_COMPARISON_WORDS = {
            "difference", "differences", "comparison", "comparisons",
            "distinction", "distinctions", "contrast", "contrasts",
            "similarity", "similarities", "relationship", "relationships",
            "pros", "cons", "benefits", "drawbacks", "advantages",
            "disadvantages", "upsides", "downsides",
        }
        _non_generic_candidates = [
            c for c in entity_candidates
            if c.text.lower() not in _GENERIC_COMPARISON_WORDS
        ]
        if _non_generic_candidates:
            entity_candidates = _non_generic_candidates

        # Finds candidates that are directly coordinated with each other
        # ("tuple AND list") via spaCy's conj dependency -- this is the
        # actual grammatical signal for "these are the two things being
        # compared", and it correctly EXCLUDES a candidate like "python"
        # that merely co-occurs elsewhere in the same sentence ("in a
        # python") without being part of the coordination. Confirmed via
        # real parse: tuple.conjuncts == [list], list.conjuncts ==
        # [tuple], python.conjuncts == [] (python is pobj of "in", not
        # conjoined to anything). This is what makes it possible to tell
        # "tuple and list" apart from "tuple and list in a python" even
        # though the second has 3 leftover candidates, not 2.
        def _conjunct_linked(candidates):
            roots = {c.root.i: c for c in candidates}
            return [
                c for c in candidates
                if any(t.i in roots for t in c.root.conjuncts)
            ]

        if entity_candidates:
            if len(entity_candidates) == 1:
                entity = entity_candidates[0].text
            elif has_comparison_marker(follow_up) and (
                _conjunct_linked(entity_candidates)
                or len(entity_candidates) == 2
            ):
                # Comparison query ("difference between tuple and list",
                # "AI Agent vs Agentic AI") -- forcing a single KeyBERT
                # winner here silently drops one side of the comparison.
                # Confirmed via two independent real sessions in
                # debug_log.txt (only 'tuple' tracked, 'list' dropped;
                # only 'Agentic AI' tracked, 'AI Agent' dropped).
                # Preserve both as one compound entity string instead of
                # picking a winner -- keeps Meaning.entity a plain str,
                # so conversation_resolver.py's current_entity,
                # evidence_evaluator.py's entity_repeated check, and
                # concept_history bucketing all keep working unchanged.
                #
                # Two ways to confirm this is a genuine 2-entity
                # comparison, not just "a comparison marker is present
                # somewhere in the sentence":
                #   1. A conj-linked pair exists ("tuple and list") --
                #      preferred, since it also correctly drops unrelated
                #      leftover candidates like "python".
                #   2. No conj pair, but exactly 2 real candidates remain
                #      after the generic-word filter -- covers "X vs Y"
                #      phrasing, where spaCy does NOT create a conj edge
                #      between X and Y (confirmed via real parse: "vs" is
                #      tagged as a preposition, not a coordinator).
                # If neither condition holds (3+ unresolved candidates,
                # no conj link -- genuinely ambiguous), falls through to
                # KeyBERT below rather than guessing.
                pair = _conjunct_linked(entity_candidates)
                chosen = pair if len(pair) >= 2 else entity_candidates
                candidate_texts = list(dict.fromkeys(
                    span.text for span in chosen
                ))
                entity = " and ".join(candidate_texts)
            else:
                # dedupe candidate strings, preserving first-seen order,
                # since doc.ents and _adjacent_noun_runs can both surface
                # the same span text
                candidate_texts = list(dict.fromkeys(
                    span.text for span in entity_candidates
                ))
                if kw_model is None:
                    # Model failed to load at import time -- go straight
                    # to the fallback, no point attempting the call.
                    entity = min(entity_candidates, key=_entity_candidate_score).text
                else:
                    try:
                        # keyphrase_ngram_range defaults to (1, 1) in
                        # KeyBERT -- confirmed against KeyBERT's own
                        # source (keybert/_model.py): it's fed straight
                        # into CountVectorizer(ngram_range=...,
                        # vocabulary=candidates), which only extracts
                        # SINGLE-WORD n-grams from the sentence to test
                        # against the candidate vocabulary. A two-word
                        # candidate like "gradient descent" can then
                        # never match anything -- there's no 2-word
                        # chunk of the sentence being generated to
                        # compare it against, so it's silently dropped
                        # from consideration entirely, not just scored
                        # low. Confirmed empirically: candidates
                        # ['gradient descent', 'local minima'] (both
                        # 2-word) returned an EMPTY ranked list; mixed
                        # candidates like ['Python code', 'gradient
                        # descent', 'scratch'] only ever returned the
                        # single-word one ('scratch'), regardless of
                        # actual relevance. Must set the range to span
                        # the real word-count of the generated
                        # candidates, or multi-word candidates are
                        # never even in the running.
                        candidate_word_counts = [len(c.split()) for c in candidate_texts]
                        ngram_range = (min(candidate_word_counts), max(candidate_word_counts))
                        ranked = kw_model.extract_keywords(
                            follow_up,
                            candidates=candidate_texts,
                            keyphrase_ngram_range=ngram_range,
                            top_n=1,
                        )
                        entity = ranked[0][0] if ranked else min(
                            entity_candidates, key=_entity_candidate_score
                        ).text
                    except Exception as e:
                        print(
                            f"[KeyBERT extraction failed ({e}), falling back "
                            "to grammar-role heuristic]"
                        )
                        entity = min(entity_candidates, key=_entity_candidate_score).text
            self_extracted = True
        elif follow_up:
            # Both NER and noun-run detection came back empty -- this
            # happens for bare single words with no surrounding context
            # (e.g. "eclipse" alone gets POS-tagged VERB by spaCy, an
            # imperative reading like "eclipse the competition", since
            # there's nothing else in the sentence to disambiguate it --
            # neither doc.ents nor a NOUN/PROPN/ADJ run touches VERB-
            # tagged tokens at all). Without this, entity stayed None for
            # exactly the inputs most likely to BE a bare entity name,
            # which is the opposite of what we want. Same last-resort
            # extract_entities() in intent_engine.py already uses.
            entity = follow_up
            self_extracted = True

    if entity:

        # If the entity was just self-extracted FROM this follow_up
        # (e.g. follow_up itself is "eclipse"), it's already present in
        # the text -- adding it again would produce "eclipse eclipse".
        # Otherwise, the entity came from prior dialogue context (this
        # follow_up is a reply like "its formation" or "how does it
        # form" that doesn't itself name the entity) -- prefer
        # substituting the actual pronoun over blindly prepending, since
        # prepending left a grammatically dangling, still-unresolved
        # pronoun sitting right next to the entity. Only prepend when
        # there's no pronoun in the follow_up to substitute.
        if self_extracted:
            resolved_text = follow_up
        else:
            substituted, replaced = _substitute_pronoun(follow_up, doc, entity)
            resolved_text = substituted if replaced else f"{entity} {follow_up}"

        return Meaning(

            entity=entity,

            follow_up=follow_up,

            resolved_text=resolved_text,

            # has_context specifically means "there's a live entity
            # carried over from a prior turn" (see
            # pronoun_structure_gate's docstring) -- a self-extracted
            # entity from THIS turn's own text isn't prior context, so
            # it must stay False here or the pronoun gate would
            # misfire on fresh, context-free messages.
            has_context=not self_extracted,

            complete=True,

            clarification_depth=clarification_depth,

            focus=focus,

            predicate=predicate,

            subject=subject
        )

    return Meaning(

        entity=None,

        follow_up=follow_up,

        resolved_text=follow_up,

        has_context=False,

        complete=False,

        clarification_depth=clarification_depth,

        focus=focus,

        predicate=predicate,

        subject=subject
    )