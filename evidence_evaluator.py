from dataclasses import dataclass


@dataclass
class Evidence:

    sufficient: bool

    reason: str


def evaluate_evidence(meaning):

    follow_up = meaning.follow_up.strip()

    if not follow_up:
        return Evidence(
            sufficient=False,
            reason="empty_follow_up"
        )

    if (
        meaning.entity
        and follow_up.lower() == meaning.entity.lower()
    ):
        return Evidence(
            sufficient=False,
            reason="entity_repeated"
        )

    # If meaning resolution successfully attached the
    # follow-up to an existing entity, the clarification
    # has been answered.
    if meaning.has_context:

        return Evidence(
            sufficient=True,
            reason="context_resolved"
        )

    return Evidence(
        sufficient=False,
        reason="needs_more_context"
    )