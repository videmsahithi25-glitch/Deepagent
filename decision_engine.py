from dataclasses import dataclass


@dataclass
class Decision:

    intent: str

    needs_clarification: bool

    reason: str


def decide_from_evidence(
    evidence,
    meaning
):

    if evidence.sufficient:

        return Decision(

            intent="",

            needs_clarification=False,

            reason=evidence.reason
        )


    return Decision(

        intent="unclear",

        needs_clarification=True,

        reason=evidence.reason
    )