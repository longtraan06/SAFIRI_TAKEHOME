from __future__ import annotations

import pandas as pd


def recommendations(snapshot: pd.Series, probability: float) -> list[str]:
    actions: list[str] = []
    if probability >= 0.29:
        actions.append(
            "Escalate operational follow-up for the predicted material-delay risk."
        )
    if float(snapshot.get("document_readiness_score", 1.0)) < 0.6:
        actions.append("Prioritize document completion and pre-clearance review.")
    if float(snapshot.get("event_completeness_score", 1.0)) < 1.0:
        actions.append(
            "Request a verified operational update because event coverage is incomplete."
        )
    return actions or ["Continue monitoring the next milestone update."]
