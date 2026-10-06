"""Bridge fixed tuned configs into the tuning package's candidate type."""

from __future__ import annotations

from feature_lag_mlp.config import TUNED_CONFIGS
from feature_lag_mlp.search_space import Candidate


def tuned_candidate(city: str) -> Candidate:
    """Return the locally confirmed tuned architecture for one city."""
    config = TUNED_CONFIGS[city]
    return Candidate(
        name=f"tuned_{city}",
        hidden_1=config.hidden_1,
        hidden_2=config.hidden_2,
        dropout_1=config.dropout_1,
        dropout_2=config.dropout_2,
        learning_rate=config.learning_rate,
    )
