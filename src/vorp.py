"""VORP engine and ADP arbitrage signals."""


def classify_signal(adp_delta: float) -> str:
    """Classify an ADP delta into an emoji signal.

    Parameters
    ----------
    adp_delta : search_rank - vorp_rank. Positive = value, negative = overpriced.

    Returns
    -------
    One of: "Major Value", "Slight Value", "Fair Value", "Overpriced", "Heavy Reach".
    """
    if adp_delta >= 10:
        return "\U0001f525 Major Value"
    if 5 <= adp_delta < 10:
        return "\u2705 Slight Value"
    if -5 < adp_delta < 5:
        return "\u2696\ufe0f Fair Value"
    if -10 < adp_delta <= -5:
        return "\u26a0\ufe0f Overpriced"
    # adp_delta <= -10
    return "\U0001f6ab Heavy Reach"
