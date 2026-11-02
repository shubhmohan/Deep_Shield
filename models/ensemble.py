"""
DeepShield - Ensemble Voting Module
Combines ViT-B/16 and EfficientNet-B4 fake-probability scores into a single
verdict + confidence + risk level, using a weighted average.

Why weighted average over simple majority vote:
Both branches output continuous probabilities, not just labels — averaging
preserves that signal instead of throwing it away, and lets you tune which
model you trust more once you've benchmarked both on your validation set.
"""

from dataclasses import dataclass


# Default weights: adjust after benchmarking on FF++/DFDC validation subset.
# Start 50/50 and shift based on which model empirically performs better.
VIT_WEIGHT = 0.5
EFFICIENTNET_WEIGHT = 0.5

RISK_THRESHOLDS = [
    (0.25, "LOW"),
    (0.50, "MEDIUM"),
    (0.75, "HIGH"),
    (1.01, "CRITICAL"),  # 1.01 so a fake_prob of exactly 1.0 still matches
]


@dataclass
class DetectionResult:
    verdict: str          # "REAL" or "FAKE"
    confidence: float     # 0-100
    fake_probability: float  # 0-1, raw ensembled score
    risk_level: str        # LOW / MEDIUM / HIGH / CRITICAL
    vit_score: float
    efficientnet_score: float


def get_risk_level(fake_prob: float) -> str:
    for threshold, label in RISK_THRESHOLDS:
        if fake_prob < threshold:
            return label
    return "CRITICAL"


def ensemble_predict(vit_fake_prob: float, efficientnet_fake_prob: float) -> DetectionResult:
    """
    Combines both branch scores into a final verdict.
    fake_prob > 0.5 → classified as FAKE
    """
    combined_fake_prob = (
        VIT_WEIGHT * vit_fake_prob + EFFICIENTNET_WEIGHT * efficientnet_fake_prob
    )

    verdict = "FAKE" if combined_fake_prob >= 0.5 else "REAL"

    # Confidence = how far the score sits from the 50/50 uncertainty line, scaled to 0-100
    confidence = abs(combined_fake_prob - 0.5) * 2 * 100

    risk_level = get_risk_level(combined_fake_prob)

    return DetectionResult(
        verdict=verdict,
        confidence=round(confidence, 2),
        fake_probability=round(combined_fake_prob, 4),
        risk_level=risk_level,
        vit_score=round(vit_fake_prob, 4),
        efficientnet_score=round(efficientnet_fake_prob, 4),
    )


if __name__ == "__main__":
    # Example: both models fairly confident it's fake
    result = ensemble_predict(vit_fake_prob=0.82, efficientnet_fake_prob=0.91)
    print(result)

    # Example: models disagree
    result2 = ensemble_predict(vit_fake_prob=0.30, efficientnet_fake_prob=0.70)
    print(result2)
