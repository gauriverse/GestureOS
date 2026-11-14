from collections import Counter, deque


class VoteSmoother:

    def __init__(self, window: int = 7, min_votes: int = 5, min_conf: float = 0.6):
        self.frames = deque(maxlen=window)
        self.min_votes = min_votes
        self.min_conf = min_conf

    def update(self, label: str | None, confidence: float) -> str | None:
        # Pass label=None and confidence=0.0 when no hand is detected
        self.frames.append((label, confidence))
        votes = [l for l, c in self.frames if l is not None and c >= self.min_conf]
        if not votes:
            return None
        top, count = Counter(votes).most_common(1)[0]
        return top if count >= self.min_votes else None