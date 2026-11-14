import time


class Cooldown:

    def __init__(self, seconds: float = 0.5):
        self.seconds = seconds
        self._last: dict[str, float] = {}

    def ready(self, name: str) -> bool:
        now = time.monotonic()
        if now - self._last.get(name, -1e9) >= self.seconds:
            self._last[name] = now
            return True
        return False