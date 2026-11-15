import time

MODES = ["desktop", "media", "presentation"]

class ModeManager:

    def __init__(self, hold_seconds: float = 1.0):
        self.index = 0
        self.hold_seconds = hold_seconds
        self._hold_start: float | None = None
        self._fired = False

    @property
    def mode(self) -> str:
        return MODES[self.index]

    def update(self, label: str | None) -> bool:
        if label != "open_palm":
            self._hold_start = None
            self._fired = False
            return False

        if self._hold_start is None:
            self._hold_start = time.monotonic()
        held = time.monotonic() - self._hold_start
        if not self._fired and held >= self.hold_seconds:
            self._fired = True
            self.index = (self.index + 1) % len(MODES)
            return True
        return False