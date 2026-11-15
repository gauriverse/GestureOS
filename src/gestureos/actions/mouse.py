import numpy as np
import pyautogui

pyautogui.FAILSAFE = True   
pyautogui.PAUSE = 0         


class MouseController:
    def __init__(self, margin: float = 0.08, smoothing: float = 0.6):
        self.screen_w, self.screen_h = pyautogui.size()
        self.margin = margin        
        self.alpha = smoothing      
        self._x = None
        self._y = None

    def move_to(self, norm_x: float, norm_y: float):
        
        m = self.margin
        u = float(np.clip((norm_x - m) / (1 - 2 * m), 0.0, 1.0))
        v = float(np.clip((norm_y - m) / (1 - 2 * m), 0.0, 1.0))
        tx, ty = u * self.screen_w, v * self.screen_h
        if self._x is None:
            self._x, self._y = tx, ty
        else:
            self._x += self.alpha * (tx - self._x)
            self._y += self.alpha * (ty - self._y)
        pyautogui.moveTo(int(self._x), int(self._y))

    def click(self):
        pyautogui.click()

    def right_click(self):
        pyautogui.rightClick()

    def scroll_up(self, amount: int = 300):
        pyautogui.scroll(amount)

    def scroll_down(self, amount: int = 300):
        pyautogui.scroll(-amount)