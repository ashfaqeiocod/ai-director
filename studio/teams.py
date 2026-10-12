"""teams.py — which team is which, from SHIRT COLOURS, learned live from the first seconds of the match.

The football detector already labels each person player / goalkeeper / referee. For every outfield player we
take the colour of the shirt (middle of the upper body, grass pixels removed) and, once enough shirts are
collected, split them into 2 groups (k-means). Each group = a team, named after its colour ("WHITE", "BLUE").
After that every player gets the nearer team colour. Only frames already played are used (causal).

What it gives the studio: team-coloured dots on the pitch map, POSSESSION (the team of the player nearest the
ball), and team names/colours for the scoreboard, in the same colour words Gemini uses ("white", "blue").
Settings: config.yaml -> studio.teams.
"""
from __future__ import annotations

import cv2
import numpy as np

# Simple colour names (OpenCV hue 0-180) for the scoreboard.
HUE_NAMES = [(10, "RED"), (22, "ORANGE"), (35, "YELLOW"), (85, "GREEN"), (100, "CYAN"), (130, "BLUE"),
             (160, "PURPLE"), (180, "RED")]


def shirt_colour(image: np.ndarray, box: list[float]) -> np.ndarray | None:
    """Median BGR colour of the shirt: the middle of the upper body, without grass-green pixels."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 14:
        return None
    crop = image[int(y1 + 0.18 * h):int(y1 + 0.5 * h), int(x1 + 0.25 * w):int(x2 - 0.25 * w)]
    if crop.size == 0:
        return None
    pixels = crop.reshape(-1, 3)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grass = (hsv[:, 0] > 30) & (hsv[:, 0] < 90) & (hsv[:, 1] > 60)
    keep = pixels[~grass]
    if len(keep) < 6:
        return None
    return np.median(keep, axis=0)


def colour_name(bgr: np.ndarray) -> str:
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if v < 60:
        return "BLACK"
    if s < 30:                                       # dull dark shirts (ISSIA purple: s 48) still have a hue
        return "WHITE" if v > 150 else "GREY"
    return next(name for top, name in HUE_NAMES if h <= top)


class TeamClassifier:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.samples: list[np.ndarray] = []
        self.centres: np.ndarray | None = None        # 2 x 3, Lab
        self.teams: list[dict] = []                    # [{"name": "WHITE", "rgb": "#e5e7eb"}, ...]

    @staticmethod
    def _lab(bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

    def observe(self, image: np.ndarray, players: list[dict]) -> list[int | None]:
        """Team (0/1) for each detected person in one camera picture (None = unknown, referee, keeper)."""
        out = []
        for p in players:
            if p.get("cls", "player") != "player":           # referees and goalkeepers wear other colours
                out.append(None)
                continue
            colour = shirt_colour(image, p["box"])
            if colour is None:
                out.append(None)
                continue
            lab = self._lab(colour)
            if self.centres is None:
                self.samples.append(lab)
                out.append(None)
                if len(self.samples) >= self.cfg["learn_samples"]:
                    self._learn()
            else:
                d = np.linalg.norm(self.centres - lab, axis=1)
                out.append(int(np.argmin(d)) if d.min() <= self.cfg["max_colour_distance"] else None)
        return out

    def _learn(self) -> None:
        data = np.asarray(self.samples, np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, labels, centres = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        self.centres = centres
        self.teams = []
        for c in centres:
            bgr = cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_LAB2BGR)[0, 0]
            self.teams.append({"name": colour_name(bgr), "rgb": "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0])})
        if self.teams[0]["name"] == self.teams[1]["name"]:          # two shades of one colour: make names unique
            self.teams[1]["name"] += " 2"

    @property
    def ready(self) -> bool:
        return self.centres is not None
