"""Распознавание рукописных цифр в реальном времени: холст Pygame + модель MNIST."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import math

import pygame

import preprocessing
from model_manager import get_model, predict

WINDOW_W, WINDOW_H = 560, 800
GRID = 28
CANVAS_PX = GRID * 14
CANVAS_POS = ((WINDOW_W - CANVAS_PX) // 2, 50)
CANVAS_RECT = pygame.Rect(CANVAS_POS, (CANVAS_PX, CANVAS_PX))
BUTTON_RECT = pygame.Rect(CANVAS_POS[0] + CANVAS_PX // 2 - 80, 452, 160, 40)
RESULT_POS = (CANVAS_POS[0], 502)
BARS_TOP, BAR_ROW_H = 540, 25
BAR_X, BAR_W = 110, 300
PERCENT_X, PERCENT_W = 420, 56
BRUSH_RADIUS = 17
PAPER = (255, 255, 255)
INK = (0, 0, 0)
PREDICT_INTERVAL_MS = 100

BG = (18, 18, 24)
PANEL = (28, 28, 38)
BORDER = (62, 62, 84)
TEXT = (238, 238, 245)
MUTED = (132, 132, 154)
ACCENT = (255, 150, 40)
BAR = (68, 128, 208)
BAR_BG = (44, 44, 60)
BTN_BG = (46, 46, 64)
BTN_HOVER = (62, 62, 86)


def make_font(size, bold=True):
    path = pygame.font.match_font("segoeui,arial,tahoma,verdana")
    if path:
        font = pygame.font.Font(path, size)
        font.set_bold(bold)
        return font
    return pygame.font.Font(None, int(size * 1.3))


class DigitRecognizer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Распознавание рукописных цифр")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()

        self.font_title = make_font(20)
        self.font_button = make_font(18)
        self.font_result = make_font(30)
        self.font_label = make_font(16)
        self.font_percent = make_font(15)
        self.font_hint = make_font(14, bold=False)

        self.canvas = pygame.Surface((CANVAS_PX, CANVAS_PX))
        self.model = get_model()

        self.reset()

    def reset(self):
        self.canvas.fill(PAPER)
        self.drawing = False
        self.last_point = None
        self.probabilities = [0.0] * 10
        self.dirty = True
        self.last_predict_at = 0

    def _draw_dot(self, point):
        pygame.draw.circle(self.canvas, INK, (int(point[0]), int(point[1])), BRUSH_RADIUS)

    def _draw_stroke(self, start, end):
        self._draw_dot(start)
        distance = math.dist(start, end)
        if distance <= BRUSH_RADIUS:
            return
        steps = int(distance / max(1.0, BRUSH_RADIUS * 0.5))
        for index in range(1, steps + 1):
            t = index / steps
            self._draw_dot(
                (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
            )

    def _local_point(self, position):
        return (position[0] - CANVAS_POS[0], position[1] - CANVAS_POS[1])

    def recognize(self):
        gray = preprocessing.canvas_to_gray(self.canvas)
        if preprocessing.is_blank(gray):
            self.probabilities = [0.0] * 10
        else:
            batch = preprocessing.prepare(gray)
            self.probabilities = list(predict(self.model, batch))
        self.dirty = False
        self.last_predict_at = pygame.time.get_ticks()

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_c:
                self.reset()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if CANVAS_RECT.collidepoint(event.pos):
                self.drawing = True
                self.last_point = self._local_point(event.pos)
                self._draw_dot(self.last_point)
                self.dirty = True
            elif BUTTON_RECT.collidepoint(event.pos):
                self.reset()
        elif event.type == pygame.MOUSEMOTION and self.drawing:
            point = self._local_point(event.pos)
            if CANVAS_RECT.collidepoint(event.pos):
                self._draw_stroke(self.last_point, point)
                self.last_point = point
                self.dirty = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.drawing:
            self.drawing = False
            self.last_point = None
            self.recognize()
        return True

    def draw_text(self, font, text, position, color):
        self.screen.blit(font.render(text, True, color), position)

    def draw_ui(self, mouse_pos):
        self.screen.fill(BG)
        self.draw_text(
            self.font_title,
            "Нарисуйте цифру 0-9 мышью",
            (CANVAS_POS[0], 16),
            TEXT,
        )

        pygame.draw.rect(self.screen, BORDER, CANVAS_RECT, 2)
        self.screen.blit(self.canvas, CANVAS_POS)

        hover = BUTTON_RECT.collidepoint(mouse_pos)
        pygame.draw.rect(self.screen, BTN_HOVER if hover else BTN_BG, BUTTON_RECT, border_radius=8)
        label = self.font_button.render("Очистить", True, TEXT)
        self.screen.blit(label, label.get_rect(center=BUTTON_RECT.center))

        best = max(range(10), key=lambda i: self.probabilities[i])
        if any(self.probabilities):
            confidence = self.probabilities[best] * 100
            result = f"Распознано: {best} ({confidence:.1f} %)"
            color = ACCENT if confidence >= 50 else MUTED
        else:
            result = "Распознано: —"
            color = MUTED
        self.draw_text(self.font_result, result, RESULT_POS, color)

        for digit in range(10):
            top = BARS_TOP + digit * BAR_ROW_H
            is_best = digit == best and any(self.probabilities)
            if is_best:
                pygame.draw.rect(
                    self.screen, PANEL, (BAR_X - 30, top - 2, BAR_W + 100, BAR_ROW_H - 2),
                    border_radius=6,
                )
            color = ACCENT if is_best else BAR
            self.draw_text(self.font_label, str(digit), (CANVAS_POS[0], top + 3), color if is_best else MUTED)

            pygame.draw.rect(self.screen, BAR_BG, (BAR_X, top + 6, BAR_W, 13), border_radius=3)
            width = int(BAR_W * self.probabilities[digit])
            if width > 0:
                pygame.draw.rect(
                    self.screen, color, (BAR_X, top + 6, max(width, 3), 13), border_radius=3
                )

            percent = f"{self.probabilities[digit] * 100:.1f}%"
            text = self.font_percent.render(percent, True, color if is_best else MUTED)
            self.screen.blit(text, (PERCENT_X + PERCENT_W - text.get_width(), top + 5))

        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if not self.handle_event(event):
                    pygame.quit()
                    return

            now = pygame.time.get_ticks()
            if self.dirty and now - self.last_predict_at >= PREDICT_INTERVAL_MS:
                self.recognize()

            self.draw_ui(pygame.mouse.get_pos())
            self.clock.tick(60)


if __name__ == "__main__":
    DigitRecognizer().run()