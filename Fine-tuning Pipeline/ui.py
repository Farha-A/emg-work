import math
import pygame


class Theme:
    BG = (246, 243, 248)
    PANEL = (255, 255, 255)
    PANEL_BORDER = (220, 213, 231)
    TEXT = (45, 40, 55)
    MUTED = (104, 96, 122)
    ACCENT = (156, 132, 255)
    ACCENT_2 = (117, 198, 198)
    DANGER = (255, 132, 162)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def lerp(a, b, t):
    return a + (b - a) * t


def wrap_text(font, text, max_width):
    words = text.split()
    lines = []
    current = ""
    for w in words:
        test = (current + " " + w).strip()
        if font.size(test)[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


class Button:
    def __init__(self, rect, label):
        # ...existing code...
        self.rect = pygame.Rect(rect)
        self.label = label
        self.hover = False

    def handle_event(self, event):
        # normalize rect in case external code assigned a tuple
        r = pygame.Rect(self.rect)
        self.rect = r
        if event.type == pygame.MOUSEMOTION:
            self.hover = r.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if r.collidepoint(event.pos):
                return True
        return False

    def draw(self, surface, font, theme: Theme, enabled=True):
        # ensure rect is a pygame.Rect and use a local variable for drawing
        r = pygame.Rect(self.rect)
        self.rect = r

        if enabled:
            base = theme.ACCENT
            fill = (
                int(lerp(base[0], 255, 0.18)) if self.hover else base[0],
                int(lerp(base[1], 255, 0.18)) if self.hover else base[1],
                int(lerp(base[2], 255, 0.18)) if self.hover else base[2],
            )
            border = (255, 255, 255)
            text_color = (255, 255, 255)
        else:
            fill = (204, 196, 219)
            border = (255, 255, 255)
            text_color = (246, 243, 248)

        pygame.draw.rect(surface, fill, r, border_radius=14)
        pygame.draw.rect(surface, border, r, width=2, border_radius=14)
        text = font.render(self.label, True, text_color)
        surface.blit(text, text.get_rect(center=r.center))


class TextField:
    def __init__(self, rect, label, multiline=False, max_chars=240):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.multiline = multiline
        self.max_chars = max_chars
        self.text = ""
        self.active = False
        self._blink = 0.0

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.active = self.rect.collidepoint(event.pos)
            return

        if event.type != pygame.KEYDOWN or not self.active:
            return

        if event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
            return

        if event.key in (pygame.K_TAB, pygame.K_ESCAPE):
            return

        if event.key == pygame.K_RETURN:
            if self.multiline and len(self.text) < self.max_chars:
                self.text += "\n"
            return

        if event.unicode and len(self.text) < self.max_chars:
            if self.multiline:
                self.text += event.unicode
            else:
                if event.unicode in ("\n", "\r"):
                    return
                self.text += event.unicode

    def update(self, dt):
        self._blink = (self._blink + dt) % 1.0

    def draw(self, surface, font_label, font_body, theme: Theme):
        label = font_label.render(self.label, True, theme.TEXT)
        surface.blit(label, (self.rect.x, self.rect.y - 26))

        border = theme.ACCENT if self.active else theme.PANEL_BORDER
        pygame.draw.rect(surface, theme.PANEL, self.rect, border_radius=12)
        pygame.draw.rect(surface, border, self.rect, width=2, border_radius=12)

        padding = 12
        inner = self.rect.inflate(-padding * 2, -padding * 2)

        lines = self.text.split("\n") if self.multiline else [self.text]
        rendered_lines = []
        for ln in lines:
            if ln.strip() == "":
                rendered_lines.append("")
            else:
                rendered_lines.extend(wrap_text(font_body, ln, inner.width))

        y = inner.y
        line_h = font_body.get_linesize()
        max_lines = max(1, inner.height // line_h)
        rendered_lines = rendered_lines[-max_lines:]
        for ln in rendered_lines:
            surf = font_body.render(ln, True, theme.MUTED)
            surface.blit(surf, (inner.x, y))
            y += line_h

        if self.active and self._blink < 0.55:
            last = rendered_lines[-1] if rendered_lines else ""
            caret_x = inner.x + font_body.size(last)[0]
            caret_y = inner.y + (len(rendered_lines) - 1) * line_h
            caret_x = min(caret_x, inner.right - 2)
            pygame.draw.line(surface, theme.TEXT, (caret_x, caret_y + 2), (caret_x, caret_y + line_h - 2), 2)


def star_points(cx, cy, outer_r, inner_r, points=5):
    pts = []
    step = math.pi / points
    angle = -math.pi / 2
    for i in range(points * 2):
        r = outer_r if i % 2 == 0 else inner_r
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        angle += step
    return pts
