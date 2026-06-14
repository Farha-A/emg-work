import csv
from pathlib import Path
import math
from collections import deque

import pygame

from constants import WIDTH, HEIGHT, FPS, TEST_DURATION
from emg import *
from ui import Theme, clamp, lerp, wrap_text, Button, TextField, star_points
from levels import build_levels, Level
from logger import Logger


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Robit Data Demo")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.theme = Theme()

        self.emg = EMGReader()
        # self.emg = EMGStub()
        # level-start button shown inside the circle; only starts timing/stream when pressed
        self.level_start_button = Button(rect=(0, 0, 160, 48), label="Begin")
        self.level_start_button_visible = False

        self.font_title = pygame.font.SysFont("Segoe UI", 44, bold=True)
        self.font_h2 = pygame.font.SysFont("Segoe UI", 26, bold=True)
        self.font_body = pygame.font.SysFont("Segoe UI", 20)
        self.font_button = pygame.font.SysFont("Segoe UI", 22, bold=True)
        self.font_button_small = pygame.font.SysFont("Segoe UI", 18, bold=True)

        self.state = "start"
        self.level_index = 0
        self.levels = build_levels()

        self.session_id = str(pygame.time.get_ticks())
        self.user_name = ""
        self.user_age = ""
        self.user_medical_history = ""
        self.intake_error = ""

        self.level_start_ts = None
        self.pending_duration_sec = None

        self.log_path = Path(__file__).with_name("circle_control_results.csv")
        self.logger = Logger(self.log_path)
        self.logger.ensure_log_header()

        self.progress = 0.12
        self.mouse_down = False
        self.click_times = []
        self.banner = ""
        self.banner_timer = 0.0
        self.plot_duration_sec = 6.0
        self.plot_sample_rate = 30
        max_samples = int(self.plot_duration_sec * self.plot_sample_rate)
        self.plot_filtered = deque(maxlen=max_samples)
        self.plot_envelope = deque(maxlen=max_samples)
        self._plot_timer = 0.0

        # Gulp Level specific state
        self.gulp_count = 0
        self.gulp_accumulated_duration = 0.0

        self.start_button = Button(
            rect=(WIDTH // 2 - 110, HEIGHT // 2 + 90, 220, 54),
            label="Start",
        )
        self.intake_continue_button = Button(
            rect=(WIDTH // 2 - 110, HEIGHT - 110, 220, 54),
            label="Continue",
        )
        self.next_button = Button(
            rect=(WIDTH // 2 - 110, HEIGHT // 2 + 120, 220, 54),
            label="Next",
        )
        self.restart_button = Button(
            rect=(WIDTH // 2 - 110, HEIGHT // 2 + 120, 220, 54),
            label="Restart",
        )

        self.field_name = TextField((120, 250, WIDTH - 240, 52), "Name")
        self.field_age = TextField((120, 340, 220, 52), "Age")
        self.field_history = TextField((120, 440, WIDTH - 240, 120), "Medical history", multiline=True, max_chars=600)

    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            self._handle_events(dt)
            self._update(dt)
            self._draw()

    def _reset_level_state(self):
        self.progress = 0.12
        self.mouse_down = False
        self.click_times = []
        self.banner = ""
        self.banner_timer = 0.0
        self.plot_filtered.clear()
        self.plot_envelope.clear()
        self._plot_timer = 0.0
        self.gulp_count = 0
        self.gulp_accumulated_duration = 0.0

    def _show_banner(self, text, seconds=1.2):
        self.banner = text
        self.banner_timer = seconds

    def _begin_level(self):
        # Enter the level screen but wait for the in-circle button to actually start timing/logging.
        self.state = "level"
        self._reset_level_state()
        self.level_start_ts = None
        self.level_start_button_visible = True
        # pending_duration_sec reset
        self.level_start_button_visible = True
        # pending_duration_sec reset
        self.pending_duration_sec = None

        level = self.levels[self.level_index]
        if level.title == "Gulp":
            self.level_start_button.label = "Hold to Record (0/3)"
        else:
            self.level_start_button.label = "Start"

    def _validate_intake(self):
        name = self.field_name.text.strip()
        age_raw = self.field_age.text.strip()

        if not name:
            return False, "Please enter a name."

        if not age_raw:
            return False, "Please enter an age."

        try:
            age_val = int(age_raw)
        except ValueError:
            return False, "Age must be a whole number."

        if age_val < 1 or age_val > 120:
            return False, "Please enter a valid age (1–120)."

        return True, ""

    def _handle_events(self, dt):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                continue

            if self.state == "start":
                if self.start_button.handle_event(event):
                    self.state = "intake"
                    self.level_index = 0
                    self.intake_error = ""
                    self.field_name.text = ""
                    self.field_age.text = ""
                    self.field_history.text = ""

            elif self.state == "intake":
                clicked_continue = self.intake_continue_button.handle_event(event)
                self.field_name.handle_event(event)
                self.field_age.handle_event(event)
                self.field_history.handle_event(event)

                ok, msg = self._validate_intake()
                enabled = ok and not self.logger.error
                if clicked_continue:
                    if enabled:
                        self.user_name = self.field_name.text.strip()
                        self.user_age = self.field_age.text.strip()
                        self.user_medical_history = self.field_history.text.strip()
                        self.intake_error = ""
                        self.level_index = 0
                        self._begin_level()
                    else:
                        self.intake_error = msg

            elif self.state == "level":
                # if level hasn't been started yet, handle the in-circle start button
                if self.level_start_button_visible:
                    # Specific logic for Gulp level
                    level = self.levels[self.level_index]
                    is_gulp = (level.title == "Gulp")

                    if is_gulp:
                        # Gulp Interaction: Press and Hold
                        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                            if self.level_start_button.handle_event(event):
                                # Start recording this trial
                                now = pygame.time.get_ticks() / 1000.0
                                self.level_start_ts = now # Use this to track start of current hold
                                try:
                                    # Use same Session ID and Level Number, data will append
                                    self.logger.start_stream(self.session_id, self.level_index + 1, lambda: (getattr(self.emg, "filtered", ""), getattr(self.emg, "envelope", "")))
                                except Exception:
                                    pass
                                self.level_start_button.label = "Recording..."
                        
                        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                            # If we were recording, stop and count it
                            if self.level_start_ts is not None:
                                now = pygame.time.get_ticks() / 1000.0
                                duration = max(0.0, now - self.level_start_ts)
                                self.gulp_accumulated_duration += duration
                                self.gulp_count += 1
                                
                                # Reset current hold timer
                                self.level_start_ts = None 
                                
                                # Stop stream for this trial
                                try:
                                    self.logger.stop_stream()
                                except Exception:
                                    pass

                                if self.gulp_count >= 3:
                                    self.pending_duration_sec = self.gulp_accumulated_duration
                                    self.state = "complete"
                                    self.level_start_button_visible = False
                                else:
                                    self.level_start_button.label = f"Hold to Record ({self.gulp_count}/3)"

                    else:
                        # Standard Interaction: Toggle / One-click start
                        clicked = self.level_start_button.handle_event(event)
                        if clicked:
                            # start timing and logging
                            now = pygame.time.get_ticks() / 1000.0
                            self.level_start_ts = now
                            try:
                                self.logger.start_stream(self.session_id, self.level_index + 1, lambda: (getattr(self.emg, "filtered", ""), getattr(self.emg, "envelope", "")))
                            except Exception:
                                pass
                            self.level_start_button_visible = False
                        # while waiting for start, ignore other gameplay input
                        continue
                        
                    # If Gulp level and button is visible, we capture input on the button,
                    # For Gulp, we just stay in this state until count >= 3.
                    if is_gulp:
                        continue

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.mouse_down = True
                    now = pygame.time.get_ticks() / 1000.0
                    self.click_times.append(now)
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.mouse_down = False

            elif self.state == "complete":
                clicked_next = self.next_button.handle_event(event)
                if clicked_next:
                    # log results (no rating)
                    level = self.levels[self.level_index]
                    try:
                        self.logger.log_level_result(
                            self.session_id,
                            self.user_name,
                            self.user_age,
                            self.user_medical_history,
                            self.level_index + 1,
                            self.pending_duration_sec or 0.0,
                            "",
                        )
                    except Exception:
                        pass
                    self.level_index += 1
                    if self.level_index >= len(self.levels):
                        self.state = "end"
                    else:
                        self._begin_level()

            elif self.state == "end":
                if self.restart_button.handle_event(event):
                    self.state = "start"
                    self.level_index = 0
                    self._reset_level_state()
                    self.user_name = ""
                    self.user_age = ""
                    self.user_medical_history = ""
                    self.pending_duration_sec = None
                    self.pending_rating = 0
                    self.rating_hover = 0

    def _update(self, dt):
        if self.banner_timer > 0:
            self.banner_timer = max(0.0, self.banner_timer - dt)

        if self.state == "intake":
            self.field_name.update(dt)
            self.field_age.update(dt)
            self.field_history.update(dt)
            return

        if self.state != "level":
            return

        level = self.levels[self.level_index]
        self.emg.update(dt)
        signal = self.emg.envelope
        now = pygame.time.get_ticks() / 1000.0
        self._plot_timer += dt
        sample_interval = 1.0 / self.plot_sample_rate
        while self._plot_timer >= sample_interval:
            self._plot_timer -= sample_interval
            self.plot_filtered.append(getattr(self.emg, "filtered", 0))
            self.plot_envelope.append(getattr(self.emg, "envelope", 0))

        # simplified: progress is purely time-based once the level is started
        if level.title == "Gulp":
            # Show progress based on completed trials
            self.progress = clamp(self.gulp_count / 3.0, 0.0, 1.0)
            return

        if self.level_start_ts is not None:
            self.progress = clamp((now - self.level_start_ts) / TEST_DURATION, 0.0, 1.0)
        else:
            self.progress = 0.0

        if self.progress >= 1.0:
            now = pygame.time.get_ticks() / 1000.0
            if self.level_start_ts is None:
                self.level_start_ts = now
            self.pending_duration_sec = max(0.0, now - self.level_start_ts)
            # stop background stream when level completes
            try:
                self.logger.stop_stream()
            except Exception:
                pass
            self.state = "complete"

    def _draw_panel(self, rect):
        pygame.draw.rect(self.screen, self.theme.PANEL, rect, border_radius=18)
        pygame.draw.rect(self.screen, self.theme.PANEL_BORDER, rect, width=2, border_radius=18)

    def _draw_emg_plot(self, rect):
        pygame.draw.rect(self.screen, self.theme.PANEL, rect, border_radius=12)
        pygame.draw.rect(self.screen, self.theme.PANEL_BORDER, rect, width=2, border_radius=12)

        label_h = self.font_body.get_linesize()
        title = self.font_body.render("EMG (last 6s)", True, self.theme.MUTED)
        self.screen.blit(title, (rect.x + 12, rect.y + 8))

        pad = 10
        plot_rect = pygame.Rect(rect.x + pad, rect.y + pad + label_h + 4, rect.width - pad * 2, rect.height - pad * 2 - label_h - 4)

        if len(self.plot_filtered) < 2 or len(self.plot_envelope) < 2:
            label = self.font_body.render("Waiting for signal...", True, self.theme.MUTED)
            self.screen.blit(label, label.get_rect(center=plot_rect.center))
            return

        max_val = max(
            200,
            max(max(self.plot_filtered), max(self.plot_envelope)),
        )
        max_val = max(1, max_val)

        # gridlines
        grid_color = self.theme.PANEL_BORDER
        for i in range(1, 4):
            y = plot_rect.y + int(i * plot_rect.height / 4)
            pygame.draw.line(self.screen, grid_color, (plot_rect.x, y), (plot_rect.right, y), 1)
        for i in range(1, 6):
            x = plot_rect.x + int(i * plot_rect.width / 6)
            pygame.draw.line(self.screen, grid_color, (x, plot_rect.y), (x, plot_rect.bottom), 1)

        # axis labels
        max_label = self.font_body.render(str(int(max_val)), True, self.theme.MUTED)
        mid_label = self.font_body.render(str(int(max_val / 2)), True, self.theme.MUTED)
        min_label = self.font_body.render("0", True, self.theme.MUTED)
        self.screen.blit(max_label, (plot_rect.x + 4, plot_rect.y + 2))
        self.screen.blit(mid_label, (plot_rect.x + 4, plot_rect.y + plot_rect.height // 2 - max_label.get_height() // 2))
        self.screen.blit(min_label, (plot_rect.x + 4, plot_rect.bottom - min_label.get_height() - 2))

        def to_points(values):
            pts = []
            n = len(values)
            for i, v in enumerate(values):
                x = plot_rect.x + int(i * (plot_rect.width - 1) / (n - 1))
                y = plot_rect.bottom - int((v / max_val) * plot_rect.height)
                pts.append((x, y))
            return pts

        filt_pts = to_points(self.plot_filtered)
        env_pts = to_points(self.plot_envelope)

        pygame.draw.lines(self.screen, self.theme.ACCENT, False, filt_pts, 2)
        pygame.draw.lines(self.screen, self.theme.ACCENT_2, False, env_pts, 2)

        legend_y = rect.y + 8
        legend_x = rect.x + rect.width - 10
        env_label = self.font_body.render("Envelope", True, self.theme.ACCENT_2)
        filt_label = self.font_body.render("Filtered", True, self.theme.ACCENT)
        self.screen.blit(env_label, (legend_x - env_label.get_width(), legend_y))
        self.screen.blit(filt_label, (legend_x - env_label.get_width() - filt_label.get_width() - 12, legend_y))

    def _draw(self):
        self.screen.fill(self.theme.BG)

        if self.state == "start":
            self._draw_start()
        elif self.state == "intake":
            # self.state = 'level'
            self._draw_intake()
        elif self.state == "level":
            self._draw_level()
        elif self.state == "complete":
            self._draw_complete()
        elif self.state == "end":
            self._draw_end()

        pygame.display.flip()

    def _draw_start(self):
        title = self.font_title.render("Circle Control", True, self.theme.TEXT)
        subtitle_lines = wrap_text(
            self.font_body,
            "Robit Data Collection Demo. Read the instruction on each level and complete the circle.",
            760,
        )
        self.screen.blit(title, title.get_rect(center=(WIDTH // 2, 160)))

        panel = pygame.Rect(WIDTH // 2 - 400, 220, 800, 220)
        self._draw_panel(panel)

        y = panel.y + 28
        for line in subtitle_lines:
            s = self.font_body.render(line, True, self.theme.MUTED)
            self.screen.blit(s, (panel.x + 28, y))
            y += 28

        hint = self.font_body.render("Tip: Put Ginger on Chicken.", True, self.theme.MUTED)
        self.screen.blit(hint, hint.get_rect(center=(WIDTH // 2, panel.bottom - 44)))

        self.start_button.draw(self.screen, self.font_button, self.theme)

    def _draw_intake(self):
        title = self.font_title.render("Participant Details", True, self.theme.TEXT)
        self.screen.blit(title, title.get_rect(center=(WIDTH // 2, 120)))

        panel = pygame.Rect(WIDTH // 2 - 420, 170, 840, 420)
        self._draw_panel(panel)

        guidance = (
            "Hello, Adventurer!"
        )
        lines = wrap_text(self.font_body, guidance, panel.width - 60)
        y = panel.y + 22
        for ln in lines:
            s = self.font_body.render(ln, True, self.theme.MUTED)
            self.screen.blit(s, (panel.x + 30, y))
            y += 24

        self.field_name.draw(self.screen, self.font_body, self.font_body, self.theme)
        self.field_age.draw(self.screen, self.font_body, self.font_body, self.theme)
        self.field_history.draw(self.screen, self.font_body, self.font_body, self.theme)

        ok, msg = self._validate_intake()
        if self.intake_error:
            msg = self.intake_error
        if msg:
            err = self.font_body.render(msg, True, self.theme.DANGER)
            self.screen.blit(err, (panel.x + 30, panel.bottom - 66))

        if self.logger.error:
            warn = self.font_body.render(f"Logging error: {self.logger.error}", True, self.theme.DANGER)
            self.screen.blit(warn, (panel.x + 30, panel.bottom - 40))
        else:
            info = self.font_body.render(f"Logging to: {self.log_path.name}", True, self.theme.MUTED)
            self.screen.blit(info, (panel.x + 30, panel.bottom - 40))

        self.intake_continue_button.draw(self.screen, self.font_button, self.theme, enabled=ok and not self.logger.error)

    def _draw_level(self):
        level = self.levels[self.level_index]

        header_rect = pygame.Rect(30, 24, WIDTH - 60, 116)
        self._draw_panel(header_rect)

        title = self.font_h2.render(level.title, True, self.theme.TEXT)
        self.screen.blit(title, (header_rect.x + 24, header_rect.y + 18))

        body_lines = wrap_text(self.font_body, level.body, header_rect.width - 48)
        y = header_rect.y + 56
        for line in body_lines[:2]:
            b = self.font_body.render(line, True, self.theme.MUTED)
            self.screen.blit(b, (header_rect.x + 24, y))
            y += 26

        arena_rect = pygame.Rect(30, 160, WIDTH - 60, HEIGHT - 190)
        self._draw_panel(arena_rect)

        left_rect = pygame.Rect(arena_rect.x + 20, arena_rect.y + 20, 360, arena_rect.height - 40)
        plot_rect = pygame.Rect(left_rect.right + 20, arena_rect.y + 20, arena_rect.width - left_rect.width - 60, arena_rect.height - 40)

        cx = left_rect.centerx
        cy = left_rect.centery - 20
        outer_r = 140
        inner_r = int(outer_r * self.progress)

        pygame.draw.circle(self.screen, self.theme.PANEL_BORDER, (cx, cy), outer_r, width=4)
        fill_color = self.theme.ACCENT_2 if self.level_index % 2 == 0 else self.theme.ACCENT
        pygame.draw.circle(self.screen, fill_color, (cx, cy), inner_r)

        # draw the centered level-start button if the level has not actually started
        if self.level_start_button_visible:
            bw, bh = 160, 48
            font = self.font_button

            if level.title == "Gulp":
                bw = 220
                font = self.font_button_small

            bx = cx - bw // 2
            by = cy - bh // 2
            # update button rect so it is centered in the circle
            self.level_start_button.rect = (bx, by, bw, bh)
            # self.level_start_button.label = "Start"  <-- REMOVED to allow dynamic labels
            self.level_start_button.draw(self.screen, font, self.theme, enabled=True)

        footer_y = left_rect.bottom - 20
        # always show progress percent (modes removed)
        pct = int(self.progress * 100)
        status = self.font_body.render(f"Progress: {pct}%", True, self.theme.MUTED)
        self.screen.blit(status, status.get_rect(center=(cx, footer_y)))
        self._draw_emg_plot(plot_rect)

        if self.banner_timer > 0 and self.banner:
            banner_rect = pygame.Rect(0, 0, 520, 44)
            banner_rect.center = (cx, arena_rect.y + 64)
            pygame.draw.rect(self.screen, (255, 255, 255), banner_rect, border_radius=12)
            pygame.draw.rect(self.screen, self.theme.DANGER, banner_rect, width=2, border_radius=12)
            t = self.font_body.render(self.banner, True, self.theme.TEXT)
            self.screen.blit(t, t.get_rect(center=banner_rect.center))

    def _star_index_from_pos(self, pos):
        panel = pygame.Rect(WIDTH // 2 - 390, HEIGHT // 2 - 180, 780, 360)
        y = panel.y + 226
        size = 22
        gap = 18
        total_w = 5 * (size * 2) + 4 * gap
        start_x = panel.centerx - total_w // 2 + size

        mx, my = pos
        if my < y - size - 12 or my > y + size + 12:
            return 0

        for i in range(5):
            cx = start_x + i * ((size * 2) + gap)
            if abs(mx - cx) <= size + 10:
                return i + 1
        return 0

    def _draw_complete(self):
        level = self.levels[self.level_index]
        panel = pygame.Rect(WIDTH // 2 - 390, HEIGHT // 2 - 180, 780, 360)
        self._draw_panel(panel)

        title = self.font_title.render("Level Complete", True, self.theme.TEXT)
        self.screen.blit(title, title.get_rect(center=(panel.centerx, panel.y + 70)))

        detail = self.font_body.render(level.title, True, self.theme.MUTED)
        self.screen.blit(detail, detail.get_rect(center=(panel.centerx, panel.y + 118)))

        duration = self.pending_duration_sec or 0.0
        time_line = self.font_body.render(f"Completion time: {duration:.2f} seconds", True, self.theme.MUTED)
        self.screen.blit(time_line, time_line.get_rect(center=(panel.centerx, panel.y + 150)))

        prompt = self.font_body.render("Press Next to continue", True, self.theme.TEXT)
        self.screen.blit(prompt, prompt.get_rect(center=(panel.centerx, panel.y + 178)))

        self.next_button.label = "Finish" if self.level_index >= len(self.levels) - 1 else "Next"
        self.next_button.draw(self.screen, self.font_button, self.theme, enabled=True)

    def _draw_end(self):
        panel = pygame.Rect(WIDTH // 2 - 390, HEIGHT // 2 - 180, 780, 360)
        self._draw_panel(panel)

        title = self.font_title.render("All Levels Complete", True, self.theme.TEXT)
        self.screen.blit(title, title.get_rect(center=(panel.centerx, panel.y + 90)))

        message = self.font_body.render("Thank you. Ask For Your Reward.", True, self.theme.MUTED)
        self.screen.blit(message, message.get_rect(center=(panel.centerx, panel.y + 145)))

        if self.logger.error:
            warn = self.font_body.render(f"Logging error: {self.logger.error}", True, self.theme.DANGER)
            self.screen.blit(warn, warn.get_rect(center=(panel.centerx, panel.y + 178)))
        else:
            info = self.font_body.render(f"File: {self.log_path.name}", True, self.theme.MUTED)
            self.screen.blit(info, info.get_rect(center=(panel.centerx, panel.y + 178)))

        self.restart_button.draw(self.screen, self.font_button, self.theme)


if __name__ == "__main__":
    Game().run()
