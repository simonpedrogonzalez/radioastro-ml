"""Asynchronous, coalescing Quarto rendering."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path


class QuartoReporter:
    def __init__(self, quarto_file: str | Path, every: int = 5):
        self.quarto_file = Path(quarto_file).expanduser().resolve()
        if every <= 0:
            raise ValueError("every must be positive")
        if self.quarto_file.suffix.lower() != ".qmd" or not self.quarto_file.is_file():
            raise FileNotFoundError(f"Quarto file not found: {self.quarto_file}")

        self.every = every
        self.completed = 0
        self._condition = threading.Condition()
        self._render_requested = False
        self._stopping = False
        self._finished = False
        self._worker = threading.Thread(target=self._work, name="quarto-reporter", daemon=True)
        self._worker.start()

    def sample_completed(self) -> None:
        with self._condition:
            if self._finished:
                return
            self.completed += 1
            if self.completed % self.every == 0:
                self._render_requested = True
                self._condition.notify()

    def finish(self) -> None:
        with self._condition:
            if self._finished:
                return
            self._finished = True
            self._render_requested = True
            self._stopping = True
            self._condition.notify()
        self._worker.join()

    def _work(self) -> None:
        while True:
            with self._condition:
                while not self._render_requested:
                    if self._stopping:
                        return
                    self._condition.wait()
                self._render_requested = False

            self._render()

    def _render(self) -> None:
        try:
            result = subprocess.run(
                ["quarto", "render", self.quarto_file.name],
                cwd=self.quarto_file.parent,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            print(f"Report render failed: {exc}")
            return

        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            print(f"Report render failed: {detail}")
            return

        print(f"Report updated: {self.quarto_file.with_suffix('.html')}")
