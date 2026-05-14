from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

from plugins.base import MorphPlugin


class PluginRegistry:
    """Stores and retrieves MorphPlugin instances by name."""

    def __init__(self) -> None:
        self._plugins: dict[str, MorphPlugin] = {}

    # ── Registration ───────────────────────────────────────────────────

    def register(self, plugin: MorphPlugin) -> None:
        if not plugin.name:
            raise ValueError(f"Plugin {type(plugin).__name__} has no name")
        self._plugins[plugin.name] = plugin

    def register_all(self, plugins: list[MorphPlugin]) -> None:
        for p in plugins:
            self.register(p)

    # ── Lookup ─────────────────────────────────────────────────────────

    def names(self) -> list[str]:
        return list(self._plugins.keys())

    def get(self, name: str) -> MorphPlugin:
        try:
            return self._plugins[name]
        except KeyError:
            raise KeyError(f"No plugin named '{name}'. Available: {self.names()}")

    def __contains__(self, name: str) -> bool:
        return name in self._plugins

    def __len__(self) -> int:
        return len(self._plugins)

    # ── Auto-discovery ─────────────────────────────────────────────────

    def load_from_directory(self, directory: Path) -> None:
        """Import every .py file in directory and register any MorphPlugin subclasses found."""
        skip = {"__init__.py", "base.py", "registry.py"}
        for path in sorted(directory.glob("*.py")):
            if path.name in skip:
                continue
            spec = importlib.util.spec_from_file_location(
                f"_user_plugin_{path.stem}", path
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
            except Exception as exc:
                print(f"[PluginRegistry] Could not load {path.name}: {exc}")
                continue
            for attr in vars(module).values():
                if (
                    isinstance(attr, type)
                    and issubclass(attr, MorphPlugin)
                    and attr is not MorphPlugin
                    and attr.name
                    and attr.name not in self._plugins
                ):
                    self.register(attr())


def build_default_registry() -> PluginRegistry:
    """Create a registry pre-loaded with all built-in plugins."""
    from plugins.crossfade import CrossfadePlugin
    from plugins.spectral_fft import SpectralFftPlugin
    from plugins.pitch_shift import PitchShiftPlugin
    from plugins.granular import GranularPlugin
    from plugins.vocoder import VocoderPlugin

    registry = PluginRegistry()
    registry.register_all([
        CrossfadePlugin(),
        SpectralFftPlugin(),
        PitchShiftPlugin(),
        GranularPlugin(),
        VocoderPlugin(),
    ])
    return registry
