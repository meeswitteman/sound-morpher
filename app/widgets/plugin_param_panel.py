from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QWidget,
)

from plugins.base import MorphPlugin, PluginParam


class PluginParamPanel(QGroupBox):
    """Auto-generated parameter controls for the active MorphPlugin."""

    params_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Algorithm Parameters", parent)
        self._form = QFormLayout(self)
        self._form.setSpacing(5)
        self._widgets: dict[str, QWidget] = {}

    # ── Public API ─────────────────────────────────────────────────────

    def load_plugin(self, plugin: MorphPlugin) -> None:
        """Rebuild UI for the given plugin's parameter list."""
        self._clear()
        for param in plugin.parameters:
            widget = self._make_widget(param)
            self._widgets[param.name] = widget
            self._form.addRow(param.label + ":", widget)
        self.setVisible(bool(plugin.parameters))

    def get_params(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, QComboBox):
                result[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                result[name] = widget.isChecked()
            elif isinstance(widget, QDoubleSpinBox):
                result[name] = widget.value()
            elif isinstance(widget, QSpinBox):
                result[name] = widget.value()
        return result

    def set_params(self, params: dict[str, Any]) -> None:
        for name, value in params.items():
            widget = self._widgets.get(name)
            if widget is None:
                continue
            if isinstance(widget, QComboBox):
                idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))

    # ── Internal ───────────────────────────────────────────────────────

    def _clear(self) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._widgets.clear()

    def _make_widget(self, param: PluginParam) -> QWidget:
        if param.type == "choice":
            w = QComboBox()
            for ch in (param.choices or []):
                w.addItem(ch)
            idx = w.findText(str(param.default))
            if idx >= 0:
                w.setCurrentIndex(idx)
            if param.tooltip:
                w.setToolTip(param.tooltip)
            w.currentIndexChanged.connect(self.params_changed)
            return w

        if param.type == "bool":
            w = QCheckBox()
            w.setChecked(bool(param.default))
            if param.tooltip:
                w.setToolTip(param.tooltip)
            w.checkStateChanged.connect(self.params_changed)
            return w

        if param.type == "float":
            w = QDoubleSpinBox()
            w.setDecimals(2)
            if param.min_val is not None:
                w.setMinimum(float(param.min_val))
            if param.max_val is not None:
                w.setMaximum(float(param.max_val))
            w.setValue(float(param.default))
            if param.tooltip:
                w.setToolTip(param.tooltip)
            w.valueChanged.connect(self.params_changed)
            return w

        # int (default fallback)
        w = QSpinBox()
        if param.min_val is not None:
            w.setMinimum(int(param.min_val))
        if param.max_val is not None:
            w.setMaximum(int(param.max_val))
        w.setValue(int(param.default))
        if param.tooltip:
            w.setToolTip(param.tooltip)
        w.valueChanged.connect(self.params_changed)
        return w
