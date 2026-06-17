from __future__ import annotations

from PyQt6 import QtCore, QtWidgets


DASHBOARD_STYLESHEET = """
QWidget {
    font-family: "Microsoft YaHei", "Segoe UI", Arial;
    color: #23272f;
    background: #f6f7f2;
}
QLabel {
    background: transparent;
}
QFrame#dashboardSidebar {
    background: #fbfcf8;
    border-right: 1px solid #d9dfd4;
}
QFrame#dashboardPanel, QFrame#metricCard, QFrame#statusPill {
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid #d9dfd4;
    border-radius: 8px;
}
QFrame#darkStage {
    background: #161a1d;
    border-radius: 7px;
}
QLabel#brandMark {
    background: #e3f3f1;
    border: 2px solid #23272f;
    font-weight: 900;
}
QLabel#brandTitle {
    font-size: 18px;
    font-weight: 800;
}
QLabel#brandSubtitle, QLabel#mutedText, QLabel#metricLabel {
    color: #667085;
}
QLabel#pageTitle {
    font-size: 28px;
    font-weight: 900;
}
QLabel#sectionTitle {
    font-size: 17px;
    font-weight: 850;
}
QLabel#metricValue {
    font-size: 18px;
    font-weight: 850;
}
QLabel#protocolBadge {
    color: #0f8b8d;
    background: #e3f3f1;
    border: 1px solid rgba(15, 139, 141, 0.24);
    border-radius: 7px;
    padding: 8px 10px;
    font-weight: 850;
}
QLabel#navItem {
    color: #667085;
    padding: 10px 12px;
    border-radius: 8px;
}
QLabel#navItemActive {
    color: #ffffff;
    background: #23272f;
    padding: 10px 12px;
    border-radius: 8px;
    font-weight: 800;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    min-height: 34px;
    border: 1px solid #d9dfd4;
    border-radius: 7px;
    background: #ffffff;
    padding: 5px 8px;
}
QTextEdit, QTextBrowser {
    border: 1px solid #d9dfd4;
    border-radius: 7px;
    background: #ffffff;
    padding: 8px;
}
QPushButton {
    min-height: 36px;
    border: 1px solid #d9dfd4;
    border-radius: 7px;
    background: #ffffff;
    padding: 7px 13px;
    font-weight: 750;
}
QPushButton#primaryButton {
    color: #ffffff;
    border-color: #0f8b8d;
    background: #0f8b8d;
}
QPushButton#softButton {
    border-color: rgba(15, 139, 141, 0.24);
    background: #e3f3f1;
}
QPushButton#dangerButton {
    color: #ffffff;
    border-color: #d83b3b;
    background: #d83b3b;
}
QPushButton:disabled {
    color: #98a2b3;
    background: #edf0ea;
}
"""


def apply_dashboard_style(widget: QtWidgets.QWidget) -> None:
    widget.setStyleSheet(DASHBOARD_STYLESHEET)


def label(text: str, object_name: str = "") -> QtWidgets.QLabel:
    widget = QtWidgets.QLabel(text)
    if object_name:
        widget.setObjectName(object_name)
    return widget


def make_sidebar(
    title: str,
    subtitle: str,
    active_label: str,
    secondary_labels: list[str],
    note: str,
) -> QtWidgets.QFrame:
    sidebar = QtWidgets.QFrame()
    sidebar.setObjectName("dashboardSidebar")
    sidebar.setFixedWidth(230)
    layout = QtWidgets.QVBoxLayout(sidebar)
    layout.setContentsMargins(18, 22, 18, 18)
    layout.setSpacing(16)

    brand_row = QtWidgets.QHBoxLayout()
    mark = label("CB", "brandMark")
    mark.setFixedSize(42, 42)
    mark.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    brand_text = QtWidgets.QVBoxLayout()
    brand_text.setSpacing(2)
    brand_text.addWidget(label(title, "brandTitle"))
    brand_text.addWidget(label(subtitle, "brandSubtitle"))
    brand_row.addWidget(mark)
    brand_row.addLayout(brand_text)
    layout.addLayout(brand_row)

    nav = QtWidgets.QVBoxLayout()
    nav.setSpacing(8)
    active = label(active_label, "navItemActive")
    nav.addWidget(active)
    for item in secondary_labels:
        nav.addWidget(label(item, "navItem"))
    layout.addLayout(nav)

    layout.addStretch(1)
    note_label = label(note, "mutedText")
    note_label.setWordWrap(True)
    note_label.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
    note_label.setContentsMargins(10, 10, 10, 10)
    layout.addWidget(note_label)
    return sidebar


def make_panel(title: str, subtitle: str = "") -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]:
    panel = QtWidgets.QFrame()
    panel.setObjectName("dashboardPanel")
    panel_layout = QtWidgets.QVBoxLayout(panel)
    panel_layout.setContentsMargins(14, 14, 14, 14)
    panel_layout.setSpacing(12)

    header = QtWidgets.QVBoxLayout()
    header.setSpacing(4)
    header.addWidget(label(title, "sectionTitle"))
    if subtitle:
        muted = label(subtitle, "mutedText")
        muted.setWordWrap(True)
        header.addWidget(muted)
    panel_layout.addLayout(header)
    return panel, panel_layout


def make_metric_card(title: str, value_widget: QtWidgets.QWidget) -> QtWidgets.QFrame:
    card = QtWidgets.QFrame()
    card.setObjectName("metricCard")
    layout = QtWidgets.QVBoxLayout(card)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(4)
    layout.addWidget(label(title, "metricLabel"))
    if isinstance(value_widget, QtWidgets.QLabel):
        value_widget.setObjectName("metricValue")
    layout.addWidget(value_widget)
    return card


def set_button_role(button: QtWidgets.QPushButton, role: str) -> None:
    if role == "primary":
        button.setObjectName("primaryButton")
    elif role == "soft":
        button.setObjectName("softButton")
    elif role == "danger":
        button.setObjectName("dangerButton")
