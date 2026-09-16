"""Small helpers for incremental experiment reports."""

from .quarto import QuartoReporter

__all__ = ["QuartoReporter", "ReportExport", "export_detached_report"]


def __getattr__(name: str):
    if name not in {"ReportExport", "export_detached_report"}:
        raise AttributeError(name)
    from .export import ReportExport, export_detached_report

    return {"ReportExport": ReportExport, "export_detached_report": export_detached_report}[name]
