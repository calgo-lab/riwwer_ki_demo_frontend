"""Session state management module."""
from .manager import StateManager
from .callbacks import on_scope_change, on_scope_change_deferred

__all__ = ["StateManager", "on_scope_change", "on_scope_change_deferred"]