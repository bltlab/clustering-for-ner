from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cluster_eval import Clusters
    from .models import Report, Settings

__all__ = ["Clusters", "Settings", "Report"]

# Map each public name to the submodule that defines it so it can be
# imported lazily. This keeps `import clustering_for_ner.clustering`
# (and, by extension, the CLI) cheap: the heavy scientific stack pulled
# in by `cluster_eval` is only imported when `Clusters` is first accessed.
_LAZY = {"Clusters": ".cluster_eval", "Settings": ".models", "Report": ".models"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
