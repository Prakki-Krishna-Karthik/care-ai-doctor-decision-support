"""Print a compact local MIMIC discovery summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

backend = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(backend))

from config import Settings  # noqa: E402
from data import DatasetIndex  # noqa: E402


if __name__ == "__main__":
    index = DatasetIndex(Settings())
    index.refresh()
    print(json.dumps({
        "image_root": str(index.settings.image_root),
        "data_root": str(index.settings.data_root),
        "images": len(index.images),
        "clinical_files": [str(path) for path in index.clinical_files],
        "clinical_rows": len(index.clinical_rows),
        "label_fields": index.label_field_counts(),
        "scan_error": index.scan_error,
    }, indent=2))
