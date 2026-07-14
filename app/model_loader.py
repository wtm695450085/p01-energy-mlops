"""Model loader - loading the sklearn pipeline from disk."""
import json
from pathlib import Path
from typing import Optional

import joblib

# Possible model locations (Docker / dev)
_MODEL_CANDIDATES = [
    Path("/app/models/champion/model.joblib"),
    Path("/app/models/model.joblib"),
    Path("models/champion/model.joblib"),
    Path("models/model.joblib"),
]

_METADATA_CANDIDATES = [
    Path("/app/models/model_metadata.json"),
    Path("models/model_metadata.json"),
]


class ModelState:
    model = None
    model_file: Optional[str] = None
    metadata: dict = {}
    error: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self.model is not None


model_state = ModelState()


def _find_file(candidates: list[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_model() -> None:
    """Loads the model from the first existing location."""
    model_path = _find_file(_MODEL_CANDIDATES)
    metadata_path = _find_file(_METADATA_CANDIDATES)

    if model_path is None:
        model_state.error = (
            "Model file missing. Run the notebook to generate models/model.joblib"
        )
        model_state.model = None
        model_state.model_file = None
        return

    try:
        model_state.model = joblib.load(model_path)
        model_state.model_file = str(model_path)
        model_state.error = None
    except Exception as e:
        model_state.error = f"Model loading error: {e}"
        model_state.model = None
        return

    if metadata_path:
        try:
            model_state.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            model_state.metadata = {}


def reload_model() -> dict:
    """Reloads the model without restarting the application (for /admin/reload-model)."""
    load_model()
    return {
        "model_loaded": model_state.loaded,
        "model_file": model_state.model_file,
        "error": model_state.error,
    }
