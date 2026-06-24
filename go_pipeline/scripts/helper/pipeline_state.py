"""
pipeline_state.py

Small, dependency-free checkpoint module for the GO pipeline.

Idea:
- For each pipeline step (main.py stage) OR each individual signature within
  a script, we store in a JSON file whether it completed successfully ("done")
  or failed ("failed").
- On the next run, this file is read: already completed items are skipped,
  failed or missing ones are retried.

Usage (see examples in the scripts):

    from pipeline_state import PipelineState

    state = PipelineState("results/.representatives_state.json")

    for sig in signatures:
        if state.is_done("representatives", sig.name):
            continue
        try:
            process(sig)
            state.mark_done("representatives", sig.name)
        except Exception as e:
            state.mark_failed("representatives", sig.name, str(e))
            continue  # move on instead of stopping the pipeline
"""

import json
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone


class PipelineState:
    def __init__(self, state_file):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                # If the state file is corrupted or truncated
                # (e.g. crash during write), we just start fresh
                # instead of breaking the pipeline again.
                return {}
        return {}

    def reload(self):
        """Reload state from disk (useful if another process updated it)."""
        self.data = self._load()

    def _save(self):
        # Atomic write: write to a temp file first, then replace the real one.
        # This keeps the state file consistent even if something crashes mid-write.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.state_file.parent), prefix=".tmp_state_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp_path, self.state_file)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def is_done(self, group: str, key: str) -> bool:
        return self.data.get(group, {}).get(key, {}).get("status") == "done"

    def is_failed(self, group: str, key: str) -> bool:
        return self.data.get(group, {}).get(key, {}).get("status") == "failed"

    def mark_done(self, group: str, key: str):
        self.data.setdefault(group, {})[key] = {
            "status": "done",
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def mark_failed(self, group: str, key: str, error: str = ""):
        self.data.setdefault(group, {})[key] = {
            "status": "failed",
            "error": str(error)[:500],
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def reset(self, group: str, key: str = None):
        """Reset state for a group or a single key to force rerun."""
        if key is None:
            self.data.pop(group, None)
        else:
            self.data.get(group, {}).pop(key, None)
        self._save()

    def summary(self, group: str):
        items = self.data.get(group, {})
        done = sum(1 for v in items.values() if v.get("status") == "done")
        failed = sum(1 for v in items.values() if v.get("status") == "failed")
        return {"done": done, "failed": failed, "total": len(items)}

    def failed_keys(self, group: str):
        items = self.data.get(group, {})
        return [k for k, v in items.items() if v.get("status") == "failed"]