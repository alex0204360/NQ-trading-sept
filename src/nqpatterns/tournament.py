"""Finite tournament state, append-only evidence, and immutable finalist selection."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def append_event(path, value):
    text = canonical(value) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(text)
        stream.flush()


@dataclass
class TournamentState:
    round_number: int = 0
    quality: float = 0.0
    diminishing_count: int = 0
    stop_reason: str | None = None
    survivors: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    def complete_round(self, number, survivors, evaluable=True):
        if self.stop_reason or number != self.round_number + 1 or number > 4:
            raise ValueError("Rounds must be sequential and cannot resume a terminal tournament")
        previous = self.quality
        for row in survivors:
            self.survivors[row["signature_id"]] = max(
                [self.survivors.get(row["signature_id"], row), row],
                key=lambda r: r["lift_confidence_interval"][0],
            )
            self.quality = max(self.quality, row["lift_confidence_interval"][0])
        improvement = self.quality - previous
        if number >= 2 and evaluable:
            self.diminishing_count = self.diminishing_count + 1 if improvement < 0.01 else 0
        self.round_number = number
        conditions = []
        if number == 4:
            conditions.append("fixed_round_limit")
        if self.diminishing_count >= 2:
            conditions.append("diminishing_returns")
        strong = [r for r in self.survivors.values() if r["conservative_lift"] >= 0.08]
        if number >= 2 and len(strong) >= 3:
            conditions.append("early_quality_attempt")
        self.stop_reason = conditions[0] if conditions else None
        self.history.append(
            {
                "round": number,
                "quality": self.quality,
                "improvement": improvement,
                "evaluable": evaluable,
                "conditions": conditions,
                "diminishing_count": self.diminishing_count,
            }
        )
        return self.stop_reason


def freeze_finalists(rows, path, provenance):
    path = Path(path)
    if path.exists():
        raise FileExistsError("A frozen evaluation manifest cannot be replaced")
    ordered = sorted(
        rows,
        key=lambda r: (
            -r["lift_confidence_interval"][0],
            -r["sample_size"],
            r.get("mean_favorable_resolution") or float("inf"),
            r["pattern_id"],
        ),
    )
    seen, finalists = set(), []
    for row in ordered:
        if row["signature_id"] not in seen:
            seen.add(row["signature_id"])
            finalists.append(row)
        if len(finalists) == 12:
            break
    manifest = {"schema_version": 1, "finalists": finalists, "provenance": provenance}
    manifest["content_sha256"] = hashlib.sha256(canonical(manifest).encode()).hexdigest()
    write_json(path, manifest)
    return manifest


def verify_freeze(path):
    manifest = json.loads(Path(path).read_text())
    expected = manifest.pop("content_sha256")
    if hashlib.sha256(canonical(manifest).encode()).hexdigest() != expected:
        raise ValueError("Frozen manifest content changed")
    manifest["content_sha256"] = expected
    return manifest
