"""Startup script for the PlanVectorizer MVP."""

from __future__ import annotations

from pathlib import Path

from app.pipeline import run_pipeline


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "input" / "source.png"
    output_path = base_dir / "output" / "result.svg"

    try:
        run_pipeline(str(input_path), str(output_path))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Pipeline failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
