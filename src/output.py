"""Write text exports without leaving a partially written destination."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator, TextIO


@contextmanager
def atomic_text_output(path: Path, *, newline: str | None = None) -> Iterator[TextIO]:
    """Replace the destination only after the temporary file closes successfully."""
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline=newline,
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            yield output_file
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
