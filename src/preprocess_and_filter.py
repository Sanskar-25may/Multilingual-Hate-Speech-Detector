"""
Phase 1 data pipeline: social-media text cleaning and TF-IDF sample selection.

Loads ``data/raw_dataset.csv``, standardizes each post, ranks documents by
aggregate TF-IDF score, retains the top 75% most informative sentences, and
writes ``data/preprocessed_filtered_dataset.csv``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
RAW_DATASET_PATH: Final[Path] = PROJECT_ROOT / "data" / "raw_dataset.csv"
OUTPUT_DATASET_PATH: Final[Path] = PROJECT_ROOT / "data" / "preprocessed_filtered_dataset.csv"
TEXT_COLUMN_CANDIDATES: Final[tuple[str, ...]] = (
    "text",
    "tweet",
    "comment",
    "sentence",
    "content",
    "post",
    "message",
)
KEEP_RATIO: Final[float] = 0.75
CLEANED_TEXT_COLUMN: Final[str] = "cleaned_text"
TFIDF_SCORE_COLUMN: Final[str] = "tfidf_score"

# Architecture §3.1 / Rule 1: mention and URL normalisation tokens.
USER_TOKEN: Final[str] = "[USER]"
URL_TOKEN: Final[str] = "[URL]"

URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:https?://|www\.)\S+",
    flags=re.IGNORECASE,
)
MENTION_PATTERN: Final[re.Pattern[str]] = re.compile(r"@\w+")
WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")
# Rule 1: strip emojis / pictographs without adding extra dependencies.
EMOJI_PATTERN: Final[re.Pattern[str]] = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)


def log(message: str) -> None:
    """Print a timestamp-free, single-line progress message."""
    print(f"[preprocess] {message}", flush=True)


def clean_social_text(text: object) -> str:
    """Standardize raw social-media text for multilingual hate-speech training.

    Steps (architecture §3.1 and Rule 1):
        1. Convert to lowercase (Romanised Hinglish / English).
        2. Replace user mentions (``@name``) with ``[USER]``.
        3. Replace URLs with ``[URL]``.
        4. Strip emojis.
        5. Collapse whitespace and strip ends.

    Args:
        text: A raw cell value from the dataset.

    Returns:
        A cleaned string. Non-string or missing values become ``""``.
    """
    if not isinstance(text, str):
        return ""

    cleaned = text.lower()
    cleaned = URL_PATTERN.sub(URL_TOKEN, cleaned)
    cleaned = MENTION_PATTERN.sub(USER_TOKEN, cleaned)
    cleaned = EMOJI_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def resolve_text_column(columns: pd.Index) -> str:
    """Pick the raw text column from common IndoHateMix / HASOC names.

    Args:
        columns: DataFrame column index.

    Returns:
        The name of the column to clean.

    Raises:
        ValueError: If no suitable text column exists.
    """
    lowered = {str(name).lower(): str(name) for name in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(
        "Could not find a text column. Expected one of "
        f"{list(TEXT_COLUMN_CANDIDATES)}; got {list(columns)}."
    )


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Load the raw CSV, validating that it exists and is non-empty.

    Args:
        path: Absolute path to ``raw_dataset.csv``.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the CSV is missing.
        ValueError: If the file is empty or unreadable as a table.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'. "
            "Place your CSV at data/raw_dataset.csv."
        )

    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Raw dataset at '{path}' is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse CSV at '{path}': {exc}") from exc

    if frame.empty:
        raise ValueError(f"Raw dataset at '{path}' contains no rows.")

    return frame


def apply_tfidf_filtering(
    frame: pd.DataFrame,
    text_column: str,
    keep_ratio: float = KEEP_RATIO,
) -> pd.DataFrame:
    """Rank documents by summed TF-IDF and keep the top ``keep_ratio``.

    Fits a ``TfidfVectorizer`` on the cleaned corpus, sums term scores per
    sentence, sorts descending, and drops the bottom 25% (architecture §3.1,
    Rule 2, Phase 1 §3).

    Args:
        frame: DataFrame containing cleaned text.
        text_column: Column to vectorize.
        keep_ratio: Fraction of highest-scoring rows to retain (default 0.75).

    Returns:
        A copy of the filtered DataFrame without the intermediate score column.

    Raises:
        ValueError: If ``keep_ratio`` is invalid or the text column is missing.
    """
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1]; got {keep_ratio}.")
    if text_column not in frame.columns:
        raise ValueError(f"Column '{text_column}' is not in the DataFrame.")

    working = frame.copy()
    n_original = len(working)
    log(f"Original dataset size: {n_original} samples")

    corpus = working[text_column].fillna("").astype(str)
    if corpus.str.strip().eq("").all():
        raise ValueError(
            f"Column '{text_column}' has no non-empty documents after cleaning."
        )

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError as exc:
        raise ValueError(
            "TF-IDF vectorization failed. The cleaned corpus may contain "
            f"only stop words or empty strings: {exc}"
        ) from exc

    # Aggregate TF-IDF(t, d) over terms t in each document d.
    working[TFIDF_SCORE_COLUMN] = tfidf_matrix.sum(axis=1).A1
    ranked = working.sort_values(by=TFIDF_SCORE_COLUMN, ascending=False)
    cutoff_index = max(1, int(len(ranked) * keep_ratio))
    filtered = ranked.iloc[:cutoff_index].drop(columns=[TFIDF_SCORE_COLUMN])

    pct = int(keep_ratio * 100)
    log(f"Filtered dataset size (top {pct}%): {len(filtered)} samples")
    log(
        f"Discarded {n_original - len(filtered)} low-information samples "
        f"({100 - pct}% tail)."
    )
    return filtered.reset_index(drop=True)


def save_filtered_dataset(frame: pd.DataFrame, path: Path) -> None:
    """Write the optimized training set to CSV.

    Args:
        frame: Filtered DataFrame.
        path: Destination path.

    Raises:
        OSError: If the parent directory cannot be created or the file
            cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    log(f"Wrote {len(frame)} rows to '{path}'")


def run_pipeline(
    input_path: Path = RAW_DATASET_PATH,
    output_path: Path = OUTPUT_DATASET_PATH,
    keep_ratio: float = KEEP_RATIO,
) -> Path:
    """Execute cleaning + TF-IDF filtering end to end.

    Args:
        input_path: Path to the raw CSV.
        output_path: Path for the filtered CSV.
        keep_ratio: Fraction of informative sentences to keep.

    Returns:
        The output path that was written.
    """
    log(f"Loading raw CSV from '{input_path}'")
    raw = load_raw_dataset(input_path)
    log(f"Loaded {len(raw)} rows with columns: {list(raw.columns)}")

    text_col = resolve_text_column(raw.columns)
    log(f"Using text column: '{text_col}'")

    log("Cleaning text (lowercase, [USER], [URL], whitespace, emojis)...")
    raw[CLEANED_TEXT_COLUMN] = raw[text_col].apply(clean_social_text)

    n_empty = int(raw[CLEANED_TEXT_COLUMN].eq("").sum())
    if n_empty:
        log(f"Warning: {n_empty} rows are empty after cleaning.")

    log("Applying 75% TF-IDF sentence-ranking filter...")
    filtered = apply_tfidf_filtering(
        raw,
        text_column=CLEANED_TEXT_COLUMN,
        keep_ratio=keep_ratio,
    )

    log(f"Saving optimized training set to '{output_path}'")
    save_filtered_dataset(filtered, output_path)
    return output_path


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    try:
        run_pipeline()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error writing output: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — last-resort safety net
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
