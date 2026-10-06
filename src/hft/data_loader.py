"""
Loader danych tikowych z HuggingFace Hub.

Odpowiedzialnosci:
- Pobieranie plikow CSV z repozytorium HuggingFace
- Wczytywanie CSV chunkami (unikniecie OOM na duzych plikach)
- Sanity checks: filtracja blednych tikow (spread < 0, volume == 0)
- Obliczenie Mid-Price = (open + close) / 2
- Optymalizacja typow: float64 -> float32 (oszczednosc ~50% RAM)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from config.settings import HFTConfig

logger = logging.getLogger(__name__)

# Kolumny do rzutowania na float32 (oszczednosc pamieci)
_FLOAT_DOWNCAST_EXCLUDE = ("time",)


class HFTDataLoader:
    """
    Loader danych tikowych — pobiera z HuggingFace i iteruje batchami.

    Uzycie:
        loader = HFTDataLoader(config)
        loader.download()  # jednorazowe pobranie
        for batch in loader.iter_batches("train"):
            process(batch)
    """

    def __init__(self, config: HFTConfig | None = None) -> None:
        self.config = config or HFTConfig()
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pobieranie danych
    # ------------------------------------------------------------------

    def download(self) -> dict[str, Path]:
        """
        Pobiera pliki CSV z HuggingFace Hub (jesli brak lokalnie).

        Returns:
            Slownik {split_name: local_path} dla train/val/test.
        """
        from huggingface_hub import hf_hub_download

        file_map = {
            "train": self.config.train_file,
            "val": self.config.val_file,
            "test": self.config.test_file,
        }

        local_paths: dict[str, Path] = {}

        for split_name, remote_path in file_map.items():
            local_file = self.config.data_dir / f"gold_{split_name}.csv"

            if local_file.exists():
                size_mb = local_file.stat().st_size / (1024 * 1024)
                logger.info(
                    "[%s] Plik juz istnieje: %s (%.1f MB) — pomijam pobieranie.",
                    split_name.upper(), local_file.name, size_mb,
                )
                local_paths[split_name] = local_file
                continue

            logger.info(
                "[%s] Pobieranie z HuggingFace: %s/%s ...",
                split_name.upper(), self.config.hf_repo, remote_path,
            )

            downloaded = hf_hub_download(
                repo_id=self.config.hf_repo,
                filename=remote_path,
                repo_type="dataset",
                local_dir=str(self.config.data_dir),
            )

            # Przenies plik do katalogu glownego data/hft/
            downloaded_path = Path(downloaded)
            if downloaded_path != local_file:
                downloaded_path.rename(local_file)

            size_mb = local_file.stat().st_size / (1024 * 1024)
            logger.info(
                "[%s] Pobrano -> %s (%.1f MB)",
                split_name.upper(), local_file.name, size_mb,
            )
            local_paths[split_name] = local_file

        return local_paths

    # ------------------------------------------------------------------
    # Iteracja batchami
    # ------------------------------------------------------------------

    def iter_batches(
        self,
        split: str = "train",
        batch_rows: int | None = None,
    ) -> Iterator[pd.DataFrame]:
        """
        Generator iterujacy po danych CSV w batchach.

        Kazdy batch przechodzi przez:
        1. Sanity checks (filtracja blednych tikow)
        2. Obliczenie mid_price
        3. Downcast float64 -> float32

        Args:
            split: Nazwa splitu ('train', 'val', 'test').
            batch_rows: Rozmiar batcha (domyslnie z configu).

        Yields:
            DataFrame z oczyszczonymi danymi (batch_rows wierszy).
        """
        batch_rows = batch_rows or self.config.batch_rows
        csv_path = self.config.data_dir / f"gold_{split}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Plik danych nie istnieje: {csv_path}. "
                f"Uruchom download() najpierw."
            )

        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Ladowanie %s chunkami po %d wierszy (plik: %.1f MB)...",
            csv_path.name, self.config.chunk_size, file_size_mb,
        )

        buffer_chunks: list[pd.DataFrame] = []
        buffer_rows = 0
        total_rows_yielded = 0
        batch_idx = 0

        for chunk in pd.read_csv(
            csv_path,
            chunksize=self.config.chunk_size,
            low_memory=False,
        ):
            # Sanity checks na chunk-level
            chunk = _apply_sanity_checks(chunk)
            chunk = _compute_mid_price(chunk)
            chunk = _downcast_floats(chunk)

            buffer_chunks.append(chunk)
            buffer_rows += len(chunk)

            # Kiedy bufor osiagnie rozmiar batcha — yield
            while buffer_rows >= batch_rows:
                combined = pd.concat(buffer_chunks, ignore_index=True)
                yield_df = combined.iloc[:batch_rows]
                remaining = combined.iloc[batch_rows:]

                batch_idx += 1
                total_rows_yielded += len(yield_df)
                logger.info(
                    "  Batch #%d: %d wierszy (lacznie: %d)",
                    batch_idx, len(yield_df), total_rows_yielded,
                )

                yield yield_df

                # Zachowaj reszte w buforze
                buffer_chunks = [remaining] if len(remaining) > 0 else []
                buffer_rows = len(remaining) if len(remaining) > 0 else 0

        # Ostatni niepelny batch (jesli jest)
        if buffer_chunks:
            combined = pd.concat(buffer_chunks, ignore_index=True)
            if len(combined) > 0:
                batch_idx += 1
                total_rows_yielded += len(combined)
                logger.info(
                    "  Batch #%d (ostatni): %d wierszy (lacznie: %d)",
                    batch_idx, len(combined), total_rows_yielded,
                )
                yield combined

        logger.info("Zakonczono ladowanie: %d batchow, %d wierszy.", batch_idx, total_rows_yielded)

    def load_full(self, split: str = "test") -> pd.DataFrame:
        """
        Wczytuje caly split do pamieci (uzyj dla val/test — mniejsze pliki).

        Args:
            split: Nazwa splitu.

        Returns:
            Pelny DataFrame po sanity checks.
        """
        csv_path = self.config.data_dir / f"gold_{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Plik nie istnieje: {csv_path}")

        logger.info("Wczytywanie pelnego splitu: %s ...", csv_path.name)
        df = pd.read_csv(csv_path, low_memory=False)
        df = _apply_sanity_checks(df)
        df = _compute_mid_price(df)
        df = _downcast_floats(df)
        logger.info("Wczytano %d wierszy z %s.", len(df), split)
        return df


# ======================================================================
# Funkcje pomocnicze (prywatne)
# ======================================================================

def _apply_sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtruje bledy brokerskie w danych tikowych.

    Usuwa wiersze gdzie:
    - spread < 0 (blad danych)
    - volume == 0 (pusty tik)
    """
    initial_rows = len(df)

    if "spread" in df.columns:
        df = df[df["spread"] >= 0]

    if "volume" in df.columns:
        df = df[df["volume"] > 0]

    removed = initial_rows - len(df)
    if removed > 0:
        logger.debug(
            "Sanity check: usunieto %d blednych wierszy (%.2f%%).",
            removed, removed / initial_rows * 100,
        )

    return df.reset_index(drop=True)


def _compute_mid_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oblicza Mid-Price jako (open + close) / 2.

    Redukuje szum Bid-Ask Bounce w danych tikowych.
    """
    if "open" in df.columns and "close" in df.columns:
        df["mid_price"] = (df["open"].values + df["close"].values) / 2.0

    return df


def _downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rzutuje kolumny float64 na float32 (oszczednosc ~50% pamieci RAM).
    """
    float_cols = df.select_dtypes(include=[np.float64]).columns
    cols_to_cast = [c for c in float_cols if c not in _FLOAT_DOWNCAST_EXCLUDE]

    if cols_to_cast:
        df[cols_to_cast] = df[cols_to_cast].astype(np.float32)

    return df
