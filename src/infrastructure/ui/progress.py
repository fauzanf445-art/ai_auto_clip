from typing import Optional, Iterable
from tqdm import tqdm

from src.domain.interfaces import IProgressReporter, IProgressBar, ILogger

# ==================================================================================
# TQDM-based Progress Reporter (for CLI)
# ==================================================================================

class TqdmProgressBar(IProgressBar):
    def __init__(self, bar: tqdm):
        self._bar = bar
    
    def update(self, n: int = 1):
        self._bar.update(n)
        
    def close(self):
        self._bar.close()

    @property
    def n(self) -> int:
        return self._bar.n

    @property
    def total(self) -> int:
        return self._bar.total or 0
        
    @total.setter
    def total(self, value: int):
        self._bar.total = value
        self._bar.refresh()

    @property
    def disable(self) -> bool:
        return self._bar.disable

class TqdmProgressReporter(IProgressReporter):
    """Implementasi IProgressReporter menggunakan library tqdm."""
    
    def sequence(self, iterable: Iterable, desc: str = "", unit: str = "it", total: Optional[int] = None) -> Iterable:
        """Membungkus iterable dengan tqdm progress bar."""
        return tqdm(iterable, desc=desc, unit=unit, total=total)
    
    def manual(self, total: int, desc: str = "", unit: str = "it", leave: bool = True) -> IProgressBar:
        """Membuat manual progress bar."""
        bar = tqdm(total=total, desc=desc, unit=unit, leave=leave)
        return TqdmProgressBar(bar)

# ==================================================================================
# Logger-based Progress Reporter (for Web/Headless)
# ==================================================================================

class LogProgressBar(IProgressBar):
    """Implementasi IProgressBar yang hanya mencetak log, tanpa bar visual."""
    def __init__(self, logger: ILogger, total: int, desc: str, unit: str):
        self._logger = logger
        self._total = total
        self._desc = desc
        self._unit = unit
        self._n = 0
        self._logger.info(f"Progress Start: {self._desc} (Total: {self._total} {self._unit})")

    def update(self, n: int = 1):
        self._n += n
        # Optional: Log setiap N update untuk menghindari spamming log
        if self._n % 10 == 0 or self._n == self._total:
             self._logger.debug(f"Progress: {self._desc} - {self._n}/{self._total} {self._unit}")

    def close(self):
        self._logger.info(f"Progress Finish: {self._desc}")

    @property
    def n(self) -> int: return self._n
    @property
    def total(self) -> int: return self._total
    @total.setter
    def total(self, value: int): self._total = value
    @property
    def disable(self) -> bool: return False

class LogProgressReporter(IProgressReporter):
    """Implementasi IProgressReporter yang menggunakan logger, cocok untuk mode non-interaktif."""
    def __init__(self, logger: ILogger):
        self._logger = logger

    def sequence(self, iterable: Iterable, desc: str = "", unit: str = "it", total: Optional[int] = None) -> Iterable:
        self._logger.info(f"Starting sequence: {desc}")
        for item in iterable:
            yield item
        self._logger.info(f"Finished sequence: {desc}")

    def manual(self, total: int, desc: str = "", unit: str = "it", leave: bool = True) -> IProgressBar:
        """Membuat manual progress bar berbasis log."""
        return LogProgressBar(self._logger, total, desc, unit)