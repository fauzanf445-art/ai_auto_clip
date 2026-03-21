import time
from typing import Any, Tuple, Type

from src.domain.interfaces import IRetryHandler, ILogger

class RetryHandler(IRetryHandler):
    """Implementasi mekanisme retry dengan exponential backoff."""
    def __init__(self, logger: ILogger, max_attempts: int = 3, initial_delay: float = 2.0, backoff_factor: float = 2.0, retry_on: Tuple[Type[Exception], ...] = (Exception,)):
        self.logger = logger
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.retry_on = retry_on

    def execute(self, func: Any, *args, **kwargs) -> Any:
        delay = self.initial_delay
        for attempt in range(1, self.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Jika error tidak termasuk dalam daftar retry_on, lempar langsung
                if not isinstance(e, self.retry_on):
                    raise e

                func_name = getattr(func, '__name__', 'Operation')
                if attempt == self.max_attempts:
                    self.logger.error(f"❌ {func_name} gagal total setelah {self.max_attempts} percobaan. Error: {e}")
                    raise e
                
                self.logger.warning(f"⚠️ {func_name} gagal (percobaan {attempt}/{self.max_attempts}): {e}. Retry dalam {delay:.2f}s...")
                time.sleep(delay)
                delay *= self.backoff_factor