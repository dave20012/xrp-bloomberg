from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable, Iterable

import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def retry(attempts: int = 3, delay: float = 1.0) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # pylint: disable=broad-except
                    last_exc = exc
                    time.sleep(delay)
            if last_exc:
                raise last_exc

        return wrapper

    return decorator


def zscore(values: Iterable[float]) -> np.ndarray:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return np.array([])
    mean = arr.mean()
    std = arr.std() or 1
    return (arr - mean) / std


def ewma(values: Iterable[float], span: int = 10) -> np.ndarray:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return np.array([])
    alpha = 2 / (span + 1)
    result = np.zeros_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result
