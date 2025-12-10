import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from core.backtest import run_backtests


def main():
    run_backtests()


if __name__ == "__main__":
    main()
