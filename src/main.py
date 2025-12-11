import sys
from src.presentation.cli import run_pipeline

if __name__ == "__main__":
    # This allows running the script directly or properly via python -m src.main
    sys.exit(run_pipeline())
