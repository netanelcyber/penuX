"""Deep learning training placeholder.

For small tabular clinical cohorts (typical AP study size: 100-2000 patients),
tree-based and logistic regression baselines are usually preferred over deep learning.

Use run_baseline.py first. Use this script only if you have a large dataset
and have already validated baseline model performance.

Deep learning for this task requires:
- A sufficiently large labeled cohort (>2000 patients recommended)
- Careful regularisation and calibration
- External validation
- Clinical review

This script will fail gracefully if torch is not installed.
"""
import sys
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    log.warning(
        "Deep learning is not the recommended first approach for small tabular AP cohorts. "
        "Run scripts/run_baseline.py first and evaluate logistic regression and tree-based models."
    )

    try:
        import torch
        log.info("PyTorch version: %s", torch.__version__)
        log.info("Deep learning support is available but no training pipeline is implemented yet.")
        log.info("Implement a custom MLP or TabNet here if baseline models are insufficient.")
    except ImportError:
        log.warning("PyTorch is not installed. Install with: pip install torch")
        log.warning("Exiting without training.")
        sys.exit(0)


if __name__ == "__main__":
    main()
