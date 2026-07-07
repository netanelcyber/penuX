"""A large zoo of classifier configurations for broad benchmarking.

Builds ~171 named (model_name, estimator) configurations spanning linear models,
SVMs, nearest-neighbors, naive Bayes, trees, bagging/boosting ensembles,
discriminant analysis, Gaussian processes, MLPs, and the three GBDT libraries
(XGBoost, LightGBM, CatBoost) already used elsewhere in this project.

This is a research/benchmarking utility, not part of the production model
registry in `penux_ap.models`.
"""
import itertools
import logging

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (
    LogisticRegression,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from penux_ap.config import RANDOM_SEED

log = logging.getLogger(__name__)


def build_model_zoo() -> list[tuple[str, object]]:
    """Return a list of (name, unfitted estimator) pairs, roughly 171 entries."""
    zoo: list[tuple[str, object]] = []

    def add(name, estimator):
        zoo.append((name, estimator))

    # Logistic regression: C x class_weight x solver
    for c, cw, solver in itertools.product(
        [0.01, 0.1, 1, 10], ["balanced", None], ["lbfgs", "liblinear"]
    ):
        add(
            f"logreg_C{c}_{cw}_{solver}",
            LogisticRegression(
                C=c, class_weight=cw, solver=solver, max_iter=2000, random_state=RANDOM_SEED
            ),
        )

    # Ridge classifier: alpha
    for alpha in [0.01, 1, 100]:
        add(f"ridge_alpha{alpha}", RidgeClassifier(alpha=alpha, random_state=RANDOM_SEED))

    # SGD: loss x alpha
    for loss, alpha in itertools.product(["log_loss", "modified_huber", "hinge"], [1e-4, 1e-3, 1e-2]):
        add(f"sgd_{loss}_a{alpha}", SGDClassifier(loss=loss, alpha=alpha, max_iter=2000, random_state=RANDOM_SEED))

    add("perceptron", Perceptron(max_iter=2000, random_state=RANDOM_SEED))

    # Linear SVC: C
    for c in [0.01, 0.1, 1, 10]:
        add(f"linear_svc_C{c}", LinearSVC(C=c, max_iter=5000, random_state=RANDOM_SEED))

    # Kernel SVC: kernel x C
    for kernel, c in itertools.product(["rbf", "linear"], [0.1, 1, 10]):
        add(f"svc_{kernel}_C{c}", SVC(kernel=kernel, C=c, probability=True, random_state=RANDOM_SEED))

    # KNN: n_neighbors x weights
    for k, weights in itertools.product([3, 5, 7, 9, 15, 25], ["uniform", "distance"]):
        add(f"knn_k{k}_{weights}", KNeighborsClassifier(n_neighbors=k, weights=weights))

    add("gaussian_nb", GaussianNB())
    add("bernoulli_nb", BernoulliNB())

    # Decision tree: criterion x max_depth
    for criterion, depth in itertools.product(["gini", "entropy"], [3, 5, 10, None]):
        add(
            f"dtree_{criterion}_d{depth}",
            DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=RANDOM_SEED),
        )
    for depth in [3, 5, 10, None]:
        add(f"extra_tree_d{depth}", ExtraTreeClassifier(max_depth=depth, random_state=RANDOM_SEED))

    # Random forest: n_estimators x max_depth x class_weight
    for n, depth, cw in itertools.product([100, 200, 500], [None, 10, 20], ["balanced", None]):
        add(
            f"rf_n{n}_d{depth}_{cw}",
            RandomForestClassifier(
                n_estimators=n, max_depth=depth, class_weight=cw, random_state=RANDOM_SEED, n_jobs=-1
            ),
        )

    # Extra trees: n_estimators x max_depth
    for n, depth in itertools.product([100, 200, 500], [None, 10, 20]):
        add(
            f"extra_trees_n{n}_d{depth}",
            ExtraTreesClassifier(n_estimators=n, max_depth=depth, random_state=RANDOM_SEED, n_jobs=-1),
        )

    # Gradient boosting (sklearn): n_estimators x learning_rate
    for n, lr in itertools.product([100, 200], [0.01, 0.05, 0.1]):
        add(
            f"gbdt_sklearn_n{n}_lr{lr}",
            GradientBoostingClassifier(n_estimators=n, learning_rate=lr, random_state=RANDOM_SEED),
        )

    # HistGradientBoosting: max_iter x learning_rate
    for n, lr in itertools.product([100, 200], [0.01, 0.05, 0.1]):
        add(
            f"histgbdt_n{n}_lr{lr}",
            HistGradientBoostingClassifier(max_iter=n, learning_rate=lr, random_state=RANDOM_SEED),
        )

    # AdaBoost: n_estimators x learning_rate
    for n, lr in itertools.product([50, 100], [0.5, 1.0]):
        add(f"adaboost_n{n}_lr{lr}", AdaBoostClassifier(n_estimators=n, learning_rate=lr, random_state=RANDOM_SEED))

    # Bagging: n_estimators
    for n in [50, 100]:
        add(f"bagging_n{n}", BaggingClassifier(n_estimators=n, random_state=RANDOM_SEED, n_jobs=-1))

    # MLP: hidden layers x activation
    for hidden, act in itertools.product([(32,), (64,), (64, 32)], ["relu", "tanh"]):
        add(
            f"mlp_{hidden}_{act}",
            MLPClassifier(hidden_layer_sizes=hidden, activation=act, max_iter=800, random_state=RANDOM_SEED),
        )
    for hidden in [(128, 64), (64, 32, 16)]:
        add(
            f"mlp_{hidden}_relu",
            MLPClassifier(hidden_layer_sizes=hidden, activation="relu", max_iter=800, random_state=RANDOM_SEED),
        )

    for solver in ["svd", "lsqr", "eigen"]:
        try:
            add(f"lda_{solver}", LinearDiscriminantAnalysis(solver=solver, shrinkage="auto" if solver != "svd" else None))
        except Exception:
            pass
    add("qda", QuadraticDiscriminantAnalysis())

    add("gaussian_process", GaussianProcessClassifier(random_state=RANDOM_SEED))

    # XGBoost: n_estimators x max_depth x learning_rate
    try:
        from xgboost import XGBClassifier
        for n, depth, lr in itertools.product([100, 200, 300], [3, 5, 7], [0.05, 0.1]):
            add(
                f"xgboost_n{n}_d{depth}_lr{lr}",
                XGBClassifier(
                    n_estimators=n, max_depth=depth, learning_rate=lr,
                    random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0,
                ),
            )
    except ImportError:
        log.info("xgboost not installed; skipping variants.")

    # LightGBM: n_estimators x num_leaves x learning_rate
    try:
        from lightgbm import LGBMClassifier
        for n, leaves, lr in itertools.product([100, 200, 300], [15, 31], [0.05, 0.1]):
            add(
                f"lightgbm_n{n}_leaves{leaves}_lr{lr}",
                LGBMClassifier(
                    n_estimators=n, num_leaves=leaves, learning_rate=lr,
                    random_state=RANDOM_SEED, verbose=-1,
                ),
            )
    except ImportError:
        log.info("lightgbm not installed; skipping variants.")

    # CatBoost: iterations x depth x learning_rate
    try:
        from catboost import CatBoostClassifier
        for n, depth, lr in itertools.product([100, 200, 300], [4, 6, 8], [0.05, 0.1]):
            add(
                f"catboost_n{n}_d{depth}_lr{lr}",
                CatBoostClassifier(
                    iterations=n, depth=depth, learning_rate=lr,
                    random_state=RANDOM_SEED, verbose=False, allow_writing_files=False,
                ),
            )
    except ImportError:
        log.info("catboost not installed; skipping variants.")

    return zoo


if __name__ == "__main__":
    zoo = build_model_zoo()
    print(f"Total model configurations: {len(zoo)}")
