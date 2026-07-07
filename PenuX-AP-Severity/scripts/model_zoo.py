"""A large zoo of classifier configurations for broad benchmarking.

Builds 784 named (model_name, estimator) configurations spanning linear models,
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
from sklearn.gaussian_process.kernels import RBF, Matern
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
    """Return a list of (name, unfitted estimator) pairs, 784 entries.

    The three GBDT headline configurations referenced by
    scripts/benchmark_model_zoo.py's GBDT_NAMES
    (xgboost_n200_d5_lr0.1, lightgbm_n200_leaves31_lr0.1, catboost_n200_d6_lr0.1)
    are preserved inside their respective expanded grids.
    """
    zoo: list[tuple[str, object]] = []

    def add(name, estimator):
        zoo.append((name, estimator))

    # Logistic regression: C x class_weight x solver (40)
    for c, cw, solver in itertools.product(
        [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], ["balanced", None], ["lbfgs", "liblinear"]
    ):
        add(
            f"logreg_C{c}_{cw}_{solver}",
            LogisticRegression(
                C=c, class_weight=cw, solver=solver, max_iter=2000, random_state=RANDOM_SEED
            ),
        )

    # Ridge classifier: alpha (15)
    for alpha in [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]:
        add(f"ridge_alpha{alpha}", RidgeClassifier(alpha=alpha, random_state=RANDOM_SEED))

    # SGD: loss x alpha x penalty (45), plus learning-rate-schedule variants (4)
    for loss, alpha, penalty in itertools.product(
        ["log_loss", "modified_huber", "hinge"], [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], ["l2", "l1", "elasticnet"]
    ):
        add(
            f"sgd_{loss}_a{alpha}_{penalty}",
            SGDClassifier(loss=loss, alpha=alpha, penalty=penalty, max_iter=2000, random_state=RANDOM_SEED),
        )
    for schedule in ["constant", "optimal", "invscaling", "adaptive"]:
        kwargs = {"eta0": 0.01} if schedule != "optimal" else {}
        add(
            f"sgd_logloss_a0.001_{schedule}",
            SGDClassifier(
                loss="log_loss", alpha=1e-3, learning_rate=schedule, max_iter=2000,
                random_state=RANDOM_SEED, **kwargs,
            ),
        )

    # Perceptron: penalty x alpha (16)
    for penalty, alpha in itertools.product([None, "l2", "l1", "elasticnet"], [1e-5, 1e-4, 1e-3, 1e-2]):
        add(
            f"perceptron_{penalty}_a{alpha}",
            Perceptron(penalty=penalty, alpha=alpha, max_iter=2000, random_state=RANDOM_SEED),
        )

    # Linear SVC: C x loss (12) + C x penalty l1 (12) = 24
    for c in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 300]:
        add(f"linear_svc_C{c}_hinge", LinearSVC(C=c, loss="hinge", max_iter=5000, random_state=RANDOM_SEED))
    for c in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 300]:
        add(
            f"linear_svc_C{c}_l1",
            LinearSVC(C=c, penalty="l1", loss="squared_hinge", dual=False, max_iter=5000, random_state=RANDOM_SEED),
        )

    # Kernel SVC: kernel x C (28) + poly degree x C (9) = 37
    for kernel, c in itertools.product(["rbf", "linear", "poly", "sigmoid"], [0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        add(f"svc_{kernel}_C{c}", SVC(kernel=kernel, C=c, probability=True, random_state=RANDOM_SEED))
    for degree, c in itertools.product([2, 3, 4], [0.1, 1, 10]):
        add(
            f"svc_poly_deg{degree}_C{c}",
            SVC(kernel="poly", degree=degree, C=c, probability=True, random_state=RANDOM_SEED),
        )

    # KNN: n_neighbors x weights (26) + metric x n_neighbors (6) = 32
    for k, weights in itertools.product([1, 3, 5, 7, 9, 11, 15, 21, 25, 35, 45, 60, 75], ["uniform", "distance"]):
        add(f"knn_k{k}_{weights}", KNeighborsClassifier(n_neighbors=k, weights=weights))
    for metric, k in itertools.product(["manhattan", "chebyshev"], [5, 15, 25]):
        add(f"knn_k{k}_{metric}", KNeighborsClassifier(n_neighbors=k, metric=metric))

    # Naive Bayes (4)
    add("gaussian_nb", GaussianNB())
    add("bernoulli_nb_t0.0", BernoulliNB(binarize=0.0))
    add("bernoulli_nb_t0.5", BernoulliNB(binarize=0.5))
    add("bernoulli_nb_t1.0", BernoulliNB(binarize=1.0))

    # Decision tree: criterion x max_depth (14) + min_samples_leaf (4) + max_features (4) = 22
    for criterion, depth in itertools.product(["gini", "entropy"], [3, 5, 7, 10, 15, 20, None]):
        add(
            f"dtree_{criterion}_d{depth}",
            DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=RANDOM_SEED),
        )
    for leaf in [1, 5, 10, 20]:
        add(
            f"dtree_gini_leaf{leaf}",
            DecisionTreeClassifier(criterion="gini", min_samples_leaf=leaf, random_state=RANDOM_SEED),
        )
    for feat, depth in itertools.product(["sqrt", "log2"], [5, 10]):
        add(
            f"dtree_gini_{feat}_d{depth}",
            DecisionTreeClassifier(criterion="gini", max_features=feat, max_depth=depth, random_state=RANDOM_SEED),
        )
    for split in [2, 5, 10, 20]:
        add(
            f"dtree_entropy_split{split}",
            DecisionTreeClassifier(criterion="entropy", min_samples_split=split, random_state=RANDOM_SEED),
        )

    # Extra tree: max_depth (7) + min_samples_leaf (3) = 10
    for depth in [3, 5, 7, 10, 15, 20, None]:
        add(f"extra_tree_d{depth}", ExtraTreeClassifier(max_depth=depth, random_state=RANDOM_SEED))
    for leaf in [5, 10, 20]:
        add(f"extra_tree_leaf{leaf}", ExtraTreeClassifier(min_samples_leaf=leaf, random_state=RANDOM_SEED))

    # Random forest: n_estimators x max_depth x class_weight (60)
    for n, depth, cw in itertools.product([100, 200, 300, 500, 800], [None, 5, 10, 15, 20, 30], ["balanced", None]):
        add(
            f"rf_n{n}_d{depth}_{cw}",
            RandomForestClassifier(
                n_estimators=n, max_depth=depth, class_weight=cw, random_state=RANDOM_SEED, n_jobs=-1
            ),
        )

    # Extra trees: n_estimators x max_depth (30)
    for n, depth in itertools.product([100, 200, 300, 500, 800], [None, 5, 10, 15, 20, 30]):
        add(
            f"extra_trees_n{n}_d{depth}",
            ExtraTreesClassifier(n_estimators=n, max_depth=depth, random_state=RANDOM_SEED, n_jobs=-1),
        )

    # Gradient boosting (sklearn): n_estimators x learning_rate x subsample (36)
    for n, lr, sub in itertools.product([100, 200, 300], [0.01, 0.03, 0.05, 0.1, 0.2, 0.3], [0.8, 1.0]):
        add(
            f"gbdt_sklearn_n{n}_lr{lr}_sub{sub}",
            GradientBoostingClassifier(n_estimators=n, learning_rate=lr, subsample=sub, random_state=RANDOM_SEED),
        )

    # HistGradientBoosting: max_iter x learning_rate (18)
    for n, lr in itertools.product([100, 200, 300], [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]):
        add(
            f"histgbdt_n{n}_lr{lr}",
            HistGradientBoostingClassifier(max_iter=n, learning_rate=lr, random_state=RANDOM_SEED),
        )

    # AdaBoost: n_estimators x learning_rate (25)
    for n, lr in itertools.product([50, 100, 200, 300, 500], [0.1, 0.3, 0.5, 1.0, 1.5]):
        add(f"adaboost_n{n}_lr{lr}", AdaBoostClassifier(n_estimators=n, learning_rate=lr, random_state=RANDOM_SEED))

    # Bagging: n_estimators x max_samples (18)
    for n, max_samples in itertools.product([10, 50, 100, 200, 300, 500], [0.5, 0.8, 1.0]):
        add(
            f"bagging_n{n}_ms{max_samples}",
            BaggingClassifier(n_estimators=n, max_samples=max_samples, random_state=RANDOM_SEED, n_jobs=-1),
        )

    # MLP: hidden layers x activation (33) + alpha variants (3) = 36
    hidden_configs = [
        (16,), (32,), (64,), (128,), (256,),
        (32, 16), (64, 32), (128, 64), (256, 128),
        (64, 32, 16), (128, 64, 32),
    ]
    for hidden, act in itertools.product(hidden_configs, ["relu", "tanh", "logistic"]):
        add(
            f"mlp_{hidden}_{act}",
            MLPClassifier(hidden_layer_sizes=hidden, activation=act, max_iter=800, random_state=RANDOM_SEED),
        )
    for alpha in [0.0001, 0.001, 0.01]:
        add(
            f"mlp_(64, 32)_relu_a{alpha}",
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=alpha, max_iter=800, random_state=RANDOM_SEED),
        )

    # Discriminant analysis (3 + 3)
    for solver in ["svd", "lsqr", "eigen"]:
        try:
            add(f"lda_{solver}", LinearDiscriminantAnalysis(solver=solver, shrinkage="auto" if solver != "svd" else None))
        except Exception:
            pass
    add("qda_reg0.0", QuadraticDiscriminantAnalysis(reg_param=0.0))
    add("qda_reg0.1", QuadraticDiscriminantAnalysis(reg_param=0.1))
    add("qda_reg0.5", QuadraticDiscriminantAnalysis(reg_param=0.5))

    # Gaussian process: kernel variants (2)
    add("gaussian_process_rbf", GaussianProcessClassifier(kernel=RBF(), random_state=RANDOM_SEED))
    add("gaussian_process_matern", GaussianProcessClassifier(kernel=Matern(), random_state=RANDOM_SEED))

    # XGBoost: n_estimators x max_depth x learning_rate (100)
    try:
        from xgboost import XGBClassifier
        for n, depth, lr in itertools.product([100, 200, 300, 500, 800], [3, 5, 7, 9, 11], [0.01, 0.05, 0.1, 0.2]):
            add(
                f"xgboost_n{n}_d{depth}_lr{lr}",
                XGBClassifier(
                    n_estimators=n, max_depth=depth, learning_rate=lr,
                    random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0,
                ),
            )
    except ImportError:
        log.info("xgboost not installed; skipping variants.")

    # LightGBM: n_estimators x num_leaves x learning_rate (100)
    try:
        from lightgbm import LGBMClassifier
        for n, leaves, lr in itertools.product([100, 200, 300, 500, 800], [15, 31, 63, 127, 255], [0.01, 0.05, 0.1, 0.2]):
            add(
                f"lightgbm_n{n}_leaves{leaves}_lr{lr}",
                LGBMClassifier(
                    n_estimators=n, num_leaves=leaves, learning_rate=lr,
                    random_state=RANDOM_SEED, verbose=-1,
                ),
            )
    except ImportError:
        log.info("lightgbm not installed; skipping variants.")

    # CatBoost: iterations x depth x learning_rate (100)
    try:
        from catboost import CatBoostClassifier
        for n, depth, lr in itertools.product([100, 200, 300, 500, 800], [4, 5, 6, 7, 8], [0.01, 0.05, 0.1, 0.2]):
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
    names = [n for n, _ in zoo]
    assert len(names) == len(set(names)), "Duplicate model names in zoo!"
