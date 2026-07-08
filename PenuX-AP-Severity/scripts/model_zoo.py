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
import os

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

# Explicit thread cap for multi-threaded estimators. Benchmark runs of this
# zoo have been run one dataset at a time (see scripts/benchmark_model_zoo.py) --
# running two such processes concurrently on a small number of cores causes
# severe thread oversubscription (XGBoost/LightGBM/CatBoost/RandomForest/
# ExtraTrees/HistGradientBoosting all parallelize internally), observed to
# slow individual model fits by 100x or more.
N_JOBS = os.cpu_count() or 1


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
    # max_iter caps worst-case runtime -- libsvm's dual solver can take minutes
    # per fold at high C with the linear kernel on near-separable data.
    for kernel, c in itertools.product(["rbf", "linear", "poly", "sigmoid"], [0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        add(f"svc_{kernel}_C{c}", SVC(kernel=kernel, C=c, probability=True, random_state=RANDOM_SEED, max_iter=2000))
    for degree, c in itertools.product([2, 3, 4], [0.1, 1, 10]):
        add(
            f"svc_poly_deg{degree}_C{c}",
            SVC(kernel="poly", degree=degree, C=c, probability=True, random_state=RANDOM_SEED, max_iter=2000),
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

    # "10K-node" trees: max_leaf_nodes controls tree size directly (a binary
    # tree with L leaves has 2L-1 total nodes, so max_leaf_nodes=5000 ->
    # ~10,000 nodes if the tree grows that large; these datasets have only
    # ~700-1300 rows, so in practice most such trees grow until every leaf is
    # pure or min_samples_leaf is hit, well short of the cap -- included to
    # explicitly test "let the tree get as big as the data allows."
    # Single trees: criterion x max_leaf_nodes (6)
    for criterion, max_leaf_nodes in itertools.product(["gini", "entropy"], [1000, 2500, 5000]):
        add(
            f"dtree_{criterion}_leafnodes{max_leaf_nodes}",
            DecisionTreeClassifier(criterion=criterion, max_leaf_nodes=max_leaf_nodes, random_state=RANDOM_SEED),
        )
    for max_leaf_nodes in [1000, 2500, 5000]:
        add(
            f"extra_tree_leafnodes{max_leaf_nodes}",
            ExtraTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=RANDOM_SEED),
        )
    # Forests of huge trees: n_estimators x max_leaf_nodes x class_weight (8)
    for n, max_leaf_nodes, cw in itertools.product([100, 300], [2500, 5000], ["balanced", None]):
        add(
            f"rf_huge_n{n}_leafnodes{max_leaf_nodes}_{cw}",
            RandomForestClassifier(
                n_estimators=n, max_leaf_nodes=max_leaf_nodes, class_weight=cw,
                random_state=RANDOM_SEED, n_jobs=N_JOBS,
            ),
        )
    # Extra trees forest of huge trees: n_estimators x max_leaf_nodes (4)
    for n, max_leaf_nodes in itertools.product([100, 300], [2500, 5000]):
        add(
            f"extra_trees_huge_n{n}_leafnodes{max_leaf_nodes}",
            ExtraTreesClassifier(n_estimators=n, max_leaf_nodes=max_leaf_nodes, random_state=RANDOM_SEED, n_jobs=N_JOBS),
        )
    # HistGradientBoosting with huge trees: max_iter x max_leaf_nodes (4)
    for n, max_leaf_nodes in itertools.product([100, 200], [1000, 5000]):
        add(
            f"histgbdt_huge_n{n}_leafnodes{max_leaf_nodes}",
            HistGradientBoostingClassifier(max_iter=n, max_leaf_nodes=max_leaf_nodes, random_state=RANDOM_SEED),
        )

    # Random forest: n_estimators x max_depth x class_weight (60)
    for n, depth, cw in itertools.product([100, 200, 300, 500, 800], [None, 5, 10, 15, 20, 30], ["balanced", None]):
        add(
            f"rf_n{n}_d{depth}_{cw}",
            RandomForestClassifier(
                n_estimators=n, max_depth=depth, class_weight=cw, random_state=RANDOM_SEED, n_jobs=N_JOBS
            ),
        )

    # Extra trees: n_estimators x max_depth (30)
    for n, depth in itertools.product([100, 200, 300, 500, 800], [None, 5, 10, 15, 20, 30]):
        add(
            f"extra_trees_n{n}_d{depth}",
            ExtraTreesClassifier(n_estimators=n, max_depth=depth, random_state=RANDOM_SEED, n_jobs=N_JOBS),
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
            BaggingClassifier(n_estimators=n, max_samples=max_samples, random_state=RANDOM_SEED, n_jobs=N_JOBS),
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

    # XGBoost: n_estimators x max_depth x learning_rate (125)
    try:
        from xgboost import XGBClassifier
        for n, depth, lr in itertools.product([100, 200, 300, 500, 800], [3, 5, 7, 9, 11], [0.001, 0.01, 0.05, 0.1, 0.2]):
            add(
                f"xgboost_n{n}_d{depth}_lr{lr}",
                XGBClassifier(
                    n_estimators=n, max_depth=depth, learning_rate=lr,
                    random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0, n_jobs=N_JOBS,
                ),
            )
    except ImportError:
        log.info("xgboost not installed; skipping variants.")

    # LightGBM: n_estimators x num_leaves x learning_rate (125)
    try:
        from lightgbm import LGBMClassifier
        for n, leaves, lr in itertools.product([100, 200, 300, 500, 800], [15, 31, 63, 127, 255], [0.001, 0.01, 0.05, 0.1, 0.2]):
            add(
                f"lightgbm_n{n}_leaves{leaves}_lr{lr}",
                LGBMClassifier(
                    n_estimators=n, num_leaves=leaves, learning_rate=lr,
                    random_state=RANDOM_SEED, verbose=-1, n_jobs=N_JOBS,
                ),
            )
    except ImportError:
        log.info("lightgbm not installed; skipping variants.")

    # CatBoost: iterations x depth x learning_rate (125)
    try:
        from catboost import CatBoostClassifier
        for n, depth, lr in itertools.product([100, 200, 300, 500, 800], [4, 5, 6, 7, 8], [0.001, 0.01, 0.05, 0.1, 0.2]):
            add(
                f"catboost_n{n}_d{depth}_lr{lr}",
                CatBoostClassifier(
                    iterations=n, depth=depth, learning_rate=lr,
                    random_state=RANDOM_SEED, verbose=False, allow_writing_files=False, thread_count=N_JOBS,
                ),
            )
    except ImportError:
        log.info("catboost not installed; skipping variants.")

    # --- Additional boosting *algorithms* (not just more hyperparameters of
    # the three above): XGBoost's DART booster (dropout trees, Rashmi &
    # Gilad-Bachrach 2015), LightGBM's DART and GOSS boosting modes (Ke et
    # al. 2017), and CatBoost's Plain boosting (the non-ordered baseline
    # ordered boosting was designed to fix, Prokhorenkova et al. 2018).

    # XGBoost DART: n_estimators x max_depth x learning_rate x rate_drop (36)
    try:
        from xgboost import XGBClassifier
        for n, depth, lr, rate_drop in itertools.product(
            [100, 200, 300], [3, 5, 7], [0.05, 0.1], [0.1, 0.3]
        ):
            add(
                f"xgboost_dart_n{n}_d{depth}_lr{lr}_drop{rate_drop}",
                XGBClassifier(
                    booster="dart", n_estimators=n, max_depth=depth, learning_rate=lr, rate_drop=rate_drop,
                    random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0, n_jobs=N_JOBS,
                ),
            )
    except ImportError:
        log.info("xgboost not installed; skipping DART variants.")

    # LightGBM DART: n_estimators x num_leaves x learning_rate (18)
    try:
        from lightgbm import LGBMClassifier
        for n, leaves, lr in itertools.product([100, 200, 300], [15, 31, 63], [0.05, 0.1]):
            add(
                f"lightgbm_dart_n{n}_leaves{leaves}_lr{lr}",
                LGBMClassifier(
                    boosting_type="dart", n_estimators=n, num_leaves=leaves, learning_rate=lr,
                    random_state=RANDOM_SEED, verbose=-1, n_jobs=N_JOBS,
                ),
            )
        # LightGBM GOSS: n_estimators x num_leaves x learning_rate (18)
        for n, leaves, lr in itertools.product([100, 200, 300], [15, 31, 63], [0.05, 0.1]):
            add(
                f"lightgbm_goss_n{n}_leaves{leaves}_lr{lr}",
                LGBMClassifier(
                    boosting_type="goss", n_estimators=n, num_leaves=leaves, learning_rate=lr,
                    random_state=RANDOM_SEED, verbose=-1, n_jobs=N_JOBS,
                ),
            )
    except ImportError:
        log.info("lightgbm not installed; skipping DART/GOSS variants.")

    # CatBoost Plain boosting: iterations x depth x learning_rate (18)
    try:
        from catboost import CatBoostClassifier
        for n, depth, lr in itertools.product([100, 200, 300], [4, 6, 8], [0.05, 0.1]):
            add(
                f"catboost_plain_n{n}_d{depth}_lr{lr}",
                CatBoostClassifier(
                    boosting_type="Plain", iterations=n, depth=depth, learning_rate=lr,
                    random_state=RANDOM_SEED, verbose=False, allow_writing_files=False, thread_count=N_JOBS,
                ),
            )
    except ImportError:
        log.info("catboost not installed; skipping Plain-boosting variants.")

    # From-scratch GBDT (penux_ap.scratch_gbdt.ScratchGBDTClassifier): classic
    # Friedman TreeBoost -- CART regression trees built from scratch and
    # split on residual variance reduction, with a Newton leaf-value
    # correction applied afterwards. Algorithmically distinct from every
    # other GBDT entry above, none of which use this project's own tree code
    # (they all call XGBoost/LightGBM/CatBoost/sklearn's C/Cython internals).
    # n_estimators x max_depth x learning_rate (12)
    from penux_ap.scratch_gbdt import ScratchGBDTClassifier
    for n, depth, lr in itertools.product([50, 100], [2, 3, 4], [0.05, 0.1]):
        add(
            f"scratch_gbdt_n{n}_d{depth}_lr{lr}",
            ScratchGBDTClassifier(n_estimators=n, max_depth=depth, learning_rate=lr),
        )

    # --- Deep learning (PyTorch, CPU): feedforward DNN and 1D ConvNet over
    # the tabular feature vector (~96 configs), added to check whether deep
    # nets can beat the GBDT/ensemble models on these small clinical datasets.
    try:
        from penux_ap.torch_models import TorchConvNetClassifier, TorchDNNClassifier

        # DNN: hidden_sizes x dropout x lr x batch_size (64)
        for hidden, dropout, lr, batch_size in itertools.product(
            [(32,), (64,), (128,), (64, 32), (128, 64), (256, 128), (64, 32, 16), (128, 64, 32)],
            [0.1, 0.3], [1e-3, 3e-3], [16, 32],
        ):
            add(
                f"dnn_{hidden}_drop{dropout}_lr{lr}_bs{batch_size}",
                TorchDNNClassifier(hidden_sizes=hidden, dropout=dropout, lr=lr, batch_size=batch_size),
            )

        # 1D ConvNet: channels x kernel_size x dropout x lr (32)
        for channels, kernel_size, dropout, lr in itertools.product(
            [(8, 16), (16, 32), (32, 64), (16, 32, 64)], [3, 5], [0.1, 0.3], [1e-3, 3e-3],
        ):
            add(
                f"convnet_{channels}_k{kernel_size}_drop{dropout}_lr{lr}",
                TorchConvNetClassifier(channels=channels, kernel_size=kernel_size, dropout=dropout, lr=lr),
            )
    except ImportError:
        log.info("torch not installed; skipping DNN/ConvNet variants.")

    return zoo


if __name__ == "__main__":
    zoo = build_model_zoo()
    print(f"Total model configurations: {len(zoo)}")
    names = [n for n, _ in zoo]
    assert len(names) == len(set(names)), "Duplicate model names in zoo!"
