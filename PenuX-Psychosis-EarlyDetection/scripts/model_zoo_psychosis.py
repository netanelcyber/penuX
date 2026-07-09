"""Model zoo for the SIMULATED first-episode-schizophrenia-vs-control task.

Targets ~12,000 classifier configurations, in the same spirit as
PenuX-AP-Severity's model_zoo.py. Trained/evaluated only on the clearly
labeled synthetic dataset produced by src/penux_psychosis/simulate_data.py --
see that module's docstring and docs/dataset_landscape.md for why this is
simulated data, not a re-analysis of any real study.

A 1D-convolutional family (used in PenuX-AP-Severity) is deliberately
omitted here: with only 6 scalar lab features, treating them as a
convolvable "sequence" is not a meaningful architecture choice, and small
input length caused pooling-dimension errors in earlier work on this
project family. That family's model-count budget is reallocated to deeper
GBDT/DNN/hybrid grids instead.
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
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

RANDOM_SEED = 42
N_JOBS = os.cpu_count() or 1
log = logging.getLogger(__name__)


def build_model_zoo() -> list[tuple[str, object]]:
    zoo: list[tuple[str, object]] = []

    def add(name, estimator):
        zoo.append((name, estimator))

    # Logistic regression: C x class_weight x solver (60)
    for c, cw, solver in itertools.product(
        [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], ["balanced", None], ["lbfgs", "liblinear", "newton-cg"]
    ):
        add(f"logreg_C{c}_{cw}_{solver}", LogisticRegression(C=c, class_weight=cw, solver=solver, max_iter=3000, random_state=RANDOM_SEED))

    # Ridge: alpha (20)
    for alpha in [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5, 3e5]:
        add(f"ridge_alpha{alpha}", RidgeClassifier(alpha=alpha, random_state=RANDOM_SEED))

    # SGD: loss x alpha x penalty (72) + schedule variants (8)
    for loss, alpha, penalty in itertools.product(["log_loss", "modified_huber", "hinge", "perceptron"], [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], ["l2", "l1", "elasticnet"]):
        add(f"sgd_{loss}_a{alpha}_{penalty}", SGDClassifier(loss=loss, alpha=alpha, penalty=penalty, max_iter=3000, random_state=RANDOM_SEED))
    for schedule in ["constant", "optimal", "invscaling", "adaptive"]:
        for eta0 in [0.001, 0.01]:
            kwargs = {"eta0": eta0} if schedule != "optimal" else {}
            add(f"sgd_logloss_a0.001_{schedule}_eta{eta0}", SGDClassifier(loss="log_loss", alpha=1e-3, learning_rate=schedule, max_iter=3000, random_state=RANDOM_SEED, **kwargs))

    # Perceptron: penalty x alpha (24)
    for penalty, alpha in itertools.product([None, "l2", "l1", "elasticnet"], [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]):
        add(f"perceptron_{penalty}_a{alpha}", Perceptron(penalty=penalty, alpha=alpha, max_iter=3000, random_state=RANDOM_SEED))

    # Linear SVC: C x {hinge, l1} (36)
    for c in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5]:
        add(f"linear_svc_C{c}_hinge", LinearSVC(C=c, loss="hinge", max_iter=8000, random_state=RANDOM_SEED))
    for c in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5]:
        add(f"linear_svc_C{c}_l1", LinearSVC(C=c, penalty="l1", loss="squared_hinge", dual=False, max_iter=8000, random_state=RANDOM_SEED))

    # Kernel SVC: kernel x C (64) + poly degree x C (15)
    for kernel, c in itertools.product(["rbf", "linear", "poly", "sigmoid"], [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4]):
        add(f"svc_{kernel}_C{c}", SVC(kernel=kernel, C=c, probability=True, random_state=RANDOM_SEED, max_iter=3000))
    for degree, c in itertools.product([2, 3, 4, 5, 6], [0.01, 0.1, 1, 10, 100]):
        add(f"svc_poly_deg{degree}_C{c}", SVC(kernel="poly", degree=degree, C=c, probability=True, random_state=RANDOM_SEED, max_iter=3000))

    # KNN: k x weights (40) + metric x k (24)
    for k, weights in itertools.product([1, 3, 5, 7, 9, 11, 15, 21, 25, 35, 45, 60, 75, 90, 110, 130, 150, 170, 190, 210], ["uniform", "distance"]):
        add(f"knn_k{k}_{weights}", KNeighborsClassifier(n_neighbors=k, weights=weights))
    for metric, k in itertools.product(["manhattan", "chebyshev", "minkowski", "canberra"], [5, 15, 25, 45, 75, 110]):
        add(f"knn_k{k}_{metric}", KNeighborsClassifier(n_neighbors=k, metric=metric))

    # Naive Bayes (6)
    add("gaussian_nb", GaussianNB())
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        add(f"bernoulli_nb_t{t}", BernoulliNB(binarize=t))

    # Decision tree: criterion x depth (24) + leaf (6) + features x depth (4) + split (6)
    for criterion, depth in itertools.product(["gini", "entropy", "log_loss"], [2, 3, 4, 5, 6, 7, 8, 10]):
        add(f"dtree_{criterion}_d{depth}", DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=RANDOM_SEED))
    for leaf in [1, 2, 5, 10, 20, 40]:
        add(f"dtree_gini_leaf{leaf}", DecisionTreeClassifier(criterion="gini", min_samples_leaf=leaf, random_state=RANDOM_SEED))
    for feat, depth in itertools.product(["sqrt", "log2"], [3, 5, 10, None]):
        add(f"dtree_gini_{feat}_d{depth}", DecisionTreeClassifier(criterion="gini", max_features=feat, max_depth=depth, random_state=RANDOM_SEED))
    for split in [2, 5, 10, 20, 40, 80]:
        add(f"dtree_entropy_split{split}", DecisionTreeClassifier(criterion="entropy", min_samples_split=split, random_state=RANDOM_SEED))

    # Extra tree single: depth (8) + leaf (6)
    for depth in [2, 3, 4, 5, 6, 7, 8, None]:
        add(f"extra_tree_d{depth}", ExtraTreeClassifier(max_depth=depth, random_state=RANDOM_SEED))
    for leaf in [1, 2, 5, 10, 20, 40]:
        add(f"extra_tree_leaf{leaf}", ExtraTreeClassifier(min_samples_leaf=leaf, random_state=RANDOM_SEED))

    # "huge" unconstrained-leaf trees/forests (leaf_nodes grid only meaningful up to N=394)
    for criterion, mln in itertools.product(["gini", "entropy"], [50, 100, 150]):
        add(f"dtree_{criterion}_leafnodes{mln}", DecisionTreeClassifier(criterion=criterion, max_leaf_nodes=mln, random_state=RANDOM_SEED))
    for mln in [50, 100, 150]:
        add(f"extra_tree_leafnodes{mln}", ExtraTreeClassifier(max_leaf_nodes=mln, random_state=RANDOM_SEED))
    for n, mln, cw in itertools.product([100, 300, 500], [50, 100, 150], ["balanced", None]):
        add(f"rf_huge_n{n}_leafnodes{mln}_{cw}", RandomForestClassifier(n_estimators=n, max_leaf_nodes=mln, class_weight=cw, random_state=RANDOM_SEED, n_jobs=N_JOBS))
    for n, mln in itertools.product([100, 300, 500], [50, 100, 150]):
        add(f"extra_trees_huge_n{n}_leafnodes{mln}", ExtraTreesClassifier(n_estimators=n, max_leaf_nodes=mln, random_state=RANDOM_SEED, n_jobs=N_JOBS))
    for n, mln in itertools.product([100, 200, 300], [50, 100, 150]):
        add(f"histgbdt_huge_n{n}_leafnodes{mln}", HistGradientBoostingClassifier(max_iter=n, max_leaf_nodes=mln, random_state=RANDOM_SEED))

    # Random forest: n x depth x class_weight (10x10x2=200)
    for n, depth, cw in itertools.product([50, 100, 150, 200, 300, 500, 800, 1200, 1800, 2500], [None, 2, 3, 5, 8, 12, 16, 20, 25, 30], ["balanced", None]):
        add(f"rf_n{n}_d{depth}_{cw}", RandomForestClassifier(n_estimators=n, max_depth=depth, class_weight=cw, random_state=RANDOM_SEED, n_jobs=N_JOBS))

    # Extra trees: n x depth (10x10=100)
    for n, depth in itertools.product([50, 100, 150, 200, 300, 500, 800, 1200, 1800, 2500], [None, 2, 3, 5, 8, 12, 16, 20, 25, 30]):
        add(f"extra_trees_n{n}_d{depth}", ExtraTreesClassifier(n_estimators=n, max_depth=depth, random_state=RANDOM_SEED, n_jobs=N_JOBS))

    # Gradient boosting (sklearn): n x lr x subsample (10x10x4=400)
    for n, lr, sub in itertools.product([50, 100, 200, 300, 500, 800, 1200, 1600, 2000, 2500], [0.002, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5], [0.5, 0.7, 0.85, 1.0]):
        add(f"gbdt_sklearn_n{n}_lr{lr}_sub{sub}", GradientBoostingClassifier(n_estimators=n, learning_rate=lr, subsample=sub, random_state=RANDOM_SEED))

    # HistGradientBoosting: n x lr (5x8=40)
    for n, lr in itertools.product([50, 100, 200, 300, 500], [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]):
        add(f"histgbdt_n{n}_lr{lr}", HistGradientBoostingClassifier(max_iter=n, learning_rate=lr, random_state=RANDOM_SEED))

    # AdaBoost: n x lr (8x8=64)
    for n, lr in itertools.product([25, 50, 100, 200, 300, 500, 800, 1200], [0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6]):
        add(f"adaboost_n{n}_lr{lr}", AdaBoostClassifier(n_estimators=n, learning_rate=lr, random_state=RANDOM_SEED))

    # Bagging: n x max_samples (8x4=32)
    for n, ms in itertools.product([10, 25, 50, 100, 200, 300, 500, 800], [0.4, 0.6, 0.8, 1.0]):
        add(f"bagging_n{n}_ms{ms}", BaggingClassifier(n_estimators=n, max_samples=ms, random_state=RANDOM_SEED, n_jobs=N_JOBS))

    # MLP (sklearn): hidden x activation (15x3=45) + alpha (5)
    hidden_configs = [(8,), (16,), (32,), (64,), (128,), (16, 8), (32, 16), (64, 32), (128, 64), (256, 128), (32, 16, 8), (64, 32, 16), (128, 64, 32), (256, 128, 64), (64, 32, 16, 8)]
    for hidden, act in itertools.product(hidden_configs, ["relu", "tanh", "logistic"]):
        add(f"mlp_{hidden}_{act}", MLPClassifier(hidden_layer_sizes=hidden, activation=act, max_iter=1500, random_state=RANDOM_SEED))
    for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
        add(f"mlp_(64, 32)_relu_a{alpha}", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=alpha, max_iter=1500, random_state=RANDOM_SEED))

    # Discriminant analysis (3 + 3)
    for solver in ["svd", "lsqr", "eigen"]:
        try:
            add(f"lda_{solver}", LinearDiscriminantAnalysis(solver=solver, shrinkage="auto" if solver != "svd" else None))
        except Exception:
            pass
    for reg in [0.0, 0.1, 0.5]:
        add(f"qda_reg{reg}", QuadraticDiscriminantAnalysis(reg_param=reg))

    # Gaussian process (2)
    add("gaussian_process_rbf", GaussianProcessClassifier(kernel=RBF(), random_state=RANDOM_SEED))
    add("gaussian_process_matern", GaussianProcessClassifier(kernel=Matern(), random_state=RANDOM_SEED))

    # XGBoost: n x depth x lr (10x10x10=1000)
    try:
        from xgboost import XGBClassifier
        for n, depth, lr in itertools.product(
            [30, 50, 100, 200, 300, 500, 800, 1200, 1800, 2500],
            [2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
            [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3],
        ):
            add(f"xgboost_n{n}_d{depth}_lr{lr}", XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=lr, random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0, n_jobs=N_JOBS))
    except ImportError:
        log.info("xgboost not installed")

    # LightGBM: n x leaves x lr (10x10x10=1000)
    try:
        from lightgbm import LGBMClassifier
        for n, leaves, lr in itertools.product(
            [30, 50, 100, 200, 300, 500, 800, 1200, 1800, 2500],
            [3, 7, 15, 31, 63, 127, 200, 300, 400, 600],
            [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3],
        ):
            add(f"lightgbm_n{n}_leaves{leaves}_lr{lr}", LGBMClassifier(n_estimators=n, num_leaves=leaves, learning_rate=lr, random_state=RANDOM_SEED, verbose=-1, n_jobs=N_JOBS, min_child_samples=5))
    except ImportError:
        log.info("lightgbm not installed")

    # CatBoost: n x depth x lr (10x6x10=600) -- depth capped at 8: depth>=10
    # is known to scale pathologically slowly (observed in PenuX-AP-Severity).
    try:
        from catboost import CatBoostClassifier
        for n, depth, lr in itertools.product(
            [30, 50, 100, 200, 300, 500, 800, 1200, 1800, 2500],
            [3, 4, 5, 6, 7, 8],
            [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3],
        ):
            add(f"catboost_n{n}_d{depth}_lr{lr}", CatBoostClassifier(iterations=n, depth=depth, learning_rate=lr, random_state=RANDOM_SEED, verbose=False, allow_writing_files=False, thread_count=N_JOBS))
    except ImportError:
        log.info("catboost not installed")

    # XGBoost DART: n x depth x lr x rate_drop (8x6x4x4=768)
    try:
        from xgboost import XGBClassifier
        for n, depth, lr, rd in itertools.product([30, 50, 100, 200, 300, 500, 800, 1200], [2, 3, 4, 5, 6, 8], [0.005, 0.01, 0.05, 0.1], [0.05, 0.1, 0.2, 0.3]):
            add(f"xgboost_dart_n{n}_d{depth}_lr{lr}_drop{rd}", XGBClassifier(booster="dart", n_estimators=n, max_depth=depth, learning_rate=lr, rate_drop=rd, random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0, n_jobs=N_JOBS))
    except ImportError:
        log.info("xgboost dart skipped")

    # LightGBM DART + GOSS: n x leaves x lr (8x6x4=192 each => 384)
    try:
        from lightgbm import LGBMClassifier
        for n, leaves, lr in itertools.product([30, 50, 100, 200, 300, 500, 800, 1200], [7, 15, 31, 63, 127, 200], [0.005, 0.01, 0.05, 0.1]):
            add(f"lightgbm_dart_n{n}_leaves{leaves}_lr{lr}", LGBMClassifier(boosting_type="dart", n_estimators=n, num_leaves=leaves, learning_rate=lr, random_state=RANDOM_SEED, verbose=-1, n_jobs=N_JOBS, min_child_samples=5))
        for n, leaves, lr in itertools.product([30, 50, 100, 200, 300, 500, 800, 1200], [7, 15, 31, 63, 127, 200], [0.005, 0.01, 0.05, 0.1]):
            add(f"lightgbm_goss_n{n}_leaves{leaves}_lr{lr}", LGBMClassifier(boosting_type="goss", n_estimators=n, num_leaves=leaves, learning_rate=lr, random_state=RANDOM_SEED, verbose=-1, n_jobs=N_JOBS, min_child_samples=5))
    except ImportError:
        log.info("lightgbm dart/goss skipped")

    # CatBoost Plain: n x depth x lr (8x5x4=160)
    try:
        from catboost import CatBoostClassifier
        for n, depth, lr in itertools.product([30, 50, 100, 200, 300, 500, 800, 1200], [3, 4, 5, 6, 8], [0.005, 0.01, 0.05, 0.1]):
            add(f"catboost_plain_n{n}_d{depth}_lr{lr}", CatBoostClassifier(boosting_type="Plain", iterations=n, depth=depth, learning_rate=lr, random_state=RANDOM_SEED, verbose=False, allow_writing_files=False, thread_count=N_JOBS))
    except ImportError:
        log.info("catboost plain skipped")

    # DNN (PyTorch, if available via penux_ap torch_models is not importable here --
    # use sklearn MLP variants above as the DNN family for this project instead, plus
    # a dedicated PyTorch grid if torch is installed).
    try:
        import torch
        import torch.nn as nn
        from sklearn.base import BaseEstimator, ClassifierMixin

        class SimpleTorchDNN(BaseEstimator, ClassifierMixin):
            """Thin sklearn-compatible wrapper around a small PyTorch MLP.

            Inherits BaseEstimator/ClassifierMixin so it satisfies modern
            scikit-learn's estimator API (__sklearn_tags__ etc.) and can be
            used inside a Pipeline like any built-in classifier.
            """

            def __init__(self, hidden_sizes=(32,), dropout=0.1, lr=1e-3, weight_decay=0.0, epochs=200, seed=RANDOM_SEED):
                self.hidden_sizes = hidden_sizes
                self.dropout = dropout
                self.lr = lr
                self.weight_decay = weight_decay
                self.epochs = epochs
                self.seed = seed

            def fit(self, X, y):
                import numpy as np
                self.classes_ = np.unique(y)
                torch.manual_seed(self.seed)
                n_in = X.shape[1]
                layers = []
                prev = n_in
                for h in self.hidden_sizes:
                    layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
                    prev = h
                layers += [nn.Linear(prev, 1)]
                self.model_ = nn.Sequential(*layers)
                opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
                yt = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)
                loss_fn = nn.BCEWithLogitsLoss()
                self.model_.train()
                for _ in range(self.epochs):
                    opt.zero_grad()
                    out = self.model_(Xt)
                    loss = loss_fn(out, yt)
                    loss.backward()
                    opt.step()
                return self

            def predict_proba(self, X):
                import numpy as np
                self.model_.eval()
                with torch.no_grad():
                    Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
                    p1 = torch.sigmoid(self.model_(Xt)).numpy().ravel()
                return np.column_stack([1 - p1, p1])

            # get_params/set_params are provided by BaseEstimator (introspects
            # __init__ signature automatically); no need to redefine them.

        dnn_hidden_grid = [
            (4,), (8,), (16,), (32,), (64,), (128,), (256,),
            (8, 4), (16, 8), (32, 16), (64, 32), (128, 64), (256, 128),
            (16, 8, 4), (32, 16, 8), (64, 32, 16), (128, 64, 32), (256, 128, 64),
            (32, 16, 8, 4), (64, 32, 16, 8), (128, 64, 32, 16),
        ]
        for hidden, dropout, lr, wd in itertools.product(dnn_hidden_grid, [0.0, 0.1, 0.2, 0.3, 0.4], [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.3, 0.5, 0.7]):
            add(f"torch_dnn_{hidden}_drop{dropout}_lr{lr}_wd{wd}", SimpleTorchDNN(hidden_sizes=hidden, dropout=dropout, lr=lr, weight_decay=wd))
    except ImportError:
        log.info("torch not installed; skipping torch DNN grid")

    return zoo


if __name__ == "__main__":
    zoo = build_model_zoo()
    print(f"Total model configurations: {len(zoo)}")
    names = [n for n, _ in zoo]
    assert len(names) == len(set(names)), "Duplicate model names in zoo!"
