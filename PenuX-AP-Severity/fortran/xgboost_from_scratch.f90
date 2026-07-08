! Gradient-boosted decision trees (XGBoost-style) for SAP prediction,
! implemented from scratch in Fortran -- no ML libraries, no BLAS/LAPACK.
!
! Implements the core of XGBoost's Exact Greedy Algorithm for Split Finding
! (Chen & Guestrin, 2016, Algorithm 1): each tree is fit to the first- and
! second-order gradient statistics (g_i, h_i) of the logistic loss against
! the current ensemble's log-odds predictions, using the regularized gain
!     Gain = 0.5*[GL^2/(HL+lambda) + GR^2/(HR+lambda) - G^2/(H+lambda)] - gamma
! and leaf weights w* = -G/(H+lambda), with additive shrinkage
! (learning_rate) across boosting rounds. This is a teaching/reference
! implementation: no histogram binning, no column subsampling, no
! sparsity-aware split finding -- just the exact greedy algorithm, which
! is tractable at this project's dataset sizes (~700-1300 rows).
!
! Companion to logreg_from_scratch.f90 (see that file's header) and to the
! scikit-learn-ecosystem XGBoost model already benchmarked in
! scripts/model_zoo.py / docs/sap_severity_gbdt_analysis_he.md.
!
! Usage:
!   gfortran -O2 -o xgboost_from_scratch xgboost_from_scratch.f90
!   ./xgboost_from_scratch ../data/public_sanitized/ap_multiml_sanitized.csv
!
! Same input assumptions as logreg_from_scratch.f90: last column is the
! raw binary target, raw value 0 = SAP, all other columns numeric, no
! missing values.
module gbdt_model
  use iso_fortran_env, only: real64
  implicit none
  integer, parameter :: dp = real64

  type :: tree_t
    integer :: n_nodes = 0
    integer, allocatable :: feature(:)
    real(dp), allocatable :: threshold(:)
    integer, allocatable :: left(:), right(:)
    logical, allocatable :: is_leaf(:)
    real(dp), allocatable :: leaf_value(:)
  end type tree_t

contains

  subroutine count_csv(fname, nrows, ncols)
    character(len=*), intent(in) :: fname
    integer, intent(out) :: nrows, ncols
    integer :: unit, ios, i, n
    character(len=20000) :: line
    open (newunit=unit, file=fname, status='old', action='read')
    read (unit, '(A)', iostat=ios) line ! header, discarded
    nrows = 0
    ncols = 0
    do
      read (unit, '(A)', iostat=ios) line
      if (ios /= 0) exit
      nrows = nrows + 1
      if (nrows == 1) then
        n = 1
        do i = 1, len_trim(line)
          if (line(i:i) == ',') n = n + 1
        end do
        ncols = n
      end if
    end do
    close (unit)
  end subroutine count_csv

  subroutine read_csv(fname, nrows, ncols, data)
    character(len=*), intent(in) :: fname
    integer, intent(in) :: nrows, ncols
    real(dp), intent(out) :: data(nrows, ncols)
    integer :: unit, ios, r, i
    character(len=20000) :: line
    open (newunit=unit, file=fname, status='old', action='read')
    read (unit, '(A)', iostat=ios) line ! header, discarded
    do r = 1, nrows
      read (unit, '(A)', iostat=ios) line
      do i = 1, len_trim(line)
        if (line(i:i) == ',') line(i:i) = ' '
      end do
      read (line, *) data(r, 1:ncols)
    end do
    close (unit)
  end subroutine read_csv

  subroutine shuffle(idx, n)
    integer, intent(in) :: n
    integer, intent(inout) :: idx(n)
    integer :: i, j, tmp
    real(dp) :: r
    do i = n, 2, -1
      call random_number(r)
      j = int(r*real(i, dp)) + 1
      tmp = idx(i); idx(i) = idx(j); idx(j) = tmp
    end do
  end subroutine shuffle

  subroutine stratified_split(y, n, train_frac, train_idx, n_train, test_idx, n_test)
    integer, intent(in) :: n
    integer, intent(in) :: y(n)
    real(dp), intent(in) :: train_frac
    integer, allocatable, intent(out) :: train_idx(:), test_idx(:)
    integer, intent(out) :: n_train, n_test
    integer, allocatable :: pos_idx(:), neg_idx(:)
    integer :: n_pos, n_neg, i, k, pos_train, neg_train

    n_pos = count(y == 1)
    n_neg = n - n_pos
    allocate (pos_idx(n_pos), neg_idx(n_neg))
    k = 0
    do i = 1, n
      if (y(i) == 1) then
        k = k + 1; pos_idx(k) = i
      end if
    end do
    k = 0
    do i = 1, n
      if (y(i) == 0) then
        k = k + 1; neg_idx(k) = i
      end if
    end do

    call shuffle(pos_idx, n_pos)
    call shuffle(neg_idx, n_neg)

    pos_train = nint(train_frac*real(n_pos, dp))
    neg_train = nint(train_frac*real(n_neg, dp))
    n_train = pos_train + neg_train
    n_test = n - n_train

    allocate (train_idx(n_train), test_idx(n_test))
    train_idx(1:pos_train) = pos_idx(1:pos_train)
    train_idx(pos_train + 1:n_train) = neg_idx(1:neg_train)
    test_idx(1:n_pos - pos_train) = pos_idx(pos_train + 1:n_pos)
    test_idx(n_pos - pos_train + 1:n_test) = neg_idx(neg_train + 1:n_neg)
  end subroutine stratified_split

  ! exp(x) via Taylor series, not the intrinsic exp(): reduce the argument
  ! by repeated halving until |r| < 0.5 (where the Taylor series converges
  ! in a handful of terms with no cancellation issues), then undo the
  ! reduction by squaring exp(r) back up: exp(x) = exp(x/2^k)^(2^k).
  elemental function taylor_exp(x) result(y)
    real(dp), intent(in) :: x
    real(dp) :: y
    real(dp) :: r, term, s
    integer :: i, k
    r = x
    k = 0
    do while (abs(r) > 0.5_dp)
      r = r*0.5_dp
      k = k + 1
    end do
    s = 1.0_dp
    term = 1.0_dp
    do i = 1, 20
      term = term*r/real(i, dp)
      s = s + term
    end do
    y = s
    do i = 1, k
      y = y*y
    end do
  end function taylor_exp

  elemental function sigmoid(z) result(s)
    real(dp), intent(in) :: z
    real(dp) :: s
    s = 1.0_dp/(1.0_dp + taylor_exp(-z))
  end function sigmoid

  ! In-place quicksort of vals(1:m), permuting ids(1:m) in lockstep.
  recursive subroutine sort_pairs(vals, ids, m)
    integer, intent(in) :: m
    real(dp), intent(inout) :: vals(m)
    integer, intent(inout) :: ids(m)
    real(dp) :: pivot, tv
    integer :: i, j, ti
    if (m <= 1) return
    pivot = vals(m/2 + 1)
    i = 1; j = m
    do
      do while (vals(i) < pivot)
        i = i + 1
      end do
      do while (vals(j) > pivot)
        j = j - 1
      end do
      if (i <= j) then
        tv = vals(i); vals(i) = vals(j); vals(j) = tv
        ti = ids(i); ids(i) = ids(j); ids(j) = ti
        i = i + 1; j = j - 1
      end if
      if (i > j) exit
    end do
    if (j > 1) call sort_pairs(vals(1:j), ids(1:j), j)
    if (i < m) call sort_pairs(vals(i:m), ids(i:m), m - i + 1)
  end subroutine sort_pairs

  ! Recursively build one tree's nodes using gradient/hessian statistics.
  ! `idx` holds the sample indices (into the full X/g/h arrays) at this node.
  recursive subroutine build_node(tree, node_id, X, grad, hess, n_full, p, idx, m, depth, &
                                   max_depth, lambda, gamma, min_child_weight)
    type(tree_t), intent(inout) :: tree
    integer, intent(in) :: node_id, n_full, p, m, depth, max_depth
    real(dp), intent(in) :: X(n_full, p), grad(n_full), hess(n_full)
    integer, intent(in) :: idx(m)
    real(dp), intent(in) :: lambda, gamma, min_child_weight

    real(dp) :: Gsum, Hsum, best_gain, gain, GL, HL, GR, HR, best_thr
    integer :: best_feat, f, k, left_id, right_id
    real(dp), allocatable :: vals(:)
    integer, allocatable :: order(:), left_idx(:), right_idx(:)
    integer :: n_left, n_right, i

    Gsum = sum(grad(idx)); Hsum = sum(hess(idx))

    if (depth >= max_depth .or. m < 2 .or. Hsum < 2.0_dp*min_child_weight) then
      tree%is_leaf(node_id) = .true.
      tree%leaf_value(node_id) = -Gsum/(Hsum + lambda)
      return
    end if

    best_gain = 0.0_dp
    best_feat = -1
    best_thr = 0.0_dp

    allocate (vals(m), order(m))
    do f = 1, p
      order = idx
      vals = X(idx, f)
      call sort_pairs(vals, order, m)

      GL = 0.0_dp; HL = 0.0_dp
      do k = 1, m - 1
        GL = GL + grad(order(k)); HL = HL + hess(order(k))
        if (vals(k) == vals(k + 1)) cycle
        if (HL < min_child_weight) cycle
        GR = Gsum - GL; HR = Hsum - HL
        if (HR < min_child_weight) cycle
        gain = 0.5_dp*(GL*GL/(HL + lambda) + GR*GR/(HR + lambda) - Gsum*Gsum/(Hsum + lambda)) - gamma
        if (gain > best_gain) then
          best_gain = gain
          best_feat = f
          best_thr = 0.5_dp*(vals(k) + vals(k + 1))
        end if
      end do
    end do
    deallocate (vals, order)

    if (best_feat == -1) then
      tree%is_leaf(node_id) = .true.
      tree%leaf_value(node_id) = -Gsum/(Hsum + lambda)
      return
    end if

    allocate (left_idx(m), right_idx(m))
    n_left = 0; n_right = 0
    do i = 1, m
      if (X(idx(i), best_feat) <= best_thr) then
        n_left = n_left + 1; left_idx(n_left) = idx(i)
      else
        n_right = n_right + 1; right_idx(n_right) = idx(i)
      end if
    end do

    tree%is_leaf(node_id) = .false.
    tree%feature(node_id) = best_feat
    tree%threshold(node_id) = best_thr

    left_id = tree%n_nodes + 1
    right_id = tree%n_nodes + 2
    tree%n_nodes = tree%n_nodes + 2
    tree%left(node_id) = left_id
    tree%right(node_id) = right_id

    call build_node(tree, left_id, X, grad, hess, n_full, p, left_idx(1:n_left), n_left, depth + 1, &
                     max_depth, lambda, gamma, min_child_weight)
    call build_node(tree, right_id, X, grad, hess, n_full, p, right_idx(1:n_right), n_right, depth + 1, &
                     max_depth, lambda, gamma, min_child_weight)
    deallocate (left_idx, right_idx)
  end subroutine build_node

  subroutine train_tree(tree, X, grad, hess, n, p, max_depth, lambda, gamma, min_child_weight)
    type(tree_t), intent(out) :: tree
    integer, intent(in) :: n, p, max_depth
    real(dp), intent(in) :: X(n, p), grad(n), hess(n)
    real(dp), intent(in) :: lambda, gamma, min_child_weight
    integer :: max_nodes, i, idx(n)

    max_nodes = 2**(max_depth + 1)
    allocate (tree%feature(max_nodes), tree%threshold(max_nodes))
    allocate (tree%left(max_nodes), tree%right(max_nodes))
    allocate (tree%is_leaf(max_nodes), tree%leaf_value(max_nodes))
    tree%feature = 0; tree%threshold = 0.0_dp
    tree%left = 0; tree%right = 0
    tree%is_leaf = .false.; tree%leaf_value = 0.0_dp
    tree%n_nodes = 1

    do i = 1, n
      idx(i) = i
    end do
    call build_node(tree, 1, X, grad, hess, n, p, idx, n, 0, max_depth, lambda, gamma, min_child_weight)
  end subroutine train_tree

  function predict_tree(tree, x_row, p) result(val)
    type(tree_t), intent(in) :: tree
    integer, intent(in) :: p
    real(dp), intent(in) :: x_row(p)
    real(dp) :: val
    integer :: node
    node = 1
    do while (.not. tree%is_leaf(node))
      if (x_row(tree%feature(node)) <= tree%threshold(node)) then
        node = tree%left(node)
      else
        node = tree%right(node)
      end if
    end do
    val = tree%leaf_value(node)
  end function predict_tree

  subroutine train_gbdt(X, y, n, p, n_estimators, max_depth, learning_rate, &
                         lambda, gamma, min_child_weight, trees, F_init, verbose)
    integer, intent(in) :: n, p, n_estimators, max_depth
    real(dp), intent(in) :: X(n, p)
    integer, intent(in) :: y(n)
    real(dp), intent(in) :: learning_rate, lambda, gamma, min_child_weight
    type(tree_t), intent(out) :: trees(n_estimators)
    real(dp), intent(out) :: F_init
    logical, intent(in) :: verbose
    real(dp) :: F(n), grad(n), hess(n), pr(n), rate, loss
    integer :: m, i
    real(dp), parameter :: eps = 1.0e-12_dp

    rate = sum(real(y, dp))/real(n, dp)
    rate = min(max(rate, 0.01_dp), 0.99_dp)
    F_init = log(rate/(1.0_dp - rate))
    F = F_init

    do m = 1, n_estimators
      pr = sigmoid(F)
      grad = pr - real(y, dp)
      hess = pr*(1.0_dp - pr)
      call train_tree(trees(m), X, grad, hess, n, p, max_depth, lambda, gamma, min_child_weight)
      do i = 1, n
        F(i) = F(i) + learning_rate*predict_tree(trees(m), X(i, :), p)
      end do
      if (verbose .and. (mod(m, 10) == 0 .or. m == n_estimators)) then
        pr = sigmoid(F)
        loss = -sum(real(y, dp)*log(pr + eps) + (1.0_dp - real(y, dp))*log(1.0_dp - pr + eps))/real(n, dp)
        print '(A,I4,A,I0,A,F10.5)', '  tree ', m, '/', n_estimators, '  train BCE loss = ', loss
      end if
    end do
  end subroutine train_gbdt

  function predict_gbdt(trees, n_estimators, learning_rate, F_init, x_row, p) result(proba)
    integer, intent(in) :: n_estimators, p
    type(tree_t), intent(in) :: trees(n_estimators)
    real(dp), intent(in) :: learning_rate, F_init, x_row(p)
    real(dp) :: proba, Fv
    integer :: m
    Fv = F_init
    do m = 1, n_estimators
      Fv = Fv + learning_rate*predict_tree(trees(m), x_row, p)
    end do
    proba = sigmoid(Fv)
  end function predict_gbdt

  function auroc(scores, y, n) result(auc)
    integer, intent(in) :: n
    real(dp), intent(in) :: scores(n)
    integer, intent(in) :: y(n)
    real(dp) :: auc
    integer :: i, j
    real(dp) :: total
    total = 0.0_dp
    do i = 1, n
      if (y(i) /= 1) cycle
      do j = 1, n
        if (y(j) /= 0) cycle
        if (scores(i) > scores(j)) then
          total = total + 1.0_dp
        else if (scores(i) == scores(j)) then
          total = total + 0.5_dp
        end if
      end do
    end do
    auc = total/real(count(y == 1), dp)/real(count(y == 0), dp)
  end function auroc

  subroutine confusion_metrics(scores, y, n, threshold, tp, fp, tn, fn, &
                                sensitivity, specificity, ppv, npv, f1, accuracy)
    integer, intent(in) :: n
    real(dp), intent(in) :: scores(n), threshold
    integer, intent(in) :: y(n)
    integer, intent(out) :: tp, fp, tn, fn
    real(dp), intent(out) :: sensitivity, specificity, ppv, npv, f1, accuracy
    integer :: i, pred

    tp = 0; fp = 0; tn = 0; fn = 0
    do i = 1, n
      if (scores(i) >= threshold) then
        pred = 1
      else
        pred = 0
      end if
      if (pred == 1 .and. y(i) == 1) tp = tp + 1
      if (pred == 1 .and. y(i) == 0) fp = fp + 1
      if (pred == 0 .and. y(i) == 0) tn = tn + 1
      if (pred == 0 .and. y(i) == 1) fn = fn + 1
    end do

    sensitivity = real(tp, dp)/real(max(tp + fn, 1), dp)
    specificity = real(tn, dp)/real(max(tn + fp, 1), dp)
    ppv = real(tp, dp)/real(max(tp + fp, 1), dp)
    npv = real(tn, dp)/real(max(tn + fn, 1), dp)
    if (ppv + sensitivity > 0.0_dp) then
      f1 = 2.0_dp*ppv*sensitivity/(ppv + sensitivity)
    else
      f1 = 0.0_dp
    end if
    accuracy = real(tp + tn, dp)/real(n, dp)
  end subroutine confusion_metrics

end module gbdt_model

program main
  use gbdt_model
  implicit none
  character(len=256) :: fname, arg
  integer :: nrows, ncols, nfeat
  real(dp), allocatable :: data(:, :), X(:, :), Xtr(:, :), Xte(:, :)
  integer, allocatable :: y(:), ytr(:), yte(:)
  integer, allocatable :: train_idx(:), test_idx(:)
  integer :: n_train, n_test, seed_size, n_estimators, max_depth, i
  real(dp) :: learning_rate, lambda, gamma, min_child_weight, F_init, threshold
  type(tree_t), allocatable :: trees(:)
  real(dp), allocatable :: proba_test(:)
  integer :: tp, fp, tn, fn
  real(dp) :: sensitivity, specificity, ppv, npv, f1, accuracy, auc
  integer, allocatable :: seed(:)
  real(dp) :: t_start, t_end

  if (command_argument_count() < 1) then
    print *, 'Usage: xgboost_from_scratch <sanitized_csv_path> [n_estimators] [max_depth] [learning_rate]'
    print *, 'Example: xgboost_from_scratch ../data/public_sanitized/ap_multiml_sanitized.csv'
    stop 1
  end if
  call get_command_argument(1, fname)

  call random_seed(size=seed_size)
  allocate (seed(seed_size))
  seed = 42
  call random_seed(put=seed)

  print '(A,A)', 'Reading: ', trim(fname)
  call count_csv(fname, nrows, ncols)
  print '(A,I0,A,I0,A)', 'Loaded ', nrows, ' rows x ', ncols, ' columns (incl. target).'

  allocate (data(nrows, ncols))
  call read_csv(fname, nrows, ncols, data)

  nfeat = ncols - 1
  allocate (X(nrows, nfeat), y(nrows))
  X = data(:, 1:nfeat)
  ! Raw target: 0 = SAP, 1 = non-SAP (both registered public datasets --
  ! see docs/dataset_sources.md). Flip so 1 = SAP.
  y = nint(1.0_dp - data(:, ncols))

  print '(A,I0,A,I0,A)', 'Target distribution: ', count(y == 1), ' SAP / ', count(y == 0), ' non-SAP'

  call stratified_split(y, nrows, 0.8_dp, train_idx, n_train, test_idx, n_test)
  print '(A,I0,A,I0)', 'Train size: ', n_train, '   Test size: ', n_test

  allocate (Xtr(n_train, nfeat), ytr(n_train), Xte(n_test, nfeat), yte(n_test))
  Xtr = X(train_idx, :)
  ytr = y(train_idx)
  Xte = X(test_idx, :)
  yte = y(test_idx)

  n_estimators = 100
  max_depth = 3
  learning_rate = 0.1_dp
  lambda = 1.0_dp
  gamma = 0.0_dp
  min_child_weight = 1.0_dp

  if (command_argument_count() >= 2) then
    call get_command_argument(2, arg); read (arg, *) n_estimators
  end if
  if (command_argument_count() >= 3) then
    call get_command_argument(3, arg); read (arg, *) max_depth
  end if
  if (command_argument_count() >= 4) then
    call get_command_argument(4, arg); read (arg, *) learning_rate
  end if

  print '(A,I0,A,I0,A,F5.3,A,F5.2)', 'Training GBDT: n_estimators=', n_estimators, &
    '  max_depth=', max_depth, '  learning_rate=', learning_rate, '  lambda=', lambda

  allocate (trees(n_estimators))
  call cpu_time(t_start)
  call train_gbdt(Xtr, ytr, n_train, nfeat, n_estimators, max_depth, learning_rate, &
                   lambda, gamma, min_child_weight, trees, F_init, .true.)
  call cpu_time(t_end)
  print '(A,F8.2,A)', 'Training time: ', t_end - t_start, ' s (CPU)'

  allocate (proba_test(n_test))
  do i = 1, n_test
    proba_test(i) = predict_gbdt(trees, n_estimators, learning_rate, F_init, Xte(i, :), nfeat)
  end do

  auc = auroc(proba_test, yte, n_test)
  threshold = 0.5_dp
  call confusion_metrics(proba_test, yte, n_test, threshold, tp, fp, tn, fn, &
                          sensitivity, specificity, ppv, npv, f1, accuracy)

  print *, ''
  print '(A)', '=== Test-set evaluation (threshold=0.5, positive class = SAP) ==='
  print '(A,F7.4)', 'AUROC       = ', auc
  print '(A,F7.4)', 'Accuracy    = ', accuracy
  print '(A,F7.4)', 'Sensitivity = ', sensitivity
  print '(A,F7.4)', 'Specificity = ', specificity
  print '(A,F7.4)', 'PPV         = ', ppv
  print '(A,F7.4)', 'NPV         = ', npv
  print '(A,F7.4)', 'F1          = ', f1
  print '(A,I0,A,I0,A,I0,A,I0)', 'TP=', tp, '  FP=', fp, '  TN=', tn, '  FN=', fn

end program main
