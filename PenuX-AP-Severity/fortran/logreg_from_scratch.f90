! Logistic regression for SAP prediction, implemented from scratch in Fortran.
!
! No ML libraries, no BLAS/LAPACK: CSV parsing, feature standardization,
! batch gradient descent, and evaluation (confusion matrix, sensitivity,
! specificity, PPV, NPV, F1, AUC) are all implemented directly below.
!
! This is a from-scratch teaching/reference implementation, not a
! replacement for the scikit-learn/XGBoost/LightGBM/CatBoost pipeline used
! elsewhere in PenuX-AP-Severity (src/penux_ap, scripts/*.py). See
! docs/sap_severity_gbdt_analysis_he.md for the full model comparison.
!
! Usage:
!   gfortran -O2 -o logreg_from_scratch logreg_from_scratch.f90
!   ./logreg_from_scratch ../data/public_sanitized/ap_multiml_sanitized.csv
!
! Assumes the input CSV's last column is the raw binary target and that
! raw value 0 denotes SAP (see docs/dataset_sources.md -- both registered
! public datasets have this reversed-vs-usual label direction). All other
! columns are treated as numeric features.
module lr_model
  use iso_fortran_env, only: real64
  implicit none
  integer, parameter :: dp = real64

contains

  subroutine count_csv(fname, nrows, ncols)
    character(len=*), intent(in) :: fname
    integer, intent(out) :: nrows, ncols
    integer :: unit, ios, i, n
    character(len=20000) :: line
    open(newunit=unit, file=fname, status='old', action='read')
    read(unit, '(A)', iostat=ios) line ! header, discarded
    nrows = 0
    ncols = 0
    do
      read(unit, '(A)', iostat=ios) line
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
    close(unit)
  end subroutine count_csv

  subroutine read_csv(fname, nrows, ncols, data)
    character(len=*), intent(in) :: fname
    integer, intent(in) :: nrows, ncols
    real(dp), intent(out) :: data(nrows, ncols)
    integer :: unit, ios, r, i
    character(len=20000) :: line
    open(newunit=unit, file=fname, status='old', action='read')
    read(unit, '(A)', iostat=ios) line ! header, discarded
    do r = 1, nrows
      read(unit, '(A)', iostat=ios) line
      do i = 1, len_trim(line)
        if (line(i:i) == ',') line(i:i) = ' '
      end do
      read(line, *) data(r, 1:ncols)
    end do
    close(unit)
  end subroutine read_csv

  subroutine standardize_fit(X, n, p, mu, sigma)
    integer, intent(in) :: n, p
    real(dp), intent(in) :: X(n, p)
    real(dp), intent(out) :: mu(p), sigma(p)
    integer :: j
    do j = 1, p
      mu(j) = sum(X(:, j)) / real(n, dp)
      sigma(j) = sqrt(sum((X(:, j) - mu(j))**2) / real(n, dp))
      if (sigma(j) < 1.0e-12_dp) sigma(j) = 1.0_dp
    end do
  end subroutine standardize_fit

  subroutine standardize_apply(X, n, p, mu, sigma)
    integer, intent(in) :: n, p
    real(dp), intent(inout) :: X(n, p)
    real(dp), intent(in) :: mu(p), sigma(p)
    integer :: j
    do j = 1, p
      X(:, j) = (X(:, j) - mu(j)) / sigma(j)
    end do
  end subroutine standardize_apply

  ! Fisher-Yates shuffle of a 1-based index array.
  subroutine shuffle(idx, n)
    integer, intent(in) :: n
    integer, intent(inout) :: idx(n)
    integer :: i, j, tmp
    real(dp) :: r
    do i = n, 2, -1
      call random_number(r)
      j = int(r * real(i, dp)) + 1
      tmp = idx(i); idx(i) = idx(j); idx(j) = tmp
    end do
  end subroutine shuffle

  ! Stratified train/test split on binary labels y (0/1).
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

    pos_train = nint(train_frac * real(n_pos, dp))
    neg_train = nint(train_frac * real(n_neg, dp))
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
      r = r * 0.5_dp
      k = k + 1
    end do
    s = 1.0_dp
    term = 1.0_dp
    do i = 1, 20
      term = term * r / real(i, dp)
      s = s + term
    end do
    y = s
    do i = 1, k
      y = y * y
    end do
  end function taylor_exp

  ! exp(x) via the [3/3] Pade approximant: a ratio of two cubic
  ! polynomials, P(r)/Q(r), which approximates e^r more accurately per
  ! polynomial degree than a plain Taylor series of the same order (a
  ! rational function can capture the "curvature" of e^x with far fewer
  ! terms than a polynomial can). Same range-reduction scheme as
  ! taylor_exp: reduce to |r|<0.5, apply the approximant, square back up.
  ! Matches the intrinsic exp() to >=7 significant decimal digits (well
  ! past the 4-decimal-digit target) for x in [-20, 20].
  elemental function pade_exp(x) result(y)
    real(dp), intent(in) :: x
    real(dp) :: y
    real(dp) :: r, r2, r3, num, den
    integer :: i, k
    r = x
    k = 0
    do while (abs(r) > 0.5_dp)
      r = r * 0.5_dp
      k = k + 1
    end do
    r2 = r * r
    r3 = r2 * r
    num = 1.0_dp + r / 2.0_dp + r2 / 10.0_dp + r3 / 120.0_dp
    den = 1.0_dp - r / 2.0_dp + r2 / 10.0_dp - r3 / 120.0_dp
    y = num / den
    do i = 1, k
      y = y * y
    end do
  end function pade_exp

  elemental function sigmoid(z) result(s)
    real(dp), intent(in) :: z
    real(dp) :: s
    s = 1.0_dp / (1.0_dp + pade_exp(-z))
  end function sigmoid

  ! Full-batch gradient descent with L2 regularization.
  subroutine train_logreg(X, y, n, p, lr, lambda, n_iter, w, b, verbose)
    integer, intent(in) :: n, p, n_iter
    real(dp), intent(in) :: X(n, p)
    integer, intent(in) :: y(n)
    real(dp), intent(in) :: lr, lambda
    real(dp), intent(out) :: w(p)
    real(dp), intent(out) :: b
    logical, intent(in) :: verbose
    real(dp) :: z(n), pr(n), grad_w(p), grad_b, loss
    integer :: it
    real(dp), parameter :: eps = 1.0e-12_dp

    w = 0.0_dp
    b = 0.0_dp
    do it = 1, n_iter
      z = matmul(X, w) + b
      pr = sigmoid(z)
      grad_w = matmul(transpose(X), pr - real(y, dp)) / real(n, dp) + lambda * w
      grad_b = sum(pr - real(y, dp)) / real(n, dp)
      w = w - lr * grad_w
      b = b - lr * grad_b
      if (verbose .and. mod(it, 500) == 0) then
        loss = -sum(real(y, dp) * log(pr + eps) + (1.0_dp - real(y, dp)) * log(1.0_dp - pr + eps)) / real(n, dp)
        print '(A,I5,A,F10.5)', '  iter ', it, '  train BCE loss = ', loss
      end if
    end do
  end subroutine train_logreg

  function predict_proba(X, n, p, w, b) result(pr)
    integer, intent(in) :: n, p
    real(dp), intent(in) :: X(n, p), w(p), b
    real(dp) :: pr(n)
    pr = sigmoid(matmul(X, w) + b)
  end function predict_proba

  ! AUROC via the Mann-Whitney U statistic (exact, ties count as 0.5).
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
    auc = total / real(count(y == 1), dp) / real(count(y == 0), dp)
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

    sensitivity = real(tp, dp) / real(max(tp + fn, 1), dp)
    specificity = real(tn, dp) / real(max(tn + fp, 1), dp)
    ppv = real(tp, dp) / real(max(tp + fp, 1), dp)
    npv = real(tn, dp) / real(max(tn + fn, 1), dp)
    if (ppv + sensitivity > 0.0_dp) then
      f1 = 2.0_dp * ppv * sensitivity / (ppv + sensitivity)
    else
      f1 = 0.0_dp
    end if
    accuracy = real(tp + tn, dp) / real(n, dp)
  end subroutine confusion_metrics

end module lr_model

program main
  use lr_model
  implicit none
  character(len=256) :: fname, arg
  integer :: nrows, ncols, nfeat
  real(dp), allocatable :: data(:, :), X(:, :), Xtr(:, :), Xte(:, :)
  integer, allocatable :: y(:), ytr(:), yte(:)
  integer, allocatable :: train_idx(:), test_idx(:)
  integer :: n_train, n_test, i, seed_size
  real(dp), allocatable :: mu(:), sigma(:), w(:)
  real(dp) :: b, lr, lambda, threshold
  real(dp), allocatable :: proba_test(:)
  integer :: tp, fp, tn, fn
  real(dp) :: sensitivity, specificity, ppv, npv, f1, accuracy, auc
  integer, allocatable :: seed(:)

  if (command_argument_count() < 1) then
    print *, 'Usage: logreg_from_scratch <sanitized_csv_path> [n_iter] [learning_rate] [l2_lambda]'
    print *, 'Example: logreg_from_scratch ../data/public_sanitized/ap_multiml_sanitized.csv'
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
  ! Raw target: 0 = SAP, 1 = non-SAP for both registered public datasets
  ! (see docs/dataset_sources.md). Flip so 1 = SAP.
  y = nint(1.0_dp - data(:, ncols))

  print '(A,I0,A,I0,A)', 'Target distribution: ', count(y == 1), ' SAP / ', count(y == 0), ' non-SAP'

  call stratified_split(y, nrows, 0.8_dp, train_idx, n_train, test_idx, n_test)
  print '(A,I0,A,I0)', 'Train size: ', n_train, '   Test size: ', n_test

  allocate (Xtr(n_train, nfeat), ytr(n_train), Xte(n_test, nfeat), yte(n_test))
  Xtr = X(train_idx, :)
  ytr = y(train_idx)
  Xte = X(test_idx, :)
  yte = y(test_idx)

  allocate (mu(nfeat), sigma(nfeat))
  call standardize_fit(Xtr, n_train, nfeat, mu, sigma)
  call standardize_apply(Xtr, n_train, nfeat, mu, sigma)
  call standardize_apply(Xte, n_test, nfeat, mu, sigma)

  lr = 0.3_dp
  lambda = 1.0e-3_dp
  if (command_argument_count() >= 3) then
    call get_command_argument(3, arg)
    read (arg, *) lr
  end if
  if (command_argument_count() >= 4) then
    call get_command_argument(4, arg)
    read (arg, *) lambda
  end if

  block
    integer :: n_iter
    n_iter = 3000
    if (command_argument_count() >= 2) then
      call get_command_argument(2, arg)
      read (arg, *) n_iter
    end if
    print '(A,I0,A,F6.3,A,ES9.2)', 'Training logistic regression: n_iter=', n_iter, '  lr=', lr, '  l2_lambda=', lambda
    allocate (w(nfeat))
    call train_logreg(Xtr, ytr, n_train, nfeat, lr, lambda, n_iter, w, b, .true.)
  end block

  allocate (proba_test(n_test))
  proba_test = predict_proba(Xte, n_test, nfeat, w, b)

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
