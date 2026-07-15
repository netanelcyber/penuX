! Feedforward neural network for SAP prediction, implemented from scratch
! in Fortran -- no ML libraries, no BLAS/LAPACK (beyond intrinsic matmul).
!
! The model is exactly a chain (composition) of functions:
!
!     F(x) = f_L( f_{L-1}( ... f_1(x) ... ) )
!
! where f_1 takes the full parameter vector x in R^p as input, and every
! single f_l (hidden layers AND the output layer alike) has the same form
!
!     f_l(a) = sigmoid(W_l a + b_l)
!
! mapping R^{h_{l-1}} -> R^{h_l} (or -> (0,1) for the final f_L, a single
! number between 0 and 1: the predicted SAP probability). This is a
! deliberate constraint, not an accident: every f_l must be a polynomial
! or a rational function (a ratio of two polynomials) -- so there is no
! ReLU (piecewise, not a single polynomial/rational expression) anywhere
! in this network. `sigmoid` is built entirely from `pade_exp`, the
! [3/3] Pade approximant for e^x (a ratio of two cubic polynomials, see
! logreg_from_scratch.f90 / xgboost_from_scratch.f90), so every f_l here
! is exactly sigmoid(linear polynomial in its input) = a rational
! function of a polynomial -- i.e. every stage of the chain is polynomial
! or rational, all the way through.
!
! Training is standard backpropagation (the chain rule applied layer by
! layer, backwards through the same composition) with batch gradient
! descent. See the "Pseudocode" section of fortran/README.md for the
! layer-by-layer forward/backward equations.
!
! Usage:
!   gfortran -O2 -o dnn_from_scratch dnn_from_scratch.f90
!   ./dnn_from_scratch ../data/public_sanitized/ap_multiml_sanitized.csv 64,32
!
! Same input assumptions as the other from-scratch programs: last CSV
! column is the raw binary target, raw value 0 = SAP, all other columns
! numeric, no missing values.
module dnn_model
  use iso_fortran_env, only: real64
  implicit none
  integer, parameter :: dp = real64

  type :: layer_t
    real(dp), allocatable :: W(:, :)   ! (in, out)
    real(dp), allocatable :: b(:)      ! (out)
    real(dp), allocatable :: A(:, :)   ! cached activation output, (n, out) -- for backprop
    real(dp), allocatable :: Z(:, :)   ! cached pre-activation, (n, out) -- for backprop
  end type layer_t

contains

  subroutine count_csv(fname, nrows, ncols)
    character(len=*), intent(in) :: fname
    integer, intent(out) :: nrows, ncols
    integer :: unit, ios, i, n
    character(len=20000) :: line
    open (newunit=unit, file=fname, status='old', action='read')
    read (unit, '(A)', iostat=ios) line
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
    read (unit, '(A)', iostat=ios) line
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

  subroutine standardize_fit(X, n, p, mu, sigma)
    integer, intent(in) :: n, p
    real(dp), intent(in) :: X(n, p)
    real(dp), intent(out) :: mu(p), sigma(p)
    integer :: j
    do j = 1, p
      mu(j) = sum(X(:, j))/real(n, dp)
      sigma(j) = sqrt(sum((X(:, j) - mu(j))**2)/real(n, dp))
      if (sigma(j) < 1.0e-12_dp) sigma(j) = 1.0_dp
    end do
  end subroutine standardize_fit

  subroutine standardize_apply(X, n, p, mu, sigma)
    integer, intent(in) :: n, p
    real(dp), intent(inout) :: X(n, p)
    real(dp), intent(in) :: mu(p), sigma(p)
    integer :: j
    do j = 1, p
      X(:, j) = (X(:, j) - mu(j))/sigma(j)
    end do
  end subroutine standardize_apply

  ! e^x via the [3/3] Pade approximant (ratio of two cubic polynomials),
  ! range-reduced by repeated halving -- see fortran/README.md.
  elemental function pade_exp(x) result(y)
    real(dp), intent(in) :: x
    real(dp) :: y
    real(dp) :: r, r2, r3, num, den
    integer :: i, k
    r = x
    k = 0
    do while (abs(r) > 0.5_dp)
      r = r*0.5_dp
      k = k + 1
    end do
    r2 = r*r
    r3 = r2*r
    num = 1.0_dp + r/2.0_dp + r2/10.0_dp + r3/120.0_dp
    den = 1.0_dp - r/2.0_dp + r2/10.0_dp - r3/120.0_dp
    y = num/den
    do i = 1, k
      y = y*y
    end do
  end function pade_exp

  elemental function sigmoid(z) result(s)
    real(dp), intent(in) :: z
    real(dp) :: s
    s = 1.0_dp/(1.0_dp + pade_exp(-z))
  end function sigmoid

  ! sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)), the standard identity --
  ! takes the already-computed activation a = sigmoid(z) to avoid a second
  ! pade_exp evaluation.
  elemental function sigmoid_prime_from_activation(a) result(d)
    real(dp), intent(in) :: a
    real(dp) :: d
    d = a*(1.0_dp - a)
  end function sigmoid_prime_from_activation

  ! Build the network: layer_sizes(1) = p (input dim), layer_sizes(k) for
  ! k=2..L-1 are hidden widths, layer_sizes(L) = 1 (output). Layer l holds
  ! W mapping layer_sizes(l) -> layer_sizes(l+1).
  subroutine init_network(net, layer_sizes, n_layers)
    integer, intent(in) :: n_layers
    integer, intent(in) :: layer_sizes(n_layers)
    type(layer_t), allocatable, intent(out) :: net(:)
    integer :: l, fan_in, fan_out, i, j
    real(dp) :: scale, u1, u2
    allocate (net(n_layers - 1))
    do l = 1, n_layers - 1
      fan_in = layer_sizes(l)
      fan_out = layer_sizes(l + 1)
      allocate (net(l)%W(fan_in, fan_out), net(l)%b(fan_out))
      ! Xavier/Glorot-style scale (appropriate for sigmoid activations,
      ! unlike He initialization which assumes ReLU).
      scale = sqrt(1.0_dp/real(fan_in, dp))
      do j = 1, fan_out
        do i = 1, fan_in
          call random_number(u1)
          call random_number(u2)
          ! Box-Muller: standard normal from two uniforms, scaled by Xavier init.
          net(l)%W(i, j) = scale*sqrt(-2.0_dp*log(u1 + 1.0e-12_dp))*cos(6.283185307179586_dp*u2)
        end do
      end do
      net(l)%b = 0.0_dp
    end do
  end subroutine init_network

  ! Forward pass: the composition F(x) = f_L(...f_1(x)...). Returns the
  ! final layer's output column (n) as the predicted probabilities, and
  ! caches every layer's Z/A (needed by backward_and_update).
  function forward(net, n_layers, X, n) result(proba)
    integer, intent(in) :: n_layers, n
    type(layer_t), intent(inout) :: net(n_layers - 1)
    real(dp), intent(in) :: X(n, size(net(1)%W, 1))
    real(dp) :: proba(n)
    real(dp), allocatable :: A_prev(:, :)
    integer :: l

    allocate (A_prev(n, size(net(1)%W, 1)))
    A_prev = X
    do l = 1, n_layers - 1
      if (allocated(net(l)%Z)) deallocate (net(l)%Z)
      if (allocated(net(l)%A)) deallocate (net(l)%A)
      allocate (net(l)%Z(n, size(net(l)%W, 2)), net(l)%A(n, size(net(l)%W, 2)))
      net(l)%Z = matmul(A_prev, net(l)%W)
      net(l)%Z(:, :) = net(l)%Z(:, :) + spread(net(l)%b, 1, n)
      net(l)%A = sigmoid(net(l)%Z)  ! every f_l uses the same rational-function activation
      deallocate (A_prev)
      allocate (A_prev(n, size(net(l)%W, 2)))
      A_prev = net(l)%A
    end do
    proba = A_prev(:, 1)
  end function forward

  ! Backpropagation: apply the chain rule through the same composition,
  ! layer by layer from f_L back to f_1, then take one gradient-descent
  ! step on every layer's (W, b).
  subroutine backward_and_update(net, n_layers, X, y, n, lr)
    integer, intent(in) :: n_layers, n
    type(layer_t), intent(inout) :: net(n_layers - 1)
    real(dp), intent(in) :: X(n, size(net(1)%W, 1)), y(n), lr
    real(dp), allocatable :: dZ(:, :), dA_prev(:, :), A_prev(:, :), w(:)
    integer :: l, fan_in, fan_out
    real(dp) :: n_pos, n_neg, w_pos, w_neg

    ! Output layer: for sigmoid + binary cross-entropy, dZ_L = (A_L - y) / n
    ! (the well-known simplification of the combined sigmoid+BCE gradient).
    ! Class-balanced weighting (like sklearn's class_weight='balanced'):
    ! without it, the ~16-19% SAP prevalence lets the network settle into
    ! predicting everyone negative (correct on average, but useless --
    ! sensitivity=0 at threshold 0.5) since that already minimizes
    ! unweighted BCE fairly well.
    n_pos = max(sum(y), 1.0_dp)
    n_neg = max(real(n, dp) - n_pos, 1.0_dp)
    w_pos = real(n, dp)/(2.0_dp*n_pos)
    w_neg = real(n, dp)/(2.0_dp*n_neg)
    allocate (w(n))
    w = merge(w_pos, w_neg, y > 0.5_dp)

    allocate (dZ(n, 1))
    dZ(:, 1) = w*(net(n_layers - 1)%A(:, 1) - y)/real(n, dp)
    deallocate (w)

    do l = n_layers - 1, 1, -1
      fan_in = size(net(l)%W, 1)
      fan_out = size(net(l)%W, 2)
      if (l == 1) then
        allocate (A_prev(n, fan_in))
        A_prev = X
      else
        allocate (A_prev(n, fan_in))
        A_prev = net(l - 1)%A
      end if

      block
        real(dp) :: dW(fan_in, fan_out), db(fan_out)
        dW = matmul(transpose(A_prev), dZ)
        db = sum(dZ, dim=1)
        net(l)%W = net(l)%W - lr*dW
        net(l)%b = net(l)%b - lr*db
      end block

      if (l > 1) then
        allocate (dA_prev(n, fan_in))
        dA_prev = matmul(dZ, transpose(net(l)%W))
        deallocate (dZ)
        allocate (dZ(n, fan_in))
        dZ = dA_prev*sigmoid_prime_from_activation(net(l - 1)%A)
        deallocate (dA_prev)
      end if
      deallocate (A_prev)
    end do
    deallocate (dZ)
  end subroutine backward_and_update

  ! Dump every layer's full W and b to a plain-text file, one layer per
  ! block: a header line "# layer L: fan_in x fan_out", then fan_in rows
  ! of fan_out space-separated W values, then one row of fan_out b values.
  subroutine save_weights(net, n_layers, fname)
    integer, intent(in) :: n_layers
    type(layer_t), intent(in) :: net(n_layers - 1)
    character(len=*), intent(in) :: fname
    integer :: unit, l, i, fan_in, fan_out
    open (newunit=unit, file=fname, status='replace', action='write')
    do l = 1, n_layers - 1
      fan_in = size(net(l)%W, 1)
      fan_out = size(net(l)%W, 2)
      write (unit, '(A,I0,A,I0,A,I0)') '# layer ', l, ': W is ', fan_in, ' x ', fan_out
      do i = 1, fan_in
        write (unit, '(*(ES16.8,1X))') net(l)%W(i, :)
      end do
      write (unit, '(A,I0,A,I0)') '# layer ', l, ': b is ', fan_out
      write (unit, '(*(ES16.8,1X))') net(l)%b
    end do
    close (unit)
  end subroutine save_weights

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

end module dnn_model

program main
  use dnn_model
  implicit none
  character(len=256) :: fname, arg, hidden_spec
  integer :: nrows, ncols, nfeat
  real(dp), allocatable :: data(:, :), X(:, :), Xtr(:, :), Xte(:, :)
  integer, allocatable :: y(:), ytr(:), yte(:)
  integer, allocatable :: train_idx(:), test_idx(:)
  integer :: n_train, n_test, seed_size, n_epochs, i, n_layers
  integer, allocatable :: layer_sizes(:), hidden_sizes(:)
  real(dp) :: lr, threshold
  type(layer_t), allocatable :: net(:)
  real(dp), allocatable :: mu(:), sigma(:), proba_train(:), proba_test(:)
  integer :: tp, fp, tn, fn, epoch
  real(dp) :: sensitivity, specificity, ppv, npv, f1, accuracy, auc, loss
  integer, allocatable :: seed(:)
  real(dp), parameter :: eps = 1.0e-12_dp

  if (command_argument_count() < 1) then
    print *, 'Usage: dnn_from_scratch <csv_path> [hidden_sizes csv, e.g. 64,32] [n_epochs] [lr]'
    print *, 'Example: dnn_from_scratch ../data/public_sanitized/ap_multiml_sanitized.csv 64,32'
    stop 1
  end if
  call get_command_argument(1, fname)

  hidden_spec = '64,32'
  if (command_argument_count() >= 2) call get_command_argument(2, hidden_spec)
  n_epochs = 300
  if (command_argument_count() >= 3) then
    call get_command_argument(3, arg); read (arg, *) n_epochs
  end if
  lr = 0.05_dp
  if (command_argument_count() >= 4) then
    call get_command_argument(4, arg); read (arg, *) lr
  end if

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
  y = nint(1.0_dp - data(:, ncols))  ! raw 0 = SAP -> flip so 1 = SAP

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

  ! Parse comma-separated hidden layer sizes, e.g. "64,32" -> [64, 32].
  block
    integer :: n_hidden, comma_pos
    character(len=256) :: rest
    n_hidden = 1
    do i = 1, len_trim(hidden_spec)
      if (hidden_spec(i:i) == ',') n_hidden = n_hidden + 1
    end do
    allocate (hidden_sizes(n_hidden))
    rest = hidden_spec
    do i = 1, n_hidden
      comma_pos = index(rest, ',')
      if (comma_pos == 0) then
        read (rest, *) hidden_sizes(i)
      else
        read (rest(1:comma_pos - 1), *) hidden_sizes(i)
        rest = rest(comma_pos + 1:)
      end if
    end do
  end block

  n_layers = size(hidden_sizes) + 2  ! input + hidden(s) + output
  allocate (layer_sizes(n_layers))
  layer_sizes(1) = nfeat
  layer_sizes(2:n_layers - 1) = hidden_sizes
  layer_sizes(n_layers) = 1

  print '(A,I0,A)', 'Network: composition of ', n_layers - 1, ' functions f_1..f_L (all rational/sigmoid):'
  do i = 1, n_layers - 1
    print '(A,I0,A,I0,A,I0,A)', '  f_', i, ': R^', layer_sizes(i), ' -> R^', layer_sizes(i + 1), &
      '   f_i(a) = sigmoid(W_i a + b_i)'
  end do

  call init_network(net, layer_sizes, n_layers)

  print '(A,I0,A,F6.3)', 'Training: n_epochs=', n_epochs, '  lr=', lr
  do epoch = 1, n_epochs
    proba_train = forward(net, n_layers, Xtr, n_train)
    call backward_and_update(net, n_layers, Xtr, real(ytr, dp), n_train, lr)
    if (mod(epoch, 50) == 0 .or. epoch == n_epochs) then
      loss = -sum(real(ytr, dp)*log(proba_train + eps) + (1.0_dp - real(ytr, dp))*log(1.0_dp - proba_train + eps))/real(n_train, dp)
      print '(A,I5,A,I0,A,F10.5)', '  epoch ', epoch, '/', n_epochs, '  train BCE loss = ', loss
    end if
  end do

  call save_weights(net, n_layers, 'dnn_weights.txt')
  print '(A)', 'Full trained weights written to dnn_weights.txt'

  proba_test = forward(net, n_layers, Xte, n_test)
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
