!************************************************************
!*      Fortran implementation of the Kalman filter
!*      and Rauch smoother
!************************************************************

module lapackMod
contains

! Returns the inverse of a matrix calculated by finding the LU
! decomposition.  Depends on LAPACK.
function inv(A) result(Ainv)
  implicit none
  double precision,intent(in) :: A(:,:)
  double precision            :: Ainv(size(A,1),size(A,2))
  double precision            :: work(size(A,1))            ! work array for LAPACK
  integer         :: n,info,ipiv(size(A,1))     ! pivot indices
  
  ! Store A in Ainv to prevent it from being overwritten by LAPACK
  Ainv = A
  n = size(A,1)
  ! SGETRF computes an LU factorization of a general M-by-N matrix A
  ! using partial pivoting with row interchanges.
  call DGETRF(n,n,Ainv,n,ipiv,info)
  if (info.ne.0) stop 'Matrix is numerically singular!'
  ! SGETRI computes the inverse of a matrix using the LU factorization
  ! computed by SGETRF.
  call DGETRI(n,Ainv,n,ipiv,work,n,info)
  if (info.ne.0) stop 'Matrix inversion failed!'
end function inv
end module lapackMod


subroutine filtersmoother(lenTraj, dim_x, dim_h, Xtplus, mutilde, R, diff, mu0, Sig0,muh, Sigh)
  implicit none
  integer,intent(in)::lenTraj,dim_x,dim_h
  double precision,dimension(0:lenTraj,dim_x),intent(in)::Xtplus
  double precision,dimension(0:lenTraj,dim_x+dim_h),intent(in)::mutilde
  double precision,dimension(dim_x+dim_h,dim_h),intent(in)::R
  double precision,dimension(dim_x+dim_h,dim_x+dim_h),intent(in)::diff
  double precision,dimension(dim_h),intent(in)::mu0
  double precision,dimension(dim_h,dim_h),intent(in)::Sig0
  double precision,dimension(0:lenTraj,2*dim_h),intent(out)::muh
  double precision,dimension(0:lenTraj,2*dim_h,2*dim_h),intent(out)::Sigh
  double precision,dimension(0:lenTraj,dim_h)::muf,mus
  double precision,dimension(0:lenTraj,dim_h,dim_h)::Sigf,Sigs
  integer::i
  
  !! Forward Proba
  muf(0,:) = mu0
  Sigf(0, :, :) = Sig0
  if (dim_x == 0) then
     !! Iterate and compute possible value for h at the same point
     do i=1,lenTraj !! for i in range(1, lenTraj):
        call filter_kalman_noiseless(muf(i - 1, :), Sigf(i - 1, :, :), mutilde(i - 1,:), R, diff, dim_h,&
             muf(i, :), Sigf(i, :, :), muh(i - 1, :), Sigh(i - 1, :, :))

     end do
  else
     !! Iterate and compute possible value for h at the same point
     do i=1,lenTraj !! for i in range(1, lenTraj):
        call filter_kalman(muf(i - 1, :), Sigf(i - 1, :, :), Xtplus(i - 1,:), mutilde(i - 1,:), R,diff, dim_x, dim_h,&
             muf(i, :), Sigf(i, :, :), muh(i - 1, :), Sigh(i - 1, :, :))

     end do
  end if

  !! The last step comes only from the forward recursion
  Sigs(lenTraj, :, :) = Sigf(lenTraj, :, :)
  mus(lenTraj, :) = muf(lenTraj, :)
  !! Backward proba
  if (dim_x == 0) then
     do i=lenTraj-1,0,-1   !!   for i in range(lenTraj - 2, -1, -1):  # From T-1 to 0
        call smoothing_rauch_noiseless(muf(i, :), Sigf(i, :, :), mus(i + 1, :), Sigs(i + 1, :, :), &
             mutilde(i,:), R, diff, dim_h, mus(i, :), Sigs(i, :, :), muh(i, :), Sigh(i, :, :)) 
     end do
  else
     do i=lenTraj-1,0,-1   !!   for i in range(lenTraj - 2, -1, -1):  # From T-1 to 0
        call smoothing_rauch(muf(i, :), Sigf(i, :, :), mus(i + 1, :), Sigs(i + 1, :, :), &
             Xtplus(i,:), mutilde(i,:), R, diff, dim_x, dim_h,&
             mus(i, :), Sigs(i, :, :), muh(i, :), Sigh(i, :, :)) 
     end do
  end if

end subroutine filtersmoother

!!$    Compute the foward step using Kalman filter, predict and update step
!!$    Parameters
!!$    ----------
!!$    mutm, Sigtm: Values of the foward distribution at t-1
!!$    Xt, mutilde_tm: Values of the trajectories at T and t-1
!!$    expAh, SST: Coefficients parameters["expA"][:, dim_x:] (dim_x+dim_h, dim_h) and SS^T (dim_x+dim_h, dim_x+dim_h)
!!$    dim_x,dim_h: Dimension of visibles and hidden variables
subroutine filter_kalman(mutm, Sigtm, Xt, mutilde_tm, expAh, SST, dim_x, dim_h, marg_mu, marg_sig, mu_pair, Sig_pair)
  use lapackMod
  implicit none
  integer,intent(in)::dim_x,dim_h
  double precision,dimension(dim_h),intent(in)::mutm
  double precision,dimension(dim_h,dim_h),intent(in)::Sigtm
  double precision,dimension(dim_x),intent(in)::Xt
  double precision,dimension(dim_x+dim_h),intent(in)::mutilde_tm
  double precision,dimension(dim_x+dim_h,dim_h),intent(in)::expAh
  double precision,dimension(dim_x+dim_h,dim_x+dim_h),intent(in)::SST
  
  double precision,dimension(dim_h),intent(out)::marg_mu
  double precision,dimension(dim_h,dim_h),intent(out)::marg_sig
  double precision,dimension(2*dim_h),intent(out)::mu_pair
  double precision,dimension(2*dim_h,2*dim_h),intent(out)::Sig_pair

  double precision,dimension(dim_x+dim_h)::mutemp
  double precision,dimension(dim_x+dim_h,dim_x+dim_h)::Sigtemp
  double precision,dimension(dim_x,dim_x)::invSYY

  !! Predict step marginalization Normal Gaussian
  mutemp = mutilde_tm + matmul(expAh, mutm)
  Sigtemp = SST + matmul(expAh, matmul(Sigtm, transpose(expAh)))
  !! Update step conditionnal Normal Gaussian
  invSYY = inv(Sigtemp(:dim_x, :dim_x))
  marg_mu = mutemp(1+dim_x:) + matmul(Sigtemp(1+dim_x:, :dim_x), matmul(invSYY, Xt - mutemp(:dim_x)))
  marg_sig = Sigtemp(1+dim_x:, 1+dim_x:) -&
       matmul(Sigtemp(1+dim_x:, :dim_x),matmul(invSYY, transpose(Sigtemp(1+dim_x:, :dim_x))))
  marg_sig = 0.5*marg_sig+0.5*transpose(marg_sig) ! To enforce symmetry
  !! Pair probability distibution Z_t,Z_{t-1}
  mu_pair(:dim_h) = marg_mu
  mu_pair(1+dim_h:) = mutm
  Sig_pair(:dim_h, :dim_h) = marg_sig
  Sig_pair(1+dim_h:, :dim_h) = matmul(expAh(1+dim_x:, :) - &
       matmul(Sigtemp(1+dim_x:, :dim_x), matmul(invSYY, expAh(:dim_x, :))), Sigtm)
  Sig_pair(:dim_h, 1+dim_h:) = transpose(Sig_pair(1+dim_h:, :dim_h))
  Sig_pair(1+dim_h:, 1+dim_h:) = Sigtm
  
end subroutine filter_kalman


!!$    Compute the backward step using Kalman smoother
subroutine smoothing_rauch(muft, Sigft, muStp, SigStp, Xtplus, mutilde_t, expAh, SST&
     , dim_x, dim_h, marg_mu, marg_sig, mu_pair, Sig_pair)
  use lapackMod
  implicit none
  integer,intent(in)::dim_x,dim_h
  double precision,dimension(dim_h),intent(in)::muft,muStp
  double precision,dimension(dim_h,dim_h),intent(in)::Sigft,SigStp
  double precision,dimension(dim_x),intent(in)::Xtplus
  double precision,dimension(dim_x+dim_h),intent(in)::mutilde_t
  double precision,dimension(dim_x+dim_h,dim_h),intent(in)::expAh
  double precision,dimension(dim_x+dim_h,dim_x+dim_h),intent(in)::SST

  double precision,dimension(dim_h),intent(out)::marg_mu
  double precision,dimension(dim_h,dim_h),intent(out)::marg_sig
  double precision,dimension(2*dim_h),intent(out)::mu_pair
  double precision,dimension(2*dim_h,2*dim_h),intent(out)::Sig_pair

  double precision,dimension(dim_x+dim_h,dim_x+dim_h)::invTemp
  double precision,dimension(dim_h,dim_x+dim_h)::R
  
  invTemp = inv(SST + matmul(expAh, matmul(Sigft, transpose(expAh))))
  R = matmul(matmul(Sigft, transpose(expAh)), invTemp)

  marg_mu =  muft + matmul(R(:, :dim_x), Xtplus) - matmul(R, matmul(expAh, muft) + mutilde_t) + matmul(R(:, 1+dim_x:), muStp)
  marg_sig = matmul(R(:, 1+dim_x:), matmul(SigStp, transpose(R(:, 1+dim_x:)))) + Sigft - matmul(matmul(R, expAh), Sigft)

  !! Pair probability distibution Z_{t+1},Z_{t}
  mu_pair(:dim_h) = muStp
  mu_pair(1+dim_h:) = marg_mu

  Sig_pair(:dim_h, :dim_h) = SigStp
  Sig_pair(1+dim_h:, :dim_h) = matmul(R(:, 1+dim_x:), SigStp)
  Sig_pair(:dim_h, 1+dim_h:) = transpose(Sig_pair(1+dim_h:, :dim_h))
  Sig_pair(1+dim_h:, 1+dim_h:) = marg_sig
  
end subroutine smoothing_rauch


!!$    Compute the foward step using Kalman filter, predict and update step
!!$    Parameters
!!$    ----------
!!$    mutm, Sigtm: Values of the foward distribution at t-1
!!$    Xt, mutilde_tm: Values of the trajectories at T and t-1
!!$    expAh, SST: Coefficients parameters["expA"][:, dim_x:] (dim_x+dim_h, dim_h) and SS^T (dim_x+dim_h, dim_x+dim_h)
!!$    dim_x,dim_h: Dimension of visibles and hidden variables
subroutine filter_kalman_noiseless(mutm, Sigtm, mutilde_tm, expAhh, SST, dim_h, marg_mu, marg_sig, mu_pair, Sig_pair)
  implicit none
  integer,intent(in)::dim_h
  double precision,dimension(dim_h),intent(in)::mutm
  double precision,dimension(dim_h,dim_h),intent(in)::Sigtm
  double precision,dimension(dim_h),intent(in)::mutilde_tm
  double precision,dimension(dim_h,dim_h),intent(in)::expAhh
  double precision,dimension(dim_h,dim_h),intent(in)::SST
  
  double precision,dimension(dim_h),intent(out)::marg_mu
  double precision,dimension(dim_h,dim_h),intent(out)::marg_sig
  double precision,dimension(2*dim_h),intent(out)::mu_pair
  double precision,dimension(2*dim_h,2*dim_h),intent(out)::Sig_pair

  !! Predict step marginalization Normal Gaussian
  marg_mu = mutilde_tm + matmul(expAhh, mutm)
  marg_sig = SST + matmul(expAhh, matmul(Sigtm, transpose(expAhh)))

  !! Pair probability distibution Z_t,Z_{t-1}
  mu_pair(:dim_h) = marg_mu
  mu_pair(1+dim_h:) = mutm
  Sig_pair(:dim_h, :dim_h) = marg_sig
  Sig_pair(1+dim_h:, :dim_h) = matmul(expAhh, Sigtm)
  Sig_pair(:dim_h, 1+dim_h:) = transpose(Sig_pair(1+dim_h:, :dim_h))
  Sig_pair(1+dim_h:, 1+dim_h:) = Sigtm
  
end subroutine filter_kalman_noiseless


!!$    Compute the backward step using Kalman smoother
subroutine smoothing_rauch_noiseless(muft, Sigft, muStp, SigStp,  mutilde_t, expAhh, SST&
     ,  dim_h, marg_mu, marg_sig, mu_pair, Sig_pair)
  use lapackMod
  implicit none
  integer,intent(in)::dim_h
  double precision,dimension(dim_h),intent(in)::muft,muStp
  double precision,dimension(dim_h,dim_h),intent(in)::Sigft,SigStp
  double precision,dimension(dim_h),intent(in)::mutilde_t
  double precision,dimension(dim_h,dim_h),intent(in)::expAhh
  double precision,dimension(dim_h,dim_h),intent(in)::SST

  double precision,dimension(dim_h),intent(out)::marg_mu
  double precision,dimension(dim_h,dim_h),intent(out)::marg_sig
  double precision,dimension(2*dim_h),intent(out)::mu_pair
  double precision,dimension(2*dim_h,2*dim_h),intent(out)::Sig_pair

  double precision,dimension(dim_h,dim_h)::invTemp
  double precision,dimension(dim_h,dim_h)::R
  
  invTemp = inv(SST + matmul(expAhh, matmul(Sigft, transpose(expAhh))))
  R = matmul(matmul(Sigft, transpose(expAhh)), invTemp)

  marg_mu =  muft - matmul(R, matmul(expAhh, muft) + mutilde_t) + matmul(R, muStp)
  marg_sig = matmul(R, matmul(SigStp, transpose(R))) + Sigft - matmul(matmul(R, expAhh), Sigft)

  !! Pair probability distibution Z_{t+1},Z_{t}
  mu_pair(:dim_h) = muStp
  mu_pair(1+dim_h:) = marg_mu

  Sig_pair(:dim_h, :dim_h) = SigStp
  Sig_pair(1+dim_h:, :dim_h) = matmul(R, SigStp)
  Sig_pair(:dim_h, 1+dim_h:) = transpose(Sig_pair(1+dim_h:, :dim_h))
  Sig_pair(1+dim_h:, 1+dim_h:) = marg_sig
  
end subroutine smoothing_rauch_noiseless



