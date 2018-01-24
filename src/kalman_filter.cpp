#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;

  MatrixXd I_in(4, 4);
  I_in << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1;
  I_ = I_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state
  */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  MatrixXd Ht = H_.transpose();
  VectorXd y = z - H_ * x_;

  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  x_ = x_ + K * y;
  P_ = (I_ - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  MatrixXd Ht = H_.transpose();
  double px, py, vx, vy;
  px = x_(0);
  py = x_(1);
  vx = x_(2);
  vy = x_(3);
  double norm = sqrt(px * px + py * py);
  VectorXd hx(3);
  hx << norm, atan2(py, px), (px * vx + py * vy) / norm;
  VectorXd y = z - hx;

  // make theta in y within (-PI, PI]
  while (y(1) > PI) {y(1) -= 2 * PI;}
  while (y(1) <= -PI) {y(1) += 2 * PI;}

  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  x_ = x_ + K * y;
  P_ = (I_ - K * H_) * P_;
}
