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
  VectorXd y = z - H_ * x_;

  UpdateCommon(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  const double eps = 0.000000001;
  double px, py, vx, vy;
  px = x_(0);
  py = x_(1);
  vx = x_(2);
  vy = x_(3);
  double rho = sqrt(px * px + py * py);
  double theta;
  if (abs(py) < eps && abs(px) < eps) { theta = z(1); } // assume nor theta error when both px and py are 0
  else { theta = atan2(py, px); }
  double rho_dot = (px * vx + py * vy) / std::max(eps, rho);
  VectorXd hx(3);
  hx << rho, theta, rho_dot;
  VectorXd y = z - hx;

  // make theta in y within (-PI, PI]
  while (y(1) > PI) { y(1) -= 2 * PI; }
  while (y(1) <= -PI) { y(1) += 2 * PI; }

  UpdateCommon(y);
}

void KalmanFilter::UpdateCommon(const Eigen::VectorXd &y) {
  MatrixXd PHt = P_ * H_.transpose();
  MatrixXd S = H_ * PHt + R_;
  MatrixXd K = PHt * S.inverse();

  x_ = x_ + K * y;
  P_ -= K * H_ * P_;
}
