#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
    cout << "ERROR: invalid number data points." << endl;
    return rmse;
  }

  //accumulate squared residuals
  for (int i = 0; i < estimations.size(); ++i) {
    VectorXd res = (estimations[i] - ground_truth[i]).array().square();
    rmse += res;
  }

  //calculate the root of mean
  rmse = (rmse / estimations.size()).array().sqrt();

  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {
  // only for transferring from Cartesian system x_state (4) to polar system state (3)

  MatrixXd Hj(3, 4);
  //recover state parameters
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  double norm2 = px * px + py * py;
  double norm = sqrt(norm2);
  double norm3 = norm2 * norm;

  //check division by zero
  if (norm2 < 0.0000001) {
    cout << "ERROR: divide by 0." << endl;
    Hj << 0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0;
  } else {
    //compute the Jacobian matrix
    Hj << px / norm, py / norm, 0, 0,
          -py / norm2, px / norm2, 0, 0,
          py * (vx * py - vy * px) / norm3, px * (vy * px - vx * py) / norm3, px / norm, py / norm;
  }

  return Hj;
}
