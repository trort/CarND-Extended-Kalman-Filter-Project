#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  noise_ax_ = 9;
  noise_ay_ = 9;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) << endl
         << measurement_pack.raw_measurements_ << endl;

    // Variables to initialize
    // state vector, use measurement
    Eigen::VectorXd x_in(4);
    // state covariance matrix, make a wild guess
    Eigen::MatrixXd P_in(4, 4);
    P_in << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;
    // F & Q will be updated in prediction
    Eigen::MatrixXd F_in;
    Eigen::MatrixXd Q_in;
    // H & R determined by sensor type
    Eigen::MatrixXd H_in;
    Eigen::MatrixXd R_in;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      double ro = measurement_pack.raw_measurements_[0];
      double theta = measurement_pack.raw_measurements_[1];
      double ro_dot = measurement_pack.raw_measurements_[2];
      x_in << ro * cos(theta), ro * sin(theta), ro_dot * cos(theta), ro_dot * sin(theta);
      R_in = R_radar_;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      x_in << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
      H_in = H_laser_;
      R_in = R_laser_;
    }

    ekf_.Init(x_in, P_in, F_in, Q_in, H_in, R_in);

    // done initializing, no need to predict or update
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  double dt2 = dt * dt;
  double dt3 = dt2 * dt;
  double dt4 = dt3 * dt;

  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1;

  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt4 * noise_ax_ / 4, 0, dt3 * noise_ax_ / 2, 0,
             0, dt4 * noise_ay_ / 4, 0, dt3 * noise_ay_ / 2,
             dt3 * noise_ax_ / 2, 0, dt2 * noise_ax_, 0,
             0, dt3 * noise_ay_ / 2, 0, dt2 * noise_ay_;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
