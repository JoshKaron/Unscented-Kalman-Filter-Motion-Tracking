#ifndef UKF_H
#define UKF_H

#include <iostream>
#include "Eigen/Dense"
#include "measurement_package.h"

class UKF {
 public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);

  bool foundProblem;
  bool verbose1;
  bool verbose2;
  std::string tag;
  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  Eigen::MatrixXd Xsig_aug_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  // Weights of sigma points
  Eigen::VectorXd weights_;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Sigma point spreading parameter
  double lambda_;

  int n_radar_;
  int n_lidar_;

  Eigen::VectorXd lidar_pred;

  Eigen::MatrixXd S_lidar;

  Eigen::MatrixXd R_lidar;

  Eigen::MatrixXd Zsig_lidar;

  Eigen::VectorXd radar_pred;

  // radar covariance
  Eigen::MatrixXd S_radar;

  Eigen::MatrixXd R_radar;

  // radar sigma points
  Eigen::MatrixXd Zsig_radar;

  void GenerateSigmaPoints();

  void AugmentedSigmaPoints();

  void SigmaPointPrediction(double delta_t);

  void PredictMeanAndCovar();

  void Initalize(MeasurementPackage meas_package);

  void InitFromLidar(MeasurementPackage meas_package);

  void InitFromRadar(MeasurementPackage meas_package);

  void Update(MeasurementPackage meas_package);

  void PredictRadar();

  void PredictLidar();
  
  void SetName(std::string name);

  double restrictToPI(double rad)
  {
    /*
    while(rad > M_PI || rad < -M_PI)
    {
      if (rad >  M_PI) { rad -= 2.0*M_PI; }
      if(fabs(rad - M_PI) < 0.0001) { break; }
      if (rad < -M_PI) { rad += 2.0*M_PI; }
      if(fabs(rad - M_PI) < 0.0001) { break; }
    }*/

    rad = fmod(rad + M_PI, 2.0*M_PI);
    if(rad < 0) {rad += 2.0*M_PI;}
    return rad - M_PI;
  }

  void CheckForProblem(std::string spot)
  {
    if(!foundProblem)
    {
      if(std::isnan(x_(0)))
      {
        std::cout << "Found a problem at " << spot << std::endl;
        foundProblem = true;
      }
    }
  }


};

#endif  // UKF_H