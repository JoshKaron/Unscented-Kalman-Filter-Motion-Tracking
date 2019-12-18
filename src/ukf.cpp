#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  foundProblem = false;
  verbose2 = false;
  verbose1 = false;
  tag = "";

  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 5.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 10.0;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  n_x_ = 5;

  n_aug_ = 7;

  n_radar_ = 3;
  
  n_lidar_ = 2;

  lambda_ = 3.0;

  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for(int i = 1; i < 2 * n_aug_ + 1; i++)
  {
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }

  R_radar = MatrixXd(n_radar_, n_radar_);
  R_radar.fill(0.0);
  R_radar(0,0) = std_radr_ * std_radr_;
  R_radar(1,1) = std_radphi_ * std_radphi_;
  R_radar(2,2) = std_radrd_ * std_radrd_;

  R_lidar = MatrixXd(n_lidar_, n_lidar_);
  R_lidar.fill(0.0);
  R_lidar(0,0) = std_laspx_ * std_laspx_;
  R_lidar(1,1) = std_laspy_ * std_laspy_;

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
    if(verbose1)
    {
      std::string sensor = meas_package.sensor_type_ == MeasurementPackage::LASER ? "Lidar" : "Radar";
      std::cout << tag << " got " << sensor <<" package" << std::endl;
    }

    // first skip if not useing that meas type
    if(!use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      return;
    }

    if(!use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      return;
    }

  if(!is_initialized_)
  {
    Initalize(meas_package);
    return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_) / 1e6;

  std::cout << "delta_t: " << delta_t << std::endl;

  Prediction(delta_t);

  Update(meas_package);

  time_us_ = meas_package.timestamp_;
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
    GenerateSigmaPoints();

    AugmentedSigmaPoints();

    SigmaPointPrediction(delta_t);

    PredictMeanAndCovar();
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

    VectorXd z = meas_package.raw_measurements_;
    
    if(verbose1)
    {
      std::cout << tag << " Update Lidar" << std::endl;
    }

    if(verbose2)
    {
      std::cout << "lidar:" << std::endl;
      std::cout << z << std::endl;
    }
    

    // cross corelation
    MatrixXd Tc = MatrixXd(n_x_, n_lidar_);
    Tc.fill(0.0);

    for(int i = 0; i < 2 * n_aug_ + 1; i++)
    {
      VectorXd z_diff = Zsig_lidar.col(i) - lidar_pred;

      VectorXd x_diff = Xsig_pred_.col(i) - x_;

      x_diff(3) = restrictToPI(x_diff(3));

      Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    MatrixXd K = Tc * S_lidar.inverse();

    VectorXd z_diff = z - lidar_pred;

    x_ = x_ + K * z_diff;

    P_ = P_ - K * S_lidar * K.transpose();

    if(verbose2)
    {
      std::cout << "x_" << std::endl;
      std::cout << x_ << std::endl;
      std::cin.get();
    }

    if(verbose2)
    {
      std::cout << "P_" << std::endl;
      std::cout << P_ << std::endl;
      std::cin.get();
    }

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

    if(verbose1)
    {
      std::cout << tag << " Update Radar" << std::endl;
    }

    VectorXd z = meas_package.raw_measurements_;

    if(verbose2)
    {
      std::cout << "radar:" << std::endl;
      std::cout << z << std::endl;
    }

    // cross corelation
    MatrixXd Tc = MatrixXd(n_x_, n_radar_);
    Tc.fill(0.0);

    for(int i = 0; i < 2 * n_aug_ + 1; i++)
    {
      VectorXd z_diff = Zsig_radar.col(i) - radar_pred;

      z_diff(1) = restrictToPI(z_diff(1));

      VectorXd x_diff = Xsig_pred_.col(i) - x_;

      x_diff(3) = restrictToPI(x_diff(3));

      Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    MatrixXd K = Tc * S_radar.inverse();

    VectorXd z_diff = z - radar_pred;

    z_diff(1) = restrictToPI(z_diff(1));

    x_ = x_ + K * z_diff;
    P_ = P_ - K * S_radar * K.transpose();

    if(verbose2)
    {
      std::cout << "x_" << std::endl;
      std::cout << x_ << std::endl;
      std::cin.get();
    }

    if(verbose2)
    {
      std::cout << "P_" << std::endl;
      std::cout << P_ << std::endl;
      std::cin.get();
    }
}

void UKF::GenerateSigmaPoints()
{
  if(verbose1)
  {
    std::cout << tag << " Gen Sigs" << std::endl;
  }


  MatrixXd A = P_.llt().matrixL();

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_x_ + 1);

  Xsig_pred_.col(0) = x_;

  for(int i = 0; i < n_x_; i++)
  {
    Xsig_pred_.col(i+1)      = x_ + sqrt(lambda_ + n_x_) * A.col(i);
    Xsig_pred_.col(i+1+n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
  }

  if(verbose2)
  {
    std::cout << "x_sig_pred" << std::endl;
    std::cout << Xsig_pred_ << std::endl;
    std::cin.get();
  }
}

void UKF::AugmentedSigmaPoints()
{
  if(verbose1)
  {
    std::cout << tag << " Aug Sigs" << std::endl;
  }

  VectorXd x_aug = VectorXd(n_aug_);

  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  x_aug.head(5) = x_;
  x_aug(5) = 0.0;
  x_aug(6) = 0.0;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  MatrixXd L = P_aug.llt().matrixL();

  Xsig_aug_.col(0) = x_aug;
  for(int i = 0; i < n_aug_; i++)
  {
    Xsig_aug_.col(i+1)        = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  if(verbose2)
  {
    std::cout << "Xsig_aug" << std::endl;
    std::cout << Xsig_aug_ << std::endl;
    std::cin.get();
  }
}

void UKF::SigmaPointPrediction(double delta_t)
{
  if(verbose1)
  {
    std::cout << tag << " Pred Sigs" << std::endl;
    std::cout << "delta_t: " << delta_t << std::endl;
  }

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double px = Xsig_aug_(0,i);
    double py = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);   

    double px_p, py_p;

    if(fabs(yawd) > 0.001)
    {
      px_p = px + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
      py_p = py + v/yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else
    {
      px_p = px + v * delta_t * cos(yaw);
      py_p = py + v * delta_t * sin(yaw);
    }
     
    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;
    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p; 
  }
    
  if(verbose2)
  {
    std::cout << "Xsig_pred" << std::endl;
    std::cout << Xsig_pred_ << std::endl;
    std::cin.get();
  }
}

void UKF::PredictMeanAndCovar()
{
  if(verbose1)
  {
    std::cout << tag << " Pred MeanVar" << std::endl;
  }

  x_.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  P_.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    x_diff(3) = restrictToPI(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }

  if(verbose2)
  {
    std::cout << "x_" << std::endl;
    std::cout << x_ << std::endl;
    std::cin.get();
  }

  if(verbose2)
  {
    std::cout << "P_" << std::endl;
    std::cout << P_ << std::endl;
    std::cin.get();
  }
}

void UKF::Initalize(MeasurementPackage meas_package)
{
  if(meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    InitFromLidar(meas_package);
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    InitFromRadar(meas_package);
  }
}

void UKF::InitFromLidar(MeasurementPackage meas_package)
{
  if(verbose1)
  {
    std::cout << tag << " Init Lidar" << std::endl;
  }

  x_(0) = meas_package.raw_measurements_(0);
  x_(1) = meas_package.raw_measurements_(1);
  x_(2) = 0.0;
  x_(3) = 0.0;
  x_(4) = 0.0;

  //P_(0,0) = std_laspx_ * std_laspx_;
  //P_(1,1) = std_laspy_ * std_laspy_;
  P_(0,0) = 1.0;
  P_(1,1) = 1.0;
  P_(2,2) = 1.0;
  P_(3,3) = 1.0;
  P_(4,4) = 1.0;

  time_us_ = meas_package.timestamp_;
  is_initialized_ = true;
}

void UKF::InitFromRadar(MeasurementPackage meas_package)
{
  if(verbose1)
  {
    std::cout << tag << " Init Radar" << std::endl;
  }

  double rho = meas_package.raw_measurements_(0);
  double phi = meas_package.raw_measurements_(1);
  double rho_dot = meas_package.raw_measurements_(2);

  x_(0) = rho * std::sin(phi);
  x_(1) = rho * std::cos(phi);
  x_(2) = 0.0;
  x_(3) = 0.0;
  x_(4) = 0.0;

  P_(0,0) = 1.0;
  P_(1,1) = 1.0;
  P_(2,2) = 1.0;
  P_(3,3) = 1.0;
  P_(4,4) = 1.0;

  time_us_ = meas_package.timestamp_;
  is_initialized_ = true;
}

void UKF::Update(MeasurementPackage meas_package)
{
  if(meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    PredictLidar();
    UpdateLidar(meas_package);
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    PredictRadar();
    UpdateRadar(meas_package);
  }
}

void UKF::PredictLidar()
{
  if(verbose1)
  {
    std::cout << tag << " Pred Lidar" << std::endl;
  }

  Zsig_lidar = MatrixXd(n_lidar_, 2 * n_aug_ + 1);

  lidar_pred = VectorXd(n_lidar_);

  S_lidar = MatrixXd(n_lidar_, n_lidar_);

  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);

    Zsig_lidar(0, i) = px;
    Zsig_lidar(1, i) = py;
  }

  lidar_pred.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    lidar_pred = lidar_pred + weights_(i) * Zsig_lidar.col(i);
  }

  S_lidar.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd diff = Zsig_lidar.col(i) - lidar_pred;

    S_lidar = S_lidar + weights_(i) * diff * diff.transpose(); 
  }

  S_lidar = S_lidar + R_lidar;

  if(verbose2)
  {
    std::cout << "lidar_pred" << std::endl;
    std::cout << lidar_pred << std::endl;
    std::cin.get();
  }

  if(verbose2)
  {
    std::cout << "S_lidar" << std::endl;
    std::cout << S_lidar << std::endl;
    std::cin.get();
  }
}

void UKF::PredictRadar()
{
  if(verbose1)
  {
    std::cout <<tag << " Pred Radar" << std::endl;
  }

  Zsig_radar = MatrixXd(n_radar_, 2 * n_aug_ + 1);

  radar_pred = VectorXd(n_radar_);

  S_radar = MatrixXd(n_radar_, n_radar_);

  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    double v1 = std::cos(yaw) * v;
    double v2 = std::sin(yaw) * v;

    Zsig_radar(0, i) = std::sqrt(px * px + py * py);
    Zsig_radar(1, i) = std::atan2(py, px);
    Zsig_radar(2, i) = (px * v1 + py * v2) / Zsig_radar(0, i);
  }

  radar_pred.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    radar_pred = radar_pred + weights_(i) * Zsig_radar.col(i);
  }

  S_radar.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd diff = Zsig_radar.col(i) - radar_pred;

    diff(1) = restrictToPI(diff(1));

    S_radar = S_radar + weights_(i) * diff * diff.transpose(); 
  }

  // add measurement noise covar
  S_radar = S_radar + R_radar;

  if(verbose2)
  {
    std::cout << "radar_pred" << std::endl;
    std::cout << radar_pred << std::endl;
    std::cin.get();
  }

  if(verbose2)
  {
    std::cout << "S_radar" << std::endl;
    std::cout << S_radar << std::endl;
    std::cin.get();
  }
}

void UKF::SetName(std::string name)
{
  tag = name;
}

