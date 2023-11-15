//
// Created by jemin on 3/1/22.
//

#ifndef RAISIM_RAISIMGYMTORCH_RAISIMGYMTORCH_ENV_ENVS_COMMON_H_
#define RAISIM_RAISIMGYMTORCH_RAISIMGYMTORCH_ENV_ENVS_COMMON_H_

#include <Eigen/Core>

using Dtype=float;
using EigenRowMajorMat=Eigen::Matrix<Dtype, -1, -1, Eigen::RowMajor>;
using EigenVec=Eigen::Matrix<Dtype, -1, 1>;
using EigenBoolVec=Eigen::Matrix<bool, -1, 1>;
using EigenDoubleVec = Eigen::Matrix<double, -1, 1>;
inline Eigen::Matrix3d skew(const Eigen::Vector3d v) {
  Eigen::Matrix3d M;
  M << 0, -v(2), v(1),
      v(2), 0, -v(0),
      -v(1), v(0), 0;

  return M;
}

inline Eigen::Vector3d rotToRotVec(const Eigen::Matrix3d& rot) {
  Eigen::Vector3d rotVec;
  Eigen::Vector3d axis;

  double angle = std::acos((std::min(std::max(rot.trace(), -1.), 1.) - 1.) / 2.);

  if (angle < 1e-12) { return Eigen::Vector3d::Zero(); }
  else {
    axis << rot(2, 1) - rot(1, 2), rot(0, 2) - rot(2, 0), rot(1, 0) - rot(0, 1);
    axis = axis / axis.norm();
    rotVec << angle * axis;
  }

  return rotVec;
}

inline Eigen::Matrix3d rotVecToRot(const Eigen::Vector3d& rotVec) {
  // Computes the vectorized exponential map for SO(3)
  Eigen::Matrix3d A = skew(rotVec);
  double theta = rotVec.norm();
  if (theta < 1e-12) {
    return Eigen::Matrix3d::Identity();
  }

  Eigen::Matrix3d R =  Eigen::Matrix3d::Identity() + (sin(theta) / theta) * A + ((1 - cos(theta)) / (theta * theta)) * A * A;

  return R;
}

/// rotation matrix & quaternion
inline Eigen::Matrix2d yawToRot2D(const double yaw) {
  Eigen::Matrix2d R;

  R << cos(yaw), -sin(yaw),
      sin(yaw), cos(yaw);

  return R;
}

inline Eigen::Vector4d rotToQuat(const Eigen::Matrix3d& rot) {
  Eigen::Vector4d q;
  double tr = rot(0, 0) + rot(1, 1) + rot(2, 2);
  if(tr > 0) {
    double S = std::sqrt(tr + 1.0) * 2;
    q(0) = S / 4.;
    q(1) = (rot(2, 1) - rot(1, 2)) / S;
    q(2) = (rot(0, 2) - rot(2, 0)) / S;
    q(3) = (rot(1, 0) - rot(0, 1)) / S;
  } else if((rot(0, 0) > rot(1, 1)) && (rot(0, 0) > rot(2, 2))) {
    double S = std::sqrt(1.0 + rot(0, 0) - rot(1, 1) - rot(2, 2)) * 2;
    q(0) = (rot(2, 1) - rot(1, 2)) / S;
    q(1) = S / 4.;
    q(2) = (rot(0, 1) + rot(1, 0)) / S;
    q(3) = (rot(0, 2) + rot(2, 0)) / S;
  } else if(rot(1, 1) > rot(2, 2)) {
    double S = std::sqrt(1.0 + rot(1, 1) - rot(0, 0) - rot(2, 2)) * 2;
    q(0) = (rot(0, 2) - rot(2, 0)) / S;
    q(1) = (rot(0, 1) + rot(1, 0)) / S;
    q(2) = S / 4.;
    q(3) = (rot(1, 2) + rot(2, 1)) / S;
  } else {
    double S = std::sqrt(1.0 + rot(2, 2) - rot(0, 0) - rot(1, 1)) * 2;
    q(0) = (rot(1, 0) - rot(0, 1)) / S;
    q(1) = (rot(0, 2) + rot(2, 0)) / S;
    q(2) = (rot(1, 2) + rot(2, 1)) / S;
    q(3) = S / 4.;
  }

  return q;
}

inline Eigen::Matrix3d quatToRot(const Eigen::VectorXd& q) {
  Eigen::Matrix3d mat;
  mat(0, 0) = 1 - 2 * (q(2)*q(2) + q(3)*q(3));
  mat(0, 1) = 2 * (q(1)*q(2) - q(0)*q(3));
  mat(0, 2) = 2 * (q(0)*q(2) + q(1)*q(3));
  mat(1,0) = 2 * (q(1)*q(2) + q(0)*q(3));
  mat(1, 1) = 1 - 2 * (q(1)*q(1) + q(3)*q(3));
  mat(1, 2) = 2 * (q(2)*q(3) - q(0)*q(1));
  mat(2, 0) = 2 * (q(1)*q(3) - q(0)*q(2));
  mat(2, 1) = 2 * (q(0)*q(1) + q(2)*q(3));
  mat(2, 2) = 1 - 2 * (q(1)*q(1) + q(2)*q(2));

  return mat;
}


/// rpy & rot, quat
/// convention: yaw pitch roll (ZYX intrinsic)
inline Eigen::Matrix3d rpyToRot(const Eigen::Vector3d& rpy) {
  Eigen::Matrix3d Rx, Ry, Rz;
  double roll = rpy(0);
  double pitch = rpy(1);
  double yaw = rpy(2);

  Rx << 1, 0, 0,
      0, cos(roll), -sin(roll),
      0, sin(roll), cos(roll);
  Ry << cos(pitch), 0, sin(pitch),
      0, 1, 0,
      -sin(pitch), 0, cos(pitch);
  Rz << cos(yaw), -sin(yaw), 0,
      sin(yaw), cos(yaw), 0,
      0, 0, 1;

  return Rz*Ry*Rx;
}

inline Eigen::Vector3d quatToRpy(const Eigen::VectorXd& q) {
  Eigen::Vector3d rpy;
  rpy(0) = std::atan2(2 * (q(2) * q(3) + q(0) * q(1)), q(0) * q(0) - q(1) * q(1) - q(2) * q(2) + q(3) * q(3));
  rpy(1) = 2 * std::atan2(std::sqrt(1 + 2 * (q(0) * q(2) - q(1) * q(3))), std::sqrt(1 - q(0) * q(2) - q(1) * q(3))) - M_PI / 2.;
  rpy(2) = std::atan2(2 * (q(1) * q(2) + q(0) * q(3)), q(0) * q(0) + q(1) * q(1) - q(2) * q(2) - q(3) * q(3));

  return rpy;
}

inline Eigen::Vector3d rotToRpy(const Eigen::Matrix3d& rot) {
  Eigen::Vector4d quat = rotToQuat(rot);
  Eigen::Vector3d rpy = quatToRpy(quat);

  return rpy;
}

inline Eigen::Vector4d rpyToQuat(const Eigen::Vector3d& rpy) {
  Eigen::Matrix3d rot = rpyToRot(rpy);
  Eigen::Vector4d quat = rotToQuat(rot);

  return quat;
}
extern int threadCount;

#endif //RAISIM_RAISIMGYMTORCH_RAISIMGYMTORCH_ENV_ENVS_COMMON_H_
