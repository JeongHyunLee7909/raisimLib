//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    anymal_ = world_->addArticulatedSystem(resourceDir_+"/raibot/urdf/raibot_simplified.urdf");
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    gc_12.setZero(12);
    gv_12.setZero(12);
    pTarget_.setZero(gcDim_); 
    vTarget_.setZero(gvDim_); 
    pTarget12_.setZero(nJoints_);
    prevpTarget12_.setZero(nJoints_);
    prevprevpTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.57, 1.0, 0.0, 0.0, 0.0, 0, 0.56, -1.12, 0, 0, 0.56, -1.12, 0, 0, 0.56, -1.12, 0, 0, 0.56, -1.12, 0;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero();
    jointPgain.tail(nJoints_).setConstant(50.0);
    jointPgain[9] = 0.0;
    jointPgain[13] = 0.0;
    jointPgain[17] = 0.0;
    jointPgain[21] = 0.0;
    jointDgain.setZero(); 
    jointDgain.tail(nJoints_).setConstant(0.2);
    jointDgain[9] = 0.0;
    jointDgain[13] = 0.0;
    jointDgain[17] = 0.0;
    jointDgain[21] = 0.0;
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 37;
    actionDim_ = 12; 
    actionMean_.setZero(actionDim_); 
    actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    command_.setZero();

    /// action scaling
    actionMean_ << 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12;
    double action_std;
    READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_FOOT"));
    footIndices_.insert(anymal_->getBodyIdx("RF_FOOT"));
    footIndices_.insert(anymal_->getBodyIdx("LH_FOOT"));
    footIndices_.insert(anymal_->getBodyIdx("RH_FOOT"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();

      Eigen::Vector3d rpy = {0., M_PI / 2., 0.};
      arrowHeadDown_ = rpyToRot(rpy);

      double xyCommandTheta = std::atan2(command_(1), command_(0));
      rpy << 0., 0., xyCommandTheta;
      Eigen::Matrix3d heading = rpyToRot(rpy);

      double omegaSign = 1.;
      omegaSign = std::copysign(omegaSign, command_(2));
      Eigen::Matrix3d omegaHeading;
      if (omegaSign == 1.) omegaHeading = Eigen::Matrix3d::Identity();
      else {
        rpy << 0., M_PI, 0.;
        omegaHeading = rpyToRot(rpy);
      }

      xyCommandArrow_ = server_->addVisualArrow("xyCommandArrow", cylinderRadius_, cylinderLength_*command_.head(2).norm(), 0.5, 0., 0., .7);      
      xyCommandArrow_->setOrientation(rotToQuat(quatToRot(gc_init_.segment(3, 4)) * heading * arrowHeadDown_));
      xyCommandArrow_->setPosition(gc_init_.head(3) + Eigen::Vector3d({0., 0., 0.3}));
      xyCommandArrow_->setCylinderSize(cylinderRadius_, cylinderLength_ * command_.head(2).norm());

      omegaCommandArrow_ = server_->addVisualArrow("omegaCommandArrow", cylinderRadius_, cylinderLength_*abs(command_(2)), 0., 0., 0.5, .7);
      omegaCommandArrow_->setOrientation(rotToQuat(quatToRot(gc_.segment(3, 4)) * omegaHeading));
      omegaCommandArrow_->setPosition(gc_.head(3) + Eigen::Vector3d({0., 0., 0.3}));
      omegaCommandArrow_->setCylinderSize(cylinderRadius_, cylinderLength_ * abs(command_(2)));

      server_->focusOn(anymal_);
    }
  }

  void init() final { }

  void reset() final {
    anymal_->setState(gc_init_, gv_init_);
    command_ << 2.0 * normDist_(gen_), 1.0 * normDist_(gen_), 1.0 * normDist_(gen_);
    if (visualizable_) {
      Eigen::Vector3d rpy;
      double xyCommandTheta = std::atan2(command_(1), command_(0));
      rpy << 0., 0., xyCommandTheta;
      Eigen::Matrix3d heading = rpyToRot(rpy);
      xyCommandArrow_->setOrientation(rotToQuat(quatToRot(gc_init_.segment(3, 4)) * heading * arrowHeadDown_));
      xyCommandArrow_->setPosition(gc_init_.head(3) + Eigen::Vector3d({0., 0., 0.3}));
      xyCommandArrow_->setCylinderSize(cylinderRadius_, cylinderLength_ * command_.head(2).norm());

      double omegaSign = 1.;
      omegaSign = std::copysign(omegaSign, command_(2));
      Eigen::Matrix3d omegaHeading;
      if (omegaSign == 1.) omegaHeading = Eigen::Matrix3d::Identity();
      else {
        rpy << 0., M_PI, 0.;
        omegaHeading = rpyToRot(rpy);
      }
      omegaCommandArrow_->setOrientation(rotToQuat(quatToRot(gc_.segment(3, 4)) * omegaHeading));
      omegaCommandArrow_->setPosition(gc_.head(3) + Eigen::Vector3d({0., 0., 0.3}));
      omegaCommandArrow_->setCylinderSize(cylinderRadius_, cylinderLength_ * abs(command_(2)));
    }
    updateObservation();
  }

  void resetTerminate() final {
    anymal_->setState(gc_init_, gv_init_);
    updateObservation();
  }

  void setCommand(const Eigen::Ref<EigenVec>& command) final {
    command_ = command.cast<double>();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;

    pTarget_.setZero();

    pTarget_[7] = pTarget12_[0];
    pTarget_[8] = pTarget12_[1];
    pTarget_[9] = pTarget12_[2];

    pTarget_[11] = pTarget12_[3];
    pTarget_[12] = pTarget12_[4];
    pTarget_[13] = pTarget12_[5];

    pTarget_[15] = pTarget12_[6];
    pTarget_[16] = pTarget12_[7];
    pTarget_[17] = pTarget12_[8];

    pTarget_[19] = pTarget12_[9];
    pTarget_[20] = pTarget12_[10];
    pTarget_[21] = pTarget12_[11];

    anymal_->setPdTarget(pTarget_, vTarget_);

    prevprevpTarget12_ = prevpTarget12_;
    prevpTarget12_ = pTarget12_;

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    updateObservation();
    
    if (visualizable_) {
      Eigen::Vector3d rpy;
      double xyCommandTheta = std::atan2(command_(1), command_(0));
      rpy << 0., 0., xyCommandTheta;
      Eigen::Matrix3d heading = rpyToRot(rpy);
      xyCommandArrow_->setOrientation(rotToQuat(quatToRot(gc_.segment(3, 4)) * heading * arrowHeadDown_));
      xyCommandArrow_->setPosition(gc_.head(3) + Eigen::Vector3d({0., 0., 0.3}));
      xyCommandArrow_->setCylinderSize(cylinderRadius_, cylinderLength_ * command_.head(2).norm());

      double omegaSign = 1.;
      omegaSign = std::copysign(omegaSign, command_(2));
      Eigen::Matrix3d omegaHeading;
      if (omegaSign == 1.) omegaHeading = Eigen::Matrix3d::Identity();
      else {
        rpy << 0., M_PI, 0.;
        omegaHeading = rpyToRot(rpy);
      }
      omegaCommandArrow_->setOrientation(rotToQuat(quatToRot(gc_.segment(3, 4)) * omegaHeading));
      omegaCommandArrow_->setPosition(gc_.head(3) + Eigen::Vector3d({0., 0., 0.3}));
      omegaCommandArrow_->setCylinderSize(cylinderRadius_, cylinderLength_ * abs(command_(2)));
    }

    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    auto body_z_dir = rot.e().row(2).transpose();
    rewards_.record("body_flat", body_z_dir(2));
    rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());
    rewards_.record("smoothness", (prevprevpTarget12_ + pTarget12_ - 2 * prevpTarget12_).squaredNorm());
    rewards_.record("command_lin", std::exp(-1.0 * (command_.head(2) - bodyLinearVel_.head(2)).squaredNorm()));
    rewards_.record("command_ang", std::exp(-1.5 * pow((command_(2) - bodyAngularVel_(2)), 2)));

    return rewards_.sum();
  }

  void updateObservation() {
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    gc_12.segment(0, 3) = gc_.tail(16).segment(0, 3);
    gc_12.segment(3, 3) = gc_.tail(16).segment(4, 3);
    gc_12.segment(6, 3) = gc_.tail(16).segment(8, 3);
    gc_12.segment(9, 3) = gc_.tail(16).segment(12, 3);

    gv_12.segment(0, 3) = gv_.tail(16).segment(0, 3);
    gv_12.segment(3, 3) = gv_.tail(16).segment(4, 3);
    gv_12.segment(6, 3) = gv_.tail(16).segment(8, 3);
    gv_12.segment(9, 3) = gv_.tail(16).segment(12, 3);

    obDouble_ << command_,
      gc_[2], /// body height
      rot.e().row(2).transpose(), /// body orientation
      gc_12, /// joint angles
      bodyLinearVel_, 
      bodyAngularVel_, /// body linear&angular velocity
      gv_12; /// joint velocity
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: anymal_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        return true;

    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() { };

 private:
  raisim::Visuals *xyCommandArrow_, *omegaCommandArrow_;
  Eigen::Vector3d command_;
  Eigen::Matrix3d arrowHeadDown_;
  double cylinderRadius_=0.2, cylinderLength_=0.1;

  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* anymal_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, prevpTarget12_, prevprevpTarget12_, vTarget_, gc_12, gv_12;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

