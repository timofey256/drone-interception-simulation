#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Pose3.hh>

using namespace gazebo;

class EnemyLinearTrajPlugin : public ModelPlugin {
public:
  void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override {
    model_ = _model;
    world_ = model_->GetWorld();

    speed_ = _sdf->HasElement("speed") ? _sdf->Get<double>("speed") : 20.0;
    dir_ = _sdf->HasElement("direction")
      ? _sdf->Get<ignition::math::Vector3d>("direction")
      : ignition::math::Vector3d(-1, 0, 0);
    if (dir_.Length() < 1e-6) dir_.Set(-1,0,0);
    dir_.Normalize();

    // Optional: constrain Z to initial altitude
    holdZ_ = _sdf->HasElement("hold_z") ? _sdf->Get<double>("hold_z")
                                        : model_->WorldPose().Pos().Z();

    updateConn_ = event::Events::ConnectWorldUpdateBegin(
      std::bind(&EnemyLinearTrajPlugin::OnUpdate, this, std::placeholders::_1));
  }

private:
  void OnUpdate(const common::UpdateInfo& info) {
    // Constant velocity along dir_, maintain Z
    ignition::math::Vector3d v = dir_ * speed_;
    auto pose = model_->WorldPose();
    // Hold altitude (simple)
    double z = pose.Pos().Z();
    double vz = (holdZ_ - z) * 2.0; // weak vertical correction
    model_->SetLinearVel(ignition::math::Vector3d(v.X(), v.Y(), vz));
  }

  physics::WorldPtr world_;
  physics::ModelPtr model_;
  event::ConnectionPtr updateConn_;
  ignition::math::Vector3d dir_;
  double speed_{20.0};
  double holdZ_{0.0};
};

GZ_REGISTER_MODEL_PLUGIN(EnemyLinearTrajPlugin)
