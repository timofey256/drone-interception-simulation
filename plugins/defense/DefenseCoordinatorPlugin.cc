#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <iostream>
#include <map>

namespace gazebo
{
  class DefenseCoordinatorPlugin : public WorldPlugin
  {
    private: physics::WorldPtr world;
    private: event::ConnectionPtr updateConnection;

    private: ignition::math::Vector3d site{0,0,0};
    private: std::vector<std::string> interceptors { 
        "interceptor_xplus", "interceptor_xminus",
        "interceptor_yplus", "interceptor_yminus" };
    private: std::string shahedName = "shahed";

    private: std::map<std::string, ignition::math::Vector3d> homePoses;
    private: double drone_activation_distance_m = 30.0;
    private: int32_t debug_counter = 0;

    public: void Load(physics::WorldPtr _world, sdf::ElementPtr /*_sdf*/)
    {
      this->world = _world;

      // record initial positions
      for (auto &name : interceptors)
      {
        auto m = _world->ModelByName(name);
        if (m)
          homePoses[name] = m->WorldPose().Pos();
      }

      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&DefenseCoordinatorPlugin::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      auto shahed = this->world->ModelByName(shahedName);
      if (!shahed) return;

      auto pos = shahed->WorldPose().Pos(); // TODO; add noise for real-world sim

      // pick interceptor
      std::string chosen = "interceptor_xplus"; // force one for testing
      auto chosenModel = this->world->ModelByName(chosen);
      if (!chosenModel) return;

      auto home = homePoses[chosen];
      auto chosenPos = chosenModel->WorldPose().Pos();

      ignition::math::Vector3d target;

      if (chosen.find("x") != std::string::npos) {
        // lock X at home.X, match Shahed Y/Z
        target = ignition::math::Vector3d(home.X(), pos.Y(), pos.Z());
      } else {
        // lock Y at home.Y, match Shahed X/Z
        target = ignition::math::Vector3d(pos.X(), home.Y(), pos.Z());
      }

      ignition::math::Vector3d diff = target - chosenPos;

      // give it a clear non-zero velocity
      double speed = 20.0;
      ignition::math::Vector3d cmd = diff.Normalized() * speed;

      auto link = chosenModel->GetLink("body");
      if (link)
          link->SetLinearVel(cmd);

      std::cout << "[DBG] chosen=" << chosen
                << " pos=" << chosenPos
                << " target=" << target
                << " cmd=" << cmd << std::endl;
    }
  };

  GZ_REGISTER_WORLD_PLUGIN(DefenseCoordinatorPlugin)
}

