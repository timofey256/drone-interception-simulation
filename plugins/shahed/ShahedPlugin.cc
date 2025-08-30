#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

namespace gazebo
{
  class ShahedPlugin : public ModelPlugin
  {
    private: physics::ModelPtr model;
    private: event::ConnectionPtr updateConnection;

    // List of waypoints (trajectory)
    private: std::vector<ignition::math::Vector3d> waypoints{
      {100, 100, 10},
      {20, 50, 10},
      {10, 0, 10},
      {0, 0, 0}
    };

    private: size_t currentIdx = 0;

    // UAV speed in m/s
    private: double speed = 10.0;

    // Threshold to consider waypoint reached
    private: double waypointTolerance = 1.0;

    public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      this->model = _model;

      if (_sdf->HasElement("speed"))
        this->speed = _sdf->Get<double>("speed");

      if (_sdf->HasElement("waypoint_tolerance"))
        this->waypointTolerance = _sdf->Get<double>("waypoint_tolerance");

      // Connect to update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&ShahedPlugin::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      if (currentIdx >= waypoints.size())
        return; // trajectory complete

      // Current position
      ignition::math::Vector3d pos = this->model->WorldPose().Pos();

      // Current target waypoint
      ignition::math::Vector3d target = waypoints[currentIdx];

      // Distance to target
      double dist = pos.Distance(target);

      if (dist < waypointTolerance)
      {
        // Move to next waypoint
        currentIdx++;
        if (currentIdx >= waypoints.size())
        {
          // Stop moving when trajectory is finished
          this->model->SetLinearVel({0,0,0});
          return;
        }
        target = waypoints[currentIdx];
      }

      // Direction toward current target
      ignition::math::Vector3d dir = (target - pos).Normalized();

      // Desired velocity
      ignition::math::Vector3d vel = dir * speed;

      // Apply velocity
      this->model->SetLinearVel(vel);
    }
  };

  GZ_REGISTER_MODEL_PLUGIN(ShahedPlugin)
}

