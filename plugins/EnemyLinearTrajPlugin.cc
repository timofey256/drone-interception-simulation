// EnemyLinearTrajPlugin.cc  (Gazebo Classic)
#include <cmath>
#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <optional>

using namespace gazebo;
using ignition::math::Vector3d;

static inline double deg2rad(double d) { return d * M_PI / 180.0; }

class EnemyLinearTrajPlugin : public ModelPlugin {
   public:
    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override {
        model_ = _model;
        world_ = model_->GetWorld();

        speed_ = _sdf->HasElement("speed") ? _sdf->Get<double>("speed") : 20.0;

        // Priority: aim_point > direction > (azimuth_deg + elevation_deg)
        if (_sdf->HasElement("aim_point")) {
            aimPoint_ = _sdf->Get<Vector3d>("aim_point");
        }
        if (_sdf->HasElement("direction")) {
            dirParam_ = _sdf->Get<Vector3d>("direction");
        }
        if (_sdf->HasElement("azimuth_deg"))
            azDeg_ = _sdf->Get<double>("azimuth_deg");
        if (_sdf->HasElement("elevation_deg"))
            elDeg_ = _sdf->Get<double>("elevation_deg");

        // Optional small randomization of direction (degrees, stddev) around
        // the chosen line
        if (_sdf->HasElement("randomize_dir_sigma_deg"))
            randSigmaDeg_ = _sdf->Get<double>("randomize_dir_sigma_deg");

        // Optional start delay
        if (_sdf->HasElement("start_delay_s"))
            startDelay_ = _sdf->Get<double>("start_delay_s");

        // Optional time-to-live (seconds); after TTL, stop
        if (_sdf->HasElement("ttl_s")) ttl_ = _sdf->Get<double>("ttl_s");

        // Make motion purely kinematic: disable gravity on all links
        for (auto& link : model_->GetLinks()) {
            if (link) link->SetGravityMode(false);
        }

        // Compute final direction at load
        computeDirection();

        t0_ = world_->SimTime().Double();
        updateConn_ = event::Events::ConnectWorldUpdateBegin(std::bind(
            &EnemyLinearTrajPlugin::OnUpdate, this, std::placeholders::_1));
    }

   private:
    void computeDirection() {
        Vector3d dir(1, 0, 0);  // default

        const Vector3d p0 = model_->WorldPose().Pos();

        if (aimPoint_) {
            dir = *aimPoint_ - p0;
        } else if (dirParam_) {
            dir = *dirParam_;
        } else {
            // Build from azimuth/elevation (deg)
            double az = deg2rad(azDeg_);  // 0° = +X, 90° = +Y
            double el = deg2rad(elDeg_);  // 0° = horizontal, +up
            double c = std::cos(el);
            dir.Set(c * std::cos(az), c * std::sin(az), std::sin(el));
        }

        if (dir.Length() < 1e-9) dir.Set(1, 0, 0);
        dir.Normalize();

        if (randSigmaDeg_ > 0.0) {
            // Small random tilt: rotate dir by tiny yaw/pitch sampled from
            // N(0,sigma)
            std::mt19937 rng{std::random_device{}()};
            std::normal_distribution<double> N01(0.0, 1.0);
            double s = deg2rad(randSigmaDeg_);
            double dyaw = s * N01(rng);
            double dpit = s * N01(rng);

            // Build orthonormal basis around dir
            Vector3d w = dir;
            w.Normalize();
            Vector3d up(0, 0, 1);
            Vector3d u = up.Cross(w);
            if (u.Length() < 1e-6)
                u = Vector3d(0, 1, 0);  // fallback if dir ~ up
            u.Normalize();
            Vector3d v = w.Cross(u);
            v.Normalize();
            // Apply small-angle perturbation in local yaw/pitch
            Vector3d d2 = w + u * dyaw + v * dpit;
            if (d2.Length() > 1e-9)
                dir_ = d2.Normalized();
            else
                dir_ = w;
        } else {
            dir_ = dir;
        }
    }

    void OnUpdate(const common::UpdateInfo& info) {
        const double t = world_->SimTime().Double();
        if ((t - t0_) < startDelay_) {
            model_->SetLinearVel(Vector3d::Zero);
            return;
        }
        if (ttl_ > 0.0 && (t - t0_) > (startDelay_ + ttl_)) {
            model_->SetLinearVel(Vector3d::Zero);
            return;
        }
        // Constant-velocity straight line in 3D
        model_->SetLinearVel(dir_ * speed_);
    }

    // Members
    physics::WorldPtr world_;
    physics::ModelPtr model_;
    event::ConnectionPtr updateConn_;

    double speed_{20.0};
    std::optional<Vector3d> aimPoint_;
    std::optional<Vector3d> dirParam_;
    double azDeg_{0.0};
    double elDeg_{0.0};
    double randSigmaDeg_{0.0};
    double startDelay_{0.0};
    double ttl_{-1.0};

    Vector3d dir_{1, 0, 0};
    double t0_{0.0};
};

GZ_REGISTER_MODEL_PLUGIN(EnemyLinearTrajPlugin)
