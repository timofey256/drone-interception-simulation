#include <cmath>
#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <random>

using ignition::math::Pose3d;
using ignition::math::Vector3d;
using namespace gazebo;

namespace {
double wrapAngle(double a) {
    // Wrap to (-pi, pi]
    while (a <= -M_PI) a += 2 * M_PI;
    while (a > M_PI) a -= 2 * M_PI;
    return a;
}
double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

// Solve earliest t >= 0 for line p0 + v*t intersecting sphere |x - C| = R
// Returns (hit, t_hit, point)
std::tuple<bool, double, Vector3d> lineSphereIntersect(const Vector3d& p0,
                                                       const Vector3d& v,
                                                       const Vector3d& C,
                                                       double R) {
    Vector3d m = p0 - C;
    double b = m.Dot(v);
    double c = m.Dot(m) - R * R;
    double vv = v.Dot(v);
    if (vv < 1e-12) return {false, 0.0, Vector3d::Zero};
    double disc = b * b - vv * c;
    if (disc < 0.0) return {false, 0.0, Vector3d::Zero};
    double sqrtDisc = std::sqrt(disc);
    // two roots: (-b - sqrtDisc)/vv and (-b + sqrtDisc)/vv
    double t1 = (-b - sqrtDisc) / vv;
    double t2 = (-b + sqrtDisc) / vv;
    double t_hit = 1e30;
    if (t1 >= 0.0)
        t_hit = t1;
    else if (t2 >= 0.0)
        t_hit = t2;
    else
        return {false, 0.0, Vector3d::Zero};
    Vector3d p = p0 + v * t_hit;
    return {true, t_hit, p};
}

// Add Gaussian noise in the tangent plane at point P on sphere (center C,
// radius R). We sample small 2D noise, displace along tangent basis, then
// renormalize to radius.
Vector3d noisyPointOnSphere(const Vector3d& P, const Vector3d& C, double R,
                            double sigma, std::mt19937& rng) {
    Vector3d r = P - C;
    Vector3d n = r;
    n.Normalize();
    // Build orthonormal basis (u,v) spanning tangent plane
    Vector3d tmp =
        std::fabs(n.Z()) < 0.9 ? Vector3d(0, 0, 1) : Vector3d(0, 1, 0);
    Vector3d u = n.Cross(tmp);
    if (u.Length() < 1e-9) u = Vector3d(1, 0, 0);
    u.Normalize();
    Vector3d v = n.Cross(u);
    v.Normalize();

    std::normal_distribution<double> N01(0.0, 1.0);
    double dx = sigma * N01(rng);
    double dy = sigma * N01(rng);
    Vector3d P2 = P + u * dx + v * dy;
    // Project back to the sphere
    Vector3d d = P2 - C;
    if (d.Length() < 1e-9) d = n;
    d.Normalize();
    return C + d * R;
}
}  // namespace

class DefenderPNPlugin : public ModelPlugin {
   public:
    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override {
        model_ = _model;
        world_ = model_->GetWorld();

        // ---- Parameters ----
        if (_sdf->HasElement("enemy_name"))
            enemyName_ = _sdf->Get<std::string>("enemy_name");
        if (_sdf->HasElement("center")) center_ = _sdf->Get<Vector3d>("center");
        if (_sdf->HasElement("radius")) radius_ = _sdf->Get<double>("radius");
        if (_sdf->HasElement("hold_alt"))
            holdAlt_ = _sdf->Get<double>("hold_alt");

        if (_sdf->HasElement("pn_N")) pnN_ = _sdf->Get<double>("pn_N");  // 3..5
        if (_sdf->HasElement("engage_radius"))
            engageRadius_ = _sdf->Get<double>("engage_radius");
        if (_sdf->HasElement("guard_noise_sigma"))
            guardNoiseSigma_ = _sdf->Get<double>("guard_noise_sigma");
        if (_sdf->HasElement("bearing_noise_sigma_deg"))
            bearingNoiseSigmaDeg_ =
                _sdf->Get<double>("bearing_noise_sigma_deg");

        if (_sdf->HasElement("v_transit"))
            vTransit_ = _sdf->Get<double>("v_transit");
        if (_sdf->HasElement("v_engage"))
            vEngage_ = _sdf->Get<double>("v_engage");
        if (_sdf->HasElement("yaw_rate_max"))
            yawRateMax_ = _sdf->Get<double>("yaw_rate_max");

        if (_sdf->HasElement("hit_distance"))
            hitDist_ = _sdf->Get<double>("hit_distance");
        if (_sdf->HasElement("timeout_s"))
            timeoutS_ = _sdf->Get<double>("timeout_s");

        rng_.seed(std::random_device{}());

        updateConn_ = event::Events::ConnectWorldUpdateBegin(std::bind(
            &DefenderPNPlugin::OnUpdate, this, std::placeholders::_1));
    }

   private:
    enum class Mode { PLAN, TRANSIT, HOLD, ENGAGE, DONE };
    void OnUpdate(const common::UpdateInfo& info) {
        if (!enemy_) {
            enemy_ = world_->ModelByName(enemyName_);
            if (!enemy_) return;  // wait until enemy is spawned
            tStart_ = world_->SimTime().Double();
        }

        const double simTime = world_->SimTime().Double();
        const double dt = (lastTime_ < 0.0) ? 0.0 : (simTime - lastTime_);
        lastTime_ = simTime;

        auto myPose = model_->WorldPose();
        auto enemyPose = enemy_->WorldPose();
        Vector3d pe = enemyPose.Pos();
        Vector3d pd = myPose.Pos();

        // success / timeout checks
        double dist = pe.Distance(pd);
        if (dist <= hitDist_ && mode_ != Mode::DONE) {
            mode_ = Mode::DONE;
            model_->SetLinearVel(Vector3d::Zero);
            gzdbg << "[DefenderPN] HIT at t=" << (simTime - tStart_)
                  << " s, dist=" << dist << "\n";
            return;
        }
        if (simTime - tStart_ > timeoutS_ && mode_ != Mode::DONE) {
            mode_ = Mode::DONE;
            model_->SetLinearVel(Vector3d::Zero);
            gzdbg << "[DefenderPN] TIMEOUT\n";
            return;
        }

        switch (mode_) {
            case Mode::PLAN:
                planGuardPoint();
                mode_ = Mode::TRANSIT;
                break;
            case Mode::TRANSIT:
                doTransit(dt);
                break;
            case Mode::HOLD:
                doHold(dt);
                break;
            case Mode::ENGAGE:
                doEngage(dt);
                break;
            case Mode::DONE:
            default:
                break;
        }
    }

    void planGuardPoint() {
        // Predict earliest intersection of enemy path with sphere
        auto ePose = enemy_->WorldPose();
        Vector3d p0 = ePose.Pos();
        // Estimate enemy velocity from previous frames if available; fallback
        // to SetLinearVel not accessible -> use model API:
        Vector3d v = enemy_->WorldLinearVel();

        if (v.Length() < 1e-3) v = Vector3d(-20, 0, 0);  // fallback

        auto [hit, tHit, pHit] = lineSphereIntersect(p0, v, center_, radius_);
        if (!hit) {
            // Fallback: aim to closest point on sphere along enemy direction
            Vector3d r = (p0 - center_);
            if (r.Length() < 1e-6) r = Vector3d(1, 0, 0);
            r.Normalize();
            pHit = center_ + r * radius_;
        }

        guardPoint_ =
            noisyPointOnSphere(pHit, center_, radius_, guardNoiseSigma_, rng_);
        // Set Z if requested
        if (std::isfinite(holdAlt_)) guardPoint_.Z() = holdAlt_;
    }

    void doTransit(double dt) {
        auto pose = model_->WorldPose();
        Vector3d p = pose.Pos();
        Vector3d e = guardPoint_ - p;
        double d = e.Length();

        if (d < 2.0) {
            mode_ = Mode::HOLD;
            return;
        }

        // Velocity towards guardPoint
        Vector3d dir = e;
        dir.Z() = 0.0;  // planar transit
        double climb = 0.0;
        if (std::isfinite(holdAlt_))
            climb = (holdAlt_ - p.Z()) * 1.0;  // mild Z control
        double L = dir.Length();
        if (L > 1e-6) dir /= L;
        Vector3d vcmd = dir * vTransit_;
        vcmd.Z() = climb;

        model_->SetLinearVel(vcmd);
        // Face direction of travel (optional)
        faceHeading(dir);
        // Engage if enemy close enough
        double distToEnemy =
            model_->WorldPose().Pos().Distance(enemy_->WorldPose().Pos());
        if (distToEnemy < engageRadius_ * 0.9) {  // enter with hysteresis
            initPNState();
            mode_ = Mode::ENGAGE;
        }
    }

    void doHold(double /*dt*/) {
        // Loiter at guard point, face enemy
        auto pose = model_->WorldPose();
        Vector3d p = pose.Pos();
        Vector3d e = guardPoint_ - p;
        Vector3d vcmd(0, 0, 0);
        // Position hold P controller
        vcmd.X() = clamp(e.X() * 0.8, -vTransit_, vTransit_);
        vcmd.Y() = clamp(e.Y() * 0.8, -vTransit_, vTransit_);
        if (std::isfinite(holdAlt_))
            vcmd.Z() = clamp((holdAlt_ - p.Z()) * 0.8, -3.0, 3.0);

        model_->SetLinearVel(vcmd);
        // Point towards enemy
        Vector3d rel = enemy_->WorldPose().Pos() - p;
        rel.Z() = 0.0;
        if (rel.Length() > 1e-6) faceHeading(rel.Normalized());

        double distToEnemy = p.Distance(enemy_->WorldPose().Pos());
        if (distToEnemy < engageRadius_) {
            initPNState();
            mode_ = Mode::ENGAGE;
        }
    }

    void initPNState() {
        lastBearing_ = std::nullopt;
        dBearingFilt_ = 0.0;
    }

    void doEngage(double dt) {
        if (dt <= 0.0) return;
        auto myPose = model_->WorldPose();
        Vector3d p = myPose.Pos();
        Vector3d pe = enemy_->WorldPose().Pos();

        // Bearing in the world XY plane, measured with noise
        Vector3d r = pe - p;
        r.Z() = 0.0;
        double bearing = std::atan2(r.Y(), r.X());  // world-frame LOS azimuth
        // Add measurement noise
        std::normal_distribution<double> N01(0.0, 1.0);
        double sig = bearingNoiseSigmaDeg_ * M_PI / 180.0;
        bearing += sig * N01(rng_);
        bearing = wrapAngle(bearing);

        // Estimate bearing rate (filtered)
        if (!lastBearing_) lastBearing_ = bearing;
        double draw =
            wrapAngle(bearing - *lastBearing_) / dt;  // raw derivative
        lastBearing_ = bearing;
        // 1st order IIR smoothing
        double alpha = 0.3;
        dBearingFilt_ = (1.0 - alpha) * dBearingFilt_ + alpha * draw;

        // PN yaw-rate command
        double yawRateCmd =
            clamp(pnN_ * dBearingFilt_, -yawRateMax_, yawRateMax_);

        // Command forward velocity and yaw-rate; simple Z hold
        Vector3d dir(std::cos(myYaw_), std::sin(myYaw_), 0.0);
        if (!std::isfinite(myYaw_)) {
            myYaw_ = myPose.Rot().Yaw();
            dir = Vector3d(std::cos(myYaw_), std::sin(myYaw_), 0.0);
        }

        // Update heading by integrating yawRateCmd (internal kinematic heading)
        myYaw_ = wrapAngle(myYaw_ + yawRateCmd * dt);

        Vector3d vcmd =
            Vector3d(std::cos(myYaw_), std::sin(myYaw_), 0.0) * vEngage_;
        if (std::isfinite(holdAlt_))
            vcmd.Z() = clamp((holdAlt_ - p.Z()) * 1.0, -3.0, 3.0);
        model_->SetLinearVel(vcmd);

        // Optional: also apply body yaw to match (visual)
        ignition::math::Quaterniond q(0, 0, myYaw_);
        model_->SetWorldPose(Pose3d(p, q));
    }

    void faceHeading(const Vector3d& dir) {
        if (dir.Length() < 1e-6) return;
        double desired = std::atan2(dir.Y(), dir.X());
        auto pose = model_->WorldPose();
        double yaw = pose.Rot().Yaw();
        // Directly set yaw (for simplicity). For dynamics, use torque/vel
        // controllers.
        model_->SetWorldPose(
            Pose3d(pose.Pos(), ignition::math::Quaterniond(0, 0, desired)));
        myYaw_ = desired;  // keep internal heading consistent
    }

    // --- Members ---
    physics::WorldPtr world_;
    physics::ModelPtr model_;
    physics::ModelPtr enemy_;
    event::ConnectionPtr updateConn_;

    std::string enemyName_{"enemy"};
    Vector3d center_{0, 0, 0};
    double radius_{300.0};
    double holdAlt_{std::numeric_limits<double>::quiet_NaN()};

    double pnN_{4.0};
    double engageRadius_{120.0};
    double guardNoiseSigma_{5.0};  // meters along tangent
    double bearingNoiseSigmaDeg_{1.5};

    double vTransit_{25.0};
    double vEngage_{35.0};
    double yawRateMax_{1.2};  // rad/s
    double hitDist_{5.0};
    double timeoutS_{120.0};

    Vector3d guardPoint_{0, 0, 0};
    Mode mode_{Mode::PLAN};

    double tStart_{0.0};
    double lastTime_{-1.0};
    double myYaw_{std::numeric_limits<double>::quiet_NaN()};
    std::optional<double> lastBearing_;
    double dBearingFilt_{0.0};

    std::mt19937 rng_;
};

GZ_REGISTER_MODEL_PLUGIN(DefenderPNPlugin)
