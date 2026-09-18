#include <viam/trajex/jacobian/jacobian.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace viam::trajex::jacobian {

namespace {

using vec3 = std::array<double, 3>;

double dot(const vec3& a, const vec3& b) {
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
}
vec3 cross(const vec3& a, const vec3& b) {
    return {(a[1] * b[2]) - (a[2] * b[1]), (a[2] * b[0]) - (a[0] * b[2]), (a[0] * b[1]) - (a[1] * b[0])};
}
double norm(const vec3& a) {
    return std::sqrt(dot(a, a));
}
vec3 normalized(const vec3& a) {
    const double n = norm(a);
    return {a[0] / n, a[1] / n, a[2] / n};
}

// Row-major 4x4 homogeneous transform on the stack. The FK walk runs once or
// twice per integration step, so its arithmetic stays on plain arrays; xtensor
// appears only at the public API boundary.
using mat4 = std::array<double, 16>;

double& at(mat4& m, std::size_t i, std::size_t j) {
    return m[(i * 4) + j];
}
double at(const mat4& m, std::size_t i, std::size_t j) {
    return m[(i * 4) + j];
}

mat4 identity4() {
    mat4 m{};
    at(m, 0, 0) = 1.0;
    at(m, 1, 1) = 1.0;
    at(m, 2, 2) = 1.0;
    at(m, 3, 3) = 1.0;
    return m;
}

// Multiply two 4x4 homogeneous transforms. Hand-rolled because xtensor has no
// matmul without xtensor-blas, which the core library avoids.
mat4 matmul4(const mat4& a, const mat4& b) {
    mat4 c{};
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < 4; ++k) {
                s += at(a, i, k) * at(b, k, j);
            }
            at(c, i, j) = s;
        }
    }
    return c;
}

// 4x4 rotation about a unit axis by theta radians (Rodrigues).
mat4 axis_rotation4(const vec3& axis, double theta) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double t = 1.0 - c;
    const double x = axis[0];
    const double y = axis[1];
    const double z = axis[2];

    mat4 r = identity4();
    at(r, 0, 0) = (t * x * x) + c;
    at(r, 0, 1) = (t * x * y) - (s * z);
    at(r, 0, 2) = (t * x * z) + (s * y);
    at(r, 1, 0) = (t * x * y) + (s * z);
    at(r, 1, 1) = (t * y * y) + c;
    at(r, 1, 2) = (t * y * z) - (s * x);
    at(r, 2, 0) = (t * x * z) - (s * y);
    at(r, 2, 1) = (t * y * z) + (s * x);
    at(r, 2, 2) = (t * z * z) + c;
    return r;
}

// 4x4 transform for a URDF link: rotation Rz(yaw) * Ry(pitch) * Rx(roll) with
// translation xyz.
mat4 link_transform(const vec3& xyz, const vec3& rpy) {
    const mat4 rx = axis_rotation4({1.0, 0.0, 0.0}, rpy[0]);
    const mat4 ry = axis_rotation4({0.0, 1.0, 0.0}, rpy[1]);
    const mat4 rz = axis_rotation4({0.0, 0.0, 1.0}, rpy[2]);
    mat4 out = matmul4(matmul4(rz, ry), rx);
    at(out, 0, 3) = xyz[0];
    at(out, 1, 3) = xyz[1];
    at(out, 2, 3) = xyz[2];
    return out;
}

// Rotate a 3-vector by the rotation part of a 4x4 transform.
vec3 rotate(const mat4& transform, const vec3& v) {
    return {
        (at(transform, 0, 0) * v[0]) + (at(transform, 0, 1) * v[1]) + (at(transform, 0, 2) * v[2]),
        (at(transform, 1, 0) * v[0]) + (at(transform, 1, 1) * v[1]) + (at(transform, 1, 2) * v[2]),
        (at(transform, 2, 0) * v[0]) + (at(transform, 2, 1) * v[1]) + (at(transform, 2, 2) * v[2]),
    };
}

vec3 translation(const mat4& transform) {
    return {at(transform, 0, 3), at(transform, 1, 3), at(transform, 2, 3)};
}

// 4x4 transform for a joint origin: rotation by a unit quaternion w, x, y, z (Hamilton
// convention), then translation.
mat4 origin_transform(const vec3& translation, const std::array<double, 4>& unit_quaternion) {
    const auto [w, x, y, z] = unit_quaternion;
    mat4 out = identity4();
    at(out, 0, 0) = 1.0 - (2.0 * ((y * y) + (z * z)));
    at(out, 0, 1) = 2.0 * ((x * y) - (w * z));
    at(out, 0, 2) = 2.0 * ((x * z) + (w * y));
    at(out, 1, 0) = 2.0 * ((x * y) + (w * z));
    at(out, 1, 1) = 1.0 - (2.0 * ((x * x) + (z * z)));
    at(out, 1, 2) = 2.0 * ((y * z) - (w * x));
    at(out, 2, 0) = 2.0 * ((x * z) - (w * y));
    at(out, 2, 1) = 2.0 * ((y * z) + (w * x));
    at(out, 2, 2) = 1.0 - (2.0 * ((x * x) + (y * y)));
    at(out, 0, 3) = translation[0];
    at(out, 1, 3) = translation[1];
    at(out, 2, 3) = translation[2];
    return out;
}

// 4x4 translation by d along a unit axis.
mat4 axis_translation4(const vec3& axis, double d) {
    mat4 t = identity4();
    at(t, 0, 3) = axis[0] * d;
    at(t, 1, 3) = axis[1] * d;
    at(t, 2, 3) = axis[2] * d;
    return t;
}

bool all_finite(const auto& values) {
    return std::ranges::all_of(values, [](double v) { return std::isfinite(v); });
}

// The linear-velocity Jacobian column of a moving joint whose world-frame axis and origin were
// captured by the forward-kinematics walk. A translating joint moves the end effector along its
// axis; a rotating joint moves it tangentially about its axis.
vec3 linear_column(const auto& joint, const vec3& p_e) {
    if (joint.translates) {
        return joint.axis;
    }
    return cross(joint.axis, {p_e[0] - joint.position[0], p_e[1] - joint.position[1], p_e[2] - joint.position[2]});
}

// Column-9 encodings of the model-table tensor, matching viam::sdk::ModelTable::JointType.
enum class model_table_joint_type : std::uint8_t {
    k_revolute = 0,
    k_continuous = 1,
    k_prismatic = 2,
    k_fixed = 3,
};

}  // namespace

// The world-frame axis and origin of each moving joint, plus the end-effector position, captured
// while evaluating the forward kinematics. With these in hand, a rotating joint's Jacobian column
// is J_v_i = axis x (p_e - position) (linear) stacked on J_w_i = axis (angular), and a
// translating joint's is J_v_i = axis stacked on zero.
//
// The joints are a std::vector of small structs, not (N, 3) xtensor arrays: they are filled one
// joint at a time during the walk and read back one joint at a time by the per-joint column
// assembly, so a joint is the natural unit. A vectorized xtensor assembly is deferred to the
// linear-algebra cleanup.
struct kinematic_chain::chain_state_ {
    struct moving_joint {
        vec3 axis;
        vec3 position;
        bool translates;
    };
    std::vector<moving_joint> joints;
    vec3 p_e;
};

// Evaluate the forward kinematics joint by joint (base to end-effector), accumulating the running
// transform. For each moving joint, capture its world-frame axis and origin before applying its
// motion. Neither depends on the joint's own motion, since a joint neither rotates nor translates
// its own axis. The q-independent terms (origin transform, unit axis) come precomputed from
// construction.
kinematic_chain::chain_state_ kinematic_chain::compute_chain_state_(const xvector<>& q) const {
    if (q.size() != actuated_count_) {
        throw std::invalid_argument("viam::trajex::jacobian: q size mismatch: expected " + std::to_string(actuated_count_) +
                                    " (actuated joints), got " + std::to_string(q.size()));
    }

    chain_state_ state;
    state.joints.reserve(actuated_count_);

    mat4 running_transform = identity4();
    std::size_t qi = 0;
    for (const joint_& joint : joints_) {
        running_transform = matmul4(running_transform, joint.origin_tf);
        if (joint.kind == joint_kind_::k_fixed) {
            continue;
        }

        const bool translates = joint.kind == joint_kind_::k_translational;
        state.joints.push_back({rotate(running_transform, joint.unit_axis), translation(running_transform), translates});
        const mat4 motion = translates ? axis_translation4(joint.unit_axis, q(qi)) : axis_rotation4(joint.unit_axis, q(qi));
        running_transform = matmul4(running_transform, motion);
        ++qi;
    }
    state.p_e = translation(running_transform);
    return state;
}

kinematic_chain::kinematic_chain(std::vector<joint_> joints) : joints_(std::move(joints)) {
    actuated_count_ =
        static_cast<std::size_t>(std::ranges::count_if(joints_, [](const joint_& joint) { return joint.kind != joint_kind_::k_fixed; }));
}

kinematic_chain kinematic_chain::from(const xmatrix<>& tensor) {
    if (tensor.shape()[1] != 10) {
        throw std::invalid_argument("viam::trajex::jacobian: expected model-table shape (n, 10), got (n, " +
                                    std::to_string(tensor.shape()[1]) + ")");
    }
    if (tensor.shape()[0] == 0) {
        throw std::invalid_argument("viam::trajex::jacobian: empty model table");
    }

    std::vector<joint_> joints;
    joints.reserve(tensor.shape()[0]);
    for (std::size_t i = 0; i < tensor.shape()[0]; ++i) {
        const double type_value = tensor(i, 9);
        const int type_int = static_cast<int>(type_value);
        if (static_cast<double>(type_int) != type_value) {
            throw std::invalid_argument("viam::trajex::jacobian: row " + std::to_string(i) + " joint type value " +
                                        std::to_string(type_value) + " is not an integer");
        }

        const vec3 xyz = {tensor(i, 0), tensor(i, 1), tensor(i, 2)};
        const vec3 rpy = {tensor(i, 3), tensor(i, 4), tensor(i, 5)};
        const vec3 axis = {tensor(i, 6), tensor(i, 7), tensor(i, 8)};
        joint_ joint{.kind = joint_kind_::k_fixed, .origin_tf = link_transform(xyz, rpy)};

        switch (static_cast<model_table_joint_type>(type_int)) {
            // A continuous joint is a revolute joint without position limits. The model table
            // carries no limits for any joint type and the Jacobian column depends only on the
            // axis and the current angle, so the two are indistinguishable to us and share a
            // case. The tensor path refuses prismatic rows, since it is kept only until its
            // callers move to kinematic_chain::builder.
            case model_table_joint_type::k_revolute:
            case model_table_joint_type::k_continuous:
                if (dot(axis, axis) == 0.0) {
                    throw std::invalid_argument("viam::trajex::jacobian: row " + std::to_string(i) +
                                                " is a revolute joint with zero-magnitude axis");
                }
                joint.kind = joint_kind_::k_rotational;
                joint.unit_axis = normalized(axis);
                break;
            case model_table_joint_type::k_fixed:
                break;
            case model_table_joint_type::k_prismatic:
                throw std::invalid_argument("viam::trajex::jacobian: row " + std::to_string(i) +
                                            " has unsupported joint type (only revolute, continuous and fixed are supported)");
            default:
                throw std::invalid_argument("viam::trajex::jacobian: row " + std::to_string(i) +
                                            " joint type does not match any joint_type value");
        }
        joints.push_back(joint);
    }
    return kinematic_chain(std::move(joints));
}

xmatrix<> kinematic_chain::jacobian(const xvector<>& q) const {
    const chain_state_ state = compute_chain_state_(q);

    xmatrix<> J = xt::zeros<double>({std::size_t{6}, actuated_count_});
    for (std::size_t i = 0; i < actuated_count_; ++i) {
        const auto& joint = state.joints[i];
        const vec3 jv = linear_column(joint, state.p_e);
        J(0, i) = jv[0];
        J(1, i) = jv[1];
        J(2, i) = jv[2];
        if (!joint.translates) {
            J(3, i) = joint.axis[0];
            J(4, i) = joint.axis[1];
            J(5, i) = joint.axis[2];
        }
    }
    return J;
}

xmatrix<> kinematic_chain::linear_jacobian(const xvector<>& q) const {
    const chain_state_ state = compute_chain_state_(q);

    xmatrix<> J = xt::zeros<double>({std::size_t{3}, actuated_count_});
    for (std::size_t i = 0; i < actuated_count_; ++i) {
        const vec3 jv = linear_column(state.joints[i], state.p_e);
        J(0, i) = jv[0];
        J(1, i) = jv[1];
        J(2, i) = jv[2];
    }
    return J;
}

std::array<double, 3> kinematic_chain::end_position(const xvector<>& q) const {
    return compute_chain_state_(q).p_e;
}

kinematic_chain::linear_velocity_gain kinematic_chain::linear_velocity_gain_at(const xvector<>& q,
                                                                               const xvector<>& q_prime,
                                                                               const xvector<>& q_double_prime) const {
    if (q_prime.size() != actuated_count_ || q_double_prime.size() != actuated_count_) {
        throw std::invalid_argument("viam::trajex::jacobian: q_prime/q_double_prime size mismatch: expected " +
                                    std::to_string(actuated_count_) + " (actuated joints)");
    }

    const chain_state_ state = compute_chain_state_(q);

    vec3 v{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < actuated_count_; ++i) {
        const vec3 col = linear_column(state.joints[i], state.p_e);
        v = {v[0] + (col[0] * q_prime(i)), v[1] + (col[1] * q_prime(i)), v[2] + (col[2] * q_prime(i))};
    }

    // Differentiate v = sum(col_i * q'_i) along the path, walking base to tip. Everything
    // downstream of joint i moves with the motion of the joints before it, so we carry that motion
    // per unit of path speed: omega is the angular velocity the rotating joints so far impart, and
    // a point p moves at omega x p - c + t, where c sums (w_j q'_j) x p_j over those rotating
    // joints and t sums w_j q'_j over the translating ones. A joint's axis then turns at omega x w,
    // and a rotating joint's lever arm r = p_e - p changes at v minus the velocity of p.
    vec3 dv{0.0, 0.0, 0.0};
    vec3 omega{0.0, 0.0, 0.0};
    vec3 c{0.0, 0.0, 0.0};
    vec3 t{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < actuated_count_; ++i) {
        const auto& joint = state.joints[i];
        const vec3& w = joint.axis;
        const vec3& p = joint.position;
        const vec3 dw = cross(omega, w);

        vec3 col = w;
        vec3 d_col = dw;
        if (!joint.translates) {
            const vec3 r = {state.p_e[0] - p[0], state.p_e[1] - p[1], state.p_e[2] - p[2]};
            col = cross(w, r);
            const vec3 cp = cross(omega, p);
            const vec3 v_point = {cp[0] - c[0] + t[0], cp[1] - c[1] + t[1], cp[2] - c[2] + t[2]};
            const vec3 dr = {v[0] - v_point[0], v[1] - v_point[1], v[2] - v_point[2]};
            const vec3 d_col_a = cross(dw, r);
            const vec3 d_col_b = cross(w, dr);
            d_col = {d_col_a[0] + d_col_b[0], d_col_a[1] + d_col_b[1], d_col_a[2] + d_col_b[2]};
        }
        dv = {dv[0] + (d_col[0] * q_prime(i)) + (col[0] * q_double_prime(i)),
              dv[1] + (d_col[1] * q_prime(i)) + (col[1] * q_double_prime(i)),
              dv[2] + (d_col[2] * q_prime(i)) + (col[2] * q_double_prime(i))};

        const vec3 wq = {w[0] * q_prime(i), w[1] * q_prime(i), w[2] * q_prime(i)};
        if (joint.translates) {
            t = {t[0] + wq[0], t[1] + wq[1], t[2] + wq[2]};
        } else {
            omega = {omega[0] + wq[0], omega[1] + wq[1], omega[2] + wq[2]};
            const vec3 wqp = cross(wq, p);
            c = {c[0] + wqp[0], c[1] + wqp[1], c[2] + wqp[2]};
        }
    }

    const double gain = norm(v);
    // The slope d||v||/ds is undefined at a singular gain (||J*f'|| -> 0): dot(v, dv) / gain is
    // 0 / 0 or x / 0. Callers evaluate the slope only where the TCP curve is the binding
    // (finite-gain) constraint, so this is not reached in normal operation, but guard the division
    // at the API boundary so a singular gain yields a defined 0 slope rather than a NaN/inf that
    // would poison the phase-plane slope downstream.
    double d_gain_ds = dot(v, dv) / gain;
    if (!std::isfinite(d_gain_ds)) {
        d_gain_ds = 0.0;
    }
    return {.gain_per_arc_unit = gain, .d_gain_ds = d_gain_ds};
}

kinematic_chain::builder& kinematic_chain::builder::extend(const joint_origin& origin) & {
    joints_.push_back({.kind = joint_kind_::k_fixed, .origin = origin, .axis = {}});
    return *this;
}

kinematic_chain::builder& kinematic_chain::builder::extend(const joint_origin& origin, const rotation_axis& axis) & {
    joints_.push_back({.kind = joint_kind_::k_rotational, .origin = origin, .axis = axis.direction});
    return *this;
}

kinematic_chain::builder& kinematic_chain::builder::extend(const joint_origin& origin, const translation_axis& axis) & {
    joints_.push_back({.kind = joint_kind_::k_translational, .origin = origin, .axis = axis.direction});
    return *this;
}

kinematic_chain kinematic_chain::builder::build() && {
    if (joints_.empty()) {
        throw std::invalid_argument("viam::trajex::jacobian: a kinematic chain needs at least one joint");
    }

    std::vector<joint_> joints;
    joints.reserve(joints_.size());
    for (std::size_t i = 0; i < joints_.size(); ++i) {
        const pending_joint_& pending = joints_[i];
        const std::string which = "viam::trajex::jacobian: joint " + std::to_string(i);

        if (!all_finite(pending.origin.translation)) {
            throw std::invalid_argument(which + " has a non-finite translation");
        }

        // Any nonzero quaternion denotes exactly one rotation, so we normalize rather than demand
        // that the caller already did. Only a zero or non-finite quaternion is refused, since
        // there is no rotation to recover from it. The norm is checked as well as the components
        // because large finite components can still overflow it.
        const auto& [w, x, y, z] = pending.origin.rotation;
        const double rotation_norm = std::sqrt((w * w) + (x * x) + (y * y) + (z * z));
        if (!all_finite(pending.origin.rotation) || !std::isfinite(rotation_norm) || rotation_norm == 0.0) {
            throw std::invalid_argument(which + " has a zero or non-finite quaternion");
        }
        const std::array<double, 4> unit_rotation = {w / rotation_norm, x / rotation_norm, y / rotation_norm, z / rotation_norm};

        joint_ joint{.kind = pending.kind, .origin_tf = origin_transform(pending.origin.translation, unit_rotation)};
        if (pending.kind != joint_kind_::k_fixed) {
            const double axis_norm = norm(pending.axis);
            if (!all_finite(pending.axis) || !std::isfinite(axis_norm) || axis_norm == 0.0) {
                throw std::invalid_argument(which + " has a zero or non-finite axis");
            }
            joint.unit_axis = normalized(pending.axis);
        }
        joints.push_back(joint);
    }
    return kinematic_chain(std::move(joints));
}

}  // namespace viam::trajex::jacobian
