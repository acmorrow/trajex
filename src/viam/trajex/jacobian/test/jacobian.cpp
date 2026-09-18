#include <viam/trajex/jacobian/jacobian.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/types/xt.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#if __has_include(<xtensor/containers/xfixed.hpp>)
#include <xtensor/containers/xfixed.hpp>
#else
#include <xtensor/xfixed.hpp>
#endif

#include <boost/test/unit_test.hpp>

namespace {

using viam::trajex::xmatrix;
using viam::trajex::xvector;
using viam::trajex::jacobian::kinematic_chain;

// One-shot convenience for tests: parse the tensor and evaluate at q.
xmatrix<> compute_jacobian(const xmatrix<>& table, const xvector<>& q) {
    return kinematic_chain::from(table).jacobian(q);
}

// Joint-type encodings for column 9 of the model-table tensor; values match
// viam::sdk::ModelTable::JointType.
constexpr double k_rev = 0.0;
constexpr double k_cont = 1.0;
constexpr double k_pris = 2.0;
constexpr double k_fix = 3.0;

xmatrix<> make_table(std::initializer_list<std::initializer_list<double>> rows) {
    const std::size_t n = rows.size();
    xmatrix<> t = xt::zeros<double>({n, std::size_t{10}});
    std::size_t i = 0;
    for (const auto& r : rows) {
        std::size_t j = 0;
        for (const double v : r) {
            t(i, j++) = v;
        }
        ++i;
    }
    return t;
}

double matrix_diff_norm(const xmatrix<>& A, const xmatrix<>& B) {
    double s = 0.0;
    const auto* a = A.begin();
    const auto* b = B.begin();
    for (; a != A.end(); ++a, ++b) {
        const double d = *a - *b;
        s += d * d;
    }
    return std::sqrt(s);
}

double vec3_diff_norm(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
}

struct twist {
    std::array<double, 3> v;
    std::array<double, 3> w;
};
twist J_times_qdot(const xmatrix<>& J, const xvector<>& q_dot) {
    twist t{{0, 0, 0}, {0, 0, 0}};
    const std::size_t n = q_dot.size();
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < 3; ++i) {
            t.v[i] += J(i, j) * q_dot(j);
        }
        for (std::size_t i = 0; i < 3; ++i) {
            t.w[i] += J(3 + i, j) * q_dot(j);
        }
    }
    return t;
}

// Two 1m planar links rotating about z, ending in a 1m fixed flange.
xmatrix<> twolink_table() {
    return xmatrix<>{
        {0, 0, 0, 0, 0, 0, 0, 0, 1, k_rev},
        {1, 0, 0, 0, 0, 0, 0, 0, 1, k_rev},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, k_fix},
    };
}

// 3 revolute joints separated by fixed 1m spacers, ending in a 1m flange.
xmatrix<> threelink_with_spacers_table() {
    return xmatrix<>{
        {0, 0, 0, 0, 0, 0, 0, 0, 1, k_rev},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, k_fix},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, k_rev},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, k_fix},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, k_rev},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, k_fix},
    };
}

// 6-revolute spatial chain mimicking a UR-like structure.
xmatrix<> sixdof_arm_table() {
    return xmatrix<>{
        {0, 0, 0.10, 0, 0, 0, 0, 0, 1, k_rev},
        {0, 0, 0.15, 0, 0, 0, 0, 1, 0, k_rev},
        {0.4, 0, 0, 0, 0, 0, 0, 1, 0, k_rev},
        {0.4, 0, 0, 0, 0, 0, 0, 1, 0, k_rev},
        {0, 0, 0.10, 0, 0, 0, 1, 0, 0, k_rev},
        {0, 0, 0.10, 0, 0, 0, 0, 0, 1, k_rev},
        {0, 0, 0.05, 0, 0, 0, 0, 0, 0, k_fix},
    };
}

// The analytic Jacobian in kinematic_chain is the being tested, so it
// cannot be its own evidence. The Jacobian's defining property is that it is
// the derivative of the forward kinematics; the numerical-consistency suite
// verifies exactly that property by finite-differencing the reference forward
// kinematics below, which is written independently of kinematic_chain.

xmatrix<> reference_identity4() {
    xmatrix<> t = xt::zeros<double>({std::size_t{4}, std::size_t{4}});
    for (std::size_t i = 0; i < 4; ++i) {
        t(i, i) = 1.0;
    }
    return t;
}

xmatrix<> reference_matmul(const xmatrix<>& a, const xmatrix<>& b) {
    xmatrix<> c = xt::zeros<double>({std::size_t{4}, std::size_t{4}});
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < 4; ++k) {
                s += a(i, k) * b(k, j);
            }
            c(i, j) = s;
        }
    }
    return c;
}

// 4x4 rotation about a unit axis by angle radians (Rodrigues).
xmatrix<> reference_axis_rotation(double x, double y, double z, double angle) {
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double t = 1.0 - c;
    xmatrix<> r = reference_identity4();
    r(0, 0) = (t * x * x) + c;
    r(0, 1) = (t * x * y) - (s * z);
    r(0, 2) = (t * x * z) + (s * y);
    r(1, 0) = (t * x * y) + (s * z);
    r(1, 1) = (t * y * y) + c;
    r(1, 2) = (t * y * z) - (s * x);
    r(2, 0) = (t * x * z) - (s * y);
    r(2, 1) = (t * y * z) + (s * x);
    r(2, 2) = (t * z * z) + c;
    return r;
}

// Tensor columns: 0..2 xyz, 3..5 rpy (fixed-axis XYZ), 6..8 axis, 9 joint
// type.
xmatrix<> forward_transform(const xmatrix<>& table, const xvector<>& q) {
    xmatrix<> T = reference_identity4();
    std::size_t qi = 0;
    const std::size_t n = table.shape()[0];
    for (std::size_t r = 0; r < n; ++r) {
        xmatrix<> link = reference_matmul(
            reference_matmul(reference_axis_rotation(0.0, 0.0, 1.0, table(r, 5)), reference_axis_rotation(0.0, 1.0, 0.0, table(r, 4))),
            reference_axis_rotation(1.0, 0.0, 0.0, table(r, 3)));
        link(0, 3) = table(r, 0);
        link(1, 3) = table(r, 1);
        link(2, 3) = table(r, 2);
        T = reference_matmul(T, link);

        if (table(r, 9) == k_rev || table(r, 9) == k_cont) {
            const double ax = table(r, 6);
            const double ay = table(r, 7);
            const double az = table(r, 8);
            const double an = std::sqrt((ax * ax) + (ay * ay) + (az * az));
            T = reference_matmul(T, reference_axis_rotation(ax / an, ay / an, az / an, q(qi)));
            ++qi;
        }
    }
    return T;
}

// A chain as the builder takes it, kept as plain data so the reference forward kinematics below
// can walk it without going through kinematic_chain.
enum class test_joint_kind : std::uint8_t { k_fixed, k_rotational, k_translational };

struct test_joint {
    test_joint_kind kind;
    std::array<double, 3> xyz;
    std::array<double, 4> wxyz;
    std::array<double, 3> axis;
};

using test_chain = std::vector<test_joint>;

constexpr std::array<double, 4> k_identity_rotation = {1.0, 0.0, 0.0, 0.0};

kinematic_chain build_chain(const test_chain& joints) {
    kinematic_chain::builder builder;
    for (const auto& j : joints) {
        const kinematic_chain::joint_origin origin{.translation = j.xyz, .rotation = j.wxyz};
        switch (j.kind) {
            case test_joint_kind::k_fixed:
                builder.extend(origin);
                break;
            case test_joint_kind::k_rotational:
                builder.extend(origin, kinematic_chain::rotation_axis{j.axis});
                break;
            case test_joint_kind::k_translational:
                builder.extend(origin, kinematic_chain::translation_axis{j.axis});
                break;
        }
    }
    return std::move(builder).build();
}

// The rotation a quaternion denotes, built by way of its axis and angle rather than the direct
// matrix formula the implementation uses, so the two are independent.
xmatrix<> reference_quaternion_rotation(const std::array<double, 4>& wxyz) {
    const double n = std::sqrt((wxyz[0] * wxyz[0]) + (wxyz[1] * wxyz[1]) + (wxyz[2] * wxyz[2]) + (wxyz[3] * wxyz[3]));
    const double w = wxyz[0] / n;
    const std::array<double, 3> v = {wxyz[1] / n, wxyz[2] / n, wxyz[3] / n};
    const double s = std::sqrt((v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]));
    if (s == 0.0) {
        return reference_identity4();
    }
    return reference_axis_rotation(v[0] / s, v[1] / s, v[2] / s, 2.0 * std::atan2(s, w));
}

xmatrix<> forward_transform(const test_chain& joints, const xvector<>& q) {
    xmatrix<> T = reference_identity4();
    std::size_t qi = 0;
    for (const auto& j : joints) {
        xmatrix<> origin = reference_quaternion_rotation(j.wxyz);
        origin(0, 3) = j.xyz[0];
        origin(1, 3) = j.xyz[1];
        origin(2, 3) = j.xyz[2];
        T = reference_matmul(T, origin);

        if (j.kind == test_joint_kind::k_fixed) {
            continue;
        }
        const double an = std::sqrt((j.axis[0] * j.axis[0]) + (j.axis[1] * j.axis[1]) + (j.axis[2] * j.axis[2]));
        if (j.kind == test_joint_kind::k_rotational) {
            T = reference_matmul(T, reference_axis_rotation(j.axis[0] / an, j.axis[1] / an, j.axis[2] / an, q(qi)));
        } else {
            xmatrix<> slide = reference_identity4();
            slide(0, 3) = q(qi) * j.axis[0] / an;
            slide(1, 3) = q(qi) * j.axis[1] / an;
            slide(2, 3) = q(qi) * j.axis[2] / an;
            T = reference_matmul(T, slide);
        }
        ++qi;
    }
    return T;
}

// The quaternion for a URDF fixed-axis XYZ rpy, which composes as Rz(yaw) * Ry(pitch) * Rx(roll).
std::array<double, 4> rpy_to_quaternion(double roll, double pitch, double yaw) {
    const double cr = std::cos(roll / 2.0);
    const double sr = std::sin(roll / 2.0);
    const double cp = std::cos(pitch / 2.0);
    const double sp = std::sin(pitch / 2.0);
    const double cy = std::cos(yaw / 2.0);
    const double sy = std::sin(yaw / 2.0);
    return {
        (cr * cp * cy) + (sr * sp * sy), (sr * cp * cy) - (cr * sp * sy), (cr * sp * cy) + (sr * cp * sy), (cr * cp * sy) - (sr * sp * cy)};
}

// The same geometry as a model table, described as a joint list for the builder.
test_chain chain_from_table(const xmatrix<>& table) {
    test_chain joints;
    for (std::size_t r = 0; r < table.shape()[0]; ++r) {
        const double type = table(r, 9);
        test_joint_kind kind = test_joint_kind::k_rotational;
        if (type == k_fix) {
            kind = test_joint_kind::k_fixed;
        } else if (type == k_pris) {
            kind = test_joint_kind::k_translational;
        }
        joints.push_back({.kind = kind,
                          .xyz = {table(r, 0), table(r, 1), table(r, 2)},
                          .wxyz = rpy_to_quaternion(table(r, 3), table(r, 4), table(r, 5)),
                          .axis = {table(r, 6), table(r, 7), table(r, 8)}});
    }
    return joints;
}

// Numerical geometric Jacobian via central differences on the reference
// forward_transform. The model is either a table or a joint list, since
// forward_transform has an overload for each.
template <typename Model>
xmatrix<> numerical_jacobian(const Model& model, const xvector<>& q, double delta = 1e-7) {
    const std::size_t n_actuated = q.size();
    xmatrix<> J_num = xt::zeros<double>({std::size_t{6}, n_actuated});

    for (std::size_t i = 0; i < n_actuated; ++i) {
        xvector<> q_plus = q;
        xvector<> q_minus = q;
        q_plus(i) += delta;
        q_minus(i) -= delta;

        const xmatrix<> Tp = forward_transform(model, q_plus);
        const xmatrix<> Tm = forward_transform(model, q_minus);

        for (std::size_t r = 0; r < 3; ++r) {
            J_num(r, i) = (Tp(r, 3) - Tm(r, 3)) / (2.0 * delta);
        }

        std::array<std::array<double, 3>, 3> dR{};
        for (std::size_t a = 0; a < 3; ++a) {
            for (std::size_t b = 0; b < 3; ++b) {
                double s = 0.0;
                for (std::size_t k = 0; k < 3; ++k) {
                    s += Tp(a, k) * Tm(b, k);
                }
                dR[a][b] = s;
            }
        }
        const double scale = 1.0 / (2.0 * delta);
        J_num(3, i) = 0.5 * (dR[2][1] - dR[1][2]) * scale;
        J_num(4, i) = 0.5 * (dR[0][2] - dR[2][0]) * scale;
        J_num(5, i) = 0.5 * (dR[1][0] - dR[0][1]) * scale;
    }
    return J_num;
}

void check_matches_numerical(const xmatrix<>& table, const xvector<>& q, double tol = 1e-6) {
    const auto J = compute_jacobian(table, q);
    const auto J_num = numerical_jacobian(table, q);
    BOOST_CHECK_SMALL(matrix_diff_norm(J, J_num), tol);
}

}  // namespace

// The tests in this suite validate that `J * q_dot` produces the expected
// Cartesian velocity.
BOOST_AUTO_TEST_SUITE(jacobian_velocity_tests)

BOOST_AUTO_TEST_CASE(twolink_base_spin_extended) {
    const auto table = twolink_table();
    const xvector<> q = xt::zeros<double>({std::size_t{2}});
    const xvector<> q_dot = {1.0, 0.0};

    const auto J = compute_jacobian(table, q);
    const auto t = J_times_qdot(J, q_dot);

    BOOST_CHECK_SMALL(vec3_diff_norm(t.v, {0.0, 2.0, 0.0}), 1e-10);
    BOOST_CHECK_SMALL(vec3_diff_norm(t.w, {0.0, 0.0, 1.0}), 1e-10);
}

BOOST_AUTO_TEST_CASE(twolink_base_spin_rotated) {
    const auto table = twolink_table();
    const xvector<> q = {std::numbers::pi / 2.0, 0.0};
    const xvector<> q_dot = {1.0, 0.0};

    const auto J = compute_jacobian(table, q);
    const auto t = J_times_qdot(J, q_dot);

    BOOST_CHECK_SMALL(vec3_diff_norm(t.v, {-2.0, 0.0, 0.0}), 1e-10);
    BOOST_CHECK_SMALL(vec3_diff_norm(t.w, {0.0, 0.0, 1.0}), 1e-10);
}

BOOST_AUTO_TEST_CASE(twolink_base_spin_bent) {
    const auto table = twolink_table();
    const xvector<> q = {0.0, std::numbers::pi / 2.0};
    const xvector<> q_dot = {1.0, 0.0};

    const auto J = compute_jacobian(table, q);
    const auto t = J_times_qdot(J, q_dot);

    BOOST_CHECK_SMALL(vec3_diff_norm(t.v, {-1.0, 1.0, 0.0}), 1e-10);
    BOOST_CHECK_SMALL(vec3_diff_norm(t.w, {0.0, 0.0, 1.0}), 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()

// The tests in this suite check kinematic_chain::jacobian against hand-computed
// Jacobian values.
BOOST_AUTO_TEST_SUITE(jacobian_ground_truth_tests)

BOOST_AUTO_TEST_CASE(twolink_zero) {
    const auto table = twolink_table();
    const xvector<> q = xt::zeros<double>({std::size_t{2}});
    const auto J = compute_jacobian(table, q);

    const xmatrix<> J_expected = {
        {0.0, 0.0},
        {2.0, 1.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {1.0, 1.0},
    };
    BOOST_CHECK_SMALL(matrix_diff_norm(J, J_expected), 1e-10);
}

BOOST_AUTO_TEST_CASE(twolink_q1_ninety) {
    const auto table = twolink_table();
    const xvector<> q = {std::numbers::pi / 2.0, 0.0};
    const auto J = compute_jacobian(table, q);

    const xmatrix<> J_expected = {
        {-2.0, -1.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {1.0, 1.0},
    };
    BOOST_CHECK_SMALL(matrix_diff_norm(J, J_expected), 1e-10);
}

BOOST_AUTO_TEST_CASE(twolink_q2_ninety) {
    const auto table = twolink_table();
    const xvector<> q = {0.0, std::numbers::pi / 2.0};
    const auto J = compute_jacobian(table, q);

    const xmatrix<> J_expected = {
        {-1.0, -1.0},
        {1.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {1.0, 1.0},
    };
    BOOST_CHECK_SMALL(matrix_diff_norm(J, J_expected), 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()

// The numerical-consistency suite depends on the reference forward kinematics
// being correct, so the tests in this suite verify a few easy cases directly.
BOOST_AUTO_TEST_SUITE(fk_tests)

BOOST_AUTO_TEST_CASE(twolink_zero_ee_at_2_0_0) {
    const auto table = twolink_table();
    const xmatrix<> T = forward_transform(table, xt::zeros<double>({std::size_t{2}}));
    BOOST_CHECK_CLOSE(T(0, 3), 2.0, 1e-9);
    BOOST_CHECK_SMALL(std::abs(T(1, 3)), 1e-12);
    BOOST_CHECK_SMALL(std::abs(T(2, 3)), 1e-12);
}

BOOST_AUTO_TEST_CASE(twolink_q1_pi_over_2_ee_at_0_2_0) {
    const auto table = twolink_table();
    const xvector<> q = {std::numbers::pi / 2.0, 0.0};
    const xmatrix<> T = forward_transform(table, q);
    BOOST_CHECK_SMALL(std::abs(T(0, 3)), 1e-9);
    BOOST_CHECK_CLOSE(T(1, 3), 2.0, 1e-9);
    BOOST_CHECK_SMALL(std::abs(T(2, 3)), 1e-12);
}

BOOST_AUTO_TEST_CASE(twolink_q2_pi_over_2_ee_at_1_1_0) {
    const auto table = twolink_table();
    const xvector<> q = {0.0, std::numbers::pi / 2.0};
    const xmatrix<> T = forward_transform(table, q);
    BOOST_CHECK_CLOSE(T(0, 3), 1.0, 1e-9);
    BOOST_CHECK_CLOSE(T(1, 3), 1.0, 1e-9);
    BOOST_CHECK_SMALL(std::abs(T(2, 3)), 1e-12);
}

BOOST_AUTO_TEST_SUITE_END()

// The tests in this suite check the analytical Jacobian against a
// central-difference numerical Jacobian.
BOOST_AUTO_TEST_SUITE(jacobian_numerical_consistency_tests)

BOOST_AUTO_TEST_CASE(twolink_zero) {
    check_matches_numerical(twolink_table(), xt::zeros<double>({std::size_t{2}}));
}

BOOST_AUTO_TEST_CASE(twolink_q1_q2_forty_five) {
    const xvector<> q = {std::numbers::pi / 4.0, std::numbers::pi / 4.0};
    check_matches_numerical(twolink_table(), q);
}

BOOST_AUTO_TEST_CASE(twolink_folded_back) {
    const xvector<> q = {0.0, std::numbers::pi};
    check_matches_numerical(twolink_table(), q);
}

BOOST_AUTO_TEST_CASE(threelink_with_spacers_zero) {
    check_matches_numerical(threelink_with_spacers_table(), xt::zeros<double>({std::size_t{3}}));
}

BOOST_AUTO_TEST_CASE(threelink_with_spacers_typical) {
    const xvector<> q = {0.3, -0.5, 0.8};
    check_matches_numerical(threelink_with_spacers_table(), q);
}

BOOST_AUTO_TEST_CASE(sixdof_zero) {
    check_matches_numerical(sixdof_arm_table(), xt::zeros<double>({std::size_t{6}}));
}

BOOST_AUTO_TEST_CASE(sixdof_typical) {
    const xvector<> q = {0.1, -0.5, 1.2, -0.3, 0.6, -0.1};
    check_matches_numerical(sixdof_arm_table(), q);
}

BOOST_AUTO_TEST_SUITE_END()

// The tests in this suite cover kinematic_chain itself: reuse of one parsed
// chain across evaluations and agreement between the two Jacobian entry
// points.
BOOST_AUTO_TEST_SUITE(kinematic_chain_tests)

BOOST_AUTO_TEST_CASE(parsed_chain_is_reusable_across_evaluations) {
    const auto chain = kinematic_chain::from(twolink_table());
    const xvector<> q1 = {0.3, -0.7};
    const xvector<> q2 = {-1.1, 0.4};

    BOOST_CHECK_SMALL(matrix_diff_norm(chain.jacobian(q1), compute_jacobian(twolink_table(), q1)), 1e-15);
    BOOST_CHECK_SMALL(matrix_diff_norm(chain.jacobian(q2), compute_jacobian(twolink_table(), q2)), 1e-15);
}

BOOST_AUTO_TEST_CASE(linear_jacobian_matches_jacobian_linear_block) {
    const auto chain = kinematic_chain::from(sixdof_arm_table());
    const xvector<> q = {0.1, -0.5, 1.2, -0.3, 0.6, -0.1};

    const auto J = chain.jacobian(q);
    const auto J_lin = chain.linear_jacobian(q);
    BOOST_REQUIRE_EQUAL(J_lin.shape()[0], 3U);
    BOOST_REQUIRE_EQUAL(J_lin.shape()[1], 6U);

    double max_abs_diff = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            max_abs_diff = std::max(max_abs_diff, std::abs(J_lin(i, j) - J(i, j)));
        }
    }
    BOOST_CHECK_SMALL(max_abs_diff, 1e-15);
}

// The table is copied element by element, never adapted, so a view or an expression is as good
// as a container.
BOOST_AUTO_TEST_CASE(from_accepts_any_rank_two_and_refuses_the_rest) {
    using namespace viam::trajex;

    const auto parseable = []<typename T>() { return requires(const T& t) { kinematic_chain::from(t); }; };

    using rank_three = xt::xtensor<double, 3>;
    using fixed_shape = xt::xtensor_fixed<double, xt::xshape<2, 10>>;

    BOOST_CHECK(parseable.template operator()<xmatrix<>>());
    BOOST_CHECK(parseable.template operator()<xt::xarray<double>>());
    BOOST_CHECK(parseable.template operator()<fixed_shape>());

    BOOST_CHECK(!parseable.template operator()<xvector<>>());
    BOOST_CHECK(!parseable.template operator()<rank_three>());
    BOOST_CHECK(!parseable.template operator()<double>());
}

BOOST_AUTO_TEST_CASE(from_validates_a_rank_known_only_at_runtime) {
    const xt::xarray<double> good = twolink_table();
    BOOST_CHECK_NO_THROW(static_cast<void>(kinematic_chain::from(good)));

    const xt::xarray<double> rank_one = {1.0, 2.0, 3.0};
    BOOST_CHECK_THROW(static_cast<void>(kinematic_chain::from(rank_one)), std::invalid_argument);

    const xt::xarray<double> rank_three = xt::zeros<double>({2, 2, 10});
    BOOST_CHECK_THROW(static_cast<void>(kinematic_chain::from(rank_three)), std::invalid_argument);
}

// Same deal for the joint vector. These run once per integration step, so an xvector has to go
// through with no copy.
BOOST_AUTO_TEST_CASE(evaluation_refuses_a_rank_that_is_not_one) {
    using namespace viam::trajex;

    const auto chain = kinematic_chain::from(twolink_table());

    const auto evaluable = [&]<typename T>() { return requires(const T& t) { chain.jacobian(t); }; };
    const auto linear_evaluable = [&]<typename T>() { return requires(const T& t) { chain.linear_jacobian(t); }; };
    const auto gain_evaluable = [&]<typename T>() { return requires(const T& t) { chain.linear_velocity_gain_at(t, t, t); }; };

    using rank_three = xt::xtensor<double, 3>;

    BOOST_CHECK(evaluable.template operator()<xvector<>>());
    BOOST_CHECK(evaluable.template operator()<xt::xarray<double>>());
    BOOST_CHECK(!evaluable.template operator()<xmatrix<>>());
    BOOST_CHECK(!evaluable.template operator()<rank_three>());

    BOOST_CHECK(linear_evaluable.template operator()<xvector<>>());
    BOOST_CHECK(!linear_evaluable.template operator()<xmatrix<>>());

    BOOST_CHECK(gain_evaluable.template operator()<xvector<>>());
    BOOST_CHECK(!gain_evaluable.template operator()<xmatrix<>>());

    // A rank settled at runtime reaches the check.
    const xt::xarray<double> good = {0.3, -0.7};
    const xt::xarray<double> two_d = {{0.3, -0.7}, {0.1, 0.2}};
    BOOST_CHECK_NO_THROW(static_cast<void>(chain.jacobian(good)));
    BOOST_CHECK_THROW(static_cast<void>(chain.jacobian(two_d)), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(chain.linear_jacobian(two_d)), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(chain.linear_velocity_gain_at(two_d, good, good)), std::invalid_argument);
}

// tcp_limits::from restates none of that, so the two must agree by construction.
BOOST_AUTO_TEST_CASE(tcp_limits_from_accepts_exactly_what_the_chain_does) {
    using namespace viam::trajex;
    using viam::trajex::totg::trajectory;

    const auto tcp_takes = []<typename T>() { return requires(const T& t) { trajectory::tcp_limits::from(t, 1.0); }; };
    const auto chain_takes = []<typename T>() { return requires(const T& t) { kinematic_chain::from(t); }; };
    const auto agrees = [&]<typename T>() { return tcp_takes.template operator()<T>() == chain_takes.template operator()<T>(); };

    using rank_three = xt::xtensor<double, 3>;
    using fixed_shape = xt::xtensor_fixed<double, xt::xshape<2, 10>>;

    BOOST_CHECK(agrees.template operator()<xmatrix<>>());
    BOOST_CHECK(agrees.template operator()<xt::xarray<double>>());
    BOOST_CHECK(agrees.template operator()<fixed_shape>());
    BOOST_CHECK(agrees.template operator()<xvector<>>());
    BOOST_CHECK(agrees.template operator()<rank_three>());

    BOOST_CHECK(tcp_takes.template operator()<xmatrix<>>());
    BOOST_CHECK(!tcp_takes.template operator()<xvector<>>());

    const xt::xarray<double> rank_one = {1.0, 2.0, 3.0};
    BOOST_CHECK_THROW(static_cast<void>(trajectory::tcp_limits::from(rank_one, 1.0)), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()

// A continuous joint is a revolute joint without position limits, and the model table carries
// no limits, so retyping a row from revolute to continuous must change nothing anyone can
// observe. These compare the two tables rather than checking a continuous chain against
// precomputed values, because indistinguishability is the property we mean to hold.
BOOST_AUTO_TEST_SUITE(continuous_joint_tests)

// Same table as the named helper, with every rotating row typed continuous instead.
xt::xarray<double> twolink_table_continuous() {
    return xt::xarray<double>{
        {0, 0, 0, 0, 0, 0, 0, 0, 1, k_cont},
        {1, 0, 0, 0, 0, 0, 0, 0, 1, k_cont},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, k_fix},
    };
}

// As sixdof_arm_table, but with the wrist rows typed continuous, which is the mix a real arm
// is most likely to present.
xt::xarray<double> sixdof_arm_table_continuous_wrist() {
    return xt::xarray<double>{
        {0, 0, 0.10, 0, 0, 0, 0, 0, 1, k_rev},
        {0, 0, 0.15, 0, 0, 0, 0, 1, 0, k_rev},
        {0.4, 0, 0, 0, 0, 0, 0, 1, 0, k_rev},
        {0.4, 0, 0, 0, 0, 0, 0, 1, 0, k_cont},
        {0, 0, 0.10, 0, 0, 0, 1, 0, 0, k_cont},
        {0, 0, 0.10, 0, 0, 0, 0, 0, 1, k_cont},
        {0, 0, 0.05, 0, 0, 0, 0, 0, 0, k_fix},
    };
}

BOOST_AUTO_TEST_CASE(jacobian_matches_revolute_twolink) {
    for (const auto& q : {xt::xarray<double>{0.0, 0.0}, xt::xarray<double>{0.3, -0.7}, xt::xarray<double>{-1.1, 2.4}}) {
        BOOST_CHECK_EQUAL(matrix_diff_norm(compute_jacobian(twolink_table(), q), compute_jacobian(twolink_table_continuous(), q)), 0.0);
    }
}

BOOST_AUTO_TEST_CASE(jacobian_matches_revolute_sixdof) {
    const xt::xarray<double> q = {0.1, -0.5, 1.2, -0.3, 0.6, -0.1};
    BOOST_CHECK_EQUAL(matrix_diff_norm(compute_jacobian(sixdof_arm_table(), q), compute_jacobian(sixdof_arm_table_continuous_wrist(), q)),
                      0.0);
}

BOOST_AUTO_TEST_CASE(linear_velocity_gain_matches_revolute) {
    const xt::xarray<double> q = {0.1, -0.5, 1.2, -0.3, 0.6, -0.1};
    const xt::xarray<double> q_prime = {0.2, 0.1, -0.3, 0.4, -0.2, 0.5};
    const xt::xarray<double> q_double_prime = {-0.1, 0.3, 0.2, -0.4, 0.1, -0.2};

    const auto revolute = kinematic_chain::from(sixdof_arm_table()).linear_velocity_gain_at(q, q_prime, q_double_prime);
    const auto continuous = kinematic_chain::from(sixdof_arm_table_continuous_wrist()).linear_velocity_gain_at(q, q_prime, q_double_prime);

    BOOST_CHECK_EQUAL(revolute.gain_per_arc_unit, continuous.gain_per_arc_unit);
    BOOST_CHECK_EQUAL(revolute.d_gain_ds, continuous.d_gain_ds);
}

// We never consult a position limit, so the representative a driver reports for an angle does
// not matter. This holds for revolute rows too; it is stated here because a continuous joint is
// the one a driver is liable to report as an accumulated angle.
BOOST_AUTO_TEST_CASE(indifferent_to_full_turn_offsets) {
    const auto chain = kinematic_chain::from(twolink_table_continuous());
    const xt::xarray<double> q = {0.3, -0.7};
    const xt::xarray<double> q_offset = {0.3 + 4.0 * std::numbers::pi, -0.7 - 2.0 * std::numbers::pi};

    BOOST_CHECK_SMALL(matrix_diff_norm(chain.jacobian(q), chain.jacobian(q_offset)), 1e-14);
}

BOOST_AUTO_TEST_SUITE_END()

// The tests in this suite validate linear_velocity_gain_at: the gain ||J*f'|| and its
// path derivative d/ds||J*f'|| against a central finite difference of the gain along the path.
BOOST_AUTO_TEST_SUITE(linear_velocity_gain_tests)

namespace {

// ||linear_jacobian(q) * q_prime|| computed independently of the method under test.
double reference_gain(const xmatrix<>& table, const xvector<>& q, const xvector<>& q_prime) {
    const auto J = kinematic_chain::from(table).linear_jacobian(q);
    std::array<double, 3> v{0.0, 0.0, 0.0};
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t j = 0; j < q_prime.size(); ++j) {
            v[r] += J(r, j) * q_prime(j);
        }
    }
    return std::sqrt((v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]));
}

void check_linear_velocity_gain(
    const xmatrix<>& table, const xvector<>& q, const xvector<>& q_prime, const xvector<>& q_double_prime, double tol = 1e-6) {
    const auto chain = kinematic_chain::from(table);
    const auto analytic = chain.linear_velocity_gain_at(q, q_prime, q_double_prime);

    BOOST_CHECK_CLOSE(analytic.gain_per_arc_unit, reference_gain(table, q, q_prime), 1e-7);

    const double h = 1e-6;
    const xvector<> q_plus = q + h * q_prime;
    const xvector<> q_minus = q - h * q_prime;
    const xvector<> qp_plus = q_prime + h * q_double_prime;
    const xvector<> qp_minus = q_prime - h * q_double_prime;
    const double numeric = (reference_gain(table, q_plus, qp_plus) - reference_gain(table, q_minus, qp_minus)) / (2.0 * h);

    BOOST_CHECK_SMALL(analytic.d_gain_ds - numeric, tol);
}

}  // namespace

BOOST_AUTO_TEST_CASE(twolink_typical) {
    check_linear_velocity_gain(twolink_table(), {0.3, -0.7}, {1.0, 0.5}, {0.2, -0.4});
}

BOOST_AUTO_TEST_CASE(threelink_with_spacers_typical) {
    check_linear_velocity_gain(threelink_with_spacers_table(), {0.3, -0.5, 0.8}, {0.6, 0.1, -0.7}, {-0.2, 0.4, 0.3});
}

BOOST_AUTO_TEST_CASE(sixdof_typical) {
    check_linear_velocity_gain(
        sixdof_arm_table(), {0.1, -0.5, 1.2, -0.3, 0.6, -0.1}, {0.2, 0.7, -0.4, 0.5, -0.6, 0.3}, {-0.1, 0.3, 0.2, -0.5, 0.4, -0.2});
}

// A singular gain (||J*f'|| -> 0, here from a zero path tangent) has an undefined slope, since
// dot(v, dv) / gain is 0 / 0. The function must return a finite, non-poisoning value rather than a
// NaN that would propagate into the phase-plane slope downstream.
BOOST_AUTO_TEST_CASE(singular_gain_yields_finite_derivative) {
    const auto chain = kinematic_chain::from(twolink_table());
    const auto vg = chain.linear_velocity_gain_at(xvector<>{0.3, -0.7}, xvector<>{0.0, 0.0}, xvector<>{0.2, -0.4});
    BOOST_CHECK_SMALL(vg.gain_per_arc_unit, 1e-15);
    BOOST_CHECK(std::isfinite(vg.d_gain_ds));
}

BOOST_AUTO_TEST_SUITE_END()

// The tests in this suite validate trajectory::tcp_limits::from: the gain callback reports what
// the chain it parsed at construction reports, and keeps doing so across calls.
BOOST_AUTO_TEST_SUITE(tcp_limits_from_tests)

BOOST_AUTO_TEST_CASE(matches_linear_velocity_gain) {
    const xmatrix<> table = twolink_table();
    const xvector<> q = {0.3, -0.7};
    const xvector<> q_prime = {0.6, 0.8};
    const xvector<> q_double_prime = {-0.2, 0.1};

    const auto limits = viam::trajex::totg::trajectory::tcp_limits::from(table, 0.5);
    const auto expected = kinematic_chain::from(table).linear_velocity_gain_at(q, q_prime, q_double_prime);

    for (int call = 0; call < 2; ++call) {
        const auto g = limits.linear_velocity_gain(q, q_prime, q_double_prime);
        BOOST_CHECK_EQUAL(g.per_arc_unit, expected.gain_per_arc_unit);
        BOOST_CHECK_EQUAL(g.d_ds, expected.d_gain_ds);
    }

    BOOST_CHECK_EQUAL(limits.max_linear_velocity, 0.5);
}

BOOST_AUTO_TEST_CASE(chain_overload_matches_linear_velocity_gain) {
    const auto chain = build_chain(chain_from_table(sixdof_arm_table()));
    const xvector<> q = {0.1, -0.5, 1.2, -0.3, 0.6, -0.1};
    const xvector<> q_prime = {0.2, 0.7, -0.4, 0.5, -0.6, 0.3};
    const xvector<> q_double_prime = {-0.1, 0.3, 0.2, -0.5, 0.4, -0.2};

    const auto limits = viam::trajex::totg::trajectory::tcp_limits::from(chain, 0.5);
    const auto expected = chain.linear_velocity_gain_at(q, q_prime, q_double_prime);
    const auto g = limits.linear_velocity_gain(q, q_prime, q_double_prime);
    BOOST_CHECK_EQUAL(g.per_arc_unit, expected.gain_per_arc_unit);
    BOOST_CHECK_EQUAL(g.d_ds, expected.d_gain_ds);
    BOOST_CHECK_EQUAL(limits.max_linear_velocity, 0.5);
}

BOOST_AUTO_TEST_CASE(rejects_malformed_tensor_at_construction) {
    const xmatrix<> bad = xt::zeros<double>({std::size_t{1}, std::size_t{9}});
    BOOST_CHECK_THROW(static_cast<void>(viam::trajex::totg::trajectory::tcp_limits::from(bad, 0.5)), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()

// The tests in this suite verify error handling for invalid inputs.
BOOST_AUTO_TEST_SUITE(jacobian_error_tests)

BOOST_AUTO_TEST_CASE(rejects_wrong_q_size) {
    const auto table = twolink_table();
    const xvector<> q_bad = {0.0, 0.0, 0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q_bad), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(linear_jacobian_rejects_wrong_q_size) {
    const auto chain = kinematic_chain::from(twolink_table());
    const xvector<> q_bad = {0.0, 0.0, 0.0};
    BOOST_CHECK_THROW(static_cast<void>(chain.linear_jacobian(q_bad)), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_zero_axis_for_continuous) {
    const auto table = make_table({{0, 0, 0, 0, 0, 0, 0, 0, 0, k_cont}});
    const xvector<> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(continuous_chain_rejects_wrong_q_size) {
    const auto table = make_table({{0, 0, 0, 0, 0, 0, 0, 0, 1, k_cont}, {1, 0, 0, 0, 0, 0, 0, 0, 1, k_cont}});
    const xvector<> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_prismatic_joint) {
    const auto table = make_table({{0, 0, 0, 0, 0, 0, 0, 0, 1, k_pris}});
    const xvector<> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_zero_axis_for_revolute) {
    const auto table = make_table({{0, 0, 0, 0, 0, 0, 0, 0, 0, k_rev}});
    const xvector<> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_wrong_column_count) {
    const xmatrix<> bad = xt::zeros<double>({std::size_t{1}, std::size_t{9}});
    const xvector<> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(bad, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_empty_table) {
    const xmatrix<> bad = xt::zeros<double>({std::size_t{0}, std::size_t{10}});
    const xvector<> q = xt::zeros<double>({std::size_t{0}});
    BOOST_CHECK_THROW(compute_jacobian(bad, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_non_integer_joint_type) {
    const auto table = make_table({{0, 0, 0, 0, 0, 0, 0, 0, 1, 0.5}});
    const xvector<> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(rejects_unknown_joint_type) {
    const auto table = make_table({{0, 0, 0, 0, 0, 0, 0, 0, 1, 7.0}});
    const xvector<> q = {0.0};
    BOOST_CHECK_THROW(compute_jacobian(table, q), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()

// The tests in this suite validate chains assembled with kinematic_chain::builder: that they agree
// with the tensor path on the geometry both can express, that prismatic joints are the derivative
// of the reference forward kinematics like everything else, and that build() refuses what it
// should.
BOOST_AUTO_TEST_SUITE(builder_tests)

namespace {

// Spatial origins with nonzero rpy throughout, so the tensor's rpy and the builder's quaternions
// both carry real rotations.
xmatrix<> rotated_origins_table() {
    return xmatrix<>{
        {0, 0, 0.10, 0.3, -0.2, 0.5, 0, 0, 1, k_rev},
        {0.05, 0, 0.15, -0.4, 0.1, 0.2, 0, 1, 0, k_rev},
        {0.4, 0.02, 0, 0.0, 0.6, -0.3, 0, 1, 0, k_cont},
        {0.1, 0, 0.05, 3.14159265, -1.570796, 0, 0, 0, 0, k_fix},
    };
}

// A SCARA: two rotations about z, a vertical prismatic quill, then a wrist rotation.
test_chain scara_chain() {
    return {
        {.kind = test_joint_kind::k_rotational, .xyz = {0, 0, 0.3}, .wxyz = k_identity_rotation, .axis = {0, 0, 1}},
        {.kind = test_joint_kind::k_rotational, .xyz = {0.35, 0, 0}, .wxyz = k_identity_rotation, .axis = {0, 0, 1}},
        {.kind = test_joint_kind::k_translational, .xyz = {0.3, 0, 0}, .wxyz = k_identity_rotation, .axis = {0, 0, -1}},
        {.kind = test_joint_kind::k_rotational, .xyz = {0, 0, -0.05}, .wxyz = k_identity_rotation, .axis = {0, 0, 1}},
        {.kind = test_joint_kind::k_fixed, .xyz = {0, 0, -0.1}, .wxyz = k_identity_rotation, .axis = {}},
    };
}

// Prismatic and rotating joints interleaved, with rotated origins and axes that are neither unit
// length nor aligned with a coordinate axis, so a translating joint's axis is carried around by
// rotations upstream of it and moves points downstream of it.
test_chain mixed_chain() {
    return {
        {.kind = test_joint_kind::k_translational, .xyz = {0, 0, 0}, .wxyz = k_identity_rotation, .axis = {2, 0, 0}},
        {.kind = test_joint_kind::k_rotational, .xyz = {0, 0, 0.2}, .wxyz = rpy_to_quaternion(0.2, 0.1, -0.3), .axis = {0, 1, 1}},
        {.kind = test_joint_kind::k_translational, .xyz = {0.3, 0.1, 0}, .wxyz = rpy_to_quaternion(-0.5, 0.4, 0.2), .axis = {1, 1, 0}},
        {.kind = test_joint_kind::k_rotational, .xyz = {0, 0.2, 0.1}, .wxyz = rpy_to_quaternion(0.7, 0, 0.3), .axis = {1, 0, 0}},
        {.kind = test_joint_kind::k_fixed, .xyz = {0.1, 0, 0.05}, .wxyz = rpy_to_quaternion(0, -0.4, 0), .axis = {}},
    };
}

void check_gain_matches_numerical(const kinematic_chain& chain,
                                  const xvector<>& q,
                                  const xvector<>& q_prime,
                                  const xvector<>& q_double_prime) {
    const auto gain_at = [&](const xvector<>& qq, const xvector<>& qp) {
        const auto J = chain.linear_jacobian(qq);
        std::array<double, 3> v{0.0, 0.0, 0.0};
        for (std::size_t r = 0; r < 3; ++r) {
            for (std::size_t j = 0; j < qp.size(); ++j) {
                v[r] += J(r, j) * qp(j);
            }
        }
        return std::sqrt((v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]));
    };
    const auto analytic = chain.linear_velocity_gain_at(q, q_prime, q_double_prime);
    BOOST_CHECK_CLOSE(analytic.gain_per_arc_unit, gain_at(q, q_prime), 1e-9);

    const double h = 1e-6;
    const double numeric =
        (gain_at(q + (h * q_prime), q_prime + (h * q_double_prime)) - gain_at(q - (h * q_prime), q_prime - (h * q_double_prime))) /
        (2.0 * h);
    BOOST_CHECK_SMALL(analytic.d_gain_ds - numeric, 1e-6);
}

}  // namespace

BOOST_AUTO_TEST_CASE(agrees_with_the_tensor_path) {
    const xvector<> q6 = {0.1, -0.5, 1.2, -0.3, 0.6, -0.1};
    const xvector<> q6_prime = {0.2, 0.7, -0.4, 0.5, -0.6, 0.3};
    const xvector<> q6_double_prime = {-0.1, 0.3, 0.2, -0.5, 0.4, -0.2};
    const xvector<> q3 = {0.4, -0.9, 0.3};
    const xvector<> q3_prime = {0.6, -0.2, 0.5};
    const xvector<> q3_double_prime = {0.1, 0.4, -0.3};

    const auto compare = [](const xmatrix<>& table, const xvector<>& q, const xvector<>& q_prime, const xvector<>& q_double_prime) {
        const auto parsed = kinematic_chain::from(table);
        const auto built = build_chain(chain_from_table(table));
        BOOST_CHECK_SMALL(matrix_diff_norm(parsed.jacobian(q), built.jacobian(q)), 1e-12);
        BOOST_CHECK_SMALL(vec3_diff_norm(parsed.end_position(q), built.end_position(q)), 1e-12);
        const auto parsed_gain = parsed.linear_velocity_gain_at(q, q_prime, q_double_prime);
        const auto built_gain = built.linear_velocity_gain_at(q, q_prime, q_double_prime);
        BOOST_CHECK_SMALL(parsed_gain.gain_per_arc_unit - built_gain.gain_per_arc_unit, 1e-12);
        BOOST_CHECK_SMALL(parsed_gain.d_gain_ds - built_gain.d_gain_ds, 1e-12);
    };
    compare(sixdof_arm_table(), q6, q6_prime, q6_double_prime);
    compare(rotated_origins_table(), q3, q3_prime, q3_double_prime);
}

BOOST_AUTO_TEST_CASE(end_position_matches_reference_forward_kinematics) {
    for (const auto& [joints, q] :
         {std::pair{scara_chain(), xvector<>{0.4, -1.1, 0.12, 0.7}}, std::pair{mixed_chain(), xvector<>{0.3, -0.6, 0.2, 1.1}}}) {
        const auto T = forward_transform(joints, q);
        BOOST_CHECK_SMALL(vec3_diff_norm(build_chain(joints).end_position(q), {T(0, 3), T(1, 3), T(2, 3)}), 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(jacobian_with_prismatic_matches_numerical) {
    for (const auto& [joints, q] : {std::pair{scara_chain(), xvector<>{0.4, -1.1, 0.12, 0.7}},
                                    std::pair{scara_chain(), xvector<>{0.0, 0.0, 0.0, 0.0}},
                                    std::pair{mixed_chain(), xvector<>{0.3, -0.6, 0.2, 1.1}},
                                    std::pair{mixed_chain(), xvector<>{-0.8, 1.4, -0.3, -0.5}}}) {
        BOOST_CHECK_SMALL(matrix_diff_norm(build_chain(joints).jacobian(q), numerical_jacobian(joints, q)), 1e-6);
    }
}

BOOST_AUTO_TEST_CASE(prismatic_column_is_the_world_axis) {
    // At q = 0 the SCARA's quill points straight down, so its linear column is -z and it
    // contributes nothing angular.
    const auto J = build_chain(scara_chain()).jacobian(xvector<>{0.0, 0.0, 0.0, 0.0});
    for (std::size_t r = 0; r < 6; ++r) {
        BOOST_CHECK_SMALL(J(r, 2) - (r == 2 ? -1.0 : 0.0), 1e-15);
    }
}

BOOST_AUTO_TEST_CASE(linear_velocity_gain_with_prismatic_matches_numerical) {
    check_gain_matches_numerical(build_chain(scara_chain()), {0.4, -1.1, 0.12, 0.7}, {0.3, -0.5, 0.8, 0.2}, {0.1, 0.2, -0.4, 0.3});
    check_gain_matches_numerical(build_chain(mixed_chain()), {0.3, -0.6, 0.2, 1.1}, {0.5, 0.4, -0.7, 0.2}, {-0.3, 0.1, 0.2, 0.6});
}

// build() normalizes, so a quaternion and an axis given at some other length describe the same
// chain.
BOOST_AUTO_TEST_CASE(normalizes_quaternions_and_axes) {
    auto scaled = mixed_chain();
    for (auto& j : scaled) {
        for (auto& c : j.wxyz) {
            c *= 3.7;
        }
        for (auto& c : j.axis) {
            c *= 0.25;
        }
    }
    const xvector<> q = {0.3, -0.6, 0.2, 1.1};
    BOOST_CHECK_SMALL(matrix_diff_norm(build_chain(mixed_chain()).jacobian(q), build_chain(scaled).jacobian(q)), 1e-14);
}

// We never consult a position limit, so the representative a driver reports for an angle does not
// matter. A continuous joint is the one a driver is liable to report as an accumulated angle.
BOOST_AUTO_TEST_CASE(indifferent_to_full_turn_offsets) {
    const auto chain = build_chain(scara_chain());
    const xvector<> q = {0.4, -1.1, 0.12, 0.7};
    const xvector<> q_offset = {0.4 + (4.0 * std::numbers::pi), -1.1 - (2.0 * std::numbers::pi), 0.12, 0.7 + (6.0 * std::numbers::pi)};
    BOOST_CHECK_SMALL(matrix_diff_norm(chain.jacobian(q), chain.jacobian(q_offset)), 1e-13);
}

BOOST_AUTO_TEST_CASE(fixed_joints_consume_no_q) {
    const auto chain = build_chain(scara_chain());
    BOOST_CHECK_EQUAL(chain.jacobian(xvector<>{0.0, 0.0, 0.0, 0.0}).shape(1), 4U);
    BOOST_CHECK_THROW(static_cast<void>(chain.jacobian(xvector<>{0.0, 0.0, 0.0, 0.0, 0.0})), std::invalid_argument);
    BOOST_CHECK_THROW(static_cast<void>(chain.end_position(xvector<>{0.0, 0.0, 0.0})), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(build_refuses_what_it_cannot_represent) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const auto refuses = [](const test_chain& joints) { BOOST_CHECK_THROW(static_cast<void>(build_chain(joints)), std::invalid_argument); };

    refuses({});
    refuses({{.kind = test_joint_kind::k_rotational, .xyz = {}, .wxyz = k_identity_rotation, .axis = {0, 0, 0}}});
    refuses({{.kind = test_joint_kind::k_translational, .xyz = {}, .wxyz = k_identity_rotation, .axis = {0, 0, 0}}});
    refuses({{.kind = test_joint_kind::k_rotational, .xyz = {}, .wxyz = k_identity_rotation, .axis = {0, nan, 1}}});
    refuses({{.kind = test_joint_kind::k_translational, .xyz = {}, .wxyz = k_identity_rotation, .axis = {inf, 0, 0}}});
    refuses({{.kind = test_joint_kind::k_fixed, .xyz = {}, .wxyz = {0, 0, 0, 0}, .axis = {}}});
    refuses({{.kind = test_joint_kind::k_fixed, .xyz = {}, .wxyz = {nan, 0, 0, 0}, .axis = {}}});
    refuses({{.kind = test_joint_kind::k_fixed, .xyz = {}, .wxyz = {1e200, 1e200, 0, 0}, .axis = {}}});
    refuses({{.kind = test_joint_kind::k_fixed, .xyz = {0, inf, 0}, .wxyz = k_identity_rotation, .axis = {}}});

    // A fixed joint has no axis to validate, and a fixed-only chain is legal: it has no q at all.
    const auto fixed_only =
        build_chain({{.kind = test_joint_kind::k_fixed, .xyz = {1, 2, 3}, .wxyz = k_identity_rotation, .axis = {0, 0, 0}}});
    BOOST_CHECK_SMALL(vec3_diff_norm(fixed_only.end_position(xvector<>(xt::zeros<double>({std::size_t{0}}))), {1, 2, 3}), 1e-15);
}

BOOST_AUTO_TEST_SUITE_END()
