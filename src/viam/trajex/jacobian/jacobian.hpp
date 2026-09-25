#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <viam/trajex/types/xt.hpp>

namespace viam::trajex::jacobian {

// TODO(RSDK-14104): Remove duplicated code that exists in the viam-cpp-sdk

///
/// A validated URDF-style serial kinematic chain, parsed from an (n, 10)
/// model-table tensor and held in chain order.
///
class kinematic_chain {
   public:
    ///
    /// Parses an (n, 10) tensor in the viam::sdk::ModelTable format into a
    /// kinematic chain.
    ///
    /// @param tensor (n, 10) tensor in the viam::sdk::ModelTable format
    /// @return Validated kinematic chain
    /// @throws std::invalid_argument on non-2D input, wrong column count, a
    ///         non-integer joint-type encoding, an empty table, an unsupported
    ///         joint type (continuous or prismatic), or a revolute row with
    ///         zero-magnitude axis
    ///
    /// Takes the table as whatever the caller holds it in rather than as an xmatrix. A
    /// parameter of concrete type would accept anything convertible to one, and that
    /// conversion fabricates or discards extents to reach rank two -- so a table of the wrong
    /// rank would arrive already reshaped, and the check below would be inspecting a shape
    /// this function invented. Deduction leaves the caller's type alone and makes the check
    /// mean something. Anything whose rank is fixed at something other than two is refused
    /// outright; anything that settles its rank at runtime reaches the check.
    ///
    template <rank_or_runtime<2> T>
    [[nodiscard]] static kinematic_chain from(const T& tensor) {
        require_rank_two_(tensor.dimension());
        return from_rank_two_(tensor);
    }

    ///
    /// Computes the geometric Jacobian at joint positions q.
    ///
    /// @param q (N_actuated,) vector with one element per revolute row in the
    ///        table, in chain order. Fixed rows do not consume a q entry.
    /// @return A (6, N_actuated) matrix where rows 0..2 are the
    ///         linear-velocity columns J_v_i = w_i x (p_e - p_i) and rows 3..5
    ///         are the angular-velocity columns J_w_i = w_i, with w_i the
    ///         world-frame axis of revolute joint i, p_i its world position,
    ///         and p_e the end-effector position.
    /// @throws std::invalid_argument on q-size mismatch, or if q is not 1-dimensional
    ///
    /// Deduces q rather than naming it, for the reason given on from(): a concrete parameter
    /// would reshape a wrong-ranked argument on the way in. Deduction also means a caller who
    /// already holds an xvector passes it with no copy, which matters because the integrator
    /// calls these once per step.
    ///
    template <rank_or_runtime<1> Q>
    [[nodiscard]] xmatrix<> jacobian(const Q& q) const {
        require_rank_one_(q.dimension());
        return jacobian_rank_one_(q);
    }

    ///
    /// Computes the linear-velocity block of the geometric Jacobian at joint
    /// positions q.
    ///
    /// @param q (N_actuated,) vector with one element per revolute row in the
    ///        table, in chain order. Fixed rows do not consume a q entry.
    /// @return A (3, N_actuated) matrix of linear-velocity columns.
    /// @throws std::invalid_argument on q-size mismatch, or if q is not 1-dimensional
    ///
    template <rank_or_runtime<1> Q>
    [[nodiscard]] xmatrix<> linear_jacobian(const Q& q) const {
        require_rank_one_(q.dimension());
        return linear_jacobian_rank_one_(q);
    }

    /// Linear velocity gain ||J_v*f'||: task-space length per unit of path arc length, in
    /// whatever length unit the model table uses, with its rate of change along the path.
    struct linear_velocity_gain {
        double gain_per_arc_unit;
        double d_gain_ds;
    };

    ///
    /// Linear velocity gain ||J_v*f'|| and its path derivative d/ds||J_v*f'||, where J_v is the
    /// (3, N_actuated) linear-velocity Jacobian, f' = q_prime is the path tangent in joint
    /// space, and f'' = q_double_prime is the path curvature. Used to evaluate the TCP velocity
    /// limit curve and its slope.
    ///
    /// @param q (N_actuated,) joint positions, in chain order
    /// @param q_prime (N_actuated,) path tangent dq/ds
    /// @param q_double_prime (N_actuated,) path curvature d^2q/ds^2
    /// @return the gain and its s-derivative
    /// @throws std::invalid_argument on a size mismatch, or if any argument is not 1-dimensional
    ///
    template <rank_or_runtime<1> Q, rank_or_runtime<1> QP, rank_or_runtime<1> QPP>
    [[nodiscard]] linear_velocity_gain linear_velocity_gain_at(const Q& q, const QP& q_prime, const QPP& q_double_prime) const {
        require_rank_one_(q.dimension());
        require_rank_one_(q_prime.dimension());
        require_rank_one_(q_double_prime.dimension());
        return linear_velocity_gain_at_rank_one_(q, q_prime, q_double_prime);
    }

   private:
    // URDF joint type, restricted to arm-relevant joints. Underlying values
    // are the column-9 wire encoding accepted by `from`, and match
    // viam::sdk::ModelTable::JointType.
    enum class joint_type_ : std::uint8_t {
        k_revolute = 0,
        k_continuous = 1,
        k_prismatic = 2,
        k_fixed = 3,
    };

    // One row of the model table: the per-joint URDF fields. xyz/rpy are the
    // joint origin relative to the parent link (rpy is fixed-axis XYZ); axis
    // is the joint axis in the local frame.
    struct joint_row_ {
        std::array<double, 3> xyz{};
        std::array<double, 3> rpy{};
        std::array<double, 3> axis{};
        joint_type_ type = joint_type_::k_fixed;
    };

    // Per-revolute-joint world-frame axes and origins plus the end-effector
    // position.
    struct chain_state_;

    // Constant per-row kinematics derived once at construction: the
    // parent-to-joint link transform (row-major 4x4) and, for revolute rows,
    // the normalized local joint axis. The forward-kinematics walk runs per
    // integration step, so its q-independent terms are not recomputed there.
    struct row_constants_ {
        std::array<double, 16> link_tf{};
        std::array<double, 3> unit_axis{};
    };

    // Validates the rows (joint types, axes) and counts actuated joints, then
    // precomputes the per-row constants; all public construction funnels
    // through here via `from`.
    explicit kinematic_chain(std::vector<joint_row_> rows);

    // Split out of `from` so the parsing stays in the source file. Converting to an xmatrix
    // here is safe in a way it would not have been at the parameter, because the rank has
    // been established by then; for a caller who already had one it is not even a copy.
    static void require_rank_two_(std::size_t dimension);
    static void require_rank_one_(std::size_t dimension);

    static kinematic_chain from_rank_two_(const xmatrix<>& tensor);

    // The real work, behind the rank checks above. A caller who already holds an xvector binds
    // straight through with no copy; one who does not pays a conversion that is safe by then.
    [[nodiscard]] xmatrix<> jacobian_rank_one_(const xvector<>& q) const;
    [[nodiscard]] xmatrix<> linear_jacobian_rank_one_(const xvector<>& q) const;
    [[nodiscard]] linear_velocity_gain linear_velocity_gain_at_rank_one_(const xvector<>& q,
                                                                         const xvector<>& q_prime,
                                                                         const xvector<>& q_double_prime) const;

    // Evaluates the forward kinematics at joint positions q, capturing the
    // per-joint quantities the Jacobian assemblies need. Throws
    // std::invalid_argument on q-size mismatch.
    chain_state_ compute_chain_state_(const xvector<>& q) const;

    std::vector<joint_row_> rows_;
    std::vector<row_constants_> constants_;
    std::size_t actuated_count_ = 0;
};

}  // namespace viam::trajex::jacobian
