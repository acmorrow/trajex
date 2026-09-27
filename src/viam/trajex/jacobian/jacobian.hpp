#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
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
    [[nodiscard]] static kinematic_chain from(const xmatrix<>& tensor);

    ///
    /// Builds from a model table the caller holds in something other than an xmatrix.
    ///
    /// A parameter of concrete type accepts anything convertible to one, and that conversion
    /// fabricates or discards extents to reach rank two -- so a table of the wrong rank would
    /// arrive already reshaped, and a check would be inspecting a shape this function invented.
    /// Deducing the caller's type leaves it alone and makes the check mean something. Anything
    /// whose rank is fixed at something other than two is refused outright; anything that
    /// settles its rank at runtime reaches the check and then converts once, here, where the
    /// conversion is visible.
    ///
    /// A caller already holding an xmatrix binds to the overload above and never reaches this.
    ///
    /// @throws std::invalid_argument if the table is not 2-dimensional, plus everything the
    ///         overload above throws
    ///
    template <xrank_same_as_or_dynamic<xmatrix<>> T>
    [[nodiscard]] static kinematic_chain from(const T& tensor) {
        if (!xrank_is_same_as<xmatrix<>>(tensor)) {
            throw std::invalid_argument("viam::trajex::jacobian: expected 2D model-table tensor, got " +
                                        std::to_string(tensor.dimension()) + "D");
        }
        return from(xmatrix<>(tensor));
    }

    // A rank fixed at anything but two would otherwise reach the xmatrix overload by conversion,
    // which fabricates or discards extents rather than failing. Deleting it here wins over that
    // conversion on an exact match, so the caller sees a refusal instead of an invented shape.
    template <xexpression_like T>
        requires(!xrank_same_as_or_dynamic<T, xmatrix<>>)
    static kinematic_chain from(const T&) = delete;

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
    /// The integrator calls this once per step holding an xvector, and binds here with no copy.
    ///
    [[nodiscard]] xmatrix<> jacobian(const xvector<>& q) const;

    ///
    /// Computes the geometric Jacobian at joint positions q held in something other than an
    /// xvector.
    ///
    /// Deduces q rather than naming it, for the reason given on from(): a concrete parameter
    /// would reshape a wrong-ranked argument on the way in.
    ///
    /// @throws std::invalid_argument if q is not 1-dimensional, plus everything the overload
    ///         above throws
    ///
    template <xrank_same_as_or_dynamic<xvector<>> Q>
    [[nodiscard]] xmatrix<> jacobian(const Q& q) const {
        require_rank_one_(q);
        return jacobian(xvector<>(q));
    }

    // Refused for the reason given on from(): a rank fixed at anything but one would otherwise
    // convert into the xvector overload rather than fail.
    template <xexpression_like Q>
        requires(!xrank_same_as_or_dynamic<Q, xvector<>>)
    xmatrix<> jacobian(const Q&) const = delete;

    ///
    /// Computes the linear-velocity block of the geometric Jacobian at joint
    /// positions q.
    ///
    /// @param q (N_actuated,) vector with one element per revolute row in the
    ///        table, in chain order. Fixed rows do not consume a q entry.
    /// @return A (3, N_actuated) matrix of linear-velocity columns.
    /// @throws std::invalid_argument on q-size mismatch
    ///
    [[nodiscard]] xmatrix<> linear_jacobian(const xvector<>& q) const;

    ///
    /// As above, for a q the caller holds in something other than an xvector.
    ///
    /// @throws std::invalid_argument if q is not 1-dimensional, plus everything the overload
    ///         above throws
    ///
    template <xrank_same_as_or_dynamic<xvector<>> Q>
    [[nodiscard]] xmatrix<> linear_jacobian(const Q& q) const {
        require_rank_one_(q);
        return linear_jacobian(xvector<>(q));
    }

    template <xexpression_like Q>
        requires(!xrank_same_as_or_dynamic<Q, xvector<>>)
    xmatrix<> linear_jacobian(const Q&) const = delete;

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
    /// @throws std::invalid_argument on a size mismatch
    ///
    [[nodiscard]] linear_velocity_gain linear_velocity_gain_at(const xvector<>& q,
                                                               const xvector<>& q_prime,
                                                               const xvector<>& q_double_prime) const;

    ///
    /// As above, for arguments the caller holds in something other than an xvector.
    ///
    /// All three deduce independently, so a caller may mix. Any argument already an xvector is
    /// converted to itself; passing three of them binds to the overload above instead.
    ///
    /// @throws std::invalid_argument if any argument is not 1-dimensional, plus everything the
    ///         overload above throws
    ///
    template <xrank_same_as_or_dynamic<xvector<>> Q, xrank_same_as_or_dynamic<xvector<>> QP, xrank_same_as_or_dynamic<xvector<>> QPP>
    [[nodiscard]] linear_velocity_gain linear_velocity_gain_at(const Q& q, const QP& q_prime, const QPP& q_double_prime) const {
        require_rank_one_(q);
        require_rank_one_(q_prime);
        require_rank_one_(q_double_prime);
        return linear_velocity_gain_at(xvector<>(q), xvector<>(q_prime), xvector<>(q_double_prime));
    }

    // Refused if any one argument is a rank fixed at something other than one, since that
    // argument alone would convert into the xvector overload rather than fail.
    template <xexpression_like Q, xexpression_like QP, xexpression_like QPP>
        requires(!(xrank_same_as_or_dynamic<Q, xvector<>> && xrank_same_as_or_dynamic<QP, xvector<>> &&
                   xrank_same_as_or_dynamic<QPP, xvector<>>))
    linear_velocity_gain linear_velocity_gain_at(const Q&, const QP&, const QPP&) const = delete;

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

    // Defined here rather than in the source file because the deduced overloads above are
    // instantiated in the caller's translation unit: a definition there would have to be
    // exported, putting a private helper into the ABI. Defined inline it is emitted weakly
    // wherever it is used and never reaches the export table.
    static void require_rank_one_(const auto& value) {
        if (!xrank_is_same_as<xvector<>>(value)) {
            throw std::invalid_argument("viam::trajex::jacobian: expected a 1D joint vector, got " + std::to_string(value.dimension()) +
                                        "D");
        }
    }

    // Evaluates the forward kinematics at joint positions q, capturing the
    // per-joint quantities the Jacobian assemblies need. Throws
    // std::invalid_argument on q-size mismatch.
    chain_state_ compute_chain_state_(const xvector<>& q) const;

    std::vector<joint_row_> rows_;
    std::vector<row_constants_> constants_;
    std::size_t actuated_count_ = 0;
};

}  // namespace viam::trajex::jacobian
