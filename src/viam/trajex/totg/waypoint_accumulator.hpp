#pragma once

#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#if __has_include(<xtensor/views/xview.hpp>)
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xslice.hpp>
#include <xtensor/views/xview.hpp>
#else
#include <xtensor/xadapt.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xview.hpp>
#endif

#include <viam/trajex/types/xt.hpp>

namespace viam::trajex::totg {

// What this class needs of a waypoint set is that it be rank two, dense, row-major and hold
// the same element type a row does. xmatrix is the obvious thing that qualifies but not the
// only one, and the two concepts below differ only in when the rank becomes known.

///
/// A waypoint set already known at compile time to be shaped like one.
///
/// Admits an adaptor over a caller's own buffer and a fixed-shape array as readily as an
/// xmatrix, none of which need converting or copying to be read row by row.
///
template <typename T>
concept fixed_rank_waypoints = same_rank_as<T, xmatrix<>> && dense_rows_of<T, xmatrix<>::value_type>;

///
/// A waypoint set that settles its rank at runtime and whose rows can still be adapted.
///
/// Worth admitting under a check rather than refusing outright: the rank such a thing will turn
/// out to have is not knowable at the call, and is usually the right one.
///
template <typename T>
concept checkable_waypoints = dynamically_ranked<T> && dense_rows_of<T, xmatrix<>::value_type>;

///
/// Accumulator for building up a sequence of waypoints.
///
/// **Lifetime requirements**: All waypoint arrays passed to this object must outlive it.
///
/// **Move semantics**: This class is move-only (not copyable).
///
/// Usage:
/// @code
///   waypoint_accumulator waypoints(initial_waypoints);
///   waypoints.add_waypoints(more_waypoints);
/// @endcode
///
class waypoint_accumulator {
   public:
    ///
    /// View type for individual waypoints.
    ///
    /// An adaptor over the row rather than a view of the array it came from, because a view
    /// carries the container type and so admits rows from one container only. A row of any
    /// row-major rank-2 array is contiguous, so the adaptor spells every such row the same way.
    ///
    using value_t = xmatrix<>::value_type;
    using waypoint_view_t = decltype(xt::adapt(std::declval<const value_t*>(), std::size_t{}, xt::no_ownership()));

    ///
    /// Constructs with initial waypoints.
    ///
    /// @param waypoints 2D array (num_waypoints, num_joints)
    /// @note The waypoints array must outlive the waypoint_accumulator object
    ///
    explicit waypoint_accumulator(const xmatrix<>& waypoints);
    explicit waypoint_accumulator(xmatrix<>&& waypoints) = delete;

    ///
    /// Constructs with initial waypoints held in some other rank-2 array.
    ///
    /// @param waypoints 2D array (num_waypoints, num_joints)
    /// @note The waypoints array must outlive the waypoint_accumulator object
    ///
    /// The xmatrix overload above is the one to reach for and the one a near miss will be
    /// diagnosed against; this exists so that a caller already holding their waypoints in an
    /// adaptor over their own memory, or in a fixed-shape array, need not copy them first.
    ///
    template <fixed_rank_waypoints T>
    explicit waypoint_accumulator(const T& waypoints) {
        require_non_empty_(waypoints);
        dof_ = waypoints.shape()[1];
        append_rows_(waypoints);
    }

    template <fixed_rank_waypoints T>
    explicit waypoint_accumulator(const T&&) = delete;

    ///
    /// Constructs with initial waypoints whose rank is only known at runtime.
    ///
    /// @param waypoints Array that must turn out to be (num_waypoints, num_joints)
    /// @throws std::invalid_argument if the array is not 2-dimensional
    /// @note The waypoints array must outlive the waypoint_accumulator object
    ///
    template <checkable_waypoints T>
    explicit waypoint_accumulator(const T& waypoints) {
        require_rank_two_(waypoints);
        require_non_empty_(waypoints);
        dof_ = waypoints.shape()[1];
        append_rows_(waypoints);
    }

    // The rows point into the argument, so a temporary must be refused here as it is for
    // xmatrix above. A const rvalue reference is not a forwarding reference, so this takes
    // rvalues only and leaves the lvalue overload alone.
    template <checkable_waypoints T>
    explicit waypoint_accumulator(const T&&) = delete;

    // What is left is what cannot be read as waypoints at all: a rank fixed at something other
    // than two, a layout whose rows are not contiguous, the wrong element type, or an
    // expression with no storage behind it. Converting any of those would invent or discard
    // extents rather than fail, so refuse it and make the caller reshape deliberately.
    // See types/xt.hpp.
    template <tensor_like T>
        requires(!fixed_rank_waypoints<T> && !checkable_waypoints<T>)
    explicit waypoint_accumulator(const T&) = delete;

    ///
    /// Constructs with a single waypoint view.
    ///
    /// @param first_waypoint View to first waypoint (establishes DOF)
    /// @note The underlying array must outlive the waypoint_accumulator object
    ///
    explicit waypoint_accumulator(const waypoint_view_t& first_waypoint);

    ///
    /// Copy constructs a waypoint_accumulator.
    ///
    waypoint_accumulator(const waypoint_accumulator&);

    ///
    /// Move constructs a waypoint_accumulator.
    ///
    waypoint_accumulator(waypoint_accumulator&&) noexcept;

    ///
    /// Copy assigns a waypoint_accumulator.
    ///
    /// @return Reference to this
    ///
    waypoint_accumulator& operator=(const waypoint_accumulator&);

    ///
    /// Move assigns a waypoint_accumulator.
    ///
    /// @return Reference to this
    ///
    waypoint_accumulator& operator=(waypoint_accumulator&&) noexcept;

    ///
    /// Adds additional waypoints.
    ///
    /// @param waypoints 2D array where each row is a waypoint, shape (num_waypoints, num_joints)
    /// @return Reference to this for method chaining
    /// @note The waypoints array must outlive the waypoint_accumulator object
    ///
    waypoint_accumulator& add_waypoints(const xmatrix<>& waypoints);
    waypoint_accumulator& add_waypoints(xmatrix<>&& waypoints) = delete;

    ///
    /// Adds additional waypoints held in some other rank-2 array.
    ///
    /// @param waypoints 2D array where each row is a waypoint, shape (num_waypoints, num_joints)
    /// @return Reference to this for method chaining
    /// @throws std::invalid_argument if the DOF differs
    /// @note The waypoints array must outlive the waypoint_accumulator object
    ///
    template <fixed_rank_waypoints T>
    waypoint_accumulator& add_waypoints(const T& waypoints) {
        require_matching_dof_(waypoints.shape()[1]);
        append_rows_(waypoints);
        return *this;
    }

    template <fixed_rank_waypoints T>
    waypoint_accumulator& add_waypoints(const T&&) = delete;

    ///
    /// Adds additional waypoints whose rank is only known at runtime.
    ///
    /// @param waypoints Array that must turn out to be (num_waypoints, num_joints)
    /// @return Reference to this for method chaining
    /// @throws std::invalid_argument if the array is not 2-dimensional or the DOF differs
    /// @note The waypoints array must outlive the waypoint_accumulator object
    ///
    template <checkable_waypoints T>
    waypoint_accumulator& add_waypoints(const T& waypoints) {
        require_rank_two_(waypoints);
        require_matching_dof_(waypoints.shape()[1]);
        append_rows_(waypoints);
        return *this;
    }

    template <checkable_waypoints T>
    waypoint_accumulator& add_waypoints(const T&&) = delete;

    template <tensor_like T>
        requires(!fixed_rank_waypoints<T> && !checkable_waypoints<T>)
    waypoint_accumulator& add_waypoints(const T&) = delete;

    ///
    /// Adds a single waypoint view.
    ///
    /// @param waypoint View to waypoint to add
    /// @return Reference to this for method chaining
    /// @throws std::invalid_argument if waypoint DOF doesn't match
    /// @note The underlying array must outlive the waypoint_accumulator object
    ///
    waypoint_accumulator& add_waypoint(const waypoint_view_t& waypoint);

    ///
    /// Gets the number of degrees of freedom.
    ///
    /// @return Number of DOF
    ///
    size_t dof() const noexcept;

    ///
    /// Gets the number of waypoints.
    ///
    /// @return Total number of waypoints
    ///
    size_t size() const noexcept;

    ///
    /// Checks if empty.
    ///
    /// @return True if no waypoints
    ///
    bool empty() const noexcept;

    ///
    /// Iterator type for waypoint views.
    ///
    using const_iterator = std::vector<waypoint_view_t>::const_iterator;

    ///
    /// Value type for waypoint views.
    ///
    using value_type = waypoint_view_t;

    ///
    /// Size type.
    ///
    using size_type = std::size_t;

    ///
    /// Gets begin iterator.
    ///
    /// @return Iterator to first waypoint
    ///
    const_iterator begin() const noexcept;

    ///
    /// Gets end iterator.
    ///
    /// @return Iterator past last waypoint
    ///
    const_iterator end() const noexcept;

    ///
    /// Gets const begin iterator.
    ///
    /// @return Const iterator to first waypoint (same as begin())
    ///
    const_iterator cbegin() const noexcept;

    ///
    /// Gets const end iterator.
    ///
    /// @return Const iterator past last waypoint (same as end())
    ///
    const_iterator cend() const noexcept;

    ///
    /// Accesses waypoint by index (no bounds checking).
    ///
    /// @param i Index of waypoint
    /// @return Reference to waypoint view
    ///
    const waypoint_view_t& operator[](size_t i) const;

    ///
    /// Accesses waypoint by index with bounds checking.
    ///
    /// @param i Index of waypoint
    /// @return Reference to waypoint view
    /// @throws std::out_of_range if i >= size()
    ///
    const waypoint_view_t& at(size_t i) const;

    ///
    /// Accesses the last waypoint.
    ///
    /// @return Reference to last waypoint view
    ///
    const waypoint_view_t& back() const noexcept;

    ///
    /// Removes the last waypoint.
    ///
    void pop_back() noexcept;

   private:
    // Shared by the runtime-checked overloads above and by the xmatrix ones in the source file,
    // so that a waypoint set is admitted on the same terms however its rank became known.
    static void require_rank_two_(const auto& waypoints) {
        if (waypoints.dimension() != 2) {
            throw std::invalid_argument{"Waypoints must be 2-dimensional"};
        }
    }

    static void require_non_empty_(const auto& waypoints) {
        if (waypoints.shape()[0] == 0) {
            throw std::invalid_argument{"Waypoints cannot be empty"};
        }
    }

    void require_matching_dof_(size_t dof) const {
        if (dof != dof_) {
            throw std::invalid_argument{"Waypoints dimensions must match existing DOF"};
        }
    }

    // Each row is contiguous because the source is dense and row-major, which is what lets the
    // row be adapted where it lies instead of copied.
    void append_rows_(const auto& waypoints) {
        const size_t num_waypoints = waypoints.shape()[0];
        for (size_t i = 0; i < num_waypoints; ++i) {
            waypoints_.push_back(xt::adapt(&waypoints(i, 0), dof_, xt::no_ownership()));
        }
    }

    size_t dof_;
    std::vector<waypoint_view_t> waypoints_;
};

// Verify waypoint_accumulator satisfies C++20 range concepts
static_assert(std::ranges::range<waypoint_accumulator>);
static_assert(std::ranges::sized_range<waypoint_accumulator>);
static_assert(std::ranges::forward_range<waypoint_accumulator>);

}  // namespace viam::trajex::totg
