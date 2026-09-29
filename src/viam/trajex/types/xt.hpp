#pragma once

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#else
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#endif

// The array types trajex holds geometry in. Extent stays dynamic because we don't know the
// degrees of freedom until runtime.

namespace viam::trajex {

///
/// One-dimensional array: a configuration, or a quantity shaped like one.
///
template <typename T = double>
using xvector = xt::xtensor<T, 1>;

///
/// Two-dimensional array: a stack of xvector rows.
///
/// Converting to or from an xvector is undefined behavior, and silent.
///
template <typename T = double>
using xmatrix = xt::xtensor<T, 2>;

///
/// Satisfied by anything xtensor will evaluate, which is also what converts to the array types
/// above uninvited.
///
template <typename T>
concept xexpression_like = xt::is_xexpression<T>::value;

///
/// The rank of a type that settles its rank at runtime, like xarray or any view.
///
inline constexpr std::size_t xrank_dynamic = xt::get_rank<xt::xarray<double>>::value;

///
/// The rank xtensor reports for T, which is xrank_dynamic when it does not fix one.
///
template <typename T>
inline constexpr std::size_t xrank_of = xt::get_rank<std::decay_t<T>>::value;

///
/// Satisfied when xtensor fixes T's rank at compile time.
///
template <typename T>
concept xranked_statically = xexpression_like<T> && xt::has_fixed_rank_t<T>::value;

///
/// Satisfied when T carries its rank as a runtime property, so only T itself can report it.
///
template <typename T>
concept xranked_dynamically = xexpression_like<T> && !xt::has_fixed_rank_t<T>::value;

///
/// Satisfied when T and U both fix their rank and the two agree.
///
template <typename T, typename U>
concept xrank_same_as = xranked_statically<T> && xranked_statically<U> && (xrank_of<T> == xrank_of<U>);

///
/// Satisfied when T already has a rank like U's, or won't know until runtime and can be checked
/// then.
///
/// Put it on a deduced parameter and check in the body. On a concrete parameter the conversion
/// has already happened.
///
template <typename T, typename U>
concept xrank_same_as_or_dynamic = xrank_same_as<T, U> || xranked_dynamically<T>;

///
/// True when value's rank matches the rank T fixes.
///
/// The runtime half of xrank_same_as, for the check a deduced overload owes in its body.
///
template <xranked_statically T>
[[nodiscard]] bool xrank_is_same_as(const auto& value) {
    return value.dimension() == xrank_of<T>;
}

///
/// An array of rank N that nobody can have handed the wrong rank.
///
/// For the places with no body to check in: a public member the caller assigns into, or the
/// return type of a callback the caller writes.
///
/// Rank already N, we take it. Not known until runtime, we check and throw. Fixed at anything
/// else, you get a compile error.
///
/// Reading is const and unguarded. It converts to a const reference. Use the named accessors
/// where that conversion isn't seen through.
///
template <typename T, std::size_t N = xrank_of<T>>
class xrank_checked {
   public:
    static_assert(N != xrank_dynamic,
                  "xrank_checked cannot infer a rank to check against from a dynamically ranked T; name one explicitly");
    static_assert(!xranked_statically<T> || xrank_of<T> == N, "xrank_checked's rank must agree with the rank T fixes");

    /// Holds a default-constructed T. That's an empty array, not a missing one.
    xrank_checked() = default;

    ///
    /// Takes ownership of a value that is already the held type.
    ///
    /// Without this a callback returning T by value hits the converting constructor below,
    /// binds a const reference, and copies where it could move. An allocation per call on a
    /// hot path.
    ///
    xrank_checked(T&& value) noexcept(std::is_nothrow_move_constructible_v<T>) : value_(std::move(value)) {}

    ///
    /// Takes a source whose rank is already known to be right. Nothing to check.
    ///
    /// Deduced so an rvalue gets consumed. Whether there is anything to consume is xtensor's
    /// business: an xarray gives up its storage, an xtensor_fixed copies.
    ///
    template <typename U>
        requires(xranked_statically<U> && xrank_of<U> == N && std::constructible_from<T, U &&>)
    xrank_checked(U&& value) : value_(std::forward<U>(value)) {}

    ///
    /// Takes a source that settles its rank at runtime.
    ///
    /// @throws std::invalid_argument if the rank turns out not to be N
    ///
    template <typename U>
        requires(xranked_dynamically<U> && std::constructible_from<T, U &&>)
    xrank_checked(U&& value) : value_(checked_(std::forward<U>(value))) {}

    // A rank fixed at something else is the UB case, so refuse it where the caller can see it.
    // U is bounded by xranked_statically so this doesn't eat construction from unrelated types,
    // or from xrank_checked itself, which would take out the copy constructor.
    template <typename U>
        requires(xranked_statically<U> && xrank_of<U> != N)
    xrank_checked(U&&) = delete;

    // Every way of reading refuses rvalues, the conversion below included. Binding to a
    // temporary's storage gives you a reference that dies at the semicolon, and nothing warns
    // you: lifetime extension does not reach through a conversion function's return.
    const T& get() const& noexcept {
        return value_;
    }
    const T& get() const&& = delete;

    const T& operator*() const& noexcept {
        return value_;
    }
    const T& operator*() const&& = delete;

    const T* operator->() const& noexcept {
        return &value_;
    }
    const T* operator->() const&& = delete;

    ///
    /// Consumes the value, moving it out.
    ///
    /// The conversion only gives you a const reference, so building your own storage from one of
    /// these copies. If you're done with the value, take it instead. The rvalue qualifier makes
    /// you say so at the call.
    ///
    T take() && {
        return std::move(value_);
    }

    operator const T&() const& noexcept {
        return value_;
    }
    operator const T&() const&& = delete;

   private:
    // Check before converting, not after: the conversion is what fabricates the extent. Returns
    // by value because the caller may have passed a temporary.
    template <typename U>
    static T checked_(U&& value) {
        if (value.dimension() != N) {
            throw std::invalid_argument{"expected a rank-" + std::to_string(N) + " array, got rank " + std::to_string(value.dimension())};
        }
        return T(std::forward<U>(value));
    }

    T value_{};
};

}  // namespace viam::trajex
