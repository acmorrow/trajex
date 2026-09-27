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

// The array types trajex holds geometry in. They are named apart from their definitions so
// that the definitions can change in one place: what a configuration is stored in is a
// decision about the whole codebase, not one taken separately by every declaration that
// mentions one.
//
// Extent is dynamic throughout, because degrees of freedom is a runtime property.

namespace viam::trajex {

///
/// One-dimensional array: a configuration, or a quantity shaped like one.
///
template <typename T = double>
using xvector = xt::xtensor<T, 1>;

///
/// Two-dimensional array: a stack of xvector rows.
///
template <typename T = double>
using xmatrix = xt::xtensor<T, 2>;

// Converting between array types of different rank is well formed and silent. The conversion
// copies exactly as many extents as the destination has, so a wider destination reads the ones
// it is missing from past the end of the source's shape and a narrower one drops those that did
// not fit, along with the elements they accounted for. xtensor has the check for this, but
// compiles it in only under XTENSOR_ENABLE_ASSERT, which no build type defines. A declaration
// can still refuse the conversion. A function body cannot, because its parameter is already
// built by the time the body runs.

///
/// Satisfied by anything xtensor will evaluate, which is also what will convert to the array
/// types above uninvited.
///
/// Spelled against xt::is_xexpression rather than deriving from xt::xexpression by hand, because
/// that trait recognises both shapes the CRTP base can take where checking one of them does not.
/// Deliberately not xt::xexpression_concept, which the version we pin does not define.
///
template <typename T>
concept xexpression_like = xt::is_xexpression<T>::value;

///
/// The rank of a type that settles its rank at runtime, such as xarray or any view.
///
/// Taken from xtensor rather than restated, so the two cannot drift. xtensor writes this
/// sentinel as a bare SIZE_MAX wherever it needs it and gives it no name of its own, so asking
/// a dynamically ranked array for its rank is the closest thing to a definition available.
///
inline constexpr std::size_t xrank_dynamic = xt::get_rank<xt::xarray<double>>::value;

///
/// The rank xtensor reports for T, which is xrank_dynamic when it does not fix one.
///
template <typename T>
inline constexpr std::size_t xrank_of = xt::get_rank<std::decay_t<T>>::value;

///
/// Satisfied when xtensor fixes T's rank at compile time, which is what makes it checkable here.
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
/// Spelling the requirement against the type an interface actually wants keeps the rank out of
/// the call site, which has no business restating it.
///
template <typename T, typename U>
concept xrank_same_as = xranked_statically<T> && xranked_statically<U> && (xrank_of<T> == xrank_of<U>);

///
/// Satisfied when T is worth accepting where a rank like U's is wanted: either it already has
/// that rank, or its rank is only known at runtime and so is worth a check.
///
/// What it excludes is the case nothing can rescue, a rank fixed at something else. Use it on a
/// parameter that is then checked in the body, rather than on a parameter of concrete type: the
/// deduction is the point, since a concrete parameter would convert before the body could look.
///
/// Spelled against U for the same reason xrank_same_as is, so the call site says which type it
/// wants a rank like rather than restating the number.
///
template <typename T, typename U>
concept xrank_same_as_or_dynamic = xrank_same_as<T, U> || xranked_dynamically<T>;

///
/// True when value's rank, however it was settled, matches the rank T fixes.
///
/// The runtime counterpart of xrank_same_as, for the check a deduced overload owes in its body.
/// Naming the type keeps the rank literal out of the call site the same way the concept does.
///
template <xranked_statically T>
[[nodiscard]] bool xrank_is_same_as(const auto& value) {
    return value.dimension() == xrank_of<T>;
}

///
/// An array of rank N, held as a T, which nothing could have given the wrong rank.
///
/// For the places a rank guard cannot go on a declaration: a public member a caller assigns
/// into, or the return type of a callback a caller supplies. There is no function body at
/// either, so the check has to live in the type that receives the value.
///
/// Every way in is checked. A source whose rank is fixed at N is taken as is, one that settles
/// its rank at runtime is checked and throws when it turns out wrong, and one fixed at any
/// other rank does not compile. What cannot happen is the silent conversion, because the
/// constructors deduce the caller's type rather than converting to a parameter.
///
/// Reading is unguarded and deliberately quiet. The value is a valid rank-N array by the time
/// anything can see it, so it converts to a const reference and needs no ceremony at the
/// hundred or so places that just pass it along. The explicit accessors exist for the places
/// the language cannot see through a conversion: template argument deduction, which is most
/// of xtensor's expression machinery, and member access.
///
/// Access is const throughout. A mutable handle would let a caller resize the array out from
/// under the guarantee, which would make the type a lie rather than a check.
///
template <typename T, std::size_t N = xrank_of<T>>
class xrank_checked {
   public:
    static_assert(N != xrank_dynamic,
                  "xrank_checked cannot infer a rank to check against from a dynamically ranked T; name one explicitly");
    static_assert(!xranked_statically<T> || xrank_of<T> == N, "xrank_checked's rank must agree with the rank T fixes");

    /// Holds a default-constructed T, which is empty rather than absent.
    xrank_checked() = default;

    ///
    /// Takes ownership of a value that is already the held type.
    ///
    /// Without this, a function returning T by value into a checked slot -- a callback whose
    /// return type is one of these, say -- would copy rather than move, because the converting
    /// constructor below binds a const reference. On a path walked once per integration step
    /// that is an allocation per call.
    ///
    xrank_checked(T&& value) noexcept(std::is_nothrow_move_constructible_v<T>) : value_(std::move(value)) {}

    ///
    /// Takes a source whose rank is already known to be right. Nothing to check.
    ///
    /// Deduces the caller's value category so an rvalue is consumed rather than copied. Whether
    /// that saves anything is xtensor's business and varies by source type -- converting an
    /// xarray to an xtensor steals its storage, converting an xtensor_fixed copies -- but
    /// forwarding costs nothing where there is nothing to steal.
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

    // A rank fixed at anything else cannot be salvaged, and converting it would invent or
    // discard extents rather than fail. Refuse it where the caller can see it. The bound on U is
    // carried by xranked_statically, which admits only expressions, so this does not reach past
    // xtensor to delete construction from unrelated types -- nor to xrank_checked itself, which
    // is what keeps these forwarding constructors from displacing the copy constructor.
    template <typename U>
        requires(xranked_statically<U> && xrank_of<U> != N)
    xrank_checked(U&&) = delete;

    // Reaching inside a temporary hands out a reference to storage that dies at the semicolon,
    // and the compiler will not say so. Refused on rvalues, which costs a named local at the
    // few places that want one and removes the whole class of mistake. The conversion below is
    // deliberately not refused: a temporary passed as an argument outlives the call, so that
    // case is safe and is the one people write.
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
    /// The conversion below hands back a const reference, so initialising owned storage from a
    /// checked value copies. Where the value is finished with -- a by-value parameter being
    /// moved into a member, say -- taking it instead is the difference between a move and an
    /// allocation. Returns by value, so there is nothing for a caller to outlive, and the
    /// rvalue qualifier means the consumption is spelled out at the call.
    ///
    T take() && {
        return std::move(value_);
    }

    operator const T&() const noexcept {
        return value_;
    }

   private:
    // Checked before the conversion rather than after, because after is too late: converting a
    // dynamically ranked array to a statically ranked one is what fabricates the extent. The
    // conversion therefore happens here, on a forwarded argument so an rvalue is consumed rather
    // than copied, and the result comes back by value -- returning a reference to the argument
    // would hand one out to whatever the caller passed, which may have been a temporary. The
    // prvalue initialises value_ directly, so the check costs nothing beyond the comparison.
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
