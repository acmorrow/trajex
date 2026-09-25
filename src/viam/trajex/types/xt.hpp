#pragma once

#include <concepts>
#include <cstddef>
#include <limits>
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
template <typename T>
concept tensor_like = std::derived_from<std::decay_t<T>, xt::xexpression<std::decay_t<T>>>;

///
/// The rank xtensor reports for T, which is dynamic_rank when it does not fix one.
///
template <typename T>
inline constexpr std::size_t rank_of = xt::get_rank<std::decay_t<T>>::value;

///
/// The rank of a type that settles its rank at runtime, such as xarray or any view.
///
inline constexpr std::size_t dynamic_rank = std::numeric_limits<std::size_t>::max();

///
/// Satisfied when xtensor fixes T's rank at compile time, which is what makes it checkable here.
///
template <typename T>
concept statically_ranked = tensor_like<T> && (rank_of<T> != dynamic_rank);

///
/// Satisfied when T carries its rank as a runtime property, so only T itself can report it.
///
template <typename T>
concept dynamically_ranked = tensor_like<T> && (rank_of<T> == dynamic_rank);

///
/// Satisfied when T and U both fix their rank and the two agree.
///
/// Spelling the requirement against the type an interface actually wants keeps the rank out of
/// the call site, which has no business restating it.
///
template <typename T, typename U>
concept same_rank_as = statically_ranked<T> && statically_ranked<U> && (rank_of<T> == rank_of<U>);

///
/// Satisfied when T is worth accepting where rank N is wanted: either it already is that rank,
/// or its rank is only known at runtime and so is worth a check.
///
/// What it excludes is the case nothing can rescue, a rank fixed at something else. Use it on a
/// parameter that is then checked in the body, rather than on a parameter of concrete type: the
/// deduction is the point, since a concrete parameter would convert before the body could look.
///
template <typename T, std::size_t N>
concept rank_or_runtime = (statically_ranked<T> && rank_of<T> == N) || dynamically_ranked<T>;

///
/// Satisfied when T holds elements of type V contiguously in row-major order.
///
/// Rows of such a thing can be handed on as adaptors over their own storage. Views and lazy
/// expressions satisfy neither half: a view may stride over the array it reads, and an
/// expression has no storage to point at until something evaluates it.
///
template <typename T, typename V>
concept dense_rows_of =
    tensor_like<T> && xt::has_data_interface<std::decay_t<T>>::value && (std::decay_t<T>::static_layout == xt::layout_type::row_major) &&
    std::same_as<typename std::decay_t<T>::value_type, V>;

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
template <typename T, std::size_t N = rank_of<T>>
class rank_checked {
   public:
    static_assert(N != dynamic_rank, "rank_checked cannot infer a rank to check against from a dynamically ranked T; name one explicitly");
    static_assert(!statically_ranked<T> || rank_of<T> == N, "rank_checked's rank must agree with the rank T fixes");

    /// Holds a default-constructed T, which is empty rather than absent.
    rank_checked() = default;

    ///
    /// Takes ownership of a value that is already the held type.
    ///
    /// Without this, a function returning T by value into a checked slot -- a callback whose
    /// return type is one of these, say -- would copy rather than move, because the converting
    /// constructor below binds a const reference. On a path walked once per integration step
    /// that is an allocation per call.
    ///
    rank_checked(T&& value) noexcept(std::is_nothrow_move_constructible_v<T>) : value_(std::move(value)) {}

    /// Takes a source whose rank is already known to be right. Nothing to check.
    template <tensor_like U>
        requires(statically_ranked<U> && rank_of<U> == N && std::constructible_from<T, const U&>)
    rank_checked(const U& value) : value_(value) {}

    ///
    /// Takes a source that settles its rank at runtime.
    ///
    /// @throws std::invalid_argument if the rank turns out not to be N
    ///
    template <tensor_like U>
        requires(dynamically_ranked<U> && std::constructible_from<T, const U&>)
    rank_checked(const U& value) : value_(checked_(value)) {}

    // A rank fixed at anything else cannot be salvaged, and converting it would invent or
    // discard extents rather than fail. Refuse it where the caller can see it.
    template <tensor_like U>
        requires(statically_ranked<U> && rank_of<U> != N)
    rank_checked(const U&) = delete;

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
    // dynamically ranked array to a statically ranked one is what fabricates the extent.
    template <typename U>
    static const U& checked_(const U& value) {
        if (value.dimension() != N) {
            throw std::invalid_argument{"expected a rank-" + std::to_string(N) + " array, got rank " + std::to_string(value.dimension())};
        }
        return value;
    }

    T value_{};
};

}  // namespace viam::trajex
