// Cost of the xtensor idioms trajex uses on small, fixed-width vectors.
//
// Every hot path in the integrator manipulates one configuration-space vector at a time,
// which for the arms we care about is six doubles. At that width the useful work is a
// handful of arithmetic operations, so whatever a container spends establishing shape,
// strides, or storage is not amortised over anything -- it is the cost. These benchmarks
// put numbers on that, so decisions about which container to hold geometry in, and whether
// to return it or fill it in place, rest on measurement rather than on reasoning about what
// the compiler ought to manage.
//
// Operations are batched over a run of rows because a six-element operation takes a couple
// of nanoseconds, which is the same order as google-benchmark's own loop overhead. Timing
// one operation per iteration would mostly measure the harness.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xfixed.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/reducers/xnorm.hpp>
#include <xtensor/views/xview.hpp>
#else
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#endif

namespace {

// Six revolute joints, matching the arms trajex is used with.
constexpr std::size_t k_dof = 6;

// Rows touched per iteration. Large enough that the harness overhead is negligible against
// the work, small enough that everything stays in L1 and we measure the operation rather
// than the memory system.
constexpr std::size_t k_rows = 256;

using fixed_row = xt::xtensor_fixed<double, xt::xshape<k_dof>>;
using raw_row = std::array<double, k_dof>;

// Arbitrary but reproducible values, and not constant across rows, so nothing folds away.
double sample_value(std::size_t row, std::size_t joint) {
    return 1.0 + (static_cast<double>((row * k_dof) + joint) * 0.125);
}

template <typename Array2D>
Array2D make_2d() {
    auto result = Array2D::from_shape({k_rows, k_dof});
    for (std::size_t row = 0; row != k_rows; ++row) {
        for (std::size_t joint = 0; joint != k_dof; ++joint) {
            result(row, joint) = sample_value(row, joint);
        }
    }
    return result;
}

template <typename Row>
std::vector<Row> make_rows() {
    std::vector<Row> result(k_rows);
    for (std::size_t row = 0; row != k_rows; ++row) {
        for (std::size_t joint = 0; joint != k_dof; ++joint) {
            result[row][joint] = sample_value(row, joint);
        }
    }
    return result;
}

//
// Copying one row. This is what `waypoint_store::append` does per waypoint, and the
// comparison that sent us here: the scalar loop measured materially faster than the view
// assignment on `xarray`.
//

template <typename Array2D>
void copy_scalar_loop(benchmark::State& state) {
    const auto source = make_2d<Array2D>();
    auto destination = make_2d<Array2D>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        for (std::size_t row = 0; row != k_rows; ++row) {
            for (std::size_t joint = 0; joint != k_dof; ++joint) {
                destination(row, joint) = source(row, joint);
            }
        }
        benchmark::DoNotOptimize(destination.data());
        benchmark::ClobberMemory();
    }
}

template <typename Array2D>
void copy_view_assign(benchmark::State& state) {
    const auto source = make_2d<Array2D>();
    auto destination = make_2d<Array2D>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        for (std::size_t row = 0; row != k_rows; ++row) {
            xt::view(destination, row, xt::all()) = xt::view(source, row, xt::all());
        }
        benchmark::DoNotOptimize(destination.data());
        benchmark::ClobberMemory();
    }
}

template <typename Row>
void copy_whole_row(benchmark::State& state) {
    const auto source = make_rows<Row>();
    auto destination = make_rows<Row>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        for (std::size_t row = 0; row != k_rows; ++row) {
            destination[row] = source[row];
        }
        benchmark::DoNotOptimize(destination.data());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(copy_scalar_loop<xt::xarray<double>>)->Name("bm_copy/xarray_scalar_loop");
BENCHMARK(copy_view_assign<xt::xarray<double>>)->Name("bm_copy/xarray_view_assign");
BENCHMARK(copy_scalar_loop<xt::xtensor<double, 2>>)->Name("bm_copy/xtensor2_scalar_loop");
BENCHMARK(copy_view_assign<xt::xtensor<double, 2>>)->Name("bm_copy/xtensor2_view_assign");
BENCHMARK(copy_whole_row<fixed_row>)->Name("bm_copy/xtensor_fixed_assign");
BENCHMARK(copy_whole_row<raw_row>)->Name("bm_copy/std_array_assign");

//
// Handing a vector back to a caller. This is the shape of `path::cursor::tangent()` and its
// siblings, which return `xt::xarray<double>` by value on every geometry query.
//
// The producers are marked noinline deliberately: the real accessors are defined in path.cpp
// and called from trajectory.cpp with no link-time optimisation, so the caller cannot see
// through them and elide the return. A benchmark that let them inline would measure
// something the integrator never gets.
//

[[gnu::noinline]] xt::xarray<double> produce_xarray(const xt::xarray<double>& source, std::size_t row) {
    auto result = xt::xarray<double>::from_shape({k_dof});
    for (std::size_t joint = 0; joint != k_dof; ++joint) {
        result(joint) = source(row, joint);
    }
    return result;
}

[[gnu::noinline]] fixed_row produce_fixed(const xt::xarray<double>& source, std::size_t row) {
    fixed_row result;
    for (std::size_t joint = 0; joint != k_dof; ++joint) {
        result[joint] = source(row, joint);
    }
    return result;
}

[[gnu::noinline]] void fill_xarray(const xt::xarray<double>& source, std::size_t row, xt::xarray<double>& out) {
    for (std::size_t joint = 0; joint != k_dof; ++joint) {
        out(joint) = source(row, joint);
    }
}

[[gnu::noinline]] void fill_span(const xt::xarray<double>& source, std::size_t row, std::span<double> out) {
    for (std::size_t joint = 0; joint != k_dof; ++joint) {
        out[joint] = source(row, joint);
    }
}

void bm_produce_xarray(benchmark::State& state) {
    const auto source = make_2d<xt::xarray<double>>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        for (std::size_t row = 0; row != k_rows; ++row) {
            auto value = produce_xarray(source, row);
            benchmark::DoNotOptimize(value.data());
        }
    }
}

void bm_produce_fixed(benchmark::State& state) {
    const auto source = make_2d<xt::xarray<double>>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        for (std::size_t row = 0; row != k_rows; ++row) {
            auto value = produce_fixed(source, row);
            benchmark::DoNotOptimize(value.data());
        }
    }
}

void bm_fill_xarray(benchmark::State& state) {
    const auto source = make_2d<xt::xarray<double>>();
    auto out = xt::xarray<double>::from_shape({k_dof});

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        for (std::size_t row = 0; row != k_rows; ++row) {
            fill_xarray(source, row, out);
            benchmark::DoNotOptimize(out.data());
        }
    }
}

void bm_fill_span(benchmark::State& state) {
    const auto source = make_2d<xt::xarray<double>>();
    raw_row out{};

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        for (std::size_t row = 0; row != k_rows; ++row) {
            fill_span(source, row, out);
            benchmark::DoNotOptimize(out.data());
        }
    }
}

BENCHMARK(bm_produce_xarray)->Name("bm_produce/return_xarray");
BENCHMARK(bm_produce_fixed)->Name("bm_produce/return_xtensor_fixed");
BENCHMARK(bm_fill_xarray)->Name("bm_produce/fill_xarray_out_param");
BENCHMARK(bm_fill_span)->Name("bm_produce/fill_span_out_param");

//
// The joint velocity limit, which is the innermost arithmetic in the integrator: the
// smallest ratio of a joint's velocity limit to the magnitude of its path derivative.
//
// The epsilon guard the real `compute_joint_velocity_limit` carries is omitted, because an
// expression form cannot branch per element. A near-zero derivative yields a huge ratio
// that the minimum discards anyway, so the arithmetic compared here is representative even
// though the semantics are not identical.
//

template <typename Vector>
void limit_scalar_loop(benchmark::State& state) {
    const auto derivatives = make_rows<Vector>();
    const auto limits = make_rows<Vector>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        double smallest = std::numeric_limits<double>::infinity();
        for (std::size_t row = 0; row != k_rows; ++row) {
            for (std::size_t joint = 0; joint != k_dof; ++joint) {
                smallest = std::min(smallest, limits[row][joint] / std::abs(derivatives[row][joint]));
            }
        }
        benchmark::DoNotOptimize(smallest);
    }
}

void limit_expression(benchmark::State& state) {
    const auto derivatives = make_rows<xt::xarray<double>>();
    const auto limits = make_rows<xt::xarray<double>>();

    for (auto unused : state) {
        benchmark::DoNotOptimize(unused);
        double smallest = std::numeric_limits<double>::infinity();
        for (std::size_t row = 0; row != k_rows; ++row) {
            smallest = std::min(smallest, xt::amin(limits[row] / xt::abs(derivatives[row]))());
        }
        benchmark::DoNotOptimize(smallest);
    }
}

BENCHMARK(limit_scalar_loop<xt::xarray<double>>)->Name("bm_limit/xarray_scalar_loop");
BENCHMARK(limit_scalar_loop<fixed_row>)->Name("bm_limit/xtensor_fixed_scalar_loop");
BENCHMARK(limit_scalar_loop<raw_row>)->Name("bm_limit/std_array_scalar_loop");
BENCHMARK(limit_expression)->Name("bm_limit/xarray_expression");

}  // namespace
