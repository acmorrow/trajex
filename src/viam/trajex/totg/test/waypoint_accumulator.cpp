// Waypoint accumulator tests
// Extracted from test.cpp lines 18-148

#include <array>
#include <concepts>
#include <vector>

#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/totg/waypoint_utils.hpp>
#include <viam/trajex/types/xt.hpp>

#if __has_include(<xtensor/core/xmath.hpp>)
#include <xtensor/containers/xfixed.hpp>
#include <xtensor/core/xmath.hpp>
#else
#include <xtensor/xfixed.hpp>
#include <xtensor/xmath.hpp>
#endif

#include <boost/test/unit_test.hpp>

using viam::trajex::xmatrix;
using viam::trajex::xvector;

BOOST_AUTO_TEST_SUITE(waypoint_accumulator_tests)

BOOST_AUTO_TEST_CASE(construct_with_single_waypoint) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {{1.0, 2.0, 3.0}};
    BOOST_CHECK_NO_THROW(waypoint_accumulator{waypoints});
}

BOOST_AUTO_TEST_CASE(construct_with_multiple_waypoints) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    BOOST_CHECK_NO_THROW(waypoint_accumulator{waypoints});
}

BOOST_AUTO_TEST_CASE(add_waypoints) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints1 = {{1.0, 2.0, 3.0}};
    waypoint_accumulator acc{waypoints1};
    BOOST_CHECK_EQUAL(acc.size(), 1);
    BOOST_CHECK_EQUAL(acc.dof(), 3);

    const xmatrix<> waypoints2 = {{4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    BOOST_CHECK_NO_THROW(acc.add_waypoints(waypoints2));
    BOOST_CHECK_EQUAL(acc.size(), 3);
}

BOOST_AUTO_TEST_CASE(validates_dof_consistency) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {{1.0, 2.0, 3.0}};
    waypoint_accumulator acc{waypoints};

    const xmatrix<> waypoints_wrong_dof = {{4.0, 5.0}};
    BOOST_CHECK_THROW(acc.add_waypoints(waypoints_wrong_dof), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(iterator_interface) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    waypoint_accumulator acc{waypoints};

    // Member function calls
    BOOST_CHECK_NO_THROW(acc.begin());
    BOOST_CHECK_NO_THROW(acc.end());
    BOOST_CHECK_NO_THROW(acc.cbegin());
    BOOST_CHECK_NO_THROW(acc.cend());

    // Verify they return the right type and work
    auto it1 = acc.begin();
    auto it2 = acc.end();
    BOOST_CHECK(it1 != it2);

    // ADL calls (unqualified)
    using std::begin;
    using std::end;
    BOOST_CHECK_NO_THROW(static_cast<void>(begin(acc)));
    BOOST_CHECK_NO_THROW(static_cast<void>(end(acc)));

    // std::cbegin/cend (qualified)
    BOOST_CHECK_NO_THROW(static_cast<void>(std::cbegin(acc)));
    BOOST_CHECK_NO_THROW(static_cast<void>(std::cend(acc)));

    // Verify range-based for works
    int count = 0;
    for (const auto& wp : acc) {
        (void)wp;  // Suppress unused warning
        ++count;
    }
    BOOST_CHECK_EQUAL(count, 2);
}

BOOST_AUTO_TEST_CASE(bounds_checked_access) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const waypoint_accumulator acc{waypoints};

    // Valid access should work
    BOOST_CHECK_NO_THROW(acc.at(0));
    BOOST_CHECK_NO_THROW(acc.at(1));

    // Out of bounds should throw
    BOOST_CHECK_THROW(acc.at(2), std::out_of_range);
    BOOST_CHECK_THROW(acc.at(100), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(unchecked_access) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const waypoint_accumulator acc{waypoints};

    // Unchecked access should work
    BOOST_CHECK_NO_THROW(acc[0]);
    BOOST_CHECK_NO_THROW(acc[1]);
}

BOOST_AUTO_TEST_CASE(empty_check) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {{1.0, 2.0, 3.0}};
    const waypoint_accumulator acc{waypoints};

    BOOST_CHECK(!acc.empty());
    BOOST_CHECK_EQUAL(acc.size(), 1);
}

BOOST_AUTO_TEST_CASE(dof_consistency_after_add) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints1 = {{1.0, 2.0, 3.0}};
    waypoint_accumulator acc{waypoints1};
    BOOST_CHECK_EQUAL(acc.dof(), 3);

    const xmatrix<> waypoints2 = {{4.0, 5.0, 6.0}};
    acc.add_waypoints(waypoints2);
    BOOST_CHECK_EQUAL(acc.dof(), 3);  // DOF should remain consistent
}

BOOST_AUTO_TEST_CASE(construct_from_single_view) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const waypoint_accumulator source{waypoints};

    // Construct new accumulator from a view
    const waypoint_accumulator from_view{source[0]};
    BOOST_CHECK_EQUAL(from_view.size(), 1);
    BOOST_CHECK_EQUAL(from_view.dof(), 3);

    // Verify the waypoint data
    BOOST_CHECK_CLOSE(from_view[0](0), 1.0, 1e-10);
    BOOST_CHECK_CLOSE(from_view[0](1), 2.0, 1e-10);
    BOOST_CHECK_CLOSE(from_view[0](2), 3.0, 1e-10);
}

BOOST_AUTO_TEST_CASE(add_waypoint_single_view) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    const waypoint_accumulator source{waypoints};

    // Build accumulator by adding views one at a time
    waypoint_accumulator result{source[0]};
    BOOST_CHECK_EQUAL(result.size(), 1);

    result.add_waypoint(source[2]);  // Skip middle waypoint
    BOOST_CHECK_EQUAL(result.size(), 2);

    // Verify we have first and third waypoints
    BOOST_CHECK_CLOSE(result[0](0), 1.0, 1e-10);
    BOOST_CHECK_CLOSE(result[1](0), 7.0, 1e-10);
}

BOOST_AUTO_TEST_CASE(add_waypoint_validates_dof) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints_3dof = {{1.0, 2.0, 3.0}};
    const xmatrix<> waypoints_2dof = {{4.0, 5.0}};

    waypoint_accumulator acc_3dof{waypoints_3dof};
    const waypoint_accumulator acc_2dof{waypoints_2dof};

    // Adding waypoint with wrong DOF should throw
    BOOST_CHECK_THROW(acc_3dof.add_waypoint(acc_2dof[0]), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(deduplicate_no_duplicates) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}};
    const waypoint_accumulator source{waypoints};

    const auto result = deduplicate_waypoints(source, 0.1);

    // All waypoints are sufficiently different, should keep all
    BOOST_CHECK_EQUAL(result.size(), 3);
    BOOST_CHECK_EQUAL(result.dof(), 3);
}

BOOST_AUTO_TEST_CASE(deduplicate_with_duplicates) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {
        {0.0, 0.0, 0.0},
        {0.05, 0.05, 0.05},  // Duplicate of first (within 0.1 tolerance)
        {1.0, 1.0, 1.0},     // Different
        {1.02, 1.02, 1.02},  // Duplicate of previous (within 0.1 tolerance)
        {2.0, 2.0, 2.0}      // Different
    };
    const waypoint_accumulator source{waypoints};

    const auto result = deduplicate_waypoints(source, 0.1);

    // Should keep waypoints 0, 2, 4 (indices in source)
    BOOST_CHECK_EQUAL(result.size(), 3);

    // Verify kept waypoints
    BOOST_CHECK_CLOSE(result[0](0), 0.0, 1e-10);
    BOOST_CHECK_CLOSE(result[1](0), 1.0, 1e-10);
    BOOST_CHECK_CLOSE(result[2](0), 2.0, 1e-10);
}

BOOST_AUTO_TEST_CASE(deduplicate_always_keeps_first) {
    using namespace viam::trajex::totg;

    const xmatrix<> waypoints = {{1.0, 1.0, 1.0}, {1.01, 1.01, 1.01}, {1.02, 1.02, 1.02}};
    const waypoint_accumulator source{waypoints};

    const auto result = deduplicate_waypoints(source, 0.1);

    // First waypoint always kept, others are duplicates
    BOOST_CHECK_EQUAL(result.size(), 1);
    BOOST_CHECK_CLOSE(result[0](0), 1.0, 1e-10);
}

BOOST_AUTO_TEST_CASE(segment_no_reversals) {
    using namespace viam::trajex::totg;

    // Simple straight line - no reversals
    const xmatrix<> waypoints = {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {3.0, 3.0, 3.0}};
    waypoint_accumulator source{waypoints};

    auto segments = segment_at_reversals(std::move(source));

    // Should be single segment containing all waypoints
    BOOST_CHECK_EQUAL(segments.size(), 1);
    BOOST_CHECK_EQUAL(segments[0].size(), 4);
}

BOOST_AUTO_TEST_CASE(segment_with_single_reversal) {
    using namespace viam::trajex::totg;

    // Forward, then reverse
    const xmatrix<> waypoints = {
        {0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0},  // Forward
        {2.0, 2.0, 2.0},  // Cusp - reversal here
        {1.5, 1.5, 1.5},  // Reverse
        {1.0, 1.0, 1.0}   // Continue reverse
    };
    waypoint_accumulator source{waypoints};

    const auto segments = segment_at_reversals(std::move(source));

    // Should have 2 segments
    BOOST_CHECK_EQUAL(segments.size(), 2);

    // First segment: waypoints 0, 1, 2 (up to and including cusp)
    BOOST_CHECK_EQUAL(segments[0].size(), 3);
    BOOST_CHECK_CLOSE(segments[0][0](0), 0.0, 1e-10);
    BOOST_CHECK_CLOSE(segments[0][2](0), 2.0, 1e-10);  // cusp

    // Second segment: waypoints 2, 3, 4 (from cusp onwards)
    BOOST_CHECK_EQUAL(segments[1].size(), 3);
    BOOST_CHECK_CLOSE(segments[1][0](0), 2.0, 1e-10);  // cusp duplicated
    BOOST_CHECK_CLOSE(segments[1][2](0), 1.0, 1e-10);
}

BOOST_AUTO_TEST_CASE(segment_with_multiple_reversals) {
    using namespace viam::trajex::totg;

    // Multiple direction changes
    const xmatrix<> waypoints = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},  // Forward
        {0.5, 0.0, 0.0},  // Reverse (cusp at waypoint 1)
        {1.0, 0.0, 0.0},  // Forward again (cusp at waypoint 2)
        {0.0, 0.0, 0.0}   // Reverse again
    };
    waypoint_accumulator source{waypoints};

    const auto segments = segment_at_reversals(std::move(source));

    // Should have 4 segments (3 reversals create 4 segments)
    BOOST_CHECK_EQUAL(segments.size(), 4);
}

// Deduplication must compare against the last kept point, not the
// original predecessor. With tolerance=10:
//   wp[0]=0  (kept)
//   wp[1]=8  (dropped: |8-0|=8 < 10)
//   wp[2]=-5 (buggy: kept because |(-5)-8|=13 > 10 vs original predecessor;
//             correct: dropped because |(-5)-0|=5 < 10 vs last kept)
//   wp[3]=20 (kept by both: distance from 0 or -5 exceeds 10)
// Correct result: [0, 20]. Buggy result: [0, -5, 20].
BOOST_AUTO_TEST_CASE(deduplicate_compares_against_last_kept) {
    using namespace viam::trajex::totg;
    const xmatrix<> waypoints = {{0.0, 0.0}, {8.0, 8.0}, {-5.0, -5.0}, {20.0, 20.0}};
    const waypoint_accumulator source{waypoints};
    const auto result = deduplicate_waypoints(source, 10.0);
    BOOST_REQUIRE_EQUAL(result.size(), 2);
    BOOST_CHECK_EQUAL(result[0](0), 0.0);
    BOOST_CHECK_EQUAL(result[1](0), 20.0);
}

// The original last waypoint must be preserved exactly as the
// destination, even if it was deduplicated away. With tolerance=10:
//   wp[0]=0  (kept)
//   wp[1]=50 (kept: |50-0|=50 > 10)
//   wp[2]=55 (dropped by loop: |55-50|=5 < 10, but is the destination)
// Pre-fix result: [0, 55]. Improved result result: [0, 50].
BOOST_AUTO_TEST_CASE(deduplicate_preserves_last_waypoint) {
    using namespace viam::trajex::totg;
    const xmatrix<> waypoints = {{0.0, 0.0}, {50.0, 50.0}, {55.0, 55.0}};
    const waypoint_accumulator source{waypoints};
    const auto result = deduplicate_waypoints(source, 10.0);
    BOOST_REQUIRE_EQUAL(result.size(), 2);
    BOOST_CHECK_EQUAL(result[0](0), 0.0);
    BOOST_CHECK_EQUAL(result[1](0), 55.0);
}

// Rank is checked where waypoints enter, because an array of the wrong rank converts rather
// than failing and the result is a waypoint set the caller did not pass. The accepting cases
// are here too: a guard that also turned away the arguments people legitimately have would be
// caught by nothing else, since the rest of the suite only ever constructs from an xmatrix<>.
BOOST_AUTO_TEST_CASE(waypoints_of_the_wrong_rank_are_rejected) {
    using namespace viam::trajex;
    using namespace viam::trajex::totg;

    const auto constructible = []<typename T>() { return std::constructible_from<waypoint_accumulator, const T&>; };
    const auto appendable = []<typename T>() { return requires(waypoint_accumulator& a, const T& t) { a.add_waypoints(t); }; };

    BOOST_CHECK(constructible.template operator()<xmatrix<>>());
    BOOST_CHECK(appendable.template operator()<xmatrix<>>());

    // Dynamic rank is admitted, because the rank it will turn out to have is not knowable here.
    // What it turns out to be is checked at runtime, in the case below.
    BOOST_CHECK(constructible.template operator()<xt::xarray<double>>());
    BOOST_CHECK(appendable.template operator()<xt::xarray<double>>());

    // Dynamic rank with nothing to adapt: a lazy expression owns no storage, and a view may
    // stride over the array it reads, so neither offers rows that can be pointed at.
    using an_expression = decltype(std::declval<const xt::xarray<double>&>() + 1.0);
    BOOST_CHECK(!constructible.template operator()<an_expression>());
    BOOST_CHECK(!appendable.template operator()<an_expression>());

    // Static but wrong rank, which is the quiet case: converting a rank 1 array to a rank 2 one
    // reads the missing extent from the adjacent stride and yields a plausible N x 1 shape.
    BOOST_CHECK(!constructible.template operator()<xvector<>>());
    BOOST_CHECK(!appendable.template operator()<xvector<>>());

    // Aliased because a comma in a template argument list is a macro argument separator.
    using rank_three = xt::xtensor<double, 3>;
    BOOST_CHECK(!constructible.template operator()<rank_three>());
    BOOST_CHECK(!appendable.template operator()<rank_three>());

    // The single-waypoint constructor takes a row adaptor. It must keep working: it is a
    // non-template overload and so wins against the deleted one.
    BOOST_CHECK(constructible.template operator()<waypoint_accumulator::waypoint_view_t>());

    // Rows are views of the caller's storage, so a temporary must be refused however it is
    // spelled. The xmatrix case has always been guarded; the runtime-checked one must be too.
    const auto constructible_from_rvalue = []<typename T>() { return std::constructible_from<waypoint_accumulator, T&&>; };
    BOOST_CHECK(!constructible_from_rvalue.template operator()<xmatrix<>>());
    BOOST_CHECK(!constructible_from_rvalue.template operator()<xt::xarray<double>>());
}

// The rank a dynamically ranked array turns out to have is checked rather than assumed, and
// the rows it yields must be the same waypoints an equivalent xmatrix would have given.
BOOST_AUTO_TEST_CASE(dynamically_ranked_waypoints_are_validated) {
    using namespace viam::trajex;
    using namespace viam::trajex::totg;

    const xt::xarray<double> good = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const waypoint_accumulator acc{good};
    BOOST_CHECK_EQUAL(acc.size(), 2);
    BOOST_CHECK_EQUAL(acc.dof(), 3);
    BOOST_CHECK_CLOSE(acc[0](0), 1.0, 1e-10);
    BOOST_CHECK_CLOSE(acc[1](2), 6.0, 1e-10);

    const xt::xarray<double> rank_one = {1.0, 2.0, 3.0};
    BOOST_CHECK_THROW(waypoint_accumulator{rank_one}, std::invalid_argument);

    const xt::xarray<double> rank_three = xt::zeros<double>({2, 2, 2});
    BOOST_CHECK_THROW(waypoint_accumulator{rank_three}, std::invalid_argument);

    // Appending checks the same thing, and the DOF besides.
    waypoint_accumulator target{good};
    const xt::xarray<double> more = {{7.0, 8.0, 9.0}};
    BOOST_CHECK_NO_THROW(target.add_waypoints(more));
    BOOST_CHECK_EQUAL(target.size(), 3);

    BOOST_CHECK_THROW(target.add_waypoints(rank_one), std::invalid_argument);

    const xt::xarray<double> wrong_dof = {{1.0, 2.0}};
    BOOST_CHECK_THROW(target.add_waypoints(wrong_dof), std::invalid_argument);
}

// A dynamically ranked source must produce rows that track it, not copies of it, since the
// accumulator's contract is that the caller's array outlives it and remains the storage.
BOOST_AUTO_TEST_CASE(dynamically_ranked_waypoints_are_not_copied) {
    using namespace viam::trajex::totg;

    xt::xarray<double> source = {{1.0, 2.0, 3.0}};
    const waypoint_accumulator acc{source};

    source(0, 1) = 99.0;
    BOOST_CHECK_CLOSE(acc[0](1), 99.0, 1e-10);
}

// An adaptor over a caller's own buffer is the case this generalization is really for: the
// waypoints never become an xtensor of ours at any point, and nothing is copied.
BOOST_AUTO_TEST_CASE(waypoints_adapted_over_caller_memory) {
    using namespace viam::trajex::totg;

    std::vector<double> buffer = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::array<std::size_t, 2> shape{2, 3};
    auto adapted = xt::adapt(buffer, shape);

    const waypoint_accumulator acc{adapted};
    BOOST_CHECK_EQUAL(acc.size(), 2);
    BOOST_CHECK_EQUAL(acc.dof(), 3);
    BOOST_CHECK_CLOSE(acc[0](0), 1.0, 1e-10);
    BOOST_CHECK_CLOSE(acc[1](2), 6.0, 1e-10);

    // Straight through to the caller's vector, with no array of ours in between.
    buffer[4] = 42.0;
    BOOST_CHECK_CLOSE(acc[1](1), 42.0, 1e-10);
}

BOOST_AUTO_TEST_CASE(waypoints_in_a_fixed_shape_array) {
    using namespace viam::trajex::totg;

    const xt::xtensor_fixed<double, xt::xshape<2, 3>> fixed = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    const waypoint_accumulator acc{fixed};
    BOOST_CHECK_EQUAL(acc.size(), 2);
    BOOST_CHECK_EQUAL(acc.dof(), 3);
    BOOST_CHECK_CLOSE(acc[0](1), 2.0, 1e-10);
    BOOST_CHECK_CLOSE(acc[1](2), 6.0, 1e-10);
}

BOOST_AUTO_TEST_CASE(appending_from_a_different_rank_two_type) {
    using namespace viam::trajex::totg;

    const xmatrix<> initial = {{1.0, 2.0, 3.0}};
    waypoint_accumulator acc{initial};

    const xt::xtensor_fixed<double, xt::xshape<1, 3>> more = {{4.0, 5.0, 6.0}};
    BOOST_CHECK_NO_THROW(acc.add_waypoints(more));
    BOOST_CHECK_EQUAL(acc.size(), 2);
    BOOST_CHECK_CLOSE(acc[1](0), 4.0, 1e-10);

    const xt::xtensor_fixed<double, xt::xshape<1, 2>> wrong_dof = {{7.0, 8.0}};
    BOOST_CHECK_THROW(acc.add_waypoints(wrong_dof), std::invalid_argument);
}

// A row adaptor is built from a pointer to the row's first element and a length, which assumes
// the source keeps each row contiguous and in order. That assumption is what admits a type at
// all, so check it directly on every kind of source admitted, rather than inferring it from the
// layout the type claims.
BOOST_AUTO_TEST_CASE(accepted_sources_store_rows_contiguously) {
    using namespace viam::trajex::totg;

    const auto check = [](const char* label, const auto& source) {
        BOOST_TEST_MESSAGE(label);
        BOOST_REQUIRE_EQUAL(source.dimension(), 2u);

        const auto rows = source.shape()[0];
        const auto cols = source.shape()[1];

        // Contiguous within a row, and rows following one another without gaps.
        BOOST_CHECK_EQUAL(static_cast<std::size_t>(source.strides()[1]), 1u);
        BOOST_CHECK_EQUAL(static_cast<std::size_t>(source.strides()[0]), cols);

        const waypoint_accumulator acc{source};
        BOOST_REQUIRE_EQUAL(acc.size(), rows);

        for (std::size_t i = 0; i < rows; ++i) {
            BOOST_REQUIRE_EQUAL(acc[i].size(), cols);
            for (std::size_t j = 0; j < cols; ++j) {
                // Same value, and the same object: a row must be the source's own storage.
                BOOST_CHECK_EQUAL(acc[i](j), source(i, j));
                BOOST_CHECK_EQUAL(&acc[i](j), &source(i, j));
            }
        }
    };

    const xmatrix<> as_xmatrix = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    check("xmatrix", as_xmatrix);

    const xt::xtensor_fixed<double, xt::xshape<2, 3>> as_fixed = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    check("xtensor_fixed", as_fixed);

    const xt::xarray<double> as_xarray = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    check("xarray, rank checked at runtime", as_xarray);

    std::vector<double> buffer = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::array<std::size_t, 2> shape{2, 3};
    const auto as_adapted = xt::adapt(buffer, shape);
    check("adaptor over a caller's vector", as_adapted);
}

// no_ownership has to mean what it says, or the accumulator would be freeing storage it was
// only ever lent.
BOOST_AUTO_TEST_CASE(rows_never_own_the_storage_they_point_at) {
    using namespace viam::trajex::totg;

    std::vector<double> buffer = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::array<std::size_t, 2> shape{2, 3};

    {
        const auto adapted = xt::adapt(buffer, shape);
        const waypoint_accumulator acc{adapted};
        BOOST_CHECK_CLOSE(acc[1](2), 6.0, 1e-10);
    }

    BOOST_REQUIRE_EQUAL(buffer.size(), 6u);
    BOOST_CHECK_CLOSE(buffer[0], 1.0, 1e-10);
    BOOST_CHECK_CLOSE(buffer[5], 6.0, 1e-10);
}

// Rank two is necessary but not sufficient: a row has to be contiguous and hold the element
// type a row adaptor holds, or there is nothing to point at.
BOOST_AUTO_TEST_CASE(rank_two_alone_does_not_qualify) {
    using namespace viam::trajex::totg;

    const auto constructible = []<typename T>() { return std::constructible_from<waypoint_accumulator, const T&>; };

    // Rows of a column-major array are strided, so they cannot be adapted in place.
    using column_major = xt::xtensor<double, 2, xt::layout_type::column_major>;
    BOOST_CHECK(!constructible.template operator()<column_major>());

    using float_matrix = xt::xtensor<float, 2>;
    BOOST_CHECK(!constructible.template operator()<float_matrix>());

    // And the ones that do qualify, for contrast.
    using fixed_shape = xt::xtensor_fixed<double, xt::xshape<2, 3>>;
    using adapted_buffer = decltype(xt::adapt(std::declval<std::vector<double>&>(), std::declval<const std::array<std::size_t, 2>&>()));
    BOOST_CHECK(constructible.template operator()<fixed_shape>());
    BOOST_CHECK(constructible.template operator()<adapted_buffer>());
}

BOOST_AUTO_TEST_SUITE_END()
