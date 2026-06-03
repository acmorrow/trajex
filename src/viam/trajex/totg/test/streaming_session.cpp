// Streaming session tests.

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/streaming/session.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/types/hertz.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(streaming_session_tests)

BOOST_AUTO_TEST_CASE(fresh_session_has_zero_current_time) {
    using namespace viam::trajex;
    totg::streaming::session sess(totg::path::options{}, totg::trajectory::options{}, types::hertz{100.0});
    BOOST_CHECK_EQUAL(sess.current_time().count(), 0.0);
}

BOOST_AUTO_TEST_SUITE_END()  // streaming_session_tests
