//go:build !windows

package streaming_test

import (
	"context"
	"math"
	"strings"
	"testing"

	"go.viam.com/test"

	trajex "github.com/viam-modules/trajex/go"
	"github.com/viam-modules/trajex/go/totg/streaming"
)

var (
	identityRotation = [4]float64{1, 0, 0, 0}
	zAxis            = [3]float64{0, 0, 1}
)

// planar2Link builds two joints rotating about z, one metre apart, with the
// end of the chain one metre past the second joint.
func planar2Link(t *testing.T) *trajex.KinematicChain {
	t.Helper()
	chain, err := trajex.NewKinematicChain([]trajex.Joint{
		trajex.RotationalJoint([3]float64{0, 0, 0}, identityRotation, zAxis),
		trajex.RotationalJoint([3]float64{1, 0, 0}, identityRotation, zAxis),
		trajex.FixedJoint([3]float64{1, 0, 0}, identityRotation),
	})
	test.That(t, err, test.ShouldBeNil)
	return chain
}

// tcpCap is in metres per second, since the planar chain is described in
// metres.
const tcpCap = 0.3

// tcpBoundOptions configures a session for the planar chain with loose joint
// limits, so that when capped is true the cap is what decides how fast a move
// goes. The caller must Close the returned map.
func tcpBoundOptions(t *testing.T, capped bool) *trajex.TensorMap {
	t.Helper()
	opts, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	test.That(t, opts.InsertFloat64s(streaming.KeyVelocityLimitsRadsPerSec, []uint64{2}, []float64{100, 100}), test.ShouldBeNil)
	test.That(t, opts.InsertFloat64s(streaming.KeyAccelerationLimitsRadsPerSec2, []uint64{2}, []float64{0.05, 0.05}), test.ShouldBeNil)
	test.That(t, opts.InsertScalarFloat64(streaming.KeyPathToleranceDeltaRads, 0.001), test.ShouldBeNil)
	test.That(t, opts.InsertScalarFloat64(streaming.KeyTrajectorySamplingFreqHz, 100.0), test.ShouldBeNil)
	if capped {
		test.That(t, opts.InsertScalarFloat64(streaming.KeyTCPMaxLinearSpeed, tcpCap), test.ShouldBeNil)
	}
	return opts
}

// peakTCPSpeedOfSession extends a session with a long move, samples all of
// it, and estimates the fastest the end of the chain moved from the
// straight-line distance between its positions at consecutive samples. A
// straight line is never longer than the curve the end actually followed, so
// the estimate can fall short of the true peak but can never exceed it.
func peakTCPSpeedOfSession(t *testing.T, sess *streaming.Session) float64 {
	t.Helper()
	batch := buildBatch(t, []float64{0, 2, 10, 0})
	defer batch.Close()
	_, err := sess.Extend(context.Background(), batch)
	test.That(t, err, test.ShouldBeNil)

	out, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer out.Close()
	test.That(t, sess.SampleAtLeast(context.Background(), sess.RemainingTotalDuration(), out), test.ShouldBeNil)

	_, times, ok, err := out.ViewFloat64s(streaming.KeySampleTimesSec)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	_, configurations, ok, err := out.ViewFloat64s(streaming.KeyConfigurationsRads)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)

	chain := planar2Link(t)
	defer chain.Close()
	peak := 0.0
	previous, err := chain.EndPosition(configurations[0:2])
	test.That(t, err, test.ShouldBeNil)
	for i := 1; i < len(times); i++ {
		current, err := chain.EndPosition(configurations[2*i : 2*i+2])
		test.That(t, err, test.ShouldBeNil)
		distance := math.Hypot(math.Hypot(current[0]-previous[0], current[1]-previous[1]), current[2]-previous[2])
		peak = math.Max(peak, distance/(times[i]-times[i-1]))
		previous = current
	}
	return peak
}

// The session copies what it needs from the chain when it is created, so we
// close each chain as soon as its session exists and expect the cap to hold
// anyway. As with Generate, the uncapped run is what shows the cap had
// something to hold back.
func TestSessionWithKinematicsRespectsTCPCap(t *testing.T) {
	newSession := func(capped bool) *streaming.Session {
		opts := tcpBoundOptions(t, capped)
		defer opts.Close()
		chain := planar2Link(t)
		defer chain.Close()
		sess, err := streaming.NewWithKinematics(chain, opts)
		test.That(t, err, test.ShouldBeNil)
		return sess
	}

	uncapped := newSession(false)
	defer uncapped.Close()
	uncappedPeak := peakTCPSpeedOfSession(t, uncapped)

	capped := newSession(true)
	defer capped.Close()
	cappedPeak := peakTCPSpeedOfSession(t, capped)

	t.Logf("peak TCP speed: uncapped %.6f, capped %.6f, cap %.6f", uncappedPeak, cappedPeak, tcpCap)
	test.That(t, uncappedPeak, test.ShouldBeGreaterThan, 2*tcpCap)
	test.That(t, cappedPeak, test.ShouldBeLessThanOrEqualTo, tcpCap*(1+1e-3))
	test.That(t, cappedPeak, test.ShouldBeGreaterThan, tcpCap*0.99)
}

func TestNewRejectsTCPCapWithoutKinematics(t *testing.T) {
	opts := tcpBoundOptions(t, true)
	defer opts.Close()

	sess, err := streaming.New(opts)
	test.That(t, sess, test.ShouldBeNil)
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, strings.Contains(err.Error(), "kinematic chain"), test.ShouldBeTrue)
}

func TestNewWithKinematicsRejectsANilOrClosedChain(t *testing.T) {
	opts := tcpBoundOptions(t, false)
	defer opts.Close()

	sess, err := streaming.NewWithKinematics(nil, opts)
	test.That(t, sess, test.ShouldBeNil)
	test.That(t, err, test.ShouldNotBeNil)

	closed := planar2Link(t)
	closed.Close()
	sess, err = streaming.NewWithKinematics(closed, opts)
	test.That(t, sess, test.ShouldBeNil)
	test.That(t, err, test.ShouldNotBeNil)
}
