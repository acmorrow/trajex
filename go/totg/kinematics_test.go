//go:build !windows && cgo

package totg_test

import (
	"context"
	"math"
	"strings"
	"testing"

	"go.viam.com/test"

	trajex "github.com/viam-modules/trajex/go"
	"github.com/viam-modules/trajex/go/totg"
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
// metres. The joint limits in tcpBoundInputs are loose and the move is long,
// so without the cap the end of the chain reaches several times this speed,
// and with it the cap is the limit that decides how fast the move goes.
const tcpCap = 0.3

// tcpBoundInputs builds the inputs for a long move of both joints of the
// planar chain. The caller must Close the returned map.
func tcpBoundInputs(t *testing.T) *trajex.TensorMap {
	t.Helper()
	in, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	test.That(t, in.InsertFloat64s(totg.KeyWaypointsRads, []uint64{2, 2}, []float64{0, 2, 10, 0}), test.ShouldBeNil)
	test.That(t, in.InsertFloat64s(totg.KeyVelocityLimitsRadsPerSec, []uint64{2}, []float64{100, 100}), test.ShouldBeNil)
	test.That(t, in.InsertFloat64s(totg.KeyAccelerationLimitsRadsPerSec2, []uint64{2}, []float64{0.05, 0.05}), test.ShouldBeNil)
	test.That(t, in.InsertScalarFloat64(totg.KeyPathToleranceDeltaRads, 0.001), test.ShouldBeNil)
	test.That(t, in.InsertScalarFloat64(totg.KeyTrajectorySamplingFreqHz, 100.0), test.ShouldBeNil)
	return in
}

// peakTCPSpeed estimates the fastest the end of the chain moved, from the
// straight-line distance between its positions at consecutive samples. The
// end actually travels along a curve between samples, and a straight line is
// never longer than the curve it cuts across, so the estimate can fall short
// of the true peak but can never exceed it. That makes it a sound check that
// a cap was respected, though not on its own a check that the cap was reached.
func peakTCPSpeed(t *testing.T, chain *trajex.KinematicChain, out *trajex.TensorMap) float64 {
	t.Helper()
	_, times, ok, err := out.ViewFloat64s(totg.KeySampleTimesSec)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)
	_, configurations, ok, err := out.ViewFloat64s(totg.KeyConfigurationsRads)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, ok, test.ShouldBeTrue)

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

func generateWithKinematics(t *testing.T, chain *trajex.KinematicChain, in *trajex.TensorMap) *trajex.TensorMap {
	t.Helper()
	out, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	test.That(t, totg.GenerateWithKinematics(context.Background(), chain, in, out), test.ShouldBeNil)
	return out
}

// We run the same move with and without the cap. A peak at or under the cap
// would also be seen if the cap did nothing at all, so the uncapped run is
// what shows the cap had something to hold back, and the capped peak landing
// just under the cap shows that it was actually reached.
func TestGenerateWithKinematicsRespectsTCPCap(t *testing.T) {
	chain := planar2Link(t)
	defer chain.Close()
	in := tcpBoundInputs(t)
	defer in.Close()

	uncapped := generateWithKinematics(t, chain, in)
	defer uncapped.Close()
	uncappedPeak := peakTCPSpeed(t, chain, uncapped)

	test.That(t, in.InsertScalarFloat64(totg.KeyTCPMaxLinearVelocity, tcpCap), test.ShouldBeNil)
	capped := generateWithKinematics(t, chain, in)
	defer capped.Close()
	cappedPeak := peakTCPSpeed(t, chain, capped)

	t.Logf("peak TCP speed: uncapped %.6f, capped %.6f, cap %.6f", uncappedPeak, cappedPeak, tcpCap)
	test.That(t, uncappedPeak, test.ShouldBeGreaterThan, 2*tcpCap)
	test.That(t, cappedPeak, test.ShouldBeLessThanOrEqualTo, tcpCap*(1+1e-3))
	test.That(t, cappedPeak, test.ShouldBeGreaterThan, tcpCap*0.99)
}

// A caller who has kinematics may pass the chain on every call, including
// calls that ask for no cap, so a chain without a cap must leave the
// trajectory exactly as Generate would have produced it.
func TestGenerateWithKinematicsAndNoCapMatchesGenerate(t *testing.T) {
	chain := planar2Link(t)
	defer chain.Close()
	in := tcpBoundInputs(t)
	defer in.Close()

	withChain := generateWithKinematics(t, chain, in)
	defer withChain.Close()
	withoutChain, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer withoutChain.Close()
	test.That(t, totg.Generate(context.Background(), in, withoutChain), test.ShouldBeNil)

	_, a, _, err := withChain.ViewFloat64s(totg.KeyConfigurationsRads)
	test.That(t, err, test.ShouldBeNil)
	_, b, _, err := withoutChain.ViewFloat64s(totg.KeyConfigurationsRads)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, a, test.ShouldResemble, b)
}

func TestGenerateRejectsTCPCapWithoutKinematics(t *testing.T) {
	in := tcpBoundInputs(t)
	defer in.Close()
	test.That(t, in.InsertScalarFloat64(totg.KeyTCPMaxLinearVelocity, tcpCap), test.ShouldBeNil)
	out, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer out.Close()

	err = totg.Generate(context.Background(), in, out)
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, strings.Contains(err.Error(), "kinematic chain"), test.ShouldBeTrue)
}

func TestGenerateWithKinematicsRejectsAChainWithTheWrongDOF(t *testing.T) {
	oneJoint, err := trajex.NewKinematicChain([]trajex.Joint{
		trajex.RotationalJoint([3]float64{0, 0, 0}, identityRotation, zAxis),
	})
	test.That(t, err, test.ShouldBeNil)
	defer oneJoint.Close()
	in := tcpBoundInputs(t)
	defer in.Close()
	out, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer out.Close()

	err = totg.GenerateWithKinematics(context.Background(), oneJoint, in, out)
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, strings.Contains(err.Error(), "degrees of freedom"), test.ShouldBeTrue)
}

func TestGenerateWithKinematicsRejectsANilOrClosedChain(t *testing.T) {
	in := tcpBoundInputs(t)
	defer in.Close()
	out, err := trajex.NewTensorMap()
	test.That(t, err, test.ShouldBeNil)
	defer out.Close()

	test.That(t, totg.GenerateWithKinematics(context.Background(), nil, in, out), test.ShouldNotBeNil)

	closed := planar2Link(t)
	closed.Close()
	test.That(t, totg.GenerateWithKinematics(context.Background(), closed, in, out), test.ShouldNotBeNil)
}
