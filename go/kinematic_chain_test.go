//go:build !windows && cgo

package trajex_test

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"testing"

	"go.viam.com/test"

	"github.com/viam-modules/trajex/go"
)

var (
	identityRotation = [4]float64{1, 0, 0, 0}
	zAxis            = [3]float64{0, 0, 1}
)

// planar2Link builds two joints rotating about z, one metre apart, with the
// end of the chain one metre past the second joint. With both angles at zero
// the end sits two metres out along x.
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

func TestKinematicChainEndPositionMatchesPlanarGeometry(t *testing.T) {
	chain := planar2Link(t)
	defer chain.Close()

	p, err := chain.EndPosition([]float64{0, 0})
	test.That(t, err, test.ShouldBeNil)
	test.That(t, p[0], test.ShouldAlmostEqual, 2.0, 1e-12)
	test.That(t, p[1], test.ShouldAlmostEqual, 0.0, 1e-12)
	test.That(t, p[2], test.ShouldAlmostEqual, 0.0, 1e-12)

	p, err = chain.EndPosition([]float64{math.Pi / 2, -math.Pi / 2})
	test.That(t, err, test.ShouldBeNil)
	test.That(t, p[0], test.ShouldAlmostEqual, 1.0, 1e-12)
	test.That(t, p[1], test.ShouldAlmostEqual, 1.0, 1e-12)
	test.That(t, p[2], test.ShouldAlmostEqual, 0.0, 1e-12)
}

// This chain slides along x and then rotates, and the rotating joint's origin
// is turned a quarter turn about x, which points that joint's z axis along
// the base's negative y axis. It checks a prismatic joint and an origin with
// a real rotation in one place.
func TestKinematicChainTranslationalAndRotatedOrigin(t *testing.T) {
	quarterAboutX := [4]float64{math.Cos(math.Pi / 4), math.Sin(math.Pi / 4), 0, 0}
	chain, err := trajex.NewKinematicChain([]trajex.Joint{
		trajex.TranslationalJoint([3]float64{0, 0, 0}, identityRotation, [3]float64{1, 0, 0}),
		trajex.RotationalJoint([3]float64{0, 0, 0}, quarterAboutX, zAxis),
		trajex.FixedJoint([3]float64{1, 0, 0}, identityRotation),
	})
	test.That(t, err, test.ShouldBeNil)
	defer chain.Close()

	// A quarter turn about the base's negative y axis carries the end of the
	// chain from one metre along x up to one metre along z, and the slide
	// moves the whole thing half a metre along x.
	p, err := chain.EndPosition([]float64{0.5, math.Pi / 2})
	test.That(t, err, test.ShouldBeNil)
	test.That(t, p[0], test.ShouldAlmostEqual, 0.5, 1e-12)
	test.That(t, p[1], test.ShouldAlmostEqual, 0.0, 1e-12)
	test.That(t, p[2], test.ShouldAlmostEqual, 1.0, 1e-12)
}

func TestNewKinematicChainNamesAnInvalidJoint(t *testing.T) {
	chain, err := trajex.NewKinematicChain([]trajex.Joint{
		trajex.FixedJoint([3]float64{0, 0, 0}, identityRotation),
		trajex.TranslationalJoint([3]float64{0, 0, 0}, identityRotation, [3]float64{0, 0, 0}),
	})
	test.That(t, chain, test.ShouldBeNil)
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, strings.Contains(err.Error(), "joint 1"), test.ShouldBeTrue)
}

func TestNewKinematicChainRejectsNoJoints(t *testing.T) {
	chain, err := trajex.NewKinematicChain(nil)
	test.That(t, chain, test.ShouldBeNil)
	test.That(t, err, test.ShouldNotBeNil)
}

func TestNewKinematicChainRejectsAZeroJoint(t *testing.T) {
	chain, err := trajex.NewKinematicChain([]trajex.Joint{
		trajex.RotationalJoint([3]float64{0, 0, 0}, identityRotation, zAxis),
		{},
	})
	test.That(t, chain, test.ShouldBeNil)
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, strings.Contains(err.Error(), "joint 1"), test.ShouldBeTrue)
}

func TestKinematicChainEndPositionRejectsAWrongJointCount(t *testing.T) {
	chain := planar2Link(t)
	defer chain.Close()

	_, err := chain.EndPosition([]float64{0, 0, 0})
	test.That(t, err, test.ShouldNotBeNil)
	_, err = chain.EndPosition(nil)
	test.That(t, err, test.ShouldNotBeNil)
}

func TestKinematicChainCloseIsNilSafeAndIdempotent(t *testing.T) {
	var nilChain *trajex.KinematicChain
	nilChain.Close()
	// The test package's ShouldBeNil does not treat a nil unsafe.Pointer as
	// nil, so we compare against nil ourselves.
	test.That(t, nilChain.UnsafeHandle() == nil, test.ShouldBeTrue)

	chain := planar2Link(t)
	chain.Close()
	chain.Close()
	test.That(t, chain.UnsafeHandle() == nil, test.ShouldBeTrue)
	_, err := chain.EndPosition([]float64{0, 0})
	test.That(t, err, test.ShouldNotBeNil)
}

// KinematicChain promises that goroutines may share a chain without
// coordinating, because nothing in it changes once it is built. On its own,
// this test only shows that concurrent calls get the right answers. Run with
// -race, it also shows that the Go side writes nothing the calls share. The
// race detector cannot see into C++, so the other half of the promise rests
// on the C ABI handling the chain as const.
func TestKinematicChainConcurrentEndPosition(t *testing.T) {
	chain := planar2Link(t)
	defer chain.Close()

	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 1000; i++ {
				q := float64(g*1000+i) * 1e-3
				p, err := chain.EndPosition([]float64{q, 0})
				if err != nil {
					errs <- err
					return
				}
				if math.Abs(p[0]-2*math.Cos(q)) > 1e-12 || math.Abs(p[1]-2*math.Sin(q)) > 1e-12 {
					errs <- fmt.Errorf("EndPosition at q=%v gave %v", q, p)
					return
				}
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}
