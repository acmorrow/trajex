//go:build !windows && cgo

package rdk_test

import (
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/golang/geo/r3"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/spatialmath"
	"go.viam.com/rdk/utils"
	"go.viam.com/test"

	trajex "github.com/viam-modules/trajex/go"
	"github.com/viam-modules/trajex/go/rdk"
)

// RDK's own forward kinematics is the reference throughout. Comparing the end
// position of the chain we build against the position m.Transform reports, at
// many joint positions, checks the order in which poses are composed, the
// quaternion convention, the sense of each axis, and the units all at once,
// which no reading of RDK's code could do as reliably.

func loadModel(t *testing.T, file string) referenceframe.Model {
	t.Helper()
	m, err := referenceframe.KinematicModelFromFile(utils.ResolveFile(file), "test")
	test.That(t, err, test.ShouldBeNil)
	return m
}

func chainFromModel(t *testing.T, m referenceframe.Model, tcp spatialmath.Pose) *trajex.KinematicChain {
	t.Helper()
	joints, err := rdk.JointsFromModel(m, tcp)
	test.That(t, err, test.ShouldBeNil)
	chain, err := trajex.NewKinematicChain(joints)
	test.That(t, err, test.ShouldBeNil)
	return chain
}

// randomPositions draws joint positions within each joint's limits. Continuous
// joints have infinite limits, so for those we draw from two full turns
// either side of zero, which is enough to show that the angle wraps the same
// way in both.
func randomPositions(m referenceframe.Model, rng *rand.Rand) []float64 {
	limits := m.DoF()
	q := make([]float64, len(limits))
	for i, limit := range limits {
		lo, hi := limit.Min, limit.Max
		if math.IsInf(lo, 0) || math.IsInf(hi, 0) {
			lo, hi = -4*math.Pi, 4*math.Pi
		}
		q[i] = lo + rng.Float64()*(hi-lo)
	}
	return q
}

func TestJointsFromModelMatchesRDKForwardKinematics(t *testing.T) {
	for _, file := range []string{
		"components/arm/kinematics/dofbot.json",
		"components/arm/kinematics/fake.json",
		"components/arm/kinematics/lite6.json",
		"components/arm/kinematics/ur5e.json",
		"components/arm/kinematics/ur7e.json",
		"components/arm/kinematics/ur20.json",
		"components/arm/kinematics/xarm6.json",
		"components/arm/kinematics/xarm7.json",
		"referenceframe/testfiles/ur5e.urdf",
		"referenceframe/testfiles/ur5eDH.json",
		"referenceframe/testfiles/example_gantry.json",
	} {
		t.Run(file, func(t *testing.T) {
			m := loadModel(t, file)
			chain := chainFromModel(t, m, spatialmath.NewZeroPose())
			defer chain.Close()

			rng := rand.New(rand.NewSource(1))
			for i := 0; i < 200; i++ {
				q := randomPositions(m, rng)
				want, err := m.Transform(q)
				test.That(t, err, test.ShouldBeNil)
				got, err := chain.EndPosition(q)
				test.That(t, err, test.ShouldBeNil)
				test.That(t, got[0], test.ShouldAlmostEqual, want.Point().X, 1e-6)
				test.That(t, got[1], test.ShouldAlmostEqual, want.Point().Y, 1e-6)
				test.That(t, got[2], test.ShouldAlmostEqual, want.Point().Z, 1e-6)
			}
		})
	}
}

// None of RDK's own models puts a rotated link before a translated one in the
// same run of links, so with them alone the order in which we compose a run
// could be reversed without any test noticing. In this model every run of
// links, before the first joint, between the joints, and after the last one,
// turns and then moves, so composing any of them backwards moves the end.
const orderSensitiveModel = `{
	"name": "order_sensitive",
	"links": [
		{"id": "base_turn", "parent": "world", "translation": {"x": 0, "y": 0, "z": 0},
		 "orientation": {"type": "euler_angles", "value": {"roll": 0, "pitch": 0, "yaw": 1.2}}},
		{"id": "base_move", "parent": "base_turn", "translation": {"x": 100, "y": 0, "z": 30}},
		{"id": "mid_turn", "parent": "spin", "translation": {"x": 0, "y": 0, "z": 0},
		 "orientation": {"type": "euler_angles", "value": {"roll": 0.7, "pitch": 0, "yaw": 0}}},
		{"id": "mid_move", "parent": "mid_turn", "translation": {"x": 0, "y": 80, "z": 0}},
		{"id": "tip_turn", "parent": "slide", "translation": {"x": 0, "y": 0, "z": 0},
		 "orientation": {"type": "euler_angles", "value": {"roll": 0, "pitch": -0.9, "yaw": 0}}},
		{"id": "tip_move", "parent": "tip_turn", "translation": {"x": 40, "y": 0, "z": 60}}
	],
	"joints": [
		{"id": "spin", "type": "revolute", "parent": "base_move", "axis": {"x": 0, "y": 0, "z": 1}, "min": -180, "max": 180},
		{"id": "slide", "type": "prismatic", "parent": "mid_move", "axis": {"x": 1, "y": 0, "z": 0}, "min": -50, "max": 50}
	]
}`

func TestJointsFromModelComposesLinksInOrder(t *testing.T) {
	m, err := referenceframe.UnmarshalModelJSON([]byte(orderSensitiveModel), "")
	test.That(t, err, test.ShouldBeNil)
	chain := chainFromModel(t, m, spatialmath.NewZeroPose())
	defer chain.Close()

	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 50; i++ {
		q := randomPositions(m, rng)
		want, err := m.Transform(q)
		test.That(t, err, test.ShouldBeNil)
		got, err := chain.EndPosition(q)
		test.That(t, err, test.ShouldBeNil)
		test.That(t, got[0], test.ShouldAlmostEqual, want.Point().X, 1e-6)
		test.That(t, got[1], test.ShouldAlmostEqual, want.Point().Y, 1e-6)
		test.That(t, got[2], test.ShouldAlmostEqual, want.Point().Z, 1e-6)
	}
}

// The tool center point is given relative to the arm's output frame, so the
// capped point should sit at that offset from wherever the arm puts its
// output. The UR5e's own model ends in a rotated link, so composing the tool
// center point on the wrong side of it would move the point.
func TestJointsFromModelWithAToolCenterPoint(t *testing.T) {
	m := loadModel(t, "components/arm/kinematics/ur5e.json")
	tcp := spatialmath.NewPose(r3.Vector{X: 10, Y: -20, Z: 150}, &spatialmath.EulerAngles{Roll: 0.3, Pitch: 0, Yaw: -0.5})
	chain := chainFromModel(t, m, tcp)
	defer chain.Close()

	rng := rand.New(rand.NewSource(2))
	for i := 0; i < 50; i++ {
		q := randomPositions(m, rng)
		output, err := m.Transform(q)
		test.That(t, err, test.ShouldBeNil)
		want := spatialmath.Compose(output, tcp).Point()
		got, err := chain.EndPosition(q)
		test.That(t, err, test.ShouldBeNil)
		test.That(t, got[0], test.ShouldAlmostEqual, want.X, 1e-6)
		test.That(t, got[1], test.ShouldAlmostEqual, want.Y, 1e-6)
		test.That(t, got[2], test.ShouldAlmostEqual, want.Z, 1e-6)
	}
}

// The tool center point is never optional, because a cap silently applied at
// the flange of an arm carrying a long tool would protect the wrong point.
func TestJointsFromModelRequiresAToolCenterPoint(t *testing.T) {
	_, err := rdk.JointsFromModel(loadModel(t, "components/arm/kinematics/ur5e.json"), nil)
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, strings.Contains(err.Error(), "NewZeroPose"), test.ShouldBeTrue)
}

func TestJointsFromModelRefusesAMimicJointOnThePath(t *testing.T) {
	_, err := rdk.JointsFromModel(loadModel(t, "referenceframe/testfiles/test_mimic_serial.json"), spatialmath.NewZeroPose())
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, strings.Contains(err.Error(), "mimics"), test.ShouldBeTrue)
}

// In both gripper models the output frame hangs directly off the base, while
// a finger elsewhere in the model still takes an input. Positions meant for
// the model would then drive a chain that has no joint for them.
func TestJointsFromModelRefusesMovingJointsOffThePath(t *testing.T) {
	for _, file := range []string{
		"referenceframe/testfiles/test_gripper.json",
		"referenceframe/testfiles/test_mimic_gripper.json",
	} {
		t.Run(file, func(t *testing.T) {
			_, err := rdk.JointsFromModel(loadModel(t, file), spatialmath.NewZeroPose())
			test.That(t, err, test.ShouldNotBeNil)
			test.That(t, strings.Contains(err.Error(), "takes inputs for joints"), test.ShouldBeTrue)
		})
	}
}

// A model assembled in code has no configuration to describe its geometry.
func TestJointsFromModelRefusesAModelWithoutConfiguration(t *testing.T) {
	joint, err := referenceframe.NewRotationalFrame("joint", spatialmath.R4AA{RZ: 1}, referenceframe.Limit{Min: -math.Pi, Max: math.Pi})
	test.That(t, err, test.ShouldBeNil)
	m, err := referenceframe.NewSerialModel("code", []referenceframe.Frame{joint})
	test.That(t, err, test.ShouldBeNil)

	_, err = rdk.JointsFromModel(m, spatialmath.NewZeroPose())
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, strings.Contains(err.Error(), "no configuration"), test.ShouldBeTrue)

	_, err = rdk.JointsFromModel(nil, spatialmath.NewZeroPose())
	test.That(t, err, test.ShouldNotBeNil)
}
