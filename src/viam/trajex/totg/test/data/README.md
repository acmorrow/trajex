# Replay records

Trajectory generation inputs, replayed by the regression tests in
`../integration.cpp` (`replay_regression_tests`) and by
`viam-trajex-totg-replay`.

Records that came from the field are kept exactly as captured. Don't
minimize, truncate, or rewrite them: we can't know in advance which
details of a record matter, and the original is the only copy. Renaming
to match the existing files is the only permitted change.

A new record that fails gets a regression test, disabled until it passes.
