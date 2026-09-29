#pragma once

namespace viam::trajex::totg::bench {

// Registers the pipeline benchmarks, which are named for the replay record they run against
// and so cannot be declared with the BENCHMARK macro.
//
// Called from main, not a static initialiser: it reads the replay records, and a failure
// there should be a reported error rather than a terminate before main.
//
// Throws if a replay record cannot be read or parsed.
void register_pipeline_benchmarks();

}  // namespace viam::trajex::totg::bench
