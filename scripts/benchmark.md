# ConditionalDists.jl with `condition` function

julia> include("benchmark_condition.jl")
  0.676896 seconds (100.00 k allocations: 134.049 MiB, 8.79% gc time)
  0.628333 seconds (100.00 k allocations: 134.049 MiB, 10.44% gc time)
  0.586335 seconds (100.00 k allocations: 134.049 MiB, 17.22% gc time)


# ConditionalDists.jl current `master`

julia> include("benchmark_master.jl")
  0.143951 seconds (47 allocations: 114.557 MiB, 1.52% gc time)
  0.141293 seconds (47 allocations: 114.557 MiB, 1.42% gc time)
  0.164857 seconds (47 allocations: 114.557 MiB, 1.25% gc time)


# ConditionalDists.jl with `BatchMvNormal`

julia> include("benchmark_condition.jl")
  0.149374 seconds (27 allocations: 114.595 MiB, 3.43% gc time)
  0.169296 seconds (27 allocations: 114.595 MiB, 4.07% gc time)
  0.152490 seconds (27 allocations: 114.595 MiB, 2.15% gc time)
