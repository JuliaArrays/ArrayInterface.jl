using ArrayInterface, ReverseDiff, Tracker, Test
x = reduce(vcat, ReverseDiff.track([4.0]))
@test ArrayInterface.aos_to_soa(x) isa ReverseDiff.TrackedArray
x = reduce(vcat, ReverseDiff.track([4.0,4.0]))
@test ArrayInterface.aos_to_soa(x) isa ReverseDiff.TrackedArray

x = identity.(Tracker.TrackedArray([4.0]))
@test ArrayInterface.aos_to_soa(x) isa Tracker.TrackedArray
x = identity.(Tracker.TrackedArray([4.0,4.0]))
@test ArrayInterface.aos_to_soa(x) isa Tracker.TrackedArray
