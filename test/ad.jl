using ArrayInterface, ReverseDiff, Tracker, Test
x = ReverseDiff.track([4.0])
@test ArrayInterface.aos_to_soa(x) isa ReverseDiff.TrackedArray
x = reduce(vcat, ReverseDiff.track([4.0,4.0]))
@test ArrayInterface.aos_to_soa(x) isa ReverseDiff.TrackedArray
x = [ReverseDiff.track([4.0])[1]]
@test ArrayInterface.aos_to_soa(x) isa ReverseDiff.TrackedArray
x = reduce(vcat, ReverseDiff.track([4.0,4.0]))
x = [x[1],x[2]]
@test ArrayInterface.aos_to_soa(x) isa ReverseDiff.TrackedArray

x = Tracker.TrackedArray([4.0])
@test ArrayInterface.aos_to_soa(x) isa Tracker.TrackedArray
x = [Tracker.TrackedArray([4.0])[1]]
@test ArrayInterface.aos_to_soa(x) isa Tracker.TrackedArray
x = Tracker.TrackedArray([4.0,4.0])
@test ArrayInterface.aos_to_soa(x) isa Tracker.TrackedArray
x = reduce(vcat, Tracker.TrackedArray([4.0,4.0]))
x = [x[1],x[2]]
@test ArrayInterface.aos_to_soa(x) isa Tracker.TrackedArray
