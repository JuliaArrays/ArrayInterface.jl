using ArrayInterface, ReverseDiff, Tracker
x = reduce(vcat, ReverseDiff.track([4.0]))
ArrayInterface.aos_to_soa(x) isa ReverseDiff.TrackedArray
x = reduce(vcat, ReverseDiff.track([4.0,4.0]))
ArrayInterface.aos_to_soa(x) isa ReverseDiff.TrackedArray

x = identity.(Tracker.TrackedArray([4.0]))
ArrayInterface.aos_to_soa(x) isa Tracker.TrackedArray
x = identity.(Tracker.TrackedArray([4.0,4.0]))
ArrayInterface.aos_to_soa(x) isa Tracker.TrackedArray
