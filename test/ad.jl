using ArrayInterface, ReverseDiff, Tracker, Test
x = ReverseDiff.track([4.0])
@test ArrayInterface.aos_to_soa(x) isa ReverseDiff.TrackedArray
x = reshape([ReverseDiff.track(rand(1, 1, 1))[1]], 1, 1, 1)
@test ArrayInterface.aos_to_soa(x) isa ReverseDiff.TrackedArray
@test ndims(ArrayInterface.aos_to_soa(x)) == 3
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

x = rand(4)
y = Tracker.TrackedReal.(rand(2,2))
@test ArrayInterface.restructure(x, y) isa Array
@test eltype(ArrayInterface.restructure(x, y)) <: Tracker.TrackedReal
@test size(ArrayInterface.restructure(x, y)) == (4,)
y = Tracker.TrackedArray(rand(2,2))
@test ArrayInterface.restructure(x, y) isa Tracker.TrackedArray
@test size(ArrayInterface.restructure(x, y)) == (4,)

x = rand(4)
y = ReverseDiff.track(rand(2,2))
@test ArrayInterface.restructure(x, y) isa ReverseDiff.TrackedArray
@test size(ArrayInterface.restructure(x, y)) == (4,)
y = ReverseDiff.track.(rand(2,2))
@test ArrayInterface.restructure(x, y) isa Array
@test eltype(ArrayInterface.restructure(x, y)) <: ReverseDiff.TrackedReal
@test size(ArrayInterface.restructure(x, y)) == (4,)
