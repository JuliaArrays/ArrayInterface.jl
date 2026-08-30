using ArrayInterface, ReverseDiff, Tracker, Test, JLArrays
x = ReverseDiff.track([4.0])
@test ArrayInterface.aos_to_soa(x) isa ReverseDiff.TrackedArray
x = reshape([ReverseDiff.track(rand(1, 1, 1))[1]], 1, 1, 1)
@test ArrayInterface.aos_to_soa(x) isa ReverseDiff.TrackedArray
@test ndims(ArrayInterface.aos_to_soa(x)) == 3
x = reduce(vcat, ReverseDiff.track([4.0, 4.0]))
@test ArrayInterface.aos_to_soa(x) isa ReverseDiff.TrackedArray
x = [ReverseDiff.track([4.0])[1]]
@test ArrayInterface.aos_to_soa(x) isa ReverseDiff.TrackedArray
x = reduce(vcat, ReverseDiff.track([4.0, 4.0]))
x = [x[1], x[2]]
@test ArrayInterface.aos_to_soa(x) isa ReverseDiff.TrackedArray

x = Tracker.TrackedArray([4.0])
@test ArrayInterface.aos_to_soa(x) isa Tracker.TrackedArray
x = [Tracker.TrackedArray([4.0])[1]]
@test ArrayInterface.aos_to_soa(x) isa Tracker.TrackedArray
x = Tracker.TrackedArray([4.0, 4.0])
@test ArrayInterface.aos_to_soa(x) isa Tracker.TrackedArray
x = reduce(vcat, Tracker.TrackedArray([4.0, 4.0]))
x = [x[1], x[2]]
@test ArrayInterface.aos_to_soa(x) isa Tracker.TrackedArray

x = rand(4)
y = Tracker.TrackedReal.(rand(2, 2))
@test ArrayInterface.restructure(x, y) isa Array
@test eltype(ArrayInterface.restructure(x, y)) <: Tracker.TrackedReal
@test size(ArrayInterface.restructure(x, y)) == (4,)
y = Tracker.TrackedArray(rand(2, 2))
@test ArrayInterface.restructure(x, y) isa Tracker.TrackedArray
@test size(ArrayInterface.restructure(x, y)) == (4,)
x = Tracker.TrackedArray(rand(4))
@test ArrayInterface.restructure(x, y) isa Tracker.TrackedArray
@test size(ArrayInterface.restructure(x, y)) == (4,)
y = Tracker.TrackedReal.(rand(2, 2))
@test ArrayInterface.restructure(x, y) isa Tracker.TrackedArray
@test size(ArrayInterface.restructure(x, y)) == (4,)

x = rand(4)
y = ReverseDiff.track(rand(2, 2))
@test ArrayInterface.restructure(x, y) isa ReverseDiff.TrackedArray
@test size(ArrayInterface.restructure(x, y)) == (4,)
x = ReverseDiff.track(rand(4))
@test ArrayInterface.restructure(x, y) isa ReverseDiff.TrackedArray
@test size(ArrayInterface.restructure(x, y)) == (4,)
y = ReverseDiff.track.(rand(2, 2))
@test ArrayInterface.restructure(x, y) isa ReverseDiff.TrackedArray
@test size(ArrayInterface.restructure(x, y)) == (4,)
x = rand(4)
@test ArrayInterface.restructure(x, y) isa Array
@test eltype(ArrayInterface.restructure(x, y)) <: ReverseDiff.TrackedReal
@test size(ArrayInterface.restructure(x, y)) == (4,)

@testset "restructure GPUArraysCore + Tracker" begin
    target = JLArray(reshape(Float32.(1:6), 2, 3))
    src = JLArray(copy(vec(Array(target))))
    src_cpu = Array(src)

    yr = ArrayInterface.restructure(target, src)
    @test yr isa JLArray
    @test size(yr) == (2, 3)
    @test Array(yr) == reshape(Array(src), 2, 3)

    y, back = Tracker.forward(src) do t
        r = ArrayInterface.restructure(target, t)
        @test Tracker.data(r) isa JLArray
        @test size(r) == (2, 3)
        sum(r)
    end
    dx = only(back(1.0f0))
    @test Tracker.data(y) == 21.0f0
    @test Array(Tracker.data(dx)) == ones(Float32, 6)

    y_cpu, back_cpu = Tracker.forward(src_cpu) do t
        r = ArrayInterface.restructure(target, t)
        @test Tracker.data(r) isa JLArray
        @test size(r) == (2, 3)
        sum(r)
    end
    dx_cpu = only(back_cpu(1.0f0))
    @test Tracker.data(y_cpu) == 21.0f0
    @test dx_cpu == ones(Float32, 6)
end
