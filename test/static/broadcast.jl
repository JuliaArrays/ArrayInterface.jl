
s5 = static(1):static(5)
s4 = static(1):static(4)
s1 = static(1):static(1)
d5 = static(1):5
d4 = static(1):static(4)
d1 = static(1):static(1)

struct DummyBroadcast <: ArrayInterface.BroadcastAxis end

struct DummyAxis end

ArrayInterface.BroadcastAxis(::Type{DummyAxis}) = DummyBroadcast()

ArrayInterface.broadcast_axis(::DummyBroadcast, x, y) = y

@inferred(ArrayInterface.broadcast_axis(s1, s1)) === s1
@inferred(ArrayInterface.broadcast_axis(s5, s5)) === s5
@inferred(ArrayInterface.broadcast_axis(s5, s1)) === s5
@inferred(ArrayInterface.broadcast_axis(s1, s5)) === s5
@inferred(ArrayInterface.broadcast_axis(s5, d5)) === s5
@inferred(ArrayInterface.broadcast_axis(d5, s5)) === s5
@inferred(ArrayInterface.broadcast_axis(d5, d1)) === d5
@inferred(ArrayInterface.broadcast_axis(d1, d5)) === d5
@inferred(ArrayInterface.broadcast_axis(s1, d5)) === d5
@inferred(ArrayInterface.broadcast_axis(d5, s1)) === d5
@inferred(ArrayInterface.broadcast_axis(s5, DummyAxis())) === s5

@test_throws DimensionMismatch ArrayInterface.broadcast_axis(s5, s4)

