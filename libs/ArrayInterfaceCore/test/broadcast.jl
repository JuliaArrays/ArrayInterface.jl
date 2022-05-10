
s5 = static(1):static(5)
s4 = static(1):static(4)
s1 = static(1):static(1)
d5 = static(1):5
d4 = static(1):static(4)
d1 = static(1):static(1)

struct DummyBroadcast <: ArrayInterfaceCore.BroadcastAxis end

struct DummyAxis end

ArrayInterfaceCore.BroadcastAxis(::Type{DummyAxis}) = DummyBroadcast()

ArrayInterfaceCore.broadcast_axis(::DummyBroadcast, x, y) = y

@inferred(ArrayInterfaceCore.broadcast_axis(s1, s1)) === s1
@inferred(ArrayInterfaceCore.broadcast_axis(s5, s5)) === s5
@inferred(ArrayInterfaceCore.broadcast_axis(s5, s1)) === s5
@inferred(ArrayInterfaceCore.broadcast_axis(s1, s5)) === s5
@inferred(ArrayInterfaceCore.broadcast_axis(s5, d5)) === s5
@inferred(ArrayInterfaceCore.broadcast_axis(d5, s5)) === s5
@inferred(ArrayInterfaceCore.broadcast_axis(d5, d1)) === d5
@inferred(ArrayInterfaceCore.broadcast_axis(d1, d5)) === d5
@inferred(ArrayInterfaceCore.broadcast_axis(s1, d5)) === d5
@inferred(ArrayInterfaceCore.broadcast_axis(d5, s1)) === d5
@inferred(ArrayInterfaceCore.broadcast_axis(s5, DummyAxis())) === s5

@test_throws DimensionMismatch ArrayInterfaceCore.broadcast_axis(s5, s4)

