
s5 = static(1):static(5)
s4 = static(1):static(4)
s1 = static(1):static(1)
d5 = static(1):5

@inferred(ArrayInterface.broadcast_axis(s5, s5)) === s5
@inferred(ArrayInterface.broadcast_axis(s5, s1)) === s5
@inferred(ArrayInterface.broadcast_axis(s1, s5)) === s5
@inferred(ArrayInterface.broadcast_axis(s5, d5)) === d5

@test_throws DimensionMismatch ArrayInterface.broadcast_axis(s5, s4)

