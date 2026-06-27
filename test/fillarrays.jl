using ArrayInterface, FillArrays, Test

@test !ArrayInterface.can_setindex(Fill(1.0, 3))
@test !ArrayInterface.can_setindex(typeof(Fill(1.0, 3)))
@test !ArrayInterface.can_setindex(Ones(3))
@test !ArrayInterface.can_setindex(Zeros(3))
@test !ArrayInterface.can_setindex(typeof(Zeros(2, 2)))

oe = OneElement(3.0, 2, 4)
@test !ArrayInterface.can_setindex(oe)
@test !ArrayInterface.can_setindex(typeof(oe))
