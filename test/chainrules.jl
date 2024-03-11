using ArrayInterface, ChainRules, Test

x = ChainRules.OneElement(3.0, (3, 3), (1:4, 1:4))

@test !ArrayInterface.can_setindex(x)
@test !ArrayInterface.can_setindex(typeof(x))
