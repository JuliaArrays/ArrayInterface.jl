using ArrayInterface, ChainRules, Test
using ComponentArrays, ChainRulesTestUtils, StaticArrays

x = ChainRules.OneElement(3.0, (3, 3), (1:4, 1:4))

@test !ArrayInterface.can_setindex(x)
@test !ArrayInterface.can_setindex(typeof(x))

arr = ComponentArray(a = 1.0, b = [2.0, 3.0], c = (; a = 4.0, b = 5.0), d = SVector{2}(6.0, 7.0))
b = zeros(length(arr))

ChainRulesTestUtils.test_rrule(ArrayInterface.restructure, arr, b)
ChainRulesTestUtils.test_rrule(ArrayInterface.restructure, b, arr)
