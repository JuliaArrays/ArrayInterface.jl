
using ArrayInterface
using OffsetArrays
using StaticArrays
using Test

oa = OffsetArray([1, 2]', 1, 1)
@test @inferred(ArrayInterface.known_size(oa)) == (1, nothing)
@test @inferred(ArrayInterface.known_length(oa)) === nothing


id = OffsetArrays.IdOffsetRange(SOneTo(10), 1)
@test @inferred(ArrayInterface.known_size(id)) == (10, )
@test @inferred(ArrayInterface.known_length(id)) == 10

