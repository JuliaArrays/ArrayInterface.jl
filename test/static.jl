
using ArrayInterface
using Static
using Test

iprod = Iterators.product(static(1):static(2), static(1):static(3), static(1):static(4))
@test @inferred(ArrayInterface.known_size(iprod)) === (2, 3, 4)

iflat = Iterators.flatten(iprod)
@test @inferred(ArrayInterface.known_size(iflat)) === (72,)

