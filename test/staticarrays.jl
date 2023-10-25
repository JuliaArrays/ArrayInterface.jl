
using ArrayInterface
using StaticArrays
using Test

so = SOneTo(10)
@test ArrayInterface.known_first(typeof(so)) == first(so)
@test ArrayInterface.known_last(typeof(so)) == last(so)
@test ArrayInterface.known_length(typeof(so)) == length(so)

su = StaticArrays.SUnitRange(2, 10)
@test ArrayInterface.known_first(typeof(su)) == first(su)
@test ArrayInterface.known_last(typeof(su)) == last(su)
@test ArrayInterface.known_length(typeof(su)) == length(su)

S = @SArray(zeros(2, 3, 4))
Sp = @view(PermutedDimsArray(S, (3, 1, 2))[2:3, 1:2, :]);
Sp2 = @view(PermutedDimsArray(S, (3, 2, 1))[2:3, :, :]);
Mp = @view(PermutedDimsArray(S, (3, 1, 2))[:, 2, :])';
Mp2 = @view(PermutedDimsArray(S, (3, 1, 2))[2:3, :, 2])';


irev = Iterators.reverse(S)
igen = Iterators.map(identity, S)
iacc = Iterators.accumulate(+, S)

ienum = enumerate(S)
ipairs = pairs(S)
izip = zip(S, S)

@test @inferred(ArrayInterface.known_size(S)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(irev)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(igen)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(iacc)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(ienum)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(izip)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(ipairs)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(zip(S, zeros(2, 3, 4, 1)))) === (2, 3, 4, 1)
@test @inferred(ArrayInterface.known_size(zip(zeros(2, 3, 4, 1), S))) === (2, 3, 4, 1)
@test @inferred(ArrayInterface.known_length(Iterators.flatten(((x, y) for x in 0:1 for y in 'a':'c')))) === nothing
@test ArrayInterface.known_length(S) == length(S)


@test @inferred(ArrayInterface.known_size(S)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(Sp)) === (nothing, nothing, 3)
@test @inferred(ArrayInterface.known_size(Sp2)) === (nothing, 3, 2)
@test ArrayInterface.known_size(Sp2, 1) === nothing
@test ArrayInterface.known_size(Sp2, 2) === 3
@test ArrayInterface.known_size(Sp2, 3) === 2
@test @inferred(ArrayInterface.known_size(Mp)) === (3, 4)
@test @inferred(ArrayInterface.known_size(Mp2)) === (2, nothing)

