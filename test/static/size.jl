
@test @inferred(ArrayInterface.static_size(sv5)) === (StaticInt(5),)
@test @inferred(ArrayInterface.static_size(v5)) === (5,)
@test @inferred(ArrayInterface.static_size(A)) === (3, 4, 5)
@test @inferred(ArrayInterface.static_size(Ap)) === (2, 5)
@test @inferred(ArrayInterface.static_size(A)) === size(A)
@test @inferred(ArrayInterface.static_size(Ap)) === size(Ap)
@test @inferred(ArrayInterface.static_size(R)) === (StaticInt(2),)
@test @inferred(ArrayInterface.static_size(Rnr)) === (StaticInt(4),)
@test @inferred(ArrayInterface.known_length(Rnr)) === 4
@test @inferred(ArrayInterface.static_size(A2)) === (4, 3, 5)
@test @inferred(ArrayInterface.static_size(A2r)) === (2, 3, 5)

@test @inferred(ArrayInterface.static_size(view(rand(4), reshape(1:4, 2, 2)))) == (2, 2)
@test @inferred(ArrayInterface.static_size(irev)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterface.static_size(iprod)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterface.static_size(iflat)) === (static(72),)
@test @inferred(ArrayInterface.static_size(igen)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterface.static_size(iacc)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterface.static_size(ienum)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterface.static_size(ipairs)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterface.static_size(izip)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterface.static_size(zip(S, A_trailingdim))) === (StaticInt(2), StaticInt(3), StaticInt(4), static(1))
@test @inferred(ArrayInterface.static_size(zip(A_trailingdim, S))) === (StaticInt(2), StaticInt(3), StaticInt(4), static(1))
@test @inferred(ArrayInterface.static_size(S)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterface.static_size(Sp)) === (2, 2, StaticInt(3))
@test @inferred(ArrayInterface.static_size(Sp2)) === (2, StaticInt(3), StaticInt(2))
@test @inferred(ArrayInterface.static_size(S)) == size(S)
@test @inferred(ArrayInterface.static_size(Sp)) == size(Sp)
@test @inferred(ArrayInterface.static_size(parent(Sp2))) === (static(4), static(3), static(2))
@test @inferred(ArrayInterface.static_size(Sp2)) == size(Sp2)
@test @inferred(ArrayInterface.static_size(Sp2, StaticInt(1))) === 2
@test @inferred(ArrayInterface.static_size(Sp2, StaticInt(2))) === StaticInt(3)
@test @inferred(ArrayInterface.static_size(Sp2, StaticInt(3))) === StaticInt(2)
@test @inferred(ArrayInterface.static_size(Wrapper(Sp2), StaticInt(3))) === StaticInt(2)
@test @inferred(ArrayInterface.static_size(Diagonal([1, 2]))) == size(Diagonal([1, 2]))
@test @inferred(ArrayInterface.static_size(Mp)) === (StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterface.static_size(Mp2)) === (StaticInt(2), 2)
@test @inferred(ArrayInterface.static_size(Mp)) == size(Mp)
@test @inferred(ArrayInterface.static_size(Mp2)) == size(Mp2)

@test @inferred(ArrayInterface.known_size(1)) === ()
@test @inferred(ArrayInterface.known_size(view(rand(4), reshape(1:4, 2, 2)))) == (nothing, nothing)
@test @inferred(ArrayInterface.known_size(A)) === (nothing, nothing, nothing)
@test @inferred(ArrayInterface.known_size(Ap)) === (nothing, nothing)
@test @inferred(ArrayInterface.known_size(Wrapper(Ap))) === (nothing, nothing)
@test @inferred(ArrayInterface.known_size(R)) === (2,)
@test @inferred(ArrayInterface.known_size(Wrapper(R))) === (2,)
@test @inferred(ArrayInterface.known_size(Rnr)) === (4,)
@test @inferred(ArrayInterface.known_size(Rnr, static(1))) === 4
@test @inferred(ArrayInterface.known_size(Ar)) === (nothing, nothing, nothing,)
@test @inferred(ArrayInterface.known_size(Ar, static(1))) === nothing
@test @inferred(ArrayInterface.known_size(Ar, static(4))) === 1
@test @inferred(ArrayInterface.known_size(A2)) === (nothing, nothing, nothing)
@test @inferred(ArrayInterface.known_size(A2r)) === (nothing, nothing, nothing)

@test @inferred(ArrayInterface.known_size(irev)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(igen)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(iprod)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(iflat)) === (72,)
@test @inferred(ArrayInterface.known_size(iacc)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(ienum)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(izip)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(ipairs)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(zip(S, A_trailingdim))) === (2, 3, 4, 1)
@test @inferred(ArrayInterface.known_size(zip(A_trailingdim, S))) === (2, 3, 4, 1)
@test @inferred(ArrayInterface.known_length(Iterators.flatten(((x, y) for x in 0:1 for y in 'a':'c')))) === nothing

@test @inferred(ArrayInterface.known_size(S)) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(Wrapper(S))) === (2, 3, 4)
@test @inferred(ArrayInterface.known_size(Sp)) === (nothing, nothing, 3)
@test @inferred(ArrayInterface.known_size(Wrapper(Sp))) === (nothing, nothing, 3)
@test @inferred(ArrayInterface.known_size(Sp2)) === (nothing, 3, 2)
@test @inferred(ArrayInterface.known_size(Sp2, StaticInt(1))) === nothing
@test @inferred(ArrayInterface.known_size(Sp2, StaticInt(2))) === 3
@test @inferred(ArrayInterface.known_size(Sp2, StaticInt(3))) === 2
@test @inferred(ArrayInterface.known_size(Mp)) === (3, 4)
@test @inferred(ArrayInterface.known_size(Mp2)) === (2, nothing)

@testset "known_length" begin
    @test ArrayInterface.known_length(1:2) === nothing
    @test ArrayInterface.known_length((1,)) == 1
    @test ArrayInterface.known_length((a=1, b=2)) == 2
    @test ArrayInterface.known_length([]) === nothing
    @test ArrayInterface.known_length(CartesianIndex((1, 2, 3))) === 3
    @test @inferred(ArrayInterface.known_length(NDIndex((1, 2, 3)))) === 3

    itr = StaticInt(1):StaticInt(10)
    @inferred(ArrayInterface.known_length((i for i in itr))) == 10
end

@testset "is_lazy_conjugate" begin
    a = rand(ComplexF64, 2)
    @test @inferred(ArrayInterface.is_lazy_conjugate(a)) == false
    b = a'
    @test @inferred(ArrayInterface.is_lazy_conjugate(b)) == true
    c = transpose(b)
    @test @inferred(ArrayInterface.is_lazy_conjugate(c)) == true
    d = c'
    @test @inferred(ArrayInterface.is_lazy_conjugate(d)) == false
    e = permutedims(d)
    @test @inferred(ArrayInterface.is_lazy_conjugate(e)) == false
    @test @inferred(ArrayInterface.is_lazy_conjugate([1, 2, 3]')) == false # We don't care about conj on `<:Real`
end
