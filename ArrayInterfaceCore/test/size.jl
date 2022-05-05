
@test @inferred(ArrayInterfaceCore.size(sv5)) === (StaticInt(5),)
@test @inferred(ArrayInterfaceCore.size(v5)) === (5,)
@test @inferred(ArrayInterfaceCore.size(A)) === (3,4,5)
@test @inferred(ArrayInterfaceCore.size(Ap)) === (2,5)
@test @inferred(ArrayInterfaceCore.size(A)) === size(A)
@test @inferred(ArrayInterfaceCore.size(Ap)) === size(Ap)
@test @inferred(ArrayInterfaceCore.size(R)) === (StaticInt(2),)
@test @inferred(ArrayInterfaceCore.size(Rnr)) === (StaticInt(4),)
@test @inferred(ArrayInterfaceCore.known_length(Rnr)) === 4
@test @inferred(ArrayInterfaceCore.size(A2)) === (4,3,5)
@test @inferred(ArrayInterfaceCore.size(A2r)) === (2,3,5)

@test @inferred(ArrayInterfaceCore.size(irev)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterfaceCore.size(iprod)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterfaceCore.size(iflat)) === (static(72),)
@test @inferred(ArrayInterfaceCore.size(igen)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterfaceCore.size(iacc)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterfaceCore.size(ienum)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterfaceCore.size(ipairs)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterfaceCore.size(izip)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterfaceCore.size(zip(S, A_trailingdim))) === (StaticInt(2), StaticInt(3), StaticInt(4), static(1))
@test @inferred(ArrayInterfaceCore.size(zip(A_trailingdim, S))) === (StaticInt(2), StaticInt(3), StaticInt(4), static(1))
@test @inferred(ArrayInterfaceCore.size(S)) === (StaticInt(2), StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterfaceCore.size(Sp)) === (2, 2, StaticInt(3))
@test @inferred(ArrayInterfaceCore.size(Sp2)) === (2, StaticInt(3), StaticInt(2))
@test @inferred(ArrayInterfaceCore.size(S)) == size(S)
@test @inferred(ArrayInterfaceCore.size(Sp)) == size(Sp)
@test @inferred(ArrayInterfaceCore.size(parent(Sp2))) === (static(4), static(3), static(2))
@test @inferred(ArrayInterfaceCore.size(Sp2)) == size(Sp2)
@test @inferred(ArrayInterfaceCore.size(Sp2, StaticInt(1))) === 2
@test @inferred(ArrayInterfaceCore.size(Sp2, StaticInt(2))) === StaticInt(3)
@test @inferred(ArrayInterfaceCore.size(Sp2, StaticInt(3))) === StaticInt(2)
@test @inferred(ArrayInterfaceCore.size(Wrapper(Sp2), StaticInt(3))) === StaticInt(2)
@test @inferred(ArrayInterfaceCore.size(Diagonal([1,2]))) == size(Diagonal([1,2]))
@test @inferred(ArrayInterfaceCore.size(Mp)) === (StaticInt(3), StaticInt(4))
@test @inferred(ArrayInterfaceCore.size(Mp2)) === (StaticInt(2), 2)
@test @inferred(ArrayInterfaceCore.size(Mp)) == size(Mp)
@test @inferred(ArrayInterfaceCore.size(Mp2)) == size(Mp2)

@test @inferred(ArrayInterfaceCore.known_size(A)) === (nothing, nothing, nothing)
@test @inferred(ArrayInterfaceCore.known_size(Ap)) === (nothing,nothing)
@test @inferred(ArrayInterfaceCore.known_size(Wrapper(Ap))) === (nothing,nothing)
@test @inferred(ArrayInterfaceCore.known_size(R)) === (2,)
@test @inferred(ArrayInterfaceCore.known_size(Wrapper(R))) === (2,)
@test @inferred(ArrayInterfaceCore.known_size(Rnr)) === (4,)
@test @inferred(ArrayInterfaceCore.known_size(Rnr, static(1))) === 4
@test @inferred(ArrayInterfaceCore.known_size(Ar)) === (nothing,nothing, nothing,)
@test @inferred(ArrayInterfaceCore.known_size(Ar, static(1))) === nothing
@test @inferred(ArrayInterfaceCore.known_size(Ar, static(4))) === 1
@test @inferred(ArrayInterfaceCore.known_size(A2)) === (nothing, nothing, nothing)
@test @inferred(ArrayInterfaceCore.known_size(A2r)) === (nothing, nothing, nothing)

@test @inferred(ArrayInterfaceCore.known_size(irev)) === (2, 3, 4)
@test @inferred(ArrayInterfaceCore.known_size(igen)) === (2, 3, 4)
@test @inferred(ArrayInterfaceCore.known_size(iprod)) === (2, 3, 4)
@test @inferred(ArrayInterfaceCore.known_size(iflat)) === (72,)
@test @inferred(ArrayInterfaceCore.known_size(iacc)) === (2, 3, 4)
@test @inferred(ArrayInterfaceCore.known_size(ienum)) === (2, 3, 4)
@test @inferred(ArrayInterfaceCore.known_size(izip)) === (2, 3, 4)
@test @inferred(ArrayInterfaceCore.known_size(ipairs)) === (2, 3, 4)
@test @inferred(ArrayInterfaceCore.known_size(zip(S, A_trailingdim))) === (2, 3, 4, 1)
@test @inferred(ArrayInterfaceCore.known_size(zip(A_trailingdim, S))) === (2, 3, 4, 1)
@test @inferred(ArrayInterfaceCore.known_length(Iterators.flatten(((x,y) for x in 0:1 for y in 'a':'c')))) === nothing

@test @inferred(ArrayInterfaceCore.known_size(S)) === (2, 3, 4)
@test @inferred(ArrayInterfaceCore.known_size(Wrapper(S))) === (2, 3, 4)
@test @inferred(ArrayInterfaceCore.known_size(Sp)) === (nothing, nothing, 3)
@test @inferred(ArrayInterfaceCore.known_size(Wrapper(Sp))) === (nothing, nothing, 3)
@test @inferred(ArrayInterfaceCore.known_size(Sp2)) === (nothing, 3, 2)
@test @inferred(ArrayInterfaceCore.known_size(Sp2, StaticInt(1))) === nothing
@test @inferred(ArrayInterfaceCore.known_size(Sp2, StaticInt(2))) === 3
@test @inferred(ArrayInterfaceCore.known_size(Sp2, StaticInt(3))) === 2
@test @inferred(ArrayInterfaceCore.known_size(Mp)) === (3, 4)
@test @inferred(ArrayInterfaceCore.known_size(Mp2)) === (2, nothing)

