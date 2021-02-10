using ArrayInterface: dimnames

@testset "dimensions" begin

###
### define wrapper with dimnames
###

struct NamedDimsWrapper{L,T,N,P<:AbstractArray{T,N}} <: ArrayInterface.AbstractArray2{T,N}
    parent::P
    NamedDimsWrapper{L}(p) where {L} = new{L,eltype(p),ndims(p),typeof(p)}(p)
end
ArrayInterface.parent_type(::Type{T}) where {P,T<:NamedDimsWrapper{<:Any,<:Any,<:Any,P}} = P
ArrayInterface.has_dimnames(::Type{T}) where {T<:NamedDimsWrapper} = true
ArrayInterface.dimnames(::Type{T}) where {L,T<:NamedDimsWrapper{L}} = static(Val(L))
function ArrayInterface.dimnames(::Type{T}, dim) where {L,T<:NamedDimsWrapper{L}}
    if ndims(T) < dim
        return static(:_)
    else
        return static(L[dim])
    end
end
ArrayInterface.has_dimnames(::Type{T}) where {T} = true
Base.parent(x::NamedDimsWrapper) = x.parent

@testset "dimension permutations" begin
    a = ones(2, 2, 2)
    perm = PermutedDimsArray(a, (3, 1, 2))
    mview = view(perm, :, 1, :)
    madj = mview'
    vview = view(madj, 1, :)
    vadj = vview'

    @test @inferred(ArrayInterface.to_parent_dims(typeof(a))) == (1, 2, 3)
    @test @inferred(ArrayInterface.to_parent_dims(typeof(perm))) == (3, 1, 2)
    @test @inferred(ArrayInterface.to_parent_dims(typeof(mview))) == (1, 3)
    @test @inferred(ArrayInterface.to_parent_dims(typeof(madj))) == (2, 1)
    @test @inferred(ArrayInterface.to_parent_dims(typeof(vview))) == (2,)
    @test @inferred(ArrayInterface.to_parent_dims(typeof(vadj))) == (2, 1)

    @test @inferred(ArrayInterface.from_parent_dims(typeof(a))) == (1, 2, 3)
    @test @inferred(ArrayInterface.from_parent_dims(typeof(perm))) == (2, 3, 1)
    @test @inferred(ArrayInterface.from_parent_dims(typeof(mview))) == (1, 0, 2)
    @test @inferred(ArrayInterface.from_parent_dims(typeof(madj))) == (2, 1)
    @test @inferred(ArrayInterface.from_parent_dims(typeof(vview))) == (0, 1)
    @test @inferred(ArrayInterface.from_parent_dims(typeof(vadj))) == (2, 1)
end

@testset "order_named_inds" begin
    n1 = (static(:x),)
    n2 = (n1..., static(:y))
    n3 = (n2..., static(:z))
    @test @inferred(ArrayInterface.order_named_inds(n1, NamedTuple{(),Tuple{}}(()) )) == ()
    @test @inferred(ArrayInterface.order_named_inds(n1, (x=2,))) == (2,)
    @test @inferred(ArrayInterface.order_named_inds(n2, (x=2,))) == (2, :)
    @test @inferred(ArrayInterface.order_named_inds(n2, (y=2,))) == (:, 2)
    @test @inferred(ArrayInterface.order_named_inds(n2, (y=20, x=30))) == (30, 20)
    @test @inferred(ArrayInterface.order_named_inds(n2, (x=30, y=20))) == (30, 20)
    @test @inferred(ArrayInterface.order_named_inds(n3, (x=30, y=20))) == (30, 20, :)

    @test_throws ErrorException ArrayInterface.order_named_inds(n2, (x=30, y=20, z=40))
end

val_has_dimnames(x) = Val(ArrayInterface.has_dimnames(x))

@testset "dimnames" begin
    d = (static(:x), static(:y))
    x = NamedDimsWrapper{d}(ones(2,2));
    y = NamedDimsWrapper{(:x,)}(ones(2));
    dnums = ntuple(+, length(d))
    @test @inferred(val_has_dimnames(x)) === Val(true)
    @test @inferred(val_has_dimnames(typeof(x))) === Val(true)
    @test @inferred(val_has_dimnames(typeof(view(x, :, 1, :)))) === Val(true)
    @test @inferred(dimnames(x)) === d
    @test @inferred(dimnames(x')) === reverse(d)
    @test @inferred(dimnames(y')) === (static(:_), static(:x))
    @test @inferred(dimnames(PermutedDimsArray(x, (2, 1)))) === reverse(d)
    @test @inferred(dimnames(PermutedDimsArray(x', (2, 1)))) === d
    @test @inferred(dimnames(view(x, :, 1))) === (static(:x),)
    @test @inferred(dimnames(view(x, :, :, :))) === (static(:x),static(:y), static(:_))
    @test @inferred(dimnames(view(x, :, 1, :))) === (static(:x), static(:_))
    @test @inferred(dimnames(x, ArrayInterface.One())) === static(:x)
end

@testset "to_dims" begin
    x = NamedDimsWrapper{(:x, :y)}(ones(2,2));
    y = NamedDimsWrapper{(:x, :y, :a, :b, :c, :d)}(ones(6));

    @test @inferred(ArrayInterface.to_dims(x, :)) == Colon()
    @test @inferred(ArrayInterface.to_dims(x, 1)) == 1
    @testset "small case" begin
        @test @inferred(ArrayInterface.to_dims(x, (:x, :y))) == (1, 2)
        @test @inferred(ArrayInterface.to_dims(x, (:y, :x))) == (2, 1)
        @test @inferred(ArrayInterface.to_dims(x, :x)) == 1
        @test @inferred(ArrayInterface.to_dims(x, :y)) == 2
        @test_throws DimensionMismatch ArrayInterface.to_dims(x, :z)  # not found
    end

    @testset "large case" begin
        @test @inferred(ArrayInterface.to_dims(y, :x)) == 1
        @test @inferred(ArrayInterface.to_dims(y, :a)) == 3
        @test @inferred(ArrayInterface.to_dims(y, :d)) == 6
        @test_throws DimensionMismatch ArrayInterface.to_dims(y, :z) # not found
    end
end

@testset "methods accepting dimnames" begin
    d = (static(:x), static(:y))
    x = NamedDimsWrapper{d}(ones(2,2));
    y = NamedDimsWrapper{(:x,)}(ones(2));
    @test @inferred(size(x, :x)) == size(parent(x), 1)
    @test @inferred(ArrayInterface.size(y')) == (1, size(parent(x), 1))
    @test @inferred(axes(x, :x)) == axes(parent(x), 1)
    @test strides(x, :x) == ArrayInterface.strides(parent(x))[1]

    x[x = 1] = [2, 3]
    @test @inferred(getindex(x, x = 1)) == [2, 3]
end

end

