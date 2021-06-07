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
ArrayInterface.has_dimnames(::Type{T}) where {T<:NamedDimsWrapper} = true
Base.parent(x::NamedDimsWrapper) = x.parent
function ArrayInterface.layout_plan(::Type{A}, s::ArrayInterface.AccessStyle) where {A<:NamedDimsWrapper}
    ArrayInterface.LayoutPlan(nothing, ArrayInterface.layout_plan(ArrayInterface.parent_type(A), s), parent)
end


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
    @test @inferred(ArrayInterface.to_parent_dims(typeof(vadj), static(1))) == 2
    @test @inferred(ArrayInterface.to_parent_dims(typeof(vadj), 1)) == 2
    @test @inferred(ArrayInterface.to_parent_dims(typeof(vadj), static(3))) == 2
    @test @inferred(ArrayInterface.to_parent_dims(typeof(vadj), 3)) == 2

    @test @inferred(ArrayInterface.from_parent_dims(typeof(a))) == (1, 2, 3)
    @test @inferred(ArrayInterface.from_parent_dims(typeof(perm))) == (2, 3, 1)
    @test @inferred(ArrayInterface.from_parent_dims(typeof(mview))) == (1, 0, 2)
    @test @inferred(ArrayInterface.from_parent_dims(typeof(madj))) == (2, 1)
    @test @inferred(ArrayInterface.from_parent_dims(typeof(vview))) == (0, 1)
    @test @inferred(ArrayInterface.from_parent_dims(typeof(vadj))) == (2,)
    @test @inferred(ArrayInterface.from_parent_dims(typeof(vadj), static(1))) == 2
    @test @inferred(ArrayInterface.from_parent_dims(typeof(vadj), 1)) == 2

    @test_throws DimensionMismatch ArrayInterface.to_parent_dims(typeof(vadj), 0)
    @test_throws DimensionMismatch ArrayInterface.to_parent_dims(typeof(vadj), static(0))

    @test_throws DimensionMismatch ArrayInterface.from_parent_dims(typeof(vadj), 0)
    @test_throws DimensionMismatch ArrayInterface.from_parent_dims(typeof(vadj), static(0))

    if VERSION ≥ v"1.6.0-DEV.1581"
        colormat = reinterpret(reshape, Float64, [(R = rand(), G = rand(), B = rand()) for i ∈ 1:100])
        @test @inferred(ArrayInterface.from_parent_dims(typeof(colormat))) === (static(2),)
        @test @inferred(ArrayInterface.to_parent_dims(typeof(colormat))) === (static(0), static(1),)

        Rr = reinterpret(reshape, Int32, ones(4))
        @test @inferred(ArrayInterface.from_parent_dims(typeof(Rr))) === (static(2),)
        @test @inferred(ArrayInterface.to_parent_dims(typeof(Rr))) === (static(0), static(1),)

        Rr = reinterpret(reshape, Int64, ones(4))
        @test @inferred(ArrayInterface.from_parent_dims(typeof(Rr))) === (static(1),)
        @test @inferred(ArrayInterface.to_parent_dims(typeof(Rr))) === (static(1),)

        Sr = reinterpret(reshape, Complex{Int64}, zeros(2, 3, 4))
        @test @inferred(ArrayInterface.from_parent_dims(typeof(Sr))) === (static(0), static(1), static(2))
        @test @inferred(ArrayInterface.to_parent_dims(typeof(Sr))) === (static(2), static(3))
    end
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
    @test @inferred(ArrayInterface.has_dimnames(ones(2,2))) === false
    @test @inferred(ArrayInterface.has_dimnames(Array{Int,2})) === false
    @test @inferred(val_has_dimnames(typeof(x))) === Val(true)
    @test @inferred(val_has_dimnames(typeof(view(x, :, 1, :)))) === Val(true)
    @test @inferred(dimnames(x)) === d
    @test @inferred(dimnames(parent(x))) === (static(:_), static(:_))
    @test @inferred(dimnames(x')) === reverse(d)
    @test @inferred(dimnames(y')) === (static(:_), static(:x))
    @test @inferred(dimnames(PermutedDimsArray(x, (2, 1)))) === reverse(d)
    @test @inferred(dimnames(PermutedDimsArray(x', (2, 1)))) === d
    @test @inferred(dimnames(view(x, :, 1))) === (static(:x),)
    @test @inferred(dimnames(view(x, :, :, :))) === (static(:x),static(:y), static(:_))
    @test @inferred(dimnames(view(x, :, 1, :))) === (static(:x), static(:_))
    @test @inferred(dimnames(x, ArrayInterface.One())) === static(:x)
    @test @inferred(dimnames(parent(x), ArrayInterface.One())) === static(:_)
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
        @test_throws DimensionMismatch ArrayInterface.to_dims(x, static(:z))  # not found
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
    @test @inferred(size(x, first(d))) == size(parent(x), 1)
    @test @inferred(ArrayInterface.size(y')) == (1, size(parent(x), 1))
    @test @inferred(axes(x, first(d))) == axes(parent(x), 1)
    @test strides(x, :x) == ArrayInterface.strides(parent(x))[1]
    @test @inferred(ArrayInterface.axes_types(x, static(:x))) <: Base.OneTo{Int}
    @test ArrayInterface.axes_types(x, :x) <: Base.OneTo{Int}
    @test @inferred(ArrayInterface.axes_types(LinearIndices{2,NTuple{2,Base.OneTo{Int}}})) <: NTuple{2,Base.OneTo{Int}}
    CI = CartesianIndices{2,Tuple{Base.OneTo{Int},UnitRange{Int}}}
    @test @inferred(ArrayInterface.axes_types(CI, static(1))) <: Base.OneTo{Int}

    x[x = 1] = [2, 3]
    @test @inferred(getindex(x, x = 1)) == [2, 3]
end

end

