using ArrayInterface: dimnames

@testset "dimensions" begin

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
    @test @inferred(ArrayInterface.order_named_inds(n1, (x=2,))) == (2,)
    @test @inferred(ArrayInterface.order_named_inds(n2, (x=2,))) == (2, :)
    @test @inferred(ArrayInterface.order_named_inds(n2, (y=2,))) == (:, 2)
    @test @inferred(ArrayInterface.order_named_inds(n2, (y=20, x=30))) == (30, 20)
    @test @inferred(ArrayInterface.order_named_inds(n2, (x=30, y=20))) == (30, 20)
    @test @inferred(ArrayInterface.order_named_inds(n3, (x=30, y=20))) == (30, 20, :)
end

struct NamedDimsWrapper{L,T,N,P<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::P
    NamedDimsWrapper{L}(p) where {L} = new{L,eltype(p),ndims(p),typeof(p)}(p)
end
ArrayInterface.parent_type(::Type{T}) where {P,T<:NamedDimsWrapper{<:Any,<:Any,<:Any,P}} = P
ArrayInterface.has_dimnames(::Type{T}) where {T<:NamedDimsWrapper} = true
ArrayInterface.dimnames(::Type{T}) where {L,T<:NamedDimsWrapper{L}} = static(Val(L))
Base.parent(x::NamedDimsWrapper) = x.parent
Base.size(x::NamedDimsWrapper) = size(parent(x))
Base.size(x::NamedDimsWrapper, d) = ArrayInterface.size(x, d)
Base.axes(x::NamedDimsWrapper) = axes(parent(x))
Base.axes(x::NamedDimsWrapper, d) = ArrayInterface.axes(x, d)
Base.strides(x::NamedDimsWrapper) = Base.strides(parent(x))
Base.strides(x::NamedDimsWrapper, d) =  ArrayInterface.strides(x, d)

Base.getindex(x::NamedDimsWrapper; kwargs...) = ArrayInterface.getindex(x; kwargs...)
Base.getindex(x::NamedDimsWrapper, args...) = ArrayInterface.getindex(x, args...)
Base.setindex!(x::NamedDimsWrapper, val; kwargs...) = ArrayInterface.setindex!(x, val; kwargs...)
Base.setindex!(x::NamedDimsWrapper, val, args...) = ArrayInterface.setindex!(x, val, args...)
function ArrayInterface.unsafe_get_element(x::NamedDimsWrapper, inds; kwargs...)
    return @inbounds(parent(x)[inds...])
end
function ArrayInterface.unsafe_set_element!(x::NamedDimsWrapper, val, inds; kwargs...)
    return @inbounds(parent(x)[inds...] = val)
end

val_has_dimnames(x) = Val(ArrayInterface.has_dimnames(x))

d = (static(:x), static(:y))
x = NamedDimsWrapper{d}(ones(2,2));
y = NamedDimsWrapper{(:x,)}(ones(2));
dnums = ntuple(+, length(d))
@test @inferred(val_has_dimnames(x)) === Val(true)
@test @inferred(val_has_dimnames(typeof(x)))  === Val(true)
@test @inferred(dimnames(x)) === d
@test @inferred(dimnames(x')) === reverse(d)
@test @inferred(dimnames(y')) === (static(:_), static(:x))
@test @inferred(dimnames(PermutedDimsArray(x, (2, 1)))) === reverse(d)
@test @inferred(dimnames(view(x, :, 1))) === (static(:x),)
@test @inferred(dimnames(view(x, :, :, :))) === (static(:x),static(:y), static(:_))
@test @inferred(dimnames(view(x, :, 1, :))) === (static(:x), static(:_))
@test @inferred(dimnames(x, ArrayInterface.One())) === static(:x)
@test @inferred(ArrayInterface.to_dims(x, d)) == dnums
@test @inferred(ArrayInterface.to_dims(x, reverse(d))) == reverse(dnums)
@test_throws ArgumentError ArrayInterface.to_dims(x, :z)


@testset "to_dims" begin
    x = NamedDimsWrapper{(:x, :y)}(ones(2,2));
    y = NamedDimsWrapper{(:x, :y, :a, :b, :c, :d)}(ones(6));
    @testset "small case" begin
        @test @inferred(ArrayInterface.to_dims(x, :x)) == 1
        @test @inferred(ArrayInterface.to_dims(x, :y)) == 2
        @test_throws ArgumentError ArrayInterface.to_dims(x, :z)  # not found
    end

    @testset "large case" begin
        @test @inferred(ArrayInterface.to_dims(y, :x)) == 1
        @test @inferred(ArrayInterface.to_dims(y, :a)) == 3
        @test @inferred(ArrayInterface.to_dims(y, :d)) == 6
        @test_throws ArgumentError ArrayInterface.to_dims(y, :z) # not found
    end
end


@test @inferred(size(x, :x)) == size(parent(x), 1)
@test @inferred(axes(x, :x)) == axes(parent(x), 1)
@test strides(x, :x) == ArrayInterface.strides(parent(x))[1]

x[x = 1] = [2, 3]
@test @inferred(getindex(x, x = 1)) == [2, 3]

#= FIXME these tests need to be replaced
@testset "order_named_inds" begin
    @test ArrayInterface.order_named_inds(Val((:x,)); x=2) == (2,)
    @test ArrayInterface.order_named_inds(Val((:x, :y)); x=2) == (2, :)
    @test ArrayInterface.order_named_inds(Val((:x, :y)); y=2, ) == (:, 2)
    @test ArrayInterface.order_named_inds(Val((:x, :y)); y=20, x=30) == (30, 20)
    @test ArrayInterface.order_named_inds(Val((:x, :y)); x=30, y=20) == (30, 20)
end

@testset "tuple_issubset" begin
    @test ArrayInterface.tuple_issubset((:a, :c), (:a, :b, :c)) == true
    @test ArrayInterface.tuple_issubset((:a, :b, :c), (:a, :c)) == false
end
=#

end
