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

@testset "to_dims" begin
    @testset "small case" begin
        @test ArrayInterface.to_dims((:x, :y), :x) == 1
        @test ArrayInterface.to_dims((:x, :y), :y) == 2
        @test_throws ArgumentError ArrayInterface.to_dims((:x, :y), :z)  # not found
    end

    @testset "large case" begin
        @test ArrayInterface.to_dims((:x, :y, :a, :b, :c, :d), :x) == 1
        @test ArrayInterface.to_dims((:x, :y, :a, :b, :c, :d), :a) == 3
        @test ArrayInterface.to_dims((:x, :y, :a, :b, :c, :d), :d) == 6
        @test_throws ArgumentError ArrayInterface.to_dims((:x, :y, :a, :b, :c, :d), :z) # not found
    end
end

struct NamedDimsWrapper{L,T,N,P<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::P
    NamedDimsWrapper{L}(p) where {L} = new{L,eltype(p),ndims(p),typeof(p)}(p)
end
ArrayInterface.parent_type(::Type{T}) where {P,T<:NamedDimsWrapper{<:Any,<:Any,<:Any,P}} = P
ArrayInterface.has_dimnames(::Type{T}) where {T<:NamedDimsWrapper} = true
ArrayInterface.dimnames(::Type{T}) where {L,T<:NamedDimsWrapper{L}} = L
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
val_dimnames(x) = Val(ArrayInterface.dimnames(x))
val_dimnames(x, d) = Val(ArrayInterface.dimnames(x, d))

d = (:x, :y)
x = NamedDimsWrapper{d}(ones(2,2))
y = NamedDimsWrapper{(:x,)}(ones(2))
dnums = ntuple(+, length(d))
@test @inferred(val_has_dimnames(x)) === Val(true)
@test @inferred(val_has_dimnames(typeof(x)))  === Val(true)
@test @inferred(val_dimnames(x)) === Val(d)
@test @inferred(val_dimnames(x')) === Val(reverse(d))
@test @inferred(val_dimnames(y')) === Val((:_, :x))
@test @inferred(val_dimnames(PermutedDimsArray(x, (2, 1)))) ===Val(reverse(d))
@test @inferred(val_dimnames(view(x, :, 1))) === Val((:x,))
@test @inferred(val_dimnames(view(x, :, :, :))) === Val((:x, :y, :_))
@test @inferred(val_dimnames(view(x, :, 1, :))) === Val((:x, :_))
@test @inferred(val_dimnames(x, ArrayInterface.One())) === Val(:x)
@test @inferred(ArrayInterface.to_dims(x, d)) === dnums
@test @inferred(ArrayInterface.to_dims(x, reverse(d))) === reverse(dnums)
@test_throws ArgumentError ArrayInterface.to_dims(x, :z)

@test @inferred(size(x, :x)) == size(parent(x), 1)
@test @inferred(axes(x, :x)) == axes(parent(x), 1)
@test strides(x, :x) == ArrayInterface.strides(parent(x))[1]

x[x = 1] = [2, 3]
@test @inferred(getindex(x, x = 1)) == [2, 3]

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

end
