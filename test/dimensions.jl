
@testset "dimensions" begin

struct NamedDimsWrapper{L,T,N,P<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::P
    NamedDimsWrapper{L}(p) where {L} = new{L,eltype(p),ndims(p),typeof(p)}(p)
end
ArrayInterface.parent_type(::Type{T}) where {P,T<:NamedDimsWrapper{<:Any,P}} = P
ArrayInterface.has_dimnames(::Type{T}) where {T<:NamedDimsWrapper} = true
ArrayInterface.dimnames(::Type{T}) where {L,T<:NamedDimsWrapper{L}} = L
Base.parent(x::NamedDimsWrapper) = x.parent
Base.size(x::NamedDimsWrapper) = size(parent(x))
Base.axes(x::NamedDimsWrapper) = axes(parent(x))
Base.strides(x::NamedDimsWrapper) = Base.strides(parent(x))

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

@test @inferred(ArrayInterface.size(x, :x)) == size(parent(x), 1)
@test @inferred(ArrayInterface.axes(x, :x)) == axes(parent(x), 1)
@test @inferred(ArrayInterface.strides(x, :x)) == strides(parent(x))[1]

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
