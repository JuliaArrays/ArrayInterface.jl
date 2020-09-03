
"""
A statically sized `Int`.
Use `Static(N)` instead of `Val(N)` when you want it to behave like a number.
"""
struct Static{N} <: Integer
    Static{N}()  where {N} = new{N::Int}()
end
Base.@pure Static(N::Int) = Static{N}()
Static(N) = Static(convert(Int,N))
Static(::Val{N}) where {N} = Static{N}()
@inline Base.Val(::Static{N}) where {N} = Val{N}()
Base.convert(::Type{T}, ::Static{N}) where {T<:Number,N} = convert(T, N)
Base.convert(::Type{Static{N}}, ::Static{N}) where {N} = Static{N}()
Base.promote_rule(::Type{<:Static}, ::Type{T}) where {T} = promote_rule(Int, T)
Base.promote_rule(::Type{T}, ::Type{<:Static}) where {T} = promote_rule(T, Int)
Base.promote_rule(::Type{<:Static}, ::Type{<:Static}) where {T} = Int
Base.:(%)(::Static{N}, ::Type{Integer}) where {N} = N

_get(::Static{N}) where {N} = N
_get(::Type{Static{N}}) where {N} = N

@inline Base.iszero(::Static{0}) = true
@inline Base.iszero(::Static) = false

Base.:(+)(i::Number, ::Static{0}) = i
Base.:(+)(::Static{0}, i::Number) = i
Base.:(+)(::Static{0}, ::Static{0}) = Static{0}()
Base.:(+)(::Static{M}, ::Static{N}) where {M,N} = Static{M + N}()

Base.:(-)(::Static{0}, i::Number) = -i
Base.:(-)(i::Number, ::Static{0}) = i
Base.:(-)(::Static{0}, ::Static{0}) = Static{0}()
Base.:(-)(::Static{M}, ::Static{N}) where {M,N} = Static{M - N}()

Base.:(*)(::Static{0}, i::Number) = Static{0}()
Base.:(*)(i::Number, ::Static{0}) = Static{0}()
Base.:(*)(::Static{0}, ::Static{0}) = Static{0}()
Base.:(*)(::Static{1}, i::Number) = i
Base.:(*)(i::Number, ::Static{1}) = i
Base.:(*)(::Static{1}, ::Static{1}) = Static{1}()
Base.:(*)(::Static{M}, ::Static{N}) where {M,N} = Static{M * N}()

Base.:(÷)(::Static{M}, ::Static{N}) where {M,N} = Static{M ÷ N}()
Base.:(%)(::Static{M}, ::Static{N}) where {M,N} = Static{M % N}()
Base.:(<<)(::Static{M}, ::Static{N}) where {M,N} = Static{M << N}()
Base.:(>>)(::Static{M}, ::Static{N}) where {M,N} = Static{M >> N}()
Base.:(>>>)(::Static{M}, ::Static{N}) where {M,N} = Static{M >>> N}()
Base.:(&)(::Static{M}, ::Static{N}) where {M,N} = Static{M & N}()
Base.:(|)(::Static{M}, ::Static{N}) where {M,N} = Static{M | N}()
Base.:(⊻)(::Static{M}, ::Static{N}) where {M,N} = Static{M ⊻ N}()

Base.:(:)(L, ::Static{U}) where {U} = OptionallyStaticUnitRange(L, Val(U))
Base.:(:)(::Static{L}, U) where {L} = OptionallyStaticUnitRange(Val(L), U)
Base.:(:)(::Static{L}, ::Static{U}) where {L,U} = OptionallyStaticUnitRange(Val(L), Val(U))

Base.:(==)(::Static{M}, ::Static{N}) where {M,N} = false
Base.:(==)(::Static{M}, ::Static{M}) where {M} = true

"""
  ntuple(f::F, ::Static{N}) -> Tuple{Vararg{Any,N}}

Fully unrolled evaluation of `1:N`.
"""
@generated function Base.ntuple(f::F, ::Static{N}) where {F,N}
    t = Expr(:tuple)
    foreach(n -> push!(t.args, Expr(:call, :f, n)), 1:N)
    Expr(:block, Expr(:meta, :inline), t)
end

