
"""
A statically sized `Int`.
Use `StaticInt(N)` instead of `Val(N)` when you want it to behave like a number.
"""
struct StaticInt{N} <: Integer
    StaticInt{N}() where {N} = new{N::Int}()
end

const Zero = StaticInt{0}
const One = StaticInt{1}

Base.@pure StaticInt(N::Int) = StaticInt{N}()
StaticInt(N::Integer) = StaticInt(convert(Int, N))
StaticInt(::StaticInt{N}) where {N} = StaticInt{N}()
StaticInt(::Val{N}) where {N} = StaticInt{N}()
# Base.Val(::StaticInt{N}) where {N} = Val{N}()
Base.convert(::Type{T}, ::StaticInt{N}) where {T<:Number,N} = convert(T, N)
# (::Type{T})(::ArrayInterface.StaticInt{N}) where {T,N} = T(N)
Base.convert(::Type{StaticInt{N}}, ::StaticInt{N}) where {N} = StaticInt{N}()

Base.promote_rule(::Type{<:StaticInt}, ::Type{T}) where {T <: AbstractIrrational} = promote_rule(Int, T)
Base.promote_rule(::Type{T}, ::Type{<:StaticInt}) where {T <: AbstractIrrational} = promote_rule(T, Int)
for (S,T) ∈ [(:Complex,:Real), (:Rational, :Integer), (:(Base.TwicePrecision),:Any)]
    @eval Base.promote_rule(::Type{$S{T}}, ::Type{<:StaticInt}) where {T <: $T} = promote_rule($S{T}, Int)
end
Base.promote_rule(::Type{Union{Nothing,Missing}}, ::Type{<:StaticInt}) = Union{Nothing, Missing, Int}
Base.promote_rule(::Type{T}, ::Type{<:StaticInt}) where {T >: Union{Missing,Nothing}} = promote_rule(T, Int)
Base.promote_rule(::Type{T}, ::Type{<:StaticInt}) where {T >: Nothing} = promote_rule(T, Int)
Base.promote_rule(::Type{T}, ::Type{<:StaticInt}) where {T >: Missing} = promote_rule(T, Int)
for T ∈ [:Bool, :Missing, :BigFloat, :BigInt, :Nothing, :Any]
# let S = :Any    
    @eval begin
        Base.promote_rule(::Type{S}, ::Type{$T}) where {S <: StaticInt} = promote_rule(Int, $T)
        Base.promote_rule(::Type{$T}, ::Type{S}) where {S <: StaticInt} = promote_rule($T, Int)
    end
end
Base.promote_rule(::Type{<:StaticInt}, ::Type{<:StaticInt}) = Int
Base.:(%)(::StaticInt{N}, ::Type{Integer}) where {N} = N

Base.eltype(::Type{T}) where {T<:StaticInt} = Int
Base.iszero(::Zero) = true
Base.iszero(::StaticInt) = false
Base.isone(::One) = true
Base.isone(::StaticInt) = false
Base.zero(::Type{T}) where {T<:StaticInt} = Zero()
Base.one(::Type{T}) where {T<:StaticInt} = One()

for T = [:Real, :Rational, :Integer]
    @eval begin
        @inline Base.:(+)(i::$T, ::Zero) = i
        @inline Base.:(+)(i::$T, ::StaticInt{M}) where {M} = i + M
        @inline Base.:(+)(::Zero, i::$T) = i
        @inline Base.:(+)(::StaticInt{M}, i::$T) where {M} = M + i
        @inline Base.:(-)(i::$T, ::Zero) = i
        @inline Base.:(-)(i::$T, ::StaticInt{M}) where {M} = i - M
        @inline Base.:(*)(i::$T, ::Zero) = Zero()
        @inline Base.:(*)(i::$T, ::One) = i
        @inline Base.:(*)(i::$T, ::StaticInt{M}) where {M} = i * M
        @inline Base.:(*)(::Zero, i::$T) = Zero()
        @inline Base.:(*)(::One, i::$T) = i
        @inline Base.:(*)(::StaticInt{M}, i::$T) where {M} = M * i
    end
end
@inline Base.:(+)(::Zero, ::Zero) = Zero()
@inline Base.:(+)(::Zero, ::StaticInt{M}) where {M} = StaticInt{M}()
@inline Base.:(+)(::StaticInt{M}, ::Zero) where {M} = StaticInt{M}()

@inline Base.:(-)(::StaticInt{M}, ::Zero) where {M} = StaticInt{M}()

@inline Base.:(*)(::Zero, ::Zero) = Zero()
@inline Base.:(*)(::One, ::Zero) = Zero()
@inline Base.:(*)(::Zero, ::One) = Zero()
@inline Base.:(*)(::One, ::One) = One()
@inline Base.:(*)(::StaticInt{M}, ::Zero) where {M} = Zero()
@inline Base.:(*)(::Zero, ::StaticInt{M}) where {M} = Zero()
@inline Base.:(*)(::StaticInt{M}, ::One) where {M} = StaticInt{M}()
@inline Base.:(*)(::One, ::StaticInt{M}) where {M} = StaticInt{M}()
for f ∈ [:(+), :(-), :(*), :(/), :(÷), :(%), :(<<), :(>>), :(>>>), :(&), :(|), :(⊻)]
    @eval @generated Base.$f(::StaticInt{M}, ::StaticInt{N}) where {M,N} = Expr(:call, Expr(:curly, :StaticInt, $f(M, N)))
end
for f ∈ [:(==), :(!=), :(<), :(≤), :(>), :(≥)]
    @eval begin
        @inline Base.$f(::StaticInt{M}, ::StaticInt{N}) where {M,N} = $f(M, N)
        @inline Base.$f(::StaticInt{M}, x::Int) where {M} = $f(M, x)
        @inline Base.$f(x::Int, ::StaticInt{M}) where {M} = $f(x, M)
    end
end

@inline function maybe_static(f::F, g::G, x) where {F, G}
    L = f(x)
    isnothing(L) ? g(x) : StaticInt(L)
end
@inline static_length(x) = maybe_static(known_length, length, x)
@inline static_first(x) = maybe_static(known_first, first, x)
@inline static_last(x) = maybe_static(known_last, last, x)
@inline static_step(x) = maybe_static(known_step, step, x)

