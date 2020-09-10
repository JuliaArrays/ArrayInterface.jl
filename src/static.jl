
"""
A statically sized `Int`.
Use `Static(N)` instead of `Val(N)` when you want it to behave like a number.
"""
struct Static{N} <: Integer
    Static{N}() where {N} = new{N::Int}()
end
Base.@pure Static(N::Int) = Static{N}()
Static(N::Integer) = Static(convert(Int, N))
Static(::Static{N}) where {N} = Static{N}()
Static(::Val{N}) where {N} = Static{N}()
Base.Val(::Static{N}) where {N} = Val{N}()
Base.convert(::Type{T}, ::Static{N}) where {T<:Number,N} = convert(T, N)
Base.convert(::Type{Static{N}}, ::Static{N}) where {N} = Static{N}()

Base.promote_rule(::Type{<:Static}, ::Type{T}) where {T <: AbstractIrrational} = promote_rule(Int, T)
Base.promote_rule(::Type{T}, ::Type{<:Static}) where {T <: AbstractIrrational} = promote_rule(T, Int)
for (S,T) ∈ [(:Complex,:Real), (:Rational, :Integer), (:(Base.TwicePrecision),:Any)]
    @eval Base.promote_rule(::Type{$S{T}}, ::Type{<:Static}) where {T <: $T} = promote_rule($S{T}, Int)
end
Base.promote_rule(::Type{Union{Nothing,Missing}}, ::Type{<:Static}) = Union{Nothing, Missing, Int}
Base.promote_rule(::Type{T}, ::Type{<:Static}) where {T >: Union{Missing,Nothing}} = promote_rule(T, Int)
Base.promote_rule(::Type{T}, ::Type{<:Static}) where {T >: Nothing} = promote_rule(T, Int)
Base.promote_rule(::Type{T}, ::Type{<:Static}) where {T >: Missing} = promote_rule(T, Int)
for T ∈ [:Bool, :Missing, :BigFloat, :BigInt, :Nothing, :Any]
# let S = :Any    
    @eval begin
        Base.promote_rule(::Type{S}, ::Type{$T}) where {S <: Static} = promote_rule(Int, $T)
        Base.promote_rule(::Type{$T}, ::Type{S}) where {S <: Static} = promote_rule($T, Int)
    end
end
Base.promote_rule(::Type{<:Static}, ::Type{<:Static}) = Int
Base.:(%)(::Static{N}, ::Type{Integer}) where {N} = N

Base.iszero(::Static{0}) = true
Base.iszero(::Static) = false
Base.isone(::Static{1}) = true
Base.isone(::Static) = false

for T = [:Real, :Rational, :Integer]
    @eval begin
        @inline Base.:(+)(i::$T, ::Static{0}) = i
        @inline Base.:(+)(i::$T, ::Static{M}) where {M} = i + M
        @inline Base.:(+)(::Static{0}, i::$T) = i
        @inline Base.:(+)(::Static{M}, i::$T) where {M} = M + i
        @inline Base.:(-)(i::$T, ::Static{0}) = i
        @inline Base.:(-)(i::$T, ::Static{M}) where {M} = i - M
        @inline Base.:(*)(i::$T, ::Static{0}) = Static{0}()
        @inline Base.:(*)(i::$T, ::Static{1}) = i
        @inline Base.:(*)(i::$T, ::Static{M}) where {M} = i * M
        @inline Base.:(*)(::Static{0}, i::$T) = Static{0}()
        @inline Base.:(*)(::Static{1}, i::$T) = i
        @inline Base.:(*)(::Static{M}, i::$T) where {M} = M * i
    end
end
@inline Base.:(+)(::Static{0}, ::Static{0}) = Static{0}()
@inline Base.:(+)(::Static{0}, ::Static{M}) where {M} = Static{M}()
@inline Base.:(+)(::Static{M}, ::Static{0}) where {M} = Static{M}()

@inline Base.:(-)(::Static{M}, ::Static{0}) where {M} = Static{M}()

@inline Base.:(*)(::Static{0}, ::Static{0}) = Static{0}()
@inline Base.:(*)(::Static{1}, ::Static{0}) = Static{0}()
@inline Base.:(*)(::Static{0}, ::Static{1}) = Static{0}()
@inline Base.:(*)(::Static{1}, ::Static{1}) = Static{1}()
@inline Base.:(*)(::Static{M}, ::Static{0}) where {M} = Static{0}()
@inline Base.:(*)(::Static{0}, ::Static{M}) where {M} = Static{0}()
@inline Base.:(*)(::Static{M}, ::Static{1}) where {M} = Static{M}()
@inline Base.:(*)(::Static{1}, ::Static{M}) where {M} = Static{M}()
for f ∈ [:(+), :(-), :(*), :(/), :(÷), :(%), :(<<), :(>>), :(>>>), :(&), :(|), :(⊻)]
    @eval @generated Base.$f(::Static{M}, ::Static{N}) where {M,N} = Expr(:call, Expr(:curly, :Static, $f(M, N)))
end
for f ∈ [:(==), :(!=), :(<), :(≤), :(>), :(≥)]
    @eval begin
        @inline Base.$f(::Static{M}, ::Static{N}) where {M,N} = $f(M, N)
        @inline Base.$f(::Static{M}, x::Int) where {M} = $f(M, x)
        @inline Base.$f(x::Int, ::Static{M}) where {M} = $f(x, M)
    end
end

@inline function maybe_static(f::F, g::G, x) where {F, G}
    L = f(x)
    isnothing(L) ? g(x) : Static(L)
end
@inline static_length(x) = maybe_static(known_length, length, x)
@inline static_first(x) = maybe_static(known_first, first, x)
@inline static_last(x) = maybe_static(known_last, last, x)
@inline static_step(x) = maybe_static(known_step, step, x)

