
"""
A statically sized `Int`.
Use `Static(N)` instead of `Val(N)` when you want it to behave like a number.
"""
struct Static{N} <: Integer
    Static{N}() where {N} = new{N::Int}()
end
Base.@pure Static(N::Int) = Static{N}()
Static(N) = Static(convert(Int, N))
Static(::Val{N}) where {N} = Static{N}()
Base.Val(::Static{N}) where {N} = Val{N}()
Base.convert(::Type{T}, ::Static{N}) where {T<:Number,N} = convert(T, N)
Base.convert(::Type{Static{N}}, ::Static{N}) where {N} = Static{N}()
Base.promote_rule(::Type{<:Static}, ::Type{T}) where {T} = promote_rule(Int, T)
Base.promote_rule(::Type{T}, ::Type{<:Static}) where {T} = promote_rule(T, Int)
Base.promote_rule(::Type{<:Static}, ::Type{<:Static}) where {T} = Int
Base.:(%)(::Static{N}, ::Type{Integer}) where {N} = N

Base.iszero(::Static{0}) = true
Base.iszero(::Static) = false
Base.isone(::Static{1}) = true
Base.isone(::Static) = false

for T ∈ [:Any, :Number, :Integer]
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
@inline Base.:(*)(::Static{0}, ::Static{0}) = Static{0}()
@inline Base.:(*)(::Static{1}, ::Static{0}) = Static{0}()
@inline Base.:(*)(::Static{0}, ::Static{1}) = Static{0}()
@inline Base.:(*)(::Static{1}, ::Static{1}) = Static{1}()
for f ∈ [:(+), :(-), :(*), :(/), :(÷), :(%), :(<<), :(>>), :(>>>), :(&), :(|), :(⊻)]
    @eval @generated Base.$f(::Static{M}, ::Static{N}) where {M,N} = Expr(:call, Expr(:curly, :Static, $f(M, N)))
end
for f ∈ [:(==), :(!=), :(<), :(≤), :(>), :(≥)]
    @eval begin
        @inline Base.$f(::Static{M}, ::Static{N}) where {M,N} = $f(M, N)
        @inline Base.$f(::Static{M}, x::Integer) where {M} = $f(M, x)
        @inline Base.$f(x::Integer, ::Static{M}) where {M} = $f(x, M)
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

