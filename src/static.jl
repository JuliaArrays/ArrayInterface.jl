
"""
A statically sized `Int`.
Use `Static(N)` instead of `Val(N)` when you want it to behave like a number.
"""
struct Static{N} <: Integer
    function Static{N}()  where {N}
        @assert isa(typeof(N), Base.BitIntegerType) "$N is not a primitive integer type"
        return new{N}()
    end
end
Base.@pure Static(N::Union{Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128}) = Static{N}()
Static(N) = Static(convert(Int, N))
Static(::Val{N}) where {N} = Static{N}()
Base.Val(::Static{N}) where {N} = Val{N}()
Base.convert(::Type{T}, ::Static{N}) where {T<:Number,N} = convert(T, N)
Base.convert(::Type{Static{N}}, ::Static{N}) where {N} = Static{N}()
Base.promote_rule(::Type{<:Static}, ::Type{T}) where {T} = promote_rule(Int, T)
Base.promote_rule(::Type{T}, ::Type{<:Static}) where {T} = promote_rule(T, Int)
Base.promote_rule(::Type{<:Static}, ::Type{<:Static}) where {T} = Int
Base.:(%)(::Static{N}, ::Type{Integer}) where {N} = N

@inline Base.iszero(::Static{M}) where {M} = iszero(M)
@inline Base.isone(::Static{M}) where {M} = isone(M)

for T ∈ [:Any, :Number]
    @eval begin
        @inline function Base.:(+)(i::$T, ::Static{M}) where {M}
            if iszero(M)
                i
            else
                i + M
            end
        end
        @inline function Base.:(+)(::Static{M}, i::$T) where {M}
            if iszero(M)
                i
            else
                M + i
            end
        end
        @inline function Base.:(-)(i::$T, ::Static{M}) where {M}
            if iszero(M)
                i
            else
                i - M
            end
        end
        @inline function Base.:(*)(i::$T, j::Static{M}) where {M}
            if iszero(M)
                j
            elseif isone(M)
                i
            else
                i * M
            end
        end
        @inline function Base.:(*)(j::Static{M}, i::$T) where {M}
            if iszero(M)
                j
            elseif isone(M)
                i
            else
                M * i
            end
        end
    end
end
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

