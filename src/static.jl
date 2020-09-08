
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

@inline Base.iszero(::Static{0}) = true
@inline Base.iszero(::Static) = false

Base.:(+)(i::Number, ::Static{0}) = i
Base.:(+)(::Static{0}, i::Number) = i
Base.:(+)(::Static{0}, i::Integer) = i
Base.:(+)(::Static{0}, ::Static{0}) = Static{0}()
Base.:(+)(::Static{N}, ::Static{0}) where {N} = Static{N}()
Base.:(+)(::Static{0}, ::Static{N}) where {N} = Static{N}()
Base.:(+)(::Static{M}, ::Static{N}) where {M,N} = Static{M + N}()

Base.:(-)(::Static{0}, i::Number) = -i
Base.:(-)(i::Number, ::Static{0}) = i
Base.:(-)(::Static{0}, ::Static{0}) = Static{0}()
Base.:(-)(::Static{0}, ::Static{N}) where {N} = Static{-N}()
Base.:(-)(::Static{N}, ::Static{0}) where {N} = Static{N}()
Base.:(-)(::Static{M}, ::Static{N}) where {M,N} = Static{M - N}()

Base.:(*)(::Static{0}, i::Number) = Static{0}()
Base.:(*)(i::Number, ::Static{0}) = Static{0}()
Base.:(*)(::Static{0}, ::Static{M}) where {M} = Static{0}()
Base.:(*)(::Static{M}, ::Static{0}) where {M} = Static{0}()
Base.:(*)(::Static{0}, ::Static{0}) = Static{0}()
Base.:(*)(::Static{1}, i::Number) = i
Base.:(*)(i::Number, ::Static{1}) = i
Base.:(*)(::Static{0}, ::Static{1}) where {M} = Static{0}()
Base.:(*)(::Static{1}, ::Static{0}) where {M} = Static{0}()
Base.:(*)(::Static{M}, ::Static{1}) where {M} = Static{M}()
Base.:(*)(::Static{1}, ::Static{M}) where {M} = Static{M}()
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

Base.:(==)(::Static{M}, ::Static{N}) where {M,N} = false
Base.:(==)(::Static{M}, ::Static{M}) where {M} = true
Base.:(≤)(::Static{M}, N::Int) where {M} = M ≤ N
Base.:(≤)(N::Int, ::Static{M}) where {M} = N ≤ M
Base.:(≥)(::Static{M}, N::Int) where {M} = M ≤ N
Base.:(≥)(N::Int, ::Static{M}) where {M} = N ≥ M

@inline function maybe_static(f::F, g::G, x) where {F, G}
    L = f(x)
    isnothing(L) ? g(x) : Static(L)
end
@inline static_length(x) = maybe_static(known_length, length, x)
@inline static_first(x) = maybe_static(known_first, first, x)
@inline static_last(x) = maybe_static(known_last, last, x)
@inline static_step(x) = maybe_static(known_step, step, x)

