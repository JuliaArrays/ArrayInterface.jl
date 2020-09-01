
"""
A statically sized `Int`.
Use `Static(N)` instead of `Val(N)` when you want it to behave like a number.
"""
struct Static{N} <: Integer
    Static{N}()  where {N} = new{N::Int}()
end
Base.@pure Static(N::Int) = Static{N}()
Base.convert(::Type{T}, ::Static{N}) where {T<:Number,N} = convert(T, N)
Base.promote_rule(::Type{<:Static}, ::Type{T}) where {T} = promote_rule(Int, T)
Base.promote_rule(::Type{T}, ::Type{<:Static}) where {T} = promote_rule(T, Int)

_get(::Static{N}) where {N} = N
_get(::Type{Static{N}}) where {N} = N

Base.:(+)(i::Number, ::Static{0}) = i
Base.:(+)(::Static{0}, i::Number) = i
Base.:(+)(::Static{0}, ::Static{0}) = Static{0}()
Base.:(+)(::Static{M}, ::Static{N}) where {M,N} = Static{M+N}()

Base.:(-)(::Static{0}, i::Number) = -i
Base.:(-)(i::Number, ::Static{0}) = i
Base.:(-)(::Static{0}, ::Static{0}) = Static{0}()
Base.:(-)(::Static{M}, ::Static{N}) where {M,N} = Static{M-N}()

Base.:(*)(::Static{0}, i::Number) = Static{0}()
Base.:(*)(i::Number, ::Static{0}) = Static{0}()
Base.:(*)(::Static{0}, ::Static{0}) = Static{0}()
Base.:(*)(::Static{1}, i::Number) = i
Base.:(*)(i::Number, ::Static{1}) = i
Base.:(*)(::Static{1}, ::Static{1}) = Static{1}()
Base.:(*)(::Static{M}, ::Static{N}) where {M,N} = Static{M*N}()

Base.:(÷)(::Static{M}, ::Static{N}) where {M,N} = Static{M÷N}()
Base.:(<<)(::Static{M}, ::Static{N}) where {M,N} = Static{M<<N}()
Base.:(>>)(::Static{M}, ::Static{N}) where {M,N} = Static{M>>N}()
Base.:(>>>)(::Static{M}, ::Static{N}) where {M,N} = Static{M>>>N}()
Base.:(&)(::Static{M}, ::Static{N}) where {M,N} = Static{M & N}()
Base.:(|)(::Static{M}, ::Static{N}) where {M,N} = Static{M | N}()
Base.:(⊻)(::Static{M}, ::Static{N}) where {M,N} = Static{M ⊻ N}()

Base.:(:)(L, ::Static{U}) where {U} = OptionallyStaticUnitRange(L, Val(U))
Base.:(:)(::Static{L}, U) where {L} = OptionallyStaticUnitRange(Val(L), U)
Base.:(:)(::Static{L}, ::Static{U}) where {L,U} = OptionallyStaticUnitRange(Val(L), Val(U))

