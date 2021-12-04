
"""
    known_size(::Type{T}) -> Tuple
    known_size(::Type{T}, dim) -> Union{Int,Nothing}

Returns the size of each dimension of `A` or along dimension `dim` of `A` that is known at
compile time. If a dimension does not have a known size along a dimension then `nothing` is
returned in its position.
"""
known_size(x) = known_size(typeof(x))
known_size(::Type{T}) where {T<:Tuple} = (known_length(T),)
@inline known_size(::Type{T}) where {T} = _maybe_known_size(Base.IteratorSize(T), T)
_maybe_known_size(::Any, ::Type) = (nothing,)
@inline function _maybe_known_size(::Base.HasShape{N}, ::Type{T}) where {N,T}
    eachop(_known_size, nstatic(Val(N)), axes_types(T))
end
_known_size(::Type{T}, dim::StaticInt) where {T} = known_length(_get_tuple(T, dim))

@inline function known_size(::Type{T}) where {T<:OptionallyStaticUnitRange}
    (_range_length(known_first(T), known_last(T)),)
end
@inline function known_size(::Type{T}) where {T<:OptionallyStaticStepRange}
    (_range_length(known_first(T), known_step(T), known_last(T)),)
end
known_size(::Type{<:Base.OneTo}) = (nothing,)
@inline known_size(x, dim) = known_size(typeof(x), dim)
@inline known_size(::Type{T}, dim) where {T} = known_size(T, to_dims(T, dim))
@inline function known_size(::Type{T}, dim::CanonicalInt) where {T}
    if ndims(T) < dim
        return 1
    else
        return getfield(known_size(T), Int(dim))
    end
end
function known_size(::Type{<:SubArray{T,N,A,I}}) where {T,N,A,I}
    eachop(_known_size, _to_sub_dims(I), I)
end
@inline known_size(::Type{T}) where {T<:VecAdjTrans} = (1, known_length(parent_type(T)))
@inline function known_size(::Type{T}) where {T<:MatAdjTrans}
    s1, s2 = known_size(parent_type(T))
    return (s2, s1)
end
@inline function known_size(::Type{R}) where {T,N,S,A,R<:ReinterpretArray{T,N,S,A}}
    if _is_reshaped(R)
        if sizeof(S) === sizeof(T)
            return known_size(A)
        elseif sizeof(S) > sizeof(T)
            return (div(sizeof(S), sizeof(T)), known_size(A)...)
        else
            return tail(known_size(A))
        end
    else
        psize = known_size(A)
        pfirst = first(psize)
        if pfirst === nothing
            return psize
        else
            return (div(pfirst * sizeof(S), sizeof(T)), tail(psize)...)
        end
    end
end
function known_size(::Type{Diagonal{T,V}}) where {T,V}
    s = known_length(V)
    return (s, s)
end
known_size(::Type{T}) where {T<:IdentityUnitRange} = (known_length(parent_type(T)),)

_sym_size(x::Tuple{Int,Int}) = x
_sym_size(x::Tuple{Int,Nothing}) = (getfield(x, 1), getfield(x, 1))
_sym_size(x::Tuple{Nothing,Int}) = (getfield(x, 2), getfield(x, 2))
_sym_size(::Tuple{Nothing,Nothing}) = (nothing, nothing)
known_size(::Type{T}) where {T<:HermOrSym} = _sym_size(known_size(parent_type(T)))
function known_size(::Type{T}) where {T<:LinearAlgebra.AbstractTriangular}
    _sym_size(known_size(parent_type(T)))
end
@inline function known_size(::Type{T}) where {T<:PermutedDimsArray}
    permute(known_size(parent_type(T)), to_parent_dims(T))
end

## iterators
known_size(::Type{<:Iterators.Accumulate{F,I,T}}) where {F,I,T} = known_size(I)
function known_size(::Type{<:Iterators.ProductIterator{T}}) where {T} 
    eachop(_known_size, nstatic(Val(known_length(T))), T)
end
known_size(::Type{<:Base.Generator{I,F}}) where {I,F} = known_size(I)
function known_size(::Type{<:Iterators.Flatten{I}}) where {I}
    if I <: Tuple
        return (_flatten_lengths(eachop(_known_size, nstatic(Val(known_length(I))), I)),)
    else
        return (_known_length((known_length(eltype(I)), known_length(I))),)
    end
end
_flatten_lengths(x::Tuple{Vararg{Int}}) = sum(x)
_flatten_lengths(x::Tuple{Vararg{Union{Nothing,Int}}}) = nothing

"""
    Size(s::Tuple{Vararg{Union{Int,StaticInt}})
    Size(A) -> Size(size(A))

Type that represents statically sized dimensions as `StaticInt`s.
"""
struct Size{N,S<:Tuple{Vararg{CanonicalInt,N}}} <: ArrayIndex{N}
    size::S

    Size{N}(s::Tuple{Vararg{CanonicalInt,N}}) where {N} = new{N,typeof(s)}(s)
    Size(s::Tuple{Vararg{CanonicalInt,N}}) where {N} = Size{N}(s)
end

@inline Size(A) = Size(_size(A, Val(known_size(A))))
@generated function _size(A, ::Val{S}) where {S}
    t = Expr(:tuple)
    for i in 1:length(S)
        si = S[i]
        if si === nothing
            push!(t.args, :(Base.size(A, $i)))
        else
            push!(t.args, :($(static(si))))
        end
    end
    Expr(:block, Expr(:meta, :inline), t)
end
Size(x, dim) = _Length(x, to_dims(x, dim))
@inline function _Length(x, dim::Union{Int,StaticInt})
    sz = known_size(x, dim)
    if sz === nothing
        return Length(Base.size(x, dim))
    else
        return Length(static(sz))
    end
end

"""
    Length(x::Union{Int,StaticInt})
    Length(A) = Length(length(A))

Type that represents statically sized dimensions as `StaticInt`s.
"""
const Length{L} = Size{1,Tuple{L}}
Length(x::CanonicalInt) = Size((x,))
@inline function Length(x)
    len = known_length(x)
    if len === nothing
        return Length(length(x))
    else
        return Length(static(len))
    end
end

Base.isequal(x::Size, y::Size) = getfield(x, :size) == getfield(y, :size)

Base.:(==)(x::Size, y::Size) = getfield(x, :size) == getfield(y, :size)

@inline Base.axes(s::Size{N}) where {N} = ntuple(i -> static(1):getfield(s.size, i), Val(N))

Base.IteratorSize(::Type{<:Size{N}}) where {N} = Base.HasShape{N}()

Base.firstindex(x::Size{N}) where {N} = ntuple(Compat.Returns(1), Val(N))
Base.firstindex(x::Size{1}) = 1

Base.lastindex(x::Size{N}) where {N} = size(x)
Base.lastindex(x::Size{1}) = length(x)

known_size(::Type{<:Size{N,S}}) where {N,S} = known(S)
Base.size(s::Size{N,NTuple{N,Int}}) where {N} = s.size
Base.size(s::Size) = map(Int, s.size)
function Base.size(s::Size{N}, dim::CanonicalInt) where {N}
    if dim > N
        return 1
    else
        return Int(getfield(s.size, Int(dim)))
    end
end

function Base.getindex(s::Size, i::AbstractCartesianIndex)
    @boundscheck checkbounds(s, i)
    i
end
function Base.getindex(s::Size, i::CanonicalInt)
    ci = _lin2sub(offsets(s), s.size, static(1))
    @boundscheck checkbounds(s, ci)
end

@generated function _lin2sub(o::O, s::S, i::I) where {O,S,I}
    out = Expr(:block, Expr(:meta, :inline))
    t = Expr(:tuple)
    iprev = :(i - 1)
    N = length(S.parameters)
    for i in 1:N
        if i === N
            push!(t.args, :($iprev + getfield(o, $i)))
        else
            len = gensym()
            inext = gensym()
            push!(out.args, :($len = getfield(s, $i)))
            push!(out.args, :($inext = div($iprev, $len)))
            push!(t.args, :($iprev - $len * $inext + getfield(o, $i)))
            iprev = inext
        end
    end
    push!(out.args, :(CartesianIndex($(t))))
    out
end

function Base.iterate(::Size{0}, done=false)
    if done
        return nothing
    else
        return (), true
    end
end
Base.iterate(x::Size{1}) = (1, 1)
@inline function Base.iterate(x::Size{1}, state::Int)
    if length(x) === state
        return nothing
    else
        new_state = state + 1
        return new_state, new_state
    end
end
@inline function Base.iterate(x::Size{N}) where {N}
    state = firstindex(x)
    return state, state
end
@inline Base.iterate(x::Size, state) = _iterate_size(x.size, state)
@generated function _iterate_size(s::S, state::NTuple{N,Int}) where {S,N}
    out = Expr(:block, Expr(:meta, :inline))
    for i in 1:N
        push!(out.args, Expr(:(=), Symbol(:state_, i), :(@inbounds(getfield(state, $i)))))
    end
    ifexpr = Expr(:return, :nothing)
    for dim in N:-1:1
        t = Expr(:tuple)
        for i in 1:N
            if i === dim
                push!(t.args, Expr(:call, :+, Symbol(:state_, i), 1))
            elseif i < dim
                push!(t.args, 1)
            else
                push!(t.args, Symbol(:state_, i))
            end
        end
        if known(S.parameters[dim]) === nothing
            dim_length = :(@inbounds(getfield(s, $dim)))
        else
            dim_length = known(S.parameters[dim])
        end
        ifexpr = Expr(:if,
            Expr(:call, :(===), Symbol(:state_, dim), dim_length),
            ifexpr,
            Expr(:block, Expr(:(=), :newstate, t), Expr(:return, Expr(:tuple, :newstate, :newstate)))
        )
    end
    push!(out.args, ifexpr)
    out
end

Base.simd_outer_range(s::Size{N}) where {N} = Size(tail(s.size))
Base.simd_inner_length(s::Size{0}, ::Tuple) = 1
Base.simd_inner_length(s::Size, ::Tuple) = Int(getfield(s.size, 1))
Base.simd_index(::Size, Ilast::Tuple, I1::Int) = (I1 + 1, Ilast...)

Base.Tuple(s::Size) = Tuple((i for i in s))
function Base.Tuple(g::Base.Generator{I,F}) where {I<:Size,F}
    L = known_length(I)
    if L === nothing
        return Tuple(collect(g)...)
    else
        return _to_tuple(g.f, g.iter)
    end
end
@generated function _to_tuple(f, ::Size{N,S}) where {N,S}
    t = Expr(:tuple)
    for i in Size(ntuple(i -> known(S.parameters[i]), N))
        push!(t.args, :(f($i)))
    end
    Expr(:block, Expr(:meta, :inline), t)
end

function Base.show(io::IO, ::MIME"text/plain", @nospecialize(s::Size))
    print(io, "Size($(join(s.size, ",")))")
end

size(x) = Size(x).size
size(x, dim) = getfield(Size(x, dim).size, 1)

