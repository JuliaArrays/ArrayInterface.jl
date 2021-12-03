
"""
    known_size(::Type{T}) -> Tuple
    known_size(::Type{T}, dim) -> Union{Int,Nothing}

Returns the size of each dimension of `A` or along dimension `dim` of `A` that is known at
compile time. If a dimension does not have a known size along a dimension then `nothing` is
returned in its position.
"""
known_size(x) = known_size(typeof(x))
known_size(::Type{T}) where {T} = eachop(_known_size, nstatic(Val(ndims(T))), axes_types(T))
_known_size(::Type{T}, dim::StaticInt) where {T} = known_length(_get_tuple(T, dim))
@inline known_size(x, dim) = known_size(typeof(x), dim)
@inline known_size(::Type{T}, dim) where {T} = known_size(T, to_dims(T, dim))
@inline function known_size(::Type{T}, dim::CanonicalInt) where {T}
    if ndims(T) < dim
        return 1
    else
        return getfield(known_size(T), Int(dim))
    end
end

function known_size(::Type{<:Iterators.ProductIterator{T}}) where {T} 
    eachop(_known_size, nstatic(Val(known_length(T))), T)
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
known_size(::Type{Slice{P}}) where {P} = (known_length(P),)

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

Base.size(s::Size{N,NTuple{N,Int}}) where {N} = s.size
Base.size(s::Size) = map(Int, s.size)
function Base.size(s::Size{N}, dim) where {N}
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
    push!(out.args, :(NDIndex($(t))))
    out
end

function Base.show(io::IO, ::MIME"text/plain", @nospecialize(s::Size))
    print(io, "Size($(join(s.size, ",")))")
end

size(x) = Size(x).size
size(x, dim) = getfield(Size(x, dim).size, 1)


Base.firstindex(x::Size{N}) where {N} = ntuple(Compat.Returns(1), Val(N))
Base.firstindex(x::Size{1}) = 1

Base.lastindex(x::Size{N}) where {N} = size(x)
Base.lastindex(x::Size{1}) = length(x)

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
@inline Base.iterate(x::Size, state) = _iterate_size(size(x), state)
@generated function _iterate_size(s::NTuple{N,Int}, state::NTuple{N,Int}) where {N}
    out = Expr(:block, Expr(:meta, :inline))
    for i in 1:N
        push!(out.args, Expr(:(=), Symbol(:state_, i), :(@inbounds(getfield(state, $i)))))
        push!(out.args, Expr(:(=), Symbol(:size_, i), :(@inbounds(getfield(s, $i)))))
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
        ifexpr = Expr(:if,
            Expr(:call, :(===), Symbol(:state_, dim), Symbol(:size_, dim)),
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

