
rangeexpr(start, st, stop) = Expr(:call, :(:), start, st, stop)
function rangeexpr(start, st::Integer, stop)
    if st === 1
        return :($start:$stop)
        return Expr(:call, :(:), start, stop)
    else
        return Expr(:call, :(:), start, st, stop)
    end
end
function rangeexpr(start::Integer, st::Integer, stop::Integer)
    if st === 1
        return Int(start):Int(stop)
    else
        return Int(start):Int(st):Int(stop)
    end
end

###
### addition
###
_addexpr(x::Symbol, y::Int) = :($x + $y)
function _addexpr(x::Expr, y::Int)
    if y === 0
        return x
    elseif x.head === :call && x.args[2] === :+
        if x.arg[2] isa Integer
            x.argp[2] += y
            return x, 0
        elseif x.arg[3] isa Integer
            x.argp[3] += y
            return x, 0
        else
            _, newy = _addexpr(x.args[2], y)
            if newy === 0
                return x, newy
            else
                _, newy = _addexpr(x.args[3], y)
                return x, newy
            end
        end
    else
        return x, y
    end
end
addexpr(x::Integer, y::Integer) = Int(x) + Int(y)
addexpr(x::Union{Symbol,Expr}, y::Union{Symbol,Expr}) = :($x + $y)
addexpr(x::Integer, y::Union{Expr,Symbol}) = addexpr(y, Int(x))
function addexpr(x::Union{Symbol,Expr}, y::Integer)
    if Int(y) === 0
        return x
    else
        newx, newy = _addexpr(x, Int(y))
        if newy === 0
            return newx
        else
            return :($x + $y)
        end
    end
end

lengthexpr(x) = length(x)
lengthexpr(x::Expr) = :(length($x))

###
### subtraction
###
subexpr(x::Integer, y::Integer) = addexpr(x, -y)
subexpr(x::Union{Symbol,Expr}, y::Union{Symbol,Expr}) = :($x - $y)
function subexpr(x::Union{Symbol,Expr}, y::Integer)
    if Int(y) === 0
        return x
    else
        return :($x - $y)
    end
end
function subexpr(x::Integer, y::Union{Symbol,Expr})
    if Int(x) === 0
        return y
    else
        return :($x - $y)
    end
end

###
### multiplication
###
mulexpr(x::Union{Symbol,Expr}, y::Union{Symbol,Expr}) = :($x * $y)
mulexpr(x::Integer, y::Integer) = Int(x) * Int(y)
mulexpr(x::Integer, y::Union{Symbol,Expr}) = mulexpr(y, x)
function mulexpr(x::Union{Symbol,Expr}, y::Integer)
    newy = Int(y)
    if Int(y) === 0
        return 0
    elseif Int(y) === 1
        return x
    else
        newx, newy = _mulexpr(x, Int(y))
        if newy === 1
            return newx
        else
            return :($x * $y)
        end
    end
end

_mulexpr(x::Symbol, y::Int) = :($x * $y), 1
function _mulexpr(x::Expr, y::Int)
    if y === 0
        return 0
    elseif y === 1
        return x
    elseif x.head === :call && x.args[2] === :*
        if x.arg[2] isa Integer
            x.argp[2] *= y
            return x, 1
        elseif x.arg[3] isa Integer
            x.argp[3] *= y
            return x, 1
        else
            _, newy = _addexpr(x.args[2], y)
            if newy === 1
                return x, newy
            else
                _, newy = _addexpr(x.args[3], y)
                return x, newy
            end
        end
    else
        return x, y
    end
end


known_struct(::Type{T}) where {T} = nothing
known_struct(::Type{T}) where {N,T<:StaticInt{N}} = StaticInt{N}()
function known_struct(::Type{T}) where {T<:OptionallyStaticRange}
    if known_first(T) === nothing
        return nothing
    else
        if known_step(T) === nothing
            return nothing
        else
            if known_last(T) === nothing
                return nothing
            else
                if known_step(T) === 1
                    return known_first(T):known_last(T)
                else
                    return known_first(T):known_step(T):known_last(T)
                end
            end
        end
    end
end

#=
const ElementMap = IndexMap{Int,Int,Nothing}
const CollectionMap = IndexMap{Int,Int,Int}
const CartesianCollectionMap{N} = IndexMap{NTuple{N,Int},Int,Int}
const CartesianElementMap{N} = IndexMap{NTuple{N,Int},Int,nothing}
const SliceMap{N} = IndexMap{NTuple{N,Int},NTuple{N,Int},NTuple{N,Int}}
=#

struct IndexExpr
    assignment::Vector{Expr}
    iterator::Expr
    strided::Union{Symbol,Expr}
end

###
### index across contiguously sliced strides
###
function index_expr(
    m::IndexMap{NTuple{N,Int},NTuple{N,Int},NTuple{N,Int}},
    isym::Symbol, ::Type{I},
    ssym::Symbol, ::Type{S},
    fsym::Symbol, ::Type{F}
) where {S,I,F,N}

    idx_i = idx_map(m)
    src_i = src_map(m)
    index_i = gensym(Symbol(isym, :_, idx_i))
    itrsym = gensym()
    index = lengthexpr(_index_expr(I, isym, src_i[1]))
    if N > 1
        for i in 2:N
            index = mulexpr(index, lengthexpr(_index_expr(I, isym, src_i[1])))
        end
    end
    # TODO can we assume that offset of the memory is always 0
    return IndexExpr(
        [Expr(:(=), index_i, rangeexpr(0, 1, subexpr(index, 1)))],
        :($itrsym = $index_i),
        mulexpr(itrsym, _index_expr(S, ssym, src_i[1]))
    )
end

###
### index across linear collection
###
function index_expr(
    m::IndexMap{Int,Int,Int},
    isym::Symbol, ::Type{I},
    ssym::Symbol, ::Type{S},
    fsym::Symbol, ::Type{F}
) where {S,I,F}

    idx_i = idx_map(m)
    src_i = src_map(m)
    index_i = gensym(Symbol(isym, :_, idx_i))
    itrsym = gensym(Symbol(:itr_, idx_i))
    index = _index_expr(I, isym, idx_i)
    offset = _index_expr(F, fsym, src_i)
    if index isa Expr || offset isa Expr
        index = :($index .- $offset)
    else
        index = index .- offset
    end
    return IndexExpr([Expr(:(=), index_i, index)], :($itrsym = $index_i), mulexpr(itrsym, _index_expr(S, ssym, src_i)))
end

###
### index across multidimensional collection
###
function index_expr(
    m::IndexMap{NTuple{N,Int},Int,Int},
    isym::Symbol, ::Type{I},
    ssym::Symbol, ::Type{S},
    fsym::Symbol, ::Type{F}
) where {I,S,F,N}

    idx_i = idx_map(x)
    src_i = src_map(x)
    index_i = gensym(Symbol(isym, :_, idx_i))
    itrsym = gensym(Symbol(:itr_, idx_i))
    str = mulexpr(subexpr(:($itrsym[1]), _index_expr(F, fsym, src_i[1])), _index_expr(S, ssym, src_i[1]))
    if N > 1
        for i in 2:N
            str = addexpr(str, mulexpr(subexpr(:($itrsym[$i]), _index_expr(F, fsym, src_i[i])), _index_expr(S, ssym, src_i[i])))
        end
    end
    return IndexExpr([Expr(:(=), index_i, :($isym[$idx_i]))], :($itrsym = $index_i), str)
end

###
### combine indices that are dropped
### 
struct DroppedExpr
    sym::Symbol
    assignment::Expr
end

function index_expr(
    maps::DroppedIndexMaps,
    isym::Symbol, ::Type{I},
    ssym::Symbol, ::Type{S},
    fsym::Symbol, ::Type{F}
) where {S,I,F}

    index_i = gensym(Symbol(isym))
    return DroppedExpr(
        index_i,
        _index_expr_dropped_indices(maps.maps, isym, I, ssym, S, fsym, F))
end
function _index_expr_dropped_indices(
    maps::Tuple{Any},
    indices_sym::Symbol, ::Type{I},
    strides_sym::Symbol, ::Type{S},
    offsets_sym::Symbol, ::Type{F}
) where {S,I,F}
    x = first(maps)
    index = _index_expr(I, indices_sym, idx_map(x))
    stride = _index_expr(S, strides_sym, src_map(x))
    offset = _index_expr(F, offsets_sym, src_map(x))
    return mulexpr(subexpr(index, offset), stride)
end

function _index_expr_dropped_indices(
    maps::Tuple{Any,Vararg},
    indices_sym::Symbol, ::Type{I},
    strides_sym::Symbol, ::Type{S},
    offsets_sym::Symbol, ::Type{F}
) where {S,I,F}

    x = first(maps)
    index = _index_expr(I, indices_sym, idx_map(x))
    stride = _index_expr(S, strides_sym, src_map(x))
    offset = _index_expr(F, offsets_sym, src_map(x))
    return addexpr(
        mulexpr(subexpr(index, offset), stride),
        _index_expr_dropped_indices(tail(maps), indices_sym, I, strides_sym, S, offsets_sym, F)
    )
end

function _index_expr(::Type{T}, sym::Symbol, i::Int) where {T}
    _index_expr(known_struct(T.parameters[i]), sym, i)
end
_index_expr(::Nothing, sym::Symbol, i::Int) = :($sym[$i])
_index_expr(x, sym::Symbol, i::Int) = x

@inline function index_expr(
    maps::Tuple{Any,Vararg},
    isym::Symbol, ::Type{I},
    ssym::Symbol, ::Type{S},
    fsym::Symbol, ::Type{F}
) where {S,I,F}

    return (
        index_expr(first(maps), isym, I, ssym, S, fsym, F),
        index_expr(tail(maps), isym, I, ssym, S, fsym, F)...
    )
end

function index_expr(
    maps::Tuple{Any},
    isym::Symbol, ::Type{I},
    ssym::Symbol, ::Type{S},
    fsym::Symbol, ::Type{F}
) where {S,I,F}

    return (index_expr(first(maps), isym, I, ssym, S, fsym, F),)
end

# dropped indices will still provide offset on the indexing within each body of higher
# positioned iterating indices
function combine_dropped_with_iterators(x::Tuple{DroppedExpr,Vararg})
    dropped = first(x)
    m = first(tail(x))
    newexpr = IndexExpr([m.assignment..., Expr(:(=), dropped.sym, dropped.assignment)], m.iterator, addexpr(dropped.sym, m.strided))
    return (newexpr, combine_dropped_with_iterators(tail(tail(x)))...)
end

combine_dropped_with_iterators(::Tuple{}) = ()
function combine_dropped_with_iterators(x::Tuple{Any,Vararg})
    return (first(x), combine_dropped_with_iterators(tail(x))...)
end

# last DroppedExpr goes at top of function body so we don't drop it
combine_dropped_with_iterators(x::Tuple{DroppedExpr}) = x
combine_dropped_with_iterators(x::Tuple{Any}) = x

function generate_pointer_index(maps::Tuple{Vararg{Any,N}}, ::Type{T}) where {N,T}
    blk = Expr(:block)
    for m in maps
        if !isa(m, DroppedExpr)
            append!(blk.args, m.assignment)
        end
    end
    bitsize = sizeof(T)

    if N > 1
        strided_assignments = [gensym(Symbol(:strided_, i)) for i in 1:(N + 1)]
        m = maps[1]
        out = Expr(:for, m.iterator,
               Expr(:block, Expr(:(=), strided_assignments[1], addexpr(m.strided, strided_assignments[2])),
                    :(dst_i === nothing && break),
                    :(unsafe_copyto!(dst + (first(dst_i) * $bitsize), src + ($(strided_assignments[1]) * $bitsize), 1)),
                    :(dst_i = iterate(dst_itr, last(dst_i)))))
        for i in 2:N
            m = maps[i]
            if i === N
                if isa(m, DroppedExpr)
                    push!(blk.args, Expr(:(=), strided_assignments[i], m.assignment))
                else
                    out = Expr(:for, m.iterator, Expr(:block, Expr(:(=), strided_assignments[i], m.strided), out))
                end
            else
                out = Expr(:for,
                        m.iterator,
                        Expr(:block, Expr(:(=), strided_assignments[i], addexpr(m.strided, strided_assignments[i+1])), out))
            end
        end
    else
        m = maps[1]
        strided_1 = gensym(:strided_1)
        out = Expr(:for, m.iterator, Expr(:block, :($strided_1 = $(m.strided)),
                                          :(dst_i === nothing && break),
                                          :(unsafe_copyto!(dst + (first(dst_i) * $bitsize), src + ($strided_1 * $bitsize), 1)),
                                          :(dst_i = iterate(dst_itr, last(dst_i)))))
    end
    push!(blk.args, out)
    return blk
end


