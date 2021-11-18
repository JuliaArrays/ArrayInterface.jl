
@propagate_inbounds @inline function to_indices(a::A, axs, inds::I) where {A,I}
   _to_indices(a, axs, inds, IndexStyle(A), ndims_index(I), ndims_shape(I))
end
@generated function _to_indices(a::A, axs, inds, ::S, ::NDIndex, ::NDIndices, ::NDShape) where {A,S,NDIndex,NDIndices,NDShape}
    blk = Expr(:block)
    t = Expr(:tuple)
    ndindex = known(NDIndex)
    ndindices = known(NDIndices)
    ndshape = known(NDShape)
    NAxes = ndims(A)
    dim = 0
    for i in 1:length(ndindex)
        ndidx = ndindex[i]
        ndshp = ndshape[i]
        if ndidx === 0  # eltype(I) <: CartesianIndex{0}
            push!(t.args, @inbounds(getfield(inds, $i)))
        elseif ndidx === 1
            dim += 1
            axexpr = _axis_expr(NAxes, dim)
            if N < dim && nout === 0
                # drop integers after bounds checking trailing dims
                push!(blk.args, :(to_index($axexpr, @inbounds(getfield(inds, $i)))))
            else
                push!(t.args, :(to_index($axexpr, @inbounds(getfield(inds, $i)))))
            end
        else
            axexpr = Expr(:tuple)
            for j in 1:ndidx
                dim += 1
                push!(axexpr.args, _axis_expr(nd, dim))
            end
            :(to_indices($(ifelse(S <: IndexLinear, :LinearIndices, :CartesianIndices))($axexpr), @inbounds(getfield(inds, $i))))
            push!(t.args, )
        end
    end
end

# code gen
@generated function _to_indices(a::A, axs, inds, ::S, ::NDIn, ::NDOut) where {A,S,NDIn,NDOut}
    nd = ndims(A)
    blk = Expr(:block)
    t = Expr(:tuple)
    dim = 0
    ndin = known(NDIn)
    ndout = known(NDOut)
    for i in 1:length(ndin)
        nidx = 
        nshp
        nout = ndout[i]
        if nin === 1
            dim += 1
            axexpr = _axis_expr(nd, dim)
            if N < dim && nout === 0
                # drop integers after bounds checking trailing dims
                push!(blk.args, :(to_index($axexpr, @inbounds(getfield(inds, $i)))))
            else
                push!(t.args, :(to_index($axexpr, @inbounds(getfield(inds, $i)))))
            end
        else
            axexpr = Expr(:tuple)
            for j in 1:nin
                dim += 1
                push!(axexpr.args, _axis_expr(nd, dim))
            end
            ICall = ifelse(S <: IndexLinear, :LinearIndices, :CartesianIndices)
            push!(t.args, :(to_index($ICall($axexpr), @inbounds(getfield(inds, $i)))))
        end
    end
    quote
        Base.@_propagate_inbounds_meta
        $blk
        $t
    end
end
function _axis_expr(nd::Int, dim::Int)
    ifelse(nd < dim, static(1):static(1), :(@inbounds(getfield(axs, $dim))))
end

# TODO manage CartesianIndex{0}
# This method just flattens out CartesianIndex, CartesianIndices, and Ellipsis. Although no
# values are ever changed and nothing new is actually created, we still get hit with some
# run time costs if we recurse using lispy approach.
@generated function _splat_indices(::StaticInt{N}, inds::I) where {N,I}
    t = Expr(:tuple)
    out = Expr(:block)
    any_splats = false
    ellipsis_position = 0
    NP = length(I.parameters)
    for i in 1:NP
        Ti = I.parameters[i] 
        if Ti <: Base.AbstractCartesianIndex && !(Ti <: CartesianIndex{0})
            argi = gensym()
            push!(out.args, Expr(:(=), argi, :(Tuple(@inbounds(getfield(inds, $i))))))
            for j in 1:ArrayInterface.known_length(Ti)
                push!(t.args, :(@inbounds(getfield($argi, $j))))
            end
            any_splats = true
        elseif Ti <: CartesianIndices && !(Ti <: CartesianIndices{0})
            argi = gensym()
            push!(out.args, Expr(:(=), argi, :(axes(@inbounds(getfield(inds, $i))))))
            for j in 1:ArrayInterface.known_length(Ti)
                push!(t.args, :(@inbounds(getfield($argi, $j))))
            end
            any_splats = true
        #=
        elseif Ti <: Ellipsis
            if ellipsis_position == 0
                ellipsis_position = i
            else
                push!(t.args, :(:))
            end
        =#
        else
            push!(t.args, :(@inbounds(getfield(inds, $i))))
        end
    end
    if ellipsis_position != 0
        nremaining = N
        for i in 1:NP
            if i != ellipsis_position
                nremaining -= ndims_index(I.parameters[i])
            end
        end
        for _ in 1:nremaining
            insert!(t.args, ellipsis_position, :(:))
        end
    end
    if any_splats
        push!(out.args, t)
        return out
    else
        return :inds
    end
end


#=

to_indices2(A, ::Tuple{}) = (@boundscheck ndims(A) === 0 || throw(BoundsError(A, ())); ())
# preserve CartesianIndices{0} as they consume a dimension.
to_indices2(A, i::Tuple{CartesianIndices{0}}) = i
to_indices2(A, i::Tuple{Slice}) = i
to_indices2(A, i::Tuple{Vararg{CanonicalInt}}) = i
to_indices2(A, i::Tuple{AbstractArray{<:Integer}}) = i
to_indices2(A, i::Tuple{LogicalIndex}) = i
to_indices2(A, i::Tuple{AbstractArray{<:AbstractCartesianIndex{N}}}) where {N} = i
@inline to_indices2(A, ::Tuple{Colon}) = (indices(A),)
@inline to_indices2(A, i::Tuple{LinearIndices}) = to_indices2(A, axes(getfield(i,1)))
@inline to_indices2(A, i::Tuple{CartesianIndices}) = to_indices2(A, axes(getfield(i,1)))
@inline to_indices2(A, i::Tuple{AbstractCartesianIndex}) = to_indices2(A, Tuple(getfield(i, 1)))
@inline to_indices2(A, i::Tuple{AbstractArray{Bool}}) = (LogicalIndex(getfield(i, 1)),)
# As an optimization, we allow trailing Array{Bool} and BitArray to be linear over trailing dimensions
@inline to_indices2(A::LinearIndices, i::Tuple{Union{Array{Bool}, BitArray}}) = (LogicalIndex{Int}(getfield(i, 1)),)
@inline to_indices2(A, i::Tuple{Any,Vararg{Any}}) = to_indices2(A, lazy_axes(A), i)

@inline function to_indices2(A, axs::Tuple{Any,Vararg{Any}}, inds::Tuple{I,Vararg{Any}}) where {I}
    N = dynamic(ndims_index(I))
    if N === 1
        return (to_index(getfield(axs, 1), getfield(inds, 1)), to_indices2(A, tail(axs), tail(inds))...)
    else
        axsfront, axstail = Base.IteratorsMD.split(axs, Val(N))
        if IndexStyle(A) === IndexLinear()
            indsfront = to_indices2(LinearIndices(axsfront), (getfield(inds, 1),))
        else
            indsfront = to_indices2(CartesianIndices(axsfront), (getfield(inds, 1),))
        end
        return (indsfront..., to_indices2(A, axstail, tail(inds))...)
    end
end
to_indices2(A, axs::Tuple{Any,Vararg{Any}}, ::Tuple{}) = ()
to_indices2(A, ::Tuple{}, ::Tuple{}) = ()
=#