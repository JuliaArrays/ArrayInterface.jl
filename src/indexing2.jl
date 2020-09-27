# FIXME this needs to be handled at some level still
#@propagate_inbounds function index_keys(axis, arg::CartesianIndex{1})
#    return index_keys(axis, first(arg.I))
#end


_maybe_tail(::Tuple{}) = ()
_maybe_tail(x::Tuple) = tail(x)

"""
    to_indices(A, axs, args)
"""
function to_indices end

@propagate_inbounds function to_indices(A, axs::Tuple, args::Tuple)
    return (to_index(first(axs), first(args)), to_indices(A, tail(axs), tail(args))...)
end
@propagate_inbounds function to_indices(A, axs::Tuple, args::Tuple{})
    @boundscheck if length(first(axs)) == 1
        throw(BoundsError(first(axs), ()))
    end
    return to_indices(A, tail(axs), args)
end
to_indices(A, axs::Tuple{}, args::Tuple{}) = ()

"""
    argdims(::Type{T}) -> Int

Whats the dimensionality of the indexing argument of type `T`?
"""
argdims(x) = argdims(typeof(x))
# single elements initially map to 1 dimension but that dimension is subsequently dropped.
argdims(::Type{T}) where {T} = 0
argdims(::Type{T}) where {T<:Colon} = 1
argdims(::Type{T}) where {T<:AbstractArray} = ndims(T)
argdims(::Type{T}) where {N,T<:CartesianIndex{N}} = N
argdims(::Type{T}) where {N,T<:AbstractArray{CartesianIndex{N}}} = N
argdims(::Type{T}) where {N,T<:AbstractArray{<:Any,N}} = N
argdims(::Type{T}) where {N,T<:LogicalIndex{<:Any,<:AbstractArray{Bool,N}}} = N

"""
    regroup_axes(A, axs, args)
"""
regroup_axes(A, axs::Tuple, args::Tuple{}) = ()
function regroup_axes(A, axs::Tuple{}, args::Tuple{Arg,Vararg{<:Any}}) where {Arg}
    return (axes(A, ndims(A) + 1), regroup_axes(A, (), tail(args))...)
end
@inline function regroup_axes(A, axs::Tuple, args::Tuple{Arg,Vararg{<:Any}}) where {Arg}
    if argdims(Arg) > 1
        axes_front, axes_tail = Base.IteratorsMD.split(axs, Val(argdims(Arg)))
        return (CartesianIndices(axes_front), regroup_axes(A, axes_tail, tail(args))...)
    else
        return (first(axs), regroup_axes(A, tail(axs), tail(args))...)
    end
end


"""
    to_index([::IndexStyle, ]axis, arg) -> index

Convert the argument `arg` that was originally passed to `getindex` for the dimension
corresponding to `axis` into a form for native indexing (`Int`, Vector{Int}, ect). New
axis types with unique behavior should use an `IndexStyle` trait:

```julia
to_index(axis::MyAxisType, arg) = to_index(IndexStyle(axis), axis, arg)
to_index(::MyIndexStyle, axis, arg) = ...
```
"""
@propagate_inbounds to_index(axis, arg) = to_index(IndexStyle(axis), axis, arg)
to_index(axis, arg::CartesianIndices{0}) = arg
to_index(::IndexStyle, axis, ::Colon) = indices(axis)
function to_index(::IndexLinear, axis, arg::Integer)
    @boundscheck if !checkindex(Bool, axis, arg)
        throw(BoundsError(axis, arg))
    end
    return Int(arg)
end
@propagate_inbounds function to_index(::IndexLinear, axis, arg::AbstractUnitRange{I}) where {I<:Integer}
    @boundscheck if !checkindex(Bool, axis, arg)
        throw(BoundsError(axis, arg))
    end
    return AbstractUnitRange{Int}(arg)
end
function to_index(::IndexLinear, axis, arg::AbstractArray{I}) where {I<:Integer}
    @boundscheck if !checkindex(Bool, axis, arg)
        throw(BoundsError(axis, arg))
    end
    return AbstractArray{Int}(arg)
end
@propagate_inbounds function to_index(::IndexLinear, axis, arg::AbstractArray{I}) where {I<:CartesianIndex}
    @boundscheck checkbounds(axis, arg)
    return @inbounds(axis[arg])
end
@propagate_inbounds function to_index(::IndexLinear, axis, arg::AbstractArray{Bool})
    @boundscheck if !checkindex(Bool, axis, arg)
        throw(BoundsError(axis, arg))
    end
    return @inbounds(axis[arg])
end


@propagate_inbounds function to_index(::IndexCartesian, axis, arg::Integer)
    @boundscheck checkbounds(axis, arg)
    return @inbounds(axis[arg])
end
@propagate_inbounds function to_index(::IndexCartesian, axis, arg::AbstractUnitRange{I}) where {I<:Integer}
    @boundscheck checkbounds(axis, arg)
    return @inbounds(axis[arg])
end
@propagate_inbounds function to_index(::IndexCartesian, axis, arg::AbstractArray{I}) where {I<:Integer}
    @boundscheck checkbounds(axis, arg)
    return @inbounds(axis[arg])
end
@propagate_inbounds function to_index(::IndexCartesian, axis, arg::AbstractArray{I}) where {I<:CartesianIndex}
    @boundscheck checkbounds(axis, arg)
    return arg
end
@propagate_inbounds function to_index(::IndexCartesian, axis, arg::AbstractArray{Bool})
    @boundscheck if !checkindex(Bool, axis, arg)
        throw(BoundsError(axis, arg))
    end
    return @inbounds(axis[arg])
end

"""
    to_axes(A, args, inds) -> new_axes

Construct new axes given the index arguments `args` and the corresponding `inds`
constructed after `to_indices(A, old_axes, args) -> inds`. This method iterates
through each pair of axes and indices, calling [`to_axis`](@ref).
"""
@inline to_axes(A, axs::Tuple, inds::Tuple) = map(to_axis, axs, inds)

"""
    to_axis(old_axis, arg, index) -> new_axis

Construct an `new_axis` for a newly constructed array that corresponds to the
previously executed `to_index(old_axis, arg) -> index`. `to_axis` assumes that
`index` has already been confirmed to be inbounds. The underlying indices of
`new_axis` begins at one and extends the length of `index` (i.e. one-based indexing).
"""
function to_axis(axis, arg, inds)
    if !can_change_size(axis) && (known_length(inds) !== nothing && known_length(axis) === known_length(inds))
        return axis
    else
        return to_axis(IndexStyle(axis), axis, arg, inds)
    end
end
function to_axis(S::IndexStyle, axis, inds)
    return reconstruct_axis(S, axis, arg, StaticInt(1):static_length(inds))
end

can_flatten(::Type{T}) where {T} = false
can_flatten(::Type{T}) where {I<:CartesianIndex,T<:AbstractArray{I}} = false
can_flatten(::Type{T}) where {T<: CartesianIndices} = true
can_flatten(::Type{T}) where {N,T<:AbstractArray{<:Any,N}} = N > 1
can_flatten(::Type{T}) where {N,T<:CartesianIndex{N}} = true

should_flatten(x) = should_flatten(typeof(x))
@generated function should_flatten(::Type{T}) where {T<:Tuple}
    for i in T.parameters
        can_flatten(i) && return true
    end
    return false
end

# `flatten_args` handles the obnoxious indexing arguments passed to `getindex` that
# don't correspond to a single dimension (CartesianIndex, CartesianIndices,
# AbstractArray{Bool}). Splitting this up from `to_indices` has two advantages:
#
# 1. It greatly simplifies `to_indices` so that things like ambiguity errors aren't as
#    likely to occur. It should only occure at the top level of any given call to getindex
#    so it ensures that most internal multidim indexing is less complicated.
# 2. When `to_axes` runs back over the arguments to construct the axes of the new
#    collection all the the indices and args should match up so that less time is
#    wasted on `IteratorsMD.split`.
flatten_args(A, args::Tuple) = flatten_args(A, axes(A), args)
@inline function flatten_args(A, axs::Tuple, args::Tuple{Arg,Vararg{Any}}) where {Arg}
    return (first(args), flatten_args(A, _maybe_tail(axs), tail(args))...)
end
@inline function flatten_args(A, axs::Tuple, args::Tuple{Arg,Vararg{Any}}) where {N,Arg<:CartesianIndex{N}}
    _, axes_tail = Base.IteratorsMD.split(axs, Val(N))
    return (first(args).I..., flatten_args(A, _maybe_tail(axs), tail(args))...)
end
@inline function flatten_args(A, axs::Tuple, args::Tuple{Arg,Vararg{Any}}) where {Arg<:CartesianIndices{0}}
    return (first(args), flatten_args(A, tail(axs), tail(args))...)
end
@inline function flatten_args(A, axs::Tuple, args::Tuple{Arg,Vararg{Any}}) where {N,Arg<:CartesianIndices{N}}
    _, axes_tail = Base.IteratorsMD.split(axs, Val(N))
    return (first(args)..., flatten_args(A, axes_tail, tail(args))...)
end
@inline function flatten_args(A, axs::Tuple, args::Tuple{Arg,Vararg{Any}}) where {N,Arg<:AbstractArray{<:Any,N}}
    _, axes_tail = Base.IteratorsMD.split(axs, Val(N))
    return (first(args), flatten_args(A, axes_tail, tail(args))...)
end
@inline function flatten_args(A, axs::Tuple, args::Tuple{Arg,Vararg{Any}}) where {N,Arg<:AbstractArray{Bool,N}}
    axes_front, axes_tail = Base.IteratorsMD.split(axs, Val(N))
    if length(args) === 1
        if IndexStyle(A) isa IndexLinear
            return (LogicalIndex{Int}(first(args)),)
        else
            return (LogicalIndex(first(args)),)
        end
    else
        return (LogicalIndex(first(args)), flatten_args(A, axes_tail, tail(args)))
    end
end
flatten_args(A, axs::Tuple, args::Tuple{}) = ()

"""
    IndexType

`IndexType` controls how indices that have been bounds checked and converted to
native axes' indices are used to return the stored values of an array. For example,
if the indices at each dimension are single integers than `IndexType(inds)` returns
`IndexElement()`. Conversely, if any of the indices are vectors then `IndexCollection()`
is returned, indicating that a new array needs to be reconstructed.
"""
abstract type IndexType end

struct IndexElement <: IndexType end

struct IndexCollection <: IndexType end


IndexType(x) = IndexType(typeof(x))
IndexType(::Type{T}) where {T<:Integer} = IndexElement()
IndexCollection(::Type{T}) where {T<:AbstractArray} = IndexCollection()
IndexType(x::IndexType) = x

IndexType(x::IndexType, y::IndexElement) = x
IndexType(x::IndexElement, y::IndexType) = y
IndexType(x::IndexElement, y::IndexElement) = x
IndexType(x::IndexCollection, y::IndexCollection) = x

# Tuple
IndexType(x::Tuple{I}) where {I} = IndexType(I)
@inline function IndexType(x::Tuple{I,Vararg{Any}}) where {I}
    return IndexType(IndexType(I), IndexType(tail(x)))
end

unsafe_getindex(A, args, inds) = unsafe_getindex(IndexType(inds), A, args, inds)
function unsafe_getindex(I::IndexElement, A, args, inds)
    if parent_type(A) <: typeof(A)
        throw(MethodError(unsafe_getindex, (I, A, args, inds)))
    else
        return unsafe_reconstruct(
            A,
            # it would be better if we could do this
            # unsafe_get_collection(parent(A), args, inds)
            @inbounds(getindex(parent(A), inds...)),
            to_axes(A, args, inds)
        )
    end
end

function unsafe_getindex(::IndexCollection, A, axs, inds)
    if parent_type(A) <: typeof(A)
        throw(MethodError(unsafe_get_collection, (A, args, inds)))
    else
        return unsafe_reconstruct(
            A,
            # it would be better if we could do this
            # unsafe_get_collection(parent(A), args, inds)
            @inbounds(getindex(parent(A), inds...)),
            to_axes(axs, inds)
        )
    end
end

### unsafe_setindex!
unsafe_setindex!(A, val, args, inds) = unsafe_setindex!(IndexType(inds), A, val, args, inds)
function unsafe_setindex!(I::IndexType, A, val, inds)
    if parent_type(A) <: typeof(A)
        throw(MethodError(unsafe_setindex!, (I, A, args, inds)))
    else
        return @inbounds(setindex!(parent(A), val, inds))
    end
end

@propagate_inbounds function getindex(A, args...)
    if should_flatten(args)
        return getindex(A, flatten_args(A, args)...)
    else
        axs = regroup_axes(A, axes(A), args)
        return unsafe_getindex(A, axs, to_indices(A, axs, args))
    end
end

function unsafe_reconstruct end

# TODO better doc
"""
    reconstruct_axis(axis, arg, inds)
"""
function reconstruct_axis(axis, inds::Slice)
    if can_change_size(axis)
        # axis is mutable and cannot be safely shared across arrays 
        return reconstruct_axis(IndexStyle(axis), axis, inds)
    else
        return axis
    end
end
function reconstruct_axis(axis, arg, inds::Slice)
    if can_change_size(axis)
        return reconstruct_axis(IndexStyle(axis), axis, arg, inds)
    else
        return axis
    end
end
function reconstruct_axis(axis, arg, inds)
    if can_change_size(axis) || known_length(axis) === nothing || known_length(axis) !== known_length(inds)
        return reconstruct_axis(IndexStyle(axis), axis, arg, inds)
    else
        return axis
    end
end
reconstruct_axis(::IndexLinear, axis, arg, inds) = typeof(axis)(inds)

###
### IndexType
###

IndexType{I<:IndexStyle,IndexMap}

@inline regroup_axes(A, axs::Tuple, args::Tuple{}) = ()
function regroup_axes(A, axs::Tuple{}, args::Tuple{Arg,Vararg{<:Any}}) where {Arg}
    return (axes(A, ndims(A) + 1), regroup_axes(A, (), tail(args))...)
end
@inline function regroup_axes(A, axs::Tuple, args::Tuple{Arg,Vararg{<:Any}}) where {Arg}
    if argdims(Arg) > 1
        axes_front, axes_tail = Base.IteratorsMD.split(axs, Val(argdims(Arg)))
        return (CartesianIndices(axes_front), regroup_axes(A, axes_tail, tail(args))...)
    else
        return (first(axs), regroup_axes(A, tail(axs), tail(args))...)
    end
end

###
### 
###
struct RangeOfRanges{T<:AbstractRange,} <: AbstractArray{T}
    first_range::FR
    gap::G
    last_range::LR
end

struct RangeOfRanges{FR<:OptionallyStaticStepRange,LR<:OptionallyStaticStepRange,G<:Integer} where {T<:}
    first_range::FR
    gap::G
    last_range::LR

    function RangeOfRanges(fr::OptionallyStaticStepRange, gap::Integer, lr::OptionallyStaticStepRange)
        start = _steprange_last(static_first(fr), gap, static_first(lr))
        stp = _steprange_last(static_step(fr), gap, static_step(lr))
        stop = _steprange_last(static_last(fr), gap, static_last(lr))
        new_lr = OptionallyStaticStepRange(start, stp, stop)
        return new{typeof(fr),typeof(gap),typeof(new_lr)}(fr, gap, new_lr)
    end
end

function Base.getindex(r::RangeOfRanges, i::Integer)
    ret = convert(Int, first(r.first_range) + (i - 1) * r.gap)
    ok = ifelse(step(v) > zero(step(v)),
                (ret <= last(r.last_range)) & (ret >= first(r.first_range)),
                (ret <= first(r.first_range)) & (ret >= last(r.last_range)))
    @boundscheck ((i > 0) & ok) || throw(BoundsError(r, i))
    return StepRange{Int,Int}(ret, Int(step(r.first_range) + (i - 1) * r.gap), Int(last(r.first_range) + (i - 1) * r.gap))
end


# ArgDims - a tuple of Int's where each is the dimension each argument maps to
# ArgMap - a tuple where each element is the dimension (Int) or dimensions (Tuple{Vararg{Int}}) that each argument maps to
struct IndexMap{I<:IndexStyle,ArgDims,ArgMap}
    function IndexMap(::Type{A}, ::Type{T}) where {A,N,T<:Tuple{Vararg{<:Any,N}}}
        s = IndexStyle(A)
        ad = ntuple(i -> argdims(s, T.parameters[i]), Val(N))
        am = ntuple(i -> _argmap(i - 1, getfield(ad, i)), Val(N))
        return new{ad, am}()
    end
    IndexMap(A, args::Tuple) = IndexMap(typeof(A), typeof(args))
end

# given each arguments number of dimensions, returns the index dimensions it corresponds to
Base.@pure function _argmap(itr::Int, nd::Int)
    if nd === 0 || nd === 1
        return itr + 1
    else
        return ntuple(i -> itr + i, nd)
    end
end


@testset "indexing" begin
    #@testset "issue #19267" begin TODO find actually description for this
    @test ndims((1:3)[:]) == 1
    @test ndims((1:3)[:,:]) == 2
    @test ndims((1:3)[:,[1],:]) == 3
    @test ndims((1:3)[:,[1],:,[1]]) == 4
    @test ndims((1:3)[:,[1],1:1,:]) == 4
    @test ndims((1:3)[:,:,1:1,:]) == 4
    @test ndims((1:3)[:,:,1:1]) == 3
    @test ndims((1:3)[:,:,1:1,:,:,[1]]) == 6
end
#=

getindex(A, args...)
    flatten_args -> getindex(A, args..)

    or

    unsafe_getindex(A, args, to_indices(A, args))


end


=#



end

# token type on which to dispatch testing methods in order to avoid potential
# name conflicts elsewhere in the base test suite
mutable struct TestAbstractArray end

## Tests for the abstract array interfaces with minimally defined array types

if !isdefined(@__MODULE__, :T24Linear)
    include("testhelpers/arrayindexingtypes.jl")
end

const can_inline = Base.JLOptions().can_inline != 0
function test_scalar_indexing(::Type{T}, shape, ::Type{TestAbstractArray}) where T
    N = prod(shape)
    A = reshape(Vector(1:N), shape)
    B = T(A)
    @test A == B
    # Test indexing up to 5 dimensions
    trailing5 = CartesianIndex(ntuple(x->1, max(ndims(B)-5, 0)))
    trailing4 = CartesianIndex(ntuple(x->1, max(ndims(B)-4, 0)))
    trailing3 = CartesianIndex(ntuple(x->1, max(ndims(B)-3, 0)))
    trailing2 = CartesianIndex(ntuple(x->1, max(ndims(B)-2, 0)))
    i=0
    for i5 = 1:size(B, 5)
        for i4 = 1:size(B, 4)
            for i3 = 1:size(B, 3)
                for i2 = 1:size(B, 2)
                    for i1 = 1:size(B, 1)
                        i += 1
                        @test A[i1,i2,i3,i4,i5,trailing5] == B[i1,i2,i3,i4,i5,trailing5] == i
                        @test A[i1,i2,i3,i4,i5,trailing5] ==
                              Base.unsafe_getindex(B, i1, i2, i3, i4, i5, trailing5) == i
                    end
                end
            end
        end
    end
    # Test linear indexing and partial linear indexing
    i=0
    for i1 = 1:length(B)
        i += 1
        @test A[i1] == B[i1] == i
    end
    i=0
    for i2 = 1:size(B, 2)
        for i1 = 1:size(B, 1)
            i += 1
            @test A[i1,i2,trailing2] == B[i1,i2,trailing2] == i
        end
    end
    @test A == B
    i=0
    for i3 = 1:size(B, 3)
        for i2 = 1:size(B, 2)
            for i1 = 1:size(B, 1)
                i += 1
                @test A[i1,i2,i3,trailing3] == B[i1,i2,i3,trailing3] == i
            end
        end
    end
    # Test zero-dimensional accesses
    @test A[1] == B[1] == 1
    # Test multidimensional scalar indexed assignment
    C = T(Int, shape)
    D1 = T(Int, shape)
    D2 = T(Int, shape)
    D3 = T(Int, shape)
    i=0
    for i5 = 1:size(B, 5)
        for i4 = 1:size(B, 4)
            for i3 = 1:size(B, 3)
                for i2 = 1:size(B, 2)
                    for i1 = 1:size(B, 1)
                        i += 1
                        C[i1,i2,i3,i4,i5,trailing5] = i
                        # test general unsafe_setindex!
                        Base.unsafe_setindex!(D1, i, i1,i2,i3,i4,i5,trailing5)
                        # test for dropping trailing dims
                        Base.unsafe_setindex!(D2, i, i1,i2,i3,i4,i5,trailing5, 1, 1, 1)
                        # test for expanding index argument to appropriate dims
                        Base.unsafe_setindex!(D3, i, i1,i2,i3,i4,trailing4)
                    end
                end
            end
        end
    end
    @test D1 == D2 == C == B == A
    @test D3[:, :, :, :, 1, trailing5] == D2[:, :, :, :, 1, trailing5]
    # Test linear indexing and partial linear indexing
    C = T(Int, shape)
    fill!(C, 0)
    @test C != B && C != A
    i=0
    for i1 = 1:length(C)
        i += 1
        C[i1] = i
    end
    @test C == B == A
    C = T(Int, shape)
    i=0
    C2 = reshape(C, Val(2))
    for i2 = 1:size(C2, 2)
        for i1 = 1:size(C2, 1)
            i += 1
            C2[i1,i2,trailing2] = i
        end
    end
    @test C == B == A
    C = T(Int, shape)
    i=0
    C3 = reshape(C, Val(3))
    for i3 = 1:size(C3, 3)
        for i2 = 1:size(C3, 2)
            for i1 = 1:size(C3, 1)
                i += 1
                C3[i1,i2,i3,trailing3] = i
            end
        end
    end
    @test C == B == A
    # Test zero-dimensional setindex
    if length(A) == 1
        A[] = 0; B[] = 0
        @test A[] == B[] == 0
        @test A == B
    else
        @test_throws BoundsError A[] = 0
        @test_throws BoundsError B[] = 0
        @test_throws BoundsError A[]
        @test_throws BoundsError B[]
    end

function test_vector_indexing(::Type{T}, shape, ::Type{TestAbstractArray}) where T
    @testset "test_vector_indexing{$(T)}" begin
        N = prod(shape)
        A = reshape(Vector(1:N), shape)
        B = T(A)
        trailing5 = CartesianIndex(ntuple(x->1, max(ndims(B)-5, 0)))
        trailing4 = CartesianIndex(ntuple(x->1, max(ndims(B)-4, 0)))
        trailing3 = CartesianIndex(ntuple(x->1, max(ndims(B)-3, 0)))
        trailing2 = CartesianIndex(ntuple(x->1, max(ndims(B)-2, 0)))
        idxs = rand(1:N, 3, 3, 3)
        @test B[idxs] == A[idxs] == idxs
        @test B[vec(idxs)] == A[vec(idxs)] == vec(idxs)
        @test B[:] == A[:] == 1:N
        @test B[1:end] == A[1:end] == 1:N
        @test B[:,:,trailing2] == A[:,:,trailing2] == B[:,:,1,trailing3] == A[:,:,1,trailing3]
            B[1:end,1:end,trailing2] == A[1:end,1:end,trailing2] == B[1:end,1:end,1,trailing3] == A[1:end,1:end,1,trailing3]

        @testset "Test with containers that aren't Int[]" begin
            @test B[[]] == A[[]] == []
            @test B[convert(Array{Any}, idxs)] == A[convert(Array{Any}, idxs)] == idxs
        end

        idx1 = rand(1:size(A, 1), 3)
        idx2 = rand(1:size(A, 2), 4, 5)
        @testset "Test adding dimensions with matrices" begin
            @test B[idx1, idx2, trailing2] == A[idx1, idx2, trailing2] == reshape(A[idx1, vec(idx2), trailing2], 3, 4, 5) == reshape(B[idx1, vec(idx2), trailing2], 3, 4, 5)
            @test B[1, idx2, trailing2] == A[1, idx2, trailing2] == reshape(A[1, vec(idx2), trailing2], 4, 5) == reshape(B[1, vec(idx2), trailing2], 4, 5)
        end
            # test removing dimensions with 0-d arrays
        @testset "test removing dimensions with 0-d arrays" begin
            idx0 = reshape([rand(1:size(A, 1))])
            @test B[idx0, idx2, trailing2] == A[idx0, idx2, trailing2] == reshape(A[idx0[], vec(idx2), trailing2], 4, 5) == reshape(B[idx0[], vec(idx2), trailing2], 4, 5)
            @test B[reshape([end]), reshape([end]), trailing2] == A[reshape([end]), reshape([end]), trailing2] == reshape([A[end,end,trailing2]]) == reshape([B[end,end,trailing2]])
        end

        mask = bitrand(shape)
        @testset "test logical indexing" begin
            @test B[mask] == A[mask] == B[findall(mask)] == A[findall(mask)] == LinearIndices(mask)[findall(mask)]
            @test B[vec(mask)] == A[vec(mask)] == LinearIndices(mask)[findall(mask)]
            mask1 = bitrand(size(A, 1))
            mask2 = bitrand(size(A, 2))
            @test B[mask1, mask2, trailing2] == A[mask1, mask2, trailing2] ==
                B[LinearIndices(mask1)[findall(mask1)], LinearIndices(mask2)[findall(mask2)], trailing2]
            @test B[mask1, 1, trailing2] == A[mask1, 1, trailing2] == LinearIndices(mask)[findall(mask1)]
        end
    end
end

mutable struct UnimplementedFastArray{T, N} <: AbstractArray{T, N} end
Base.IndexStyle(::UnimplementedFastArray) = Base.IndexLinear()

mutable struct UnimplementedSlowArray{T, N} <: AbstractArray{T, N} end
Base.IndexStyle(::UnimplementedSlowArray) = Base.IndexCartesian()

mutable struct UnimplementedArray{T, N} <: AbstractArray{T, N} end

function test_getindex_internals(::Type{T}, shape, ::Type{TestAbstractArray}) where T
    N = prod(shape)
    A = reshape(Vector(1:N), shape)
    B = T(A)

    @test getindex(A, 1) == 1
    @test getindex(B, 1) == 1
    @test Base.unsafe_getindex(A, 1) == 1
    @test Base.unsafe_getindex(B, 1) == 1
end

function test_getindex_internals(::Type{TestAbstractArray})
    U = UnimplementedFastArray{Int, 2}()
    V = UnimplementedSlowArray{Int, 2}()
    @test_throws ErrorException getindex(U, 1)
    @test_throws ErrorException Base.unsafe_getindex(U, 1)
    @test_throws ErrorException getindex(V, 1, 1)
    @test_throws ErrorException Base.unsafe_getindex(V, 1, 1)
end

function test_setindex!_internals(::Type{T}, shape, ::Type{TestAbstractArray}) where T
    N = prod(shape)
    A = reshape(Vector(1:N), shape)
    B = T(A)

    Base.unsafe_setindex!(B, 2, 1)
    @test B[1] == 2
end

function test_setindex!_internals(::Type{TestAbstractArray})
    U = UnimplementedFastArray{Int, 2}()
    V = UnimplementedSlowArray{Int, 2}()
    @test_throws ErrorException setindex!(U, 0, 1)
    @test_throws ErrorException Base.unsafe_setindex!(U, 0, 1)
    @test_throws ErrorException setindex!(V, 0, 1, 1)
    @test_throws ErrorException Base.unsafe_setindex!(V, 0, 1, 1)
end

function test_ind2sub(::Type{TestAbstractArray})
    n = rand(2:5)
    dims = tuple(rand(1:5, n)...)
    len = prod(dims)
    A = reshape(Vector(1:len), dims...)
    I = CartesianIndices(dims)
    for i in 1:len
        @test A[I[i]] == A[i]
    end
end

# A custom linear slow array that insists upon Cartesian indexing
mutable struct TSlowNIndexes{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end
Base.IndexStyle(::Type{A}) where {A<:TSlowNIndexes} = Base.IndexCartesian()
Base.size(A::TSlowNIndexes) = size(A.data)
Base.getindex(A::TSlowNIndexes, index::Int...) = error("Must use $(ndims(A)) indices")
Base.getindex(A::TSlowNIndexes{T,2}, i::Int, j::Int) where {T} = A.data[i,j]

function test_UInt_indexing(::Type{TestAbstractArray})
    A = [1:100...]
    _A = Expr(:quote, A)
    for i in 1:100
        _i8 = convert(UInt8, i)
        _i16 = convert(UInt16, i)
        _i32 = convert(UInt32, i)
        for _i in (_i8, _i16, _i32)
            @eval begin
                @test $_A[$_i] == $i
            end
        end
    end
end

# Issue 13315
function test_13315(::Type{TestAbstractArray})
    U = UInt(1):UInt(2)
    @test [U;[U;]] == [UInt(1), UInt(2), UInt(1), UInt(2)]
end

# checksquare
function test_checksquare()
    @test LinearAlgebra.checksquare(zeros(2,2)) == 2
    @test LinearAlgebra.checksquare(zeros(2,2),zeros(3,3)) == [2,3]
    @test_throws DimensionMismatch LinearAlgebra.checksquare(zeros(2,3))
end

#----- run tests -------------------------------------------------------------#

@testset for T in (T24Linear, TSlow), shape in ((24,), (2, 12), (2,3,4), (1,2,3,4), (4,3,2,1))
    test_scalar_indexing(T, shape, TestAbstractArray)
    test_vector_indexing(T, shape, TestAbstractArray)
    test_primitives(T, shape, TestAbstractArray)
    test_getindex_internals(T, shape, TestAbstractArray)
    test_setindex!_internals(T, shape, TestAbstractArray)
end
test_in_bounds(TestAbstractArray)
test_getindex_internals(TestAbstractArray)
test_setindex!_internals(TestAbstractArray)
test_get(TestAbstractArray)
test_cat(TestAbstractArray)
test_ind2sub(TestAbstractArray)

include("generic_map_tests.jl")
generic_map_tests(map, map!)
@test_throws ArgumentError map!(-, [1])

test_UInt_indexing(TestAbstractArray)
test_13315(TestAbstractArray)
test_checksquare()

A = TSlowNIndexes(rand(2,2))
@test_throws ErrorException A[1]
@test A[1,1] == A.data[1]
@test first(A) == A.data[1]

@testset "#16381" begin
    @inferred size(rand(3,2,1))
    @inferred size(rand(3,2,1), 2)

    @test @inferred(axes(rand(3,2)))    == (1:3,1:2)
    @test @inferred(axes(rand(3,2,1)))  == (1:3,1:2,1:1)
    @test @inferred(axes(rand(3,2), 1)) == 1:3
    @test @inferred(axes(rand(3,2), 2)) == 1:2
    @test @inferred(axes(rand(3,2), 3)) == 1:1
end

@testset "#17088" begin
    n = 10
    M = rand(n, n)
    @testset "vector of vectors" begin
        v = [[M]; [M]] # using vcat
        @test size(v) == (2,)
        @test !issparse(v)
    end
    @testset "matrix of vectors" begin
        m1 = [[M] [M]] # using hcat
        m2 = [[M] [M];] # using hvcat
        @test m1 == m2
        @test size(m1) == (1,2)
        @test !issparse(m1)
        @test !issparse(m2)
    end
end

@testset "isinteger and isreal" begin
    @test all(isinteger, Diagonal(rand(1:5,5)))
    @test isreal(Diagonal(rand(5)))
end

@testset "issue #19267" begin
    @test ndims((1:3)[:]) == 1
    @test ndims((1:3)[:,:]) == 2
    @test ndims((1:3)[:,[1],:]) == 3
    @test ndims((1:3)[:,[1],:,[1]]) == 4
    @test ndims((1:3)[:,[1],1:1,:]) == 4
    @test ndims((1:3)[:,:,1:1,:]) == 4
    @test ndims((1:3)[:,:,1:1]) == 3
    @test ndims((1:3)[:,:,1:1,:,:,[1]]) == 6
end

struct Strider{T,N} <: AbstractArray{T,N}
    data::Vector{T}
    offset::Int
    strides::NTuple{N,Int}
    size::NTuple{N,Int}
end
function Strider{T}(strides::NTuple{N}, size::NTuple{N}) where {T,N}
    offset = 1-sum(strides .* (strides .< 0) .* (size .- 1))
    data = Array{T}(undef, sum(abs.(strides) .* (size .- 1)) + 1)
    return Strider{T, N, Vector{T}}(data, offset, strides, size)
end
function Strider(vec::AbstractArray{T}, strides::NTuple{N}, size::NTuple{N}) where {T,N}
    offset = 1-sum(strides .* (strides .< 0) .* (size .- 1))
    @assert length(vec) >= sum(abs.(strides) .* (size .- 1)) + 1
    return Strider{T, N}(vec, offset, strides, size)
end
Base.size(S::Strider) = S.size
function Base.getindex(S::Strider{<:Any,N}, I::Vararg{Int,N}) where {N}
    return S.data[sum(S.strides .* (I .- 1)) + S.offset]
end
Base.strides(S::Strider) = S.strides
Base.elsize(::Type{<:Strider{T}}) where {T} = Base.elsize(Vector{T})
Base.unsafe_convert(::Type{Ptr{T}}, S::Strider{T}) where {T} = pointer(S.data, S.offset)

@testset "Simple 3d strided views and permutes" for sz in ((5, 3, 2), (7, 11, 13))
    A = collect(reshape(1:prod(sz), sz))
    S = Strider(vec(A), strides(A), sz)
    @test pointer(A) == pointer(S)
    for i in 1:prod(sz)
        @test pointer(A, i) == pointer(S, i)
        @test A[i] == S[i]
    end
    for idxs in ((1:sz[1], 1:sz[2], 1:sz[3]),
                 (1:sz[1], 2:2:sz[2], sz[3]:-1:1),
                 (2:2:sz[1]-1, sz[2]:-1:1, sz[3]:-2:2),
                 (sz[1]:-1:1, sz[2]:-1:1, sz[3]:-1:1),
                 (sz[1]-1:-3:1, sz[2]:-2:3, 1:sz[3]),)
        Ai = A[idxs...]
        Av = view(A, idxs...)
        Sv = view(S, idxs...)
        Ss = Strider{Int, 3}(vec(A), sum((first.(idxs).-1).*strides(A))+1, strides(Av), length.(idxs))
        @test pointer(Av) == pointer(Sv) == pointer(Ss)
        for i in 1:length(Av)
            @test pointer(Av, i) == pointer(Sv, i) == pointer(Ss, i)
            @test Ai[i] == Av[i] == Sv[i] == Ss[i]
        end
        for perm in ((3, 2, 1), (2, 1, 3), (3, 1, 2))
            P = permutedims(A, perm)
            Ap = Base.PermutedDimsArray(A, perm)
            Sp = Base.PermutedDimsArray(S, perm)
            Ps = Strider{Int, 3}(vec(A), 1, strides(A)[collect(perm)], sz[collect(perm)])
            @test pointer(Ap) == pointer(Sp) == pointer(Ps)
            for i in 1:length(Ap)
                # This is intentionally disabled due to ambiguity
                @test_broken pointer(Ap, i) == pointer(Sp, i) == pointer(Ps, i)
                @test P[i] == Ap[i] == Sp[i] == Ps[i]
            end
            Pv = view(P, idxs[collect(perm)]...)
            Pi = P[idxs[collect(perm)]...]
            Apv = view(Ap, idxs[collect(perm)]...)
            Spv = view(Sp, idxs[collect(perm)]...)
            Pvs = Strider{Int, 3}(vec(A), sum((first.(idxs).-1).*strides(A))+1, strides(Apv), size(Apv))
            @test pointer(Apv) == pointer(Spv) == pointer(Pvs)
            for i in 1:length(Apv)
                @test pointer(Apv, i) == pointer(Spv, i) == pointer(Pvs, i)
                @test Pi[i] == Pv[i] == Apv[i] == Spv[i] == Pvs[i]
            end
            Vp = permutedims(Av, perm)
            Ip = permutedims(Ai, perm)
            Avp = Base.PermutedDimsArray(Av, perm)
            Svp = Base.PermutedDimsArray(Sv, perm)
            @test pointer(Avp) == pointer(Svp)
            for i in 1:length(Avp)
                # This is intentionally disabled due to ambiguity
                @test_broken pointer(Avp, i) == pointer(Svp, i)
                @test Ip[i] == Vp[i] == Avp[i] == Svp[i]
            end
        end
    end
end

@testset "simple 2d strided views, permutes, transposes" for sz in ((5, 3), (7, 11))
    A = collect(reshape(1:prod(sz), sz))
    S = Strider(vec(A), strides(A), sz)
    @test pointer(A) == pointer(S)
    for i in 1:prod(sz)
        @test pointer(A, i) == pointer(S, i)
        @test A[i] == S[i]
    end
    for idxs in ((1:sz[1], 1:sz[2]),
                 (1:sz[1], 2:2:sz[2]),
                 (2:2:sz[1]-1, sz[2]:-1:1),
                 (sz[1]:-1:1, sz[2]:-1:1),
                 (sz[1]-1:-3:1, sz[2]:-2:3),)
        Av = view(A, idxs...)
        Sv = view(S, idxs...)
        Ss = Strider{Int, 2}(vec(A), sum((first.(idxs).-1).*strides(A))+1, strides(Av), length.(idxs))
        @test pointer(Av) == pointer(Sv) == pointer(Ss)
        for i in 1:length(Av)
            @test pointer(Av, i) == pointer(Sv, i) == pointer(Ss, i)
            @test Av[i] == Sv[i] == Ss[i]
        end
        perm = (2, 1)
        P = permutedims(A, perm)
        Ap = Base.PermutedDimsArray(A, perm)
        At = transpose(A)
        Aa = adjoint(A)
        St = transpose(A)
        Sa = adjoint(A)
        Sp = Base.PermutedDimsArray(S, perm)
        Ps = Strider{Int, 2}(vec(A), 1, strides(A)[collect(perm)], sz[collect(perm)])
        @test pointer(Ap) == pointer(Sp) == pointer(Ps) == pointer(At) == pointer(Aa)
        for i in 1:length(Ap)
            # This is intentionally disabled due to ambiguity
            @test_broken pointer(Ap, i) == pointer(Sp, i) == pointer(Ps, i) == pointer(At, i) == pointer(Aa, i) == pointer(St, i) == pointer(Sa, i)
            @test pointer(Ps, i) == pointer(At, i) == pointer(Aa, i) == pointer(St, i) == pointer(Sa, i)
            @test P[i] == Ap[i] == Sp[i] == Ps[i] == At[i] == Aa[i] == St[i] == Sa[i]
        end
        Pv = view(P, idxs[collect(perm)]...)
        Apv = view(Ap, idxs[collect(perm)]...)
        Atv = view(At, idxs[collect(perm)]...)
        Ata = view(Aa, idxs[collect(perm)]...)
        Stv = view(St, idxs[collect(perm)]...)
        Sta = view(Sa, idxs[collect(perm)]...)
        Spv = view(Sp, idxs[collect(perm)]...)
        Pvs = Strider{Int, 2}(vec(A), sum((first.(idxs).-1).*strides(A))+1, strides(Apv), size(Apv))
        @test pointer(Apv) == pointer(Spv) == pointer(Pvs) == pointer(Atv) == pointer(Ata)
        for i in 1:length(Apv)
            @test pointer(Apv, i) == pointer(Spv, i) == pointer(Pvs, i) == pointer(Atv, i) == pointer(Ata, i) == pointer(Stv, i) == pointer(Sta, i)
            @test Pv[i] == Apv[i] == Spv[i] == Pvs[i] == Atv[i] == Ata[i] == Stv[i] == Sta[i]
        end
        Vp = permutedims(Av, perm)
        Avp = Base.PermutedDimsArray(Av, perm)
        Avt = transpose(Av)
        Ava = adjoint(Av)
        Svt = transpose(Sv)
        Sva = adjoint(Sv)
        Svp = Base.PermutedDimsArray(Sv, perm)
        @test pointer(Avp) == pointer(Svp) == pointer(Avt) == pointer(Ava)
        for i in 1:length(Avp)
            # This is intentionally disabled due to ambiguity
            @test_broken pointer(Avp, i) == pointer(Svp, i) == pointer(Avt, i) == pointer(Ava, i) == pointer(Svt, i) == pointer(Sva, i)
            @test pointer(Avt, i) == pointer(Ava, i) == pointer(Svt, i) == pointer(Sva, i)
            @test Vp[i] == Avp[i] == Svp[i] == Avt[i] == Ava[i] == Svt[i] == Sva[i]
        end
    end
end

@testset "first/last n elements of $(typeof(itr))" for itr in (collect(1:9),
                                                               [1 4 7; 2 5 8; 3 6 9],
                                                               ntuple(identity, 9))
    @test first(itr, 6) == [itr[1:6]...]
    @test first(itr, 25) == [itr[:]...]
    @test first(itr, 25) !== itr
    @test first(itr, 1) == [itr[1]]
    @test_throws ArgumentError first(itr, -6)
    @test last(itr, 6) == [itr[end-5:end]...]
    @test last(itr, 25) == [itr[:]...]
    @test last(itr, 25) !== itr
    @test last(itr, 1) == [itr[end]]
    @test_throws ArgumentError last(itr, -6)
end
