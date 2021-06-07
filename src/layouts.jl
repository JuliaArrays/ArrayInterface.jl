
function _stride_index_type(::Type{T}) where {T}
    StrideIndex{
        ndims(T),
        known(stride_rank(T)),
        known(contiguous_axis(T)),
        _int_or_static_int(known_strides(T)),
        _int_or_static_int(known_offsets(T)),
        _int_or_static_int(known_offset1(T))
    }
end

function _nd_linear_index_type(::Type{T}) where {T}
    NDLinearIndex{
        ndims(T),
        _int_or_static_int(known_offset1(T)),
        _int_or_static_int(known_offsets(T)),
        _int_or_static_int(known_size(T))
    }
end
function _linear_nd_index_type(::Type{T}) where {T}
    LinearNDIndex{
        ndims(T),
        _int_or_static_int(known_offset1(T)),
        _int_or_static_int(known_offsets(T)),
        _int_or_static_int(known_size(T))
    }
end

""" AccessStyle """
abstract type AccessStyle end

struct LinearAccess <: AccessStyle end

struct CartesianAccess <: AccessStyle end

struct UnorderedAccess <: AccessStyle end

AccessStyle(x) = AccessStyle(typeof(x))
AccessStyle(::Type{<:CanonicalInt}) = LinearAccess()
AccessStyle(::Type{<:AbstractCartesianIndex}) = CartesianAccess()
AccessStyle(::Type{<:AbstractArray{<:CanonicalInt}}) = LinearAccess()
AccessStyle(::Type{<:AbstractArray{<:AbstractCartesianIndex}}) = CartesianAccess()
AccessStyle(::Type{<:Tuple{I}}) where {I} = AccessStyle(I)
AccessStyle(::Type{<:Tuple{I,Vararg{Any}}}) where {I} = CartesianAccess()

"""
    LayoutPlan(::Type{T}, a::AccessStyle)

Provides a unique set of instructions for constructing a layout for any instance of the type
`T` conditional on the `a`.

* `index` : constructs the index transformation `index(::T)`. If `nothing` then there is no
  transformation to an index passed between `T` and its referene data.
* `plan` : adds additional index transformations to the layout of `T`.
* `link` : if the `plan` field requires processing of any instance of `T` prior to execution
  then `link` may contain the appropriate methods for doing so. For example, if `T` wraps
  a parent type than `link` might be `parent`.
"""
struct LayoutPlan{I,P,L}
    index::I
    plan::P
    link::L

    LayoutPlan(index::I, plan::P, link::L) where {I,P,L} = new{I,P,L}(index, plan, link)
end

const WrapperPlan{P,L} = LayoutPlan{Nothing,P,L}
# if the only thing that exists is a link function then no layout is actually produced
const NoopPlan{L} = LayoutPlan{Nothing,Nothing,L}

LayoutPlan(index, plan) = LayoutPlan(index, plan, nothing)

# get rid of link when it doesn't go anywhere
LayoutPlan(index, ::Nothing, link) = LayoutPlan(index, nothing, nothing)

LayoutPlan(::Nothing, ::NoopPlan, link) = LayoutPlan(nothing, nothing, link)
function LayoutPlan(::Nothing, plan::WrapperPlan, link)
    LayoutPlan(nothing, plan.plan, _combine_links(link, plan.link))
end
_combine_links(link1, link2) = (link1, link2)
_combine_links(link1, link2::Tuple) = (link1, link2...)
_combine_links(::Nothing, link2) = link2
_combine_links(::Nothing, link2::Tuple) = link2

@generated function layout(a::A, ::S) where {A,S}
    p = layout_plan(A, S())
    if p === nothing
        return nothing
    else
        Expr(:call, :LinkedIndex, Expr(:tuple, _layoutexpr(:a, p)...))
    end
end

_layoutexpr(sym, p) = [:($p($sym))]
_layoutexpr(sym, p::LayoutPlan{I,Nothing,Nothing}) where {I} = [:($(p.index)($sym))]
function _layoutexpr(sym, p::LayoutPlan{I,P,Nothing}) where {I,P}
    [_indexexpr(sym, p.index), _layoutexpr(sym, p.plan)...]
end
function _layoutexpr(sym, p::LayoutPlan{I,P,L}) where {I,P,L}
    [_indexexpr(sym, p.index), _layoutexpr(_linkexpr(sym, p.link), p.plan)...]
end
_linkexpr(sym, link::Tuple{Any}) = :($(first(link))(sym))
_linkexpr(sym, link::Tuple{Any,Vararg{Any}}) = _linkexpr(:($(first(link))(sym)), tail(link))
_indexexpr(sym, index) = __indexexpr(is_static(index), sym, index)
__indexexpr(::True, sym, index) = :($(index()))
__indexexpr(::False, sym, index) = :($(index)($sym))

""" layout_plan """
layout_plan(::Any, ::AccessStyle) = nothing
function layout_plan(::Type{A}, ::UnorderedAccess) where {A<:ReshapedArray}
    p = layout_plan(parent_type(A), UnorderedAccess())
    if p === nothing
        return nothing
    else
        return LayoutPlan(nothing, p, parent)
    end
end
function layout_plan(::Type{A}, ::LinearAccess) where {A<:ReshapedArray}
    p = layout_plan(parent_type(A), LinearAccess())
    if p === nothing
        return nothing
    else
        return LayoutPlan(nothing, p, parent)
    end
end
function layout_plan(::Type{A}, ::CartesianAccess) where {A<:ReshapedArray}
    p = layout_plan(parent_type(A), LinearAccess())
    if p === nothing
        return nothing
    else
        return LayoutPlan(_nd_linear_index_type(A), p)
    end
end

function layout_plan(::Type{A}, ::UnorderedAccess) where {A<:Union{PermutedDimsArray,Adjoint,Transpose}}
    p = layout_plan(parent_type(A), UnorderedAccess())
    if p === nothing
        return nothing
    else
        return LayoutPlan(nothing, p, parent)
    end
end
function layout_plan(::Type{A}, ::CartesianAccess) where {A<:Union{PermutedDimsArray,Adjoint,Transpose}}
    combine_layouts(A, layout_plan(parent_type(A), CartesianAccess()))
end
function layout_plan(::Type{A}, ::LinearAccess) where {A<:Union{PermutedDimsArray,Adjoint,Transpose}}
    LayoutPlan(
        _linear_nd_index_type(A),
        layout_plan(A, CartesianAccess())
    )
end

function layout_plan(::Type{A}, ::UnorderedAccess) where {A<:SubArray}
    I = _sub_indices_types(A) <: Tuple{Vararg{Slice}}
    if I <: Tuple{Vararg{Slice}}
        return LayoutPlan(nothing, layout_plan(parent_type(A), UnorderedAccess()), parent)
    else
        access = _view_access(I)
        if LinearAccess() === access
            return LayoutPlan(LinearIndices, layout_plan(parent_type(A), access), parent)
        else
            return LayoutPlan(CartesianIndices, layout_plan(parent_type(A), access), parent)
        end
    end
end
function layout_plan(::Type{A}, ::CartesianAccess) where {A<:SubArray}
    combine_layouts(A, layout_plan(parent_type(A), CartesianAccess()))
end
function layout_plan(::Type{A}, ::LinearAccess) where {A<:SubArray}
    access = _view_access(_sub_indices_types(A))
    if LinearAccess() === access
        return combine_layouts(A, layout_plan(parent_type(A), access))
    else
        return LayoutPlan(_linear_nd_index_type(A), combine_layouts(A, layout_plan(parent_type(A), access)))
    end
end
_view_access(::Type{Tuple{}}) = LinearAccess()
_view_access(::Type{I}) where {I<:Tuple{Real, Vararg{Any}}} = _view_access(Base.tuple_type_tail(I))
_view_access(::Type{I}) where {I<:Tuple{Slice, Slice, Vararg{Any}}} = _view_access(Base.tuple_type_tail(I))
_view_access(::Type{I}) where {I<:Tuple{Slice, AbstractUnitRange, Vararg{Real}}} = LinearAccess()
_view_access(::Type{I}) where {I<:Tuple{Slice, Slice, Vararg{Real}}} = LinearAccess()
_view_access(::Type{I}) where {I<:Tuple{AbstractRange, Vararg{Real}}} = LinearAccess()
_view_access(::Type{I}) where {I<:Tuple{Vararg{Any}}} = CartesianAccess()
_view_access(::Type{I}) where {I<:Tuple{AbstractArray,Vararg{Any}}} = CartesianAccess()

layout_plan(::Type{A}, ::UnorderedAccess) where {A<:DenseArray} = indices
layout_plan(::Type{A}, ::LinearAccess) where {A<:DenseArray} = LinearIndex{StaticInt{0}}
layout_plan(::Type{A}, ::CartesianAccess) where {A<:DenseArray} = _stride_index_type(A)

layout_plan(::Type{A}, ::IndexLinear) where {A<:AbstractRange} = LinearIndex{StaticInt{0}}
layout_plan(::Type{A}, ::UnorderedAccess) where {A<:AbstractRange} = indices

""" combine_layouts """
combine_layouts(::Type{A}, ::Nothing) where {A} = nothing
combine_layouts(::Type{A}, ::NoopPlan) where {A} = nothing
function combine_layouts(::Type{A}, p::WrapperPlan) where {A}
    _insert_previous_links(combine_layouts(A, p.plan), p.link)
end
_insert_previous_links(p, links) = p
function _insert_previous_links(p::LayoutPlan, links)
    LayoutPlan(p.index, p.plan, _combine_links(p.link, links))
end
function combine_layouts(::Type{A}, ::Type{I}) where {A<:VecAdjTrans,I<:ArrayIndex{1}}
    LayoutPlan(TransposedVectorIndex, I, parent)
end
function combine_layouts(::Type{A}, ::Type{I}) where {A<:VecAdjTrans,I<:StrideIndex{1}}
    _stride_index_type(A)
end

function combine_layouts(::Type{A}, ::Type{I}) where {A<:MatAdjTrans,I<:ArrayIndex{2}}
    return LayoutPlan(PermutedIndex{2,(2,1)}, I, parent)
end
function combine_layouts(::Type{A}, ::Type{I}) where {A<:MatAdjTrans,I<:StrideIndex{2}}
    _stride_index_type(A)
end

function combine_layouts(::Type{A}, ::Type{I}) where {N,perm,A<:PermutedDimsArray{<:Any,N,perm},I<:ArrayIndex{N}}
    return LayoutPlan(PermutedIndex{N,perm}, I, parent)
end
function combine_layouts(::Type{A}, ::Type{I}) where {N,perm,A<:PermutedDimsArray{<:Any,N,perm},I<:StrideIndex{N}}
    _stride_index_type(A)
end

combine_layouts(::Type{A}, ::Type{I}) where {A<:ReshapedArray,I<:ArrayIndex{1}} = I

# TODO LinearIndex and LinearSubIndex may be static but SubArray doesn't account for this
function combine_layouts(::Type{A}, ::Type{I}) where {A<:Base.FastSubArray,I<:ArrayIndex{1}}
    LayoutPlan(LinearSubIndex{Int,Int}, I, parent)
end
function combine_layouts(::Type{A}, ::Type{I}) where {A<:Base.FastContiguousSubArray,I<:ArrayIndex{1}}
    LayoutPlan(LinearIndex{Int}, I, parent)
end

function combine_layouts(::Type{A}, ::Type{I}) where {N,A<:SubArray{<:Any,N},I<:StrideIndex}
    if defines_strides(A)
        return _stride_index_type(A)
    else
        return LayoutPlan(SubIndex{N,_sub_indices_types(A)}, I, parent)
    end
end

