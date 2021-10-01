
include("access_styles.jl")
include("layouts.jl")

""" instantiate(lyt::Layouted) """
@inline function instantiate(x::Layouted{S,P}) where {S,P}
    lyt = _instantiate(P, layout(parent(x), S()))
    return Layouted{typeof(AccessStyle(lyt))}(
        parent(lyt),
        combined_index(getfield(lyt, :indices), getfield(x, :indices)),
        combined_transform(getfield(lyt, :f), getfield(x, :f))
    )
end
@inline _instantiate(::Type{P1}, lyt::Layouted{S,P2}) where {P1,S,P2} = instantiate(lyt)
_instantiate(::Type{P}, lyt::Layouted{S,P}) where {P,S} = lyt

@inline function instantiate(x::Layouted{AccessElement{N},P,<:AbstractCartesianIndex{N}}) where {P,N}
    Layouted{AccessElement{N}}(instantiate(layout(parent(x), AccessElement{N}())), getfield(x, :indices), getfield(x, :f))
end
@inline function instantiate(x::Layouted{AccessElement{1},P,<:CanonicalInt}) where {P}
    Layouted{AccessElement{1}}(instantiate(layout(parent(x), AccessElement{1}())), getfield(x, :indices), getfield(x, :f))
end
@inline function instantiate(x::Layouted{S,P,I}) where {S<:AccessIndices,P,I<:Tuple}
    Layouted{S}(
        instantiate(layout(parent(x), AccessElement{dynamic(ndims_index(I))}())),
        getfield(x, :indices),
        getfield(x, :f)
    )
end

# combine element transforms between arrays
combined_transform(::typeof(identity), y) = y
@inline function combined_transform(x::ComposedFunction, y)
    getfield(x, :outer) ∘ combined_transform(getfield(x, :inner), y)
end
@inline combined_transform(x, y) = _combined_transform(x, y)
_combined_transform(x, ::typeof(identity)) = x
@inline function _combined_transform(x, y::ComposedFunction)
    combined_index(x, getfield(y, :outer)) ∘ getfield(y, :inner)
end

function Base.showarg(io::IO, x::StrideIndex{N,R,C}, toplevel) where {N,R,C}
    print(io, "StrideIndex{$N,$R,$C}(")
    print(io, strides(x))
    print(io, ", ")
    print(io, offsets(x))
    print(io, ")")
end
function Base.showarg(io::IO, x::SubIndex, toplevel)
    print(io, "SubIndex{$(ndims(x))}(")
    print_index(io, getfield(x, :indices))
    print(io, ")")
end
function Base.showarg(io::IO, x::LinearSubIndex, toplevel)
    print(io, "LinearSubIndex(offset=$(getfield(x, :offset)),stride=$(getfield(x, :stride)))")
end
function Base.showarg(io::IO, x::CombinedIndex, toplevel)
    print(io, "combine(")
    print_index(io, x.i1)
    print(io, ", ")
    print_index(io, x.i2)
    print(io, ")")
end

function Base.showarg(io::IO, x::Layouted{S}, toplevel) where {S}
    print(io, "Layouted{$S}(")
    print_index(io, parent(x))
    print(io, ", ")
    print_index(io, x.indices)
    print(io, ", ")
    print_index(io, x.f)
    print(io, ")")
end

Base.show(io::IO, ::MIME"text/plain", x::Layouted) = Base.showarg(io, x, true)
Base.show(io::IO, ::MIME"text/plain", x::StrideIndex) = Base.showarg(io, x, true)
Base.show(io::IO, ::MIME"text/plain", x::SubIndex) = Base.showarg(io, x, true)

print_index(io, x::CartesianIndices) = print(io, "::CartesianIndices{$(ndims(x))}")
print_index(io, x) = Base.showarg(io, x, false)

