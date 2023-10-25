module ArrayInterfaceOffsetArraysExt

if isdefined(Base, :get_extension) 
    using ArrayInterface
    using OffsetArrays
else 
    using ..ArrayInterface
    using ..OffsetArrays
end

ArrayInterface.parent_type(@nospecialize T::Type{<:OffsetArrays.IdOffsetRange}) = fieldtype(T, :parent)
ArrayInterface.parent_type(@nospecialize T::Type{<:OffsetArray}) = fieldtype(T, :parent)

function ArrayInterface.known_size(@nospecialize T::Type{<:OffsetArrays.IdOffsetRange})
    ArrayInterface.known_size(ArrayInterface.parent_type(T))
end
function ArrayInterface.known_size(@nospecialize T::Type{<:OffsetArray})
    ArrayInterface.known_size(ArrayInterface.parent_type(T))
end

end
