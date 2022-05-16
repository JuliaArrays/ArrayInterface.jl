module ArrayInterfaceLabelledArrays

using ArrayInterfaceCore
using LabelledArrays

function ArrayInterfaceCore.ismutable(::Type{<:LArray{T,N,Syms}}) where {T,N,Syms}
    ArrayInterfaceCore.ismutable(T)
end
ArrayInterfaceCore.can_setindex(::Type{<:SLArray}) = false

end # module
