module ArrayInterfaceFillArraysExt

using ArrayInterface: ArrayInterface
using FillArrays: AbstractFill, OneElement

ArrayInterface.can_setindex(::Type{<:AbstractFill}) = false
ArrayInterface.can_setindex(::Type{<:OneElement}) = false

end
