module ArrayInterface

using Requires

function ismutable end

"""
    ismutable(x::DataType)

Query whether a type is mutable or not, see
https://github.com/JuliaDiffEq/RecursiveArrayTools.jl/issues/19.
"""
Base.@pure ismutable(x::DataType) = x.mutable
ismutable(x) = ismutable(typeof(x))

ismutable(::Type{Array}) = true
ismutable(::Type{<:Number}) = false


function __init__()

  @require StaticArrays="90137ffa-7385-5640-81b9-e52037218182" begin
    ismutable(::Type{StaticArrays.StaticArray}) = false
    ismutable(::Type{StaticArrays.MArray}) = true
  end

  @require LabelledArrays="2ee39098-c373-598a-b85f-a56591580800" begin
    ismutable(::Type{LabelledArrays.LArray{T,N,Syms}}) where {T,N,Syms} = ismutable(T)
  end
  
  @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
    ismutable(x::Flux.Tracker.TrackedArray) = false
  end
end

end
