module ArrayInterface

using Requires

function ismutable end

"""
    ismutable(x::DataType)

Query whether a type is mutable or not, see
https://github.com/JuliaDiffEq/RecursiveArrayTools.jl/issues/19.
"""
Base.@pure ismutable(x::DataType) = x.mutable

ismutable(x::Array) = true


function __init__()

  @require StaticArrays="90137ffa-7385-5640-81b9-e52037218182" begin
    ismutable(x::StaticArrays.StaticArray) = false
    ismutable(x::StaticArrays.MArray) = true
  end

  @require LabelledArrays="2ee39098-c373-598a-b85f-a56591580800" begin
    ismutable(x::LVector) = ismutable(x.__x)
  end
  
  @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
    ismutable(x::Flux.Tracker.TrackedArray) = false
  end
end

end
