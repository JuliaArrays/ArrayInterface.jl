module ArrayInterface

using Requires

function ismutable end
ismutable(x::Array) = true


function __init__()

  @require StaticArrays="90137ffa-7385-5640-81b9-e52037218182" begin
    ismutable(x::StaticArrays.StaticArray) = false
    ismutable(x::StaticArrays.MArray) = true
  end

  @require LabelledArrays="2ee39098-c373-598a-b85f-a56591580800" begin
    ismutable(x::LVector) = ismutable(x.__x)
  end
end

end
