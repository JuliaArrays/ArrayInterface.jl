using ArrayInterface, Test

@test ArrayInterface.ismutable(rand(3))

using StaticArrays
ArrayInterface.ismutable(@SVector [1,2,3]) == false
ArrayInterface.ismutable(@MVector [1,2,3]) == true
