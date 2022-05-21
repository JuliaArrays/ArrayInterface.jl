using ArrayInterface
using ArrayInterfaceStaticArrays
using LinearAlgebra
using StaticArrays
using Static
using Test

x = @SVector [1,2,3]
@test ArrayInterface.ismutable(x) == false
@test ArrayInterface.ismutable(view(x, 1:2)) == false
@test ArrayInterface.can_setindex(typeof(x)) == false
@test ArrayInterface.buffer(x) == x.data
@test @inferred(ArrayInterface.device(typeof(x))) === ArrayInterface.CPUTuple()

x = @MVector [1,2,3]
@test ArrayInterface.ismutable(x) == true
@test ArrayInterface.ismutable(view(x, 1:2)) == true
@test @inferred(ArrayInterface.device(typeof(x))) === ArrayInterface.CPUPointer()

A = @SMatrix(randn(5, 5))
@test ArrayInterface.lu_instance(A) isa typeof(lu(A))
A = @MMatrix(randn(5, 5))
@test ArrayInterface.lu_instance(A) isa typeof(lu(A))

@test isone(ArrayInterface.known_first(typeof(StaticArrays.SOneTo(7))))
@test ArrayInterface.known_last(typeof(StaticArrays.SOneTo(7))) == 7
@test ArrayInterface.known_length(typeof(StaticArrays.SOneTo(7))) == 7

@test ArrayInterface.parent_type(SizedVector{1, Int, Vector{Int}}) <: Vector{Int}
@test ArrayInterface.known_length(@inferred(ArrayInterface.indices(SOneTo(7)))) == 7

x = view(SArray{Tuple{3,3,3}}(ones(3,3,3)), :, SOneTo(2), 2)
@test @inferred(ArrayInterface.known_length(x)) == 6
@test @inferred(ArrayInterface.known_length(x')) == 6

v = @SVector rand(8);
A = @MMatrix rand(7, 6);
T = SizedArray{Tuple{5,4,3}}(zeros(5,4,3));
@test @inferred(ArrayInterface.length(v)) === StaticInt(8)
@test @inferred(ArrayInterface.length(A)) === StaticInt(42)
@test @inferred(ArrayInterface.length(T)) === StaticInt(60)

x = @SMatrix rand(Float32, 2, 2)
y = @SVector rand(4)
yr = ArrayInterface.restructure(x, y)
@test yr isa SMatrix{2, 2}
@test Base.size(yr) == (2,2)
@test vec(yr) == vec(y)
z = rand(4)
zr = ArrayInterface.restructure(x, z)
@test zr isa SMatrix{2, 2}
@test Base.size(zr) == (2,2)
@test vec(zr) == vec(z)

Am = @MMatrix rand(2,10);
@test @inferred(ArrayInterface.strides(view(Am,1,:))) === (StaticInt(2),)

@test @inferred(ArrayInterface.contiguous_axis(@SArray(zeros(2,2,2)))) === ArrayInterface.StaticInt(1)
@test @inferred(ArrayInterface.contiguous_axis_indicator(@SArray(zeros(2,2,2)))) == (true,false,false)
@test @inferred(ArrayInterface.contiguous_batch_size(@SArray(zeros(2,2,2)))) === ArrayInterface.StaticInt(0)
@test @inferred(ArrayInterface.stride_rank(@SArray(zeros(2,2,2)))) == (1, 2, 3)
@test @inferred(ArrayInterface.is_column_major(@SArray(zeros(2,2,2)))) === True()
@test @inferred(ArrayInterface.dense_dims(@SArray(zeros(2,2,2)))) == (true,true,true)

