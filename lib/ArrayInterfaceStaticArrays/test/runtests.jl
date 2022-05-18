using ArrayInterfaceCore
using ArrayInterfaceStaticArrays
using LinearAlgebra
using StaticArrays
using Static
using Test

x = @SVector [1,2,3]
@test ArrayInterfaceCore.ismutable(x) == false
@test ArrayInterfaceCore.ismutable(view(x, 1:2)) == false
@test ArrayInterfaceCore.can_setindex(typeof(x)) == false
@test ArrayInterfaceCore.buffer(x) == x.data
@test @inferred(ArrayInterfaceCore.device(typeof(x))) === ArrayInterfaceCore.CPUTuple()

x = @MVector [1,2,3]
@test ArrayInterfaceCore.ismutable(x) == true
@test ArrayInterfaceCore.ismutable(view(x, 1:2)) == true
@test @inferred(ArrayInterfaceCore.device(typeof(x))) === ArrayInterfaceCore.CPUPointer()

A = @SMatrix(randn(5, 5))
@test ArrayInterfaceCore.lu_instance(A) isa typeof(lu(A))
A = @MMatrix(randn(5, 5))
@test ArrayInterfaceCore.lu_instance(A) isa typeof(lu(A))

@test isone(ArrayInterfaceCore.known_first(typeof(StaticArrays.SOneTo(7))))
@test ArrayInterfaceCore.known_last(typeof(StaticArrays.SOneTo(7))) == 7
@test ArrayInterfaceCore.known_length(typeof(StaticArrays.SOneTo(7))) == 7

@test ArrayInterfaceCore.parent_type(SizedVector{1, Int, Vector{Int}}) <: Vector{Int}
@test ArrayInterfaceCore.known_length(@inferred(ArrayInterfaceCore.indices(SOneTo(7)))) == 7

x = view(SArray{Tuple{3,3,3}}(ones(3,3,3)), :, SOneTo(2), 2)
@test @inferred(ArrayInterfaceCore.known_length(x)) == 6
@test @inferred(ArrayInterfaceCore.known_length(x')) == 6

v = @SVector rand(8);
A = @MMatrix rand(7, 6);
T = SizedArray{Tuple{5,4,3}}(zeros(5,4,3));
@test @inferred(ArrayInterfaceCore.length(v)) === StaticInt(8)
@test @inferred(ArrayInterfaceCore.length(A)) === StaticInt(42)
@test @inferred(ArrayInterfaceCore.length(T)) === StaticInt(60)

x = @SMatrix rand(Float32, 2, 2)
y = @SVector rand(4)
yr = ArrayInterfaceCore.restructure(x, y)
@test yr isa SMatrix{2, 2}
@test Base.size(yr) == (2,2)
@test vec(yr) == vec(y)
z = rand(4)
zr = ArrayInterfaceCore.restructure(x, z)
@test zr isa SMatrix{2, 2}
@test Base.size(zr) == (2,2)
@test vec(zr) == vec(z)

Am = @MMatrix rand(2,10);
@test @inferred(ArrayInterfaceCore.strides(view(Am,1,:))) === (StaticInt(2),)

@test @inferred(ArrayInterfaceCore.contiguous_axis(@SArray(zeros(2,2,2)))) === ArrayInterfaceCore.StaticInt(1)
@test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(@SArray(zeros(2,2,2)))) == (true,false,false)
@test @inferred(ArrayInterfaceCore.contiguous_batch_size(@SArray(zeros(2,2,2)))) === ArrayInterfaceCore.StaticInt(0)
@test @inferred(ArrayInterfaceCore.stride_rank(@SArray(zeros(2,2,2)))) == (1, 2, 3)
@test @inferred(ArrayInterfaceCore.is_column_major(@SArray(zeros(2,2,2)))) === True()
@test @inferred(ArrayInterfaceCore.dense_dims(@SArray(zeros(2,2,2)))) == (true,true,true)

