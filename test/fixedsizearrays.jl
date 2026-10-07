using ArrayInterface, FixedSizeArrays, LinearAlgebra, Test

@testset "$T" for T in (Float64, Float32, ComplexF64, Int, BigFloat)
    F = FixedSizeMatrix{T}(T.(rand(1:5, 4, 4)) + 10I)

    luF = ArrayInterface.lu_instance(F)
    @test luF isa typeof(lu(F, check = false))
    @test size(luF.factors) == (0, 0)

    qrF = ArrayInterface.qr_instance(F)
    @test qrF isa typeof(qr(F))
    @test size(qrF.factors) == (0, 0)

    qrpF = ArrayInterface.qr_instance(F, ColumnNorm())
    @test qrpF isa typeof(qr(F, ColumnNorm()))
    @test size(qrpF.factors) == (0, 0)
end
