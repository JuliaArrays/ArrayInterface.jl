using ArrayInterface
using BandedMatrices
using Test

function checkequal(idx1::ArrayInterface.BandedMatrixIndex,
    idx2::ArrayInterface.BandedMatrixIndex)
    return idx1.rowsize == idx2.rowsize && idx1.colsize == idx2.colsize &&
           idx1.bandinds == idx2.bandinds && idx1.bandsizes == idx2.bandsizes &&
           idx1.isrow == idx2.isrow && idx1.count == idx2.count
end

B = BandedMatrix(Ones(5, 5), (-1, 2))
B[band(1)] .= [1, 2, 3, 4]
B[band(2)] .= [5, 6, 7]
@test ArrayInterface.has_sparsestruct(B)
rowind, colind = ArrayInterface.findstructralnz(B)
@test [B[rowind[i], colind[i]] for i in 1:length(rowind)] == [5, 6, 7, 1, 2, 3, 4]
B = BandedMatrix(Ones(4, 6), (-1, 2))
B[band(1)] .= [1, 2, 3, 4]
B[band(2)] .= [5, 6, 7, 8]
rowind, colind = ArrayInterface.findstructralnz(B)
@test [B[rowind[i], colind[i]] for i in 1:length(rowind)] == [5, 6, 7, 8, 1, 2, 3, 4]
@test ArrayInterface.isstructured(typeof(B))
@test ArrayInterface.fast_matrix_colors(typeof(B))

for op in (adjoint, transpose)
    B = BandedMatrix(Ones(5, 5), (-1, 2))
    B[band(1)] .= [1, 2, 3, 4]
    B[band(2)] .= [5, 6, 7]
    B′ = op(B)
    @test ArrayInterface.has_sparsestruct(B′)
    rowind′, colind′ = ArrayInterface.findstructralnz(B′)
    rowind′′, colind′′ = ArrayInterface.findstructralnz(BandedMatrix(B′))
    @test checkequal(rowind′, rowind′′)
    @test checkequal(colind′, colind′′)

    B = BandedMatrix(Ones(4, 6), (-1, 2))
    B[band(1)] .= [1, 2, 3, 4]
    B[band(2)] .= [5, 6, 7, 8]
    B′ = op(B)
    rowind′, colind′ = ArrayInterface.findstructralnz(B′)
    rowind′′, colind′′ = ArrayInterface.findstructralnz(BandedMatrix(B′))
    @test checkequal(rowind′, rowind′′)
    @test checkequal(colind′, colind′′)

    @test ArrayInterface.isstructured(typeof(B′))
    @test ArrayInterface.fast_matrix_colors(typeof(B′))

    @test ArrayInterface.matrix_colors(B′) == ArrayInterface.matrix_colors(BandedMatrix(B′))
end
