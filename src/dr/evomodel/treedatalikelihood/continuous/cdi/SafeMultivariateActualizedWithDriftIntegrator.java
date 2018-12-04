package dr.evomodel.treedatalikelihood.continuous.cdi;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import static dr.math.matrixAlgebra.missingData.MissingOps.*;

/**
 * @author Marc A. Suchard
 * @author Paul Bastide
 */

public class SafeMultivariateActualizedWithDriftIntegrator extends SafeMultivariateDiagonalActualizedWithDriftIntegrator {

    private static boolean DEBUG = false;

    public SafeMultivariateActualizedWithDriftIntegrator(PrecisionType precisionType,
                                                         int numTraits, int dimTrait, int dimProcess,
                                                         int bufferCount, int diffusionCount,
                                                         boolean isActualizationSymmetric) {
        super(precisionType, numTraits, dimTrait, dimProcess, bufferCount, diffusionCount);

        allocateStorage();

        this.isActualizationSymmetric = isActualizationSymmetric;

        System.err.println("Trying SafeMultivariateActualizedWithDriftIntegrator");
    }

    @Override
    public void getBranchActualization(int bufferIndex, double[] actualization) {

        if (bufferIndex == -1) {
            throw new RuntimeException("Not yet implemented");
        }

        assert (actualization != null);
        assert (actualization.length >= dimTrait * dimTrait);

        System.arraycopy(actualizations, bufferIndex * dimTrait * dimTrait,
                actualization, 0, dimTrait * dimTrait);
    }

    @Override
    public void getBranchExpectation(double[] actualization, double[] parentValue, double[] displacement,
                                     double[] expectation) {

        assert (expectation != null);
        assert (expectation.length >= dimTrait);

        assert (actualization != null);
        assert (actualization.length >= dimTrait * dimTrait);

        assert (parentValue != null);
        assert (parentValue.length >= dimTrait);

        assert (displacement != null);
        assert (displacement.length >= dimTrait);

        DMatrixRMaj branchExpectationMatrix = new DMatrixRMaj(dimTrait, 1);
        CommonOps_DDRM.mult(wrap(actualization, 0, dimTrait, dimTrait),
                wrap(parentValue, 0, dimTrait, 1),
                branchExpectationMatrix);
        CommonOps_DDRM.addEquals(branchExpectationMatrix, wrap(displacement, 0, dimTrait, 1));

        unwrap(branchExpectationMatrix, expectation, 0);
    }

    private static final boolean TIMING = false;

    private void allocateStorage() {

        actualizations = new double[dimTrait * dimTrait * bufferCount];

        matrixQdiPip = new DMatrixRMaj(dimTrait, dimTrait);
        matrixQdjPjp = new DMatrixRMaj(dimTrait, dimTrait);

        matrixNiacc = new DMatrixRMaj(dimTrait, 1);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Setting variances, displacement and actualization vectors
    ///////////////////////////////////////////////////////////////////////////

    @Override
    public void setDiffusionStationaryVariance(int precisionIndex, final double[] alphaEig, final double[] alphaRot) {

        int dim = alphaEig.length;
        assert alphaRot.length == dim * dim;

        super.setDiffusionStationaryVariance(precisionIndex, alphaEig, alphaRot);

        // Transform back in original space
        final int offset = dimProcess * dimProcess * precisionIndex;
        transformMatrixBack(stationaryVariances, offset, alphaRot, 0);

        if (DEBUG) {
            System.err.println("At precision index: " + precisionIndex);
            System.err.println("stationary variance: " + wrap(stationaryVariances, offset, dim, dim));
        }
    }

    @Override
    void setStationaryVariance(int offset, double[] scales, int matrixSize, double[] rotation) {
        assert (rotation.length == matrixSize);

        DMatrixRMaj rotMat = wrap(rotation, 0, dimProcess, dimProcess);
        DMatrixRMaj variance = wrap(inverseDiffusions, offset, dimProcess, dimProcess);
        transformMatrix(variance, rotMat);
        double[] transVar = new double[matrixSize];
        unwrap(variance, transVar, 0);
        scaleInv(transVar, 0, scales, stationaryVariances, offset, matrixSize);
    }

    private void transformMatrix(DMatrixRMaj matrix, DMatrixRMaj rotation) {
        if (isActualizationSymmetric) {
            transformMatrixSymmetric(matrix, rotation);
        } else {
            transformMatrixGeneral(matrix, rotation);
        }
    }

    private void transformMatrixGeneral(DMatrixRMaj matrix, DMatrixRMaj rotation) {
        DMatrixRMaj tmp = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.invert(rotation); // Warning: side effect on rotation matrix.
        CommonOps_DDRM.mult(rotation, matrix, tmp);
        CommonOps_DDRM.multTransB(tmp, rotation, matrix);
    }

    private void transformMatrixSymmetric(DMatrixRMaj matrix, DMatrixRMaj rotation) {
        DMatrixRMaj tmp = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.multTransA(rotation, matrix, tmp);
        CommonOps_DDRM.mult(tmp, rotation, matrix);
    }

    private void transformMatrixBack(double[] matrixDouble, int matrixOffset,
                                     double[] rotationDouble, int rotationOffset) {
        DMatrixRMaj matrix = wrap(matrixDouble, matrixOffset, dimProcess, dimProcess);
        DMatrixRMaj rotation = wrap(rotationDouble, rotationOffset, dimProcess, dimProcess);
        transformMatrixBack(matrix, rotation);
        unwrap(matrix, matrixDouble, matrixOffset);
    }

    private void transformMatrixBack(DMatrixRMaj matrix, DMatrixRMaj rotation) {
        DMatrixRMaj tmp = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.multTransB(matrix, rotation, tmp);
        CommonOps_DDRM.mult(rotation, tmp, matrix);
    }

    private void transformDiagonalMatrixBack(double[] diagonalMatrix, double[] matrixDestination, int matrixOffset,
                                             double[] rotationDouble, int rotationOffset) {
        DMatrixRMaj matrix = wrapDiagonal(diagonalMatrix, matrixOffset, dimProcess);
        DMatrixRMaj rotation = wrap(rotationDouble, rotationOffset, dimProcess, dimProcess);
        transformMatrixBase(matrix, rotation);
        unwrap(matrix, matrixDestination, matrixOffset);
    }

    private DMatrixRMaj getInverseSelectionStrength(double[] diagonalMatrix, double[] rotationDouble) {
        DMatrixRMaj matrix = wrapDiagonalInverse(diagonalMatrix, 0, dimProcess);
        DMatrixRMaj rotation = wrap(rotationDouble, 0, dimProcess, dimProcess);
        transformMatrixBase(matrix, rotation);
        return matrix;
    }

    private void transformMatrixBase(DMatrixRMaj matrix, DMatrixRMaj rotation) {
        if (isActualizationSymmetric) {
            transformMatrixBack(matrix, rotation);
        } else {
            transformMatrixBaseGeneral(matrix, rotation);
        }
    }

    private void transformMatrixBaseGeneral(DMatrixRMaj matrix, DMatrixRMaj rotation) {
        DMatrixRMaj tmp = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.mult(rotation, matrix, tmp);
        CommonOps_DDRM.invert(rotation); // Warning: side effect on rotation matrix.
        CommonOps_DDRM.mult(tmp, rotation, matrix);
    }

    @Override
    void computeOUActualization(final double[] diagonalStrengthOfSelectionMatrix,
                                final double[] rotation,
                                final double edgeLength,
                                final int scaledOffsetDiagonal,
                                final int scaledOffset) {
        double[] diagonalActualizations = new double[dimTrait];
        computeOUDiagonalActualization(diagonalStrengthOfSelectionMatrix, edgeLength, dimProcess,
                diagonalActualizations, 0);
        transformDiagonalMatrixBack(diagonalActualizations, actualizations, scaledOffset, rotation, 0);
    }

    @Override
    void computeOUVarianceBranch(final int sourceOffset,
                                 final int destinationOffset,
                                 final int destinationOffsetDiagonal,
                                 final double edgeLength) {
        DMatrixRMaj actualization = wrap(actualizations, destinationOffset, dimProcess, dimProcess);
        DMatrixRMaj variance = wrap(stationaryVariances, sourceOffset, dimProcess, dimProcess);
        DMatrixRMaj temp = new DMatrixRMaj(dimProcess, dimProcess);

        CommonOps_DDRM.multTransB(variance, actualization, temp);
        CommonOps_DDRM.multAdd(-1.0, actualization, temp, variance);

        unwrap(variance, variances, destinationOffset);
    }

    @Override
    void computeOUActualizedDisplacement(final double[] optimalRates,
                                         final int offset,
                                         final int actualizationOffset,
                                         final int pio) {
        DMatrixRMaj actualization = wrap(actualizations, actualizationOffset, dimProcess, dimProcess);
        DMatrixRMaj optVal = wrap(optimalRates, offset, dimProcess, 1);
        DMatrixRMaj temp = CommonOps_DDRM.identity(dimProcess);
        DMatrixRMaj displacement = new DMatrixRMaj(dimProcess, 1);

        CommonOps_DDRM.addEquals(temp, -1.0, actualization);
        CommonOps_DDRM.mult(temp, optVal, displacement);

        unwrap(displacement, displacements, pio);
    }

    private void computeIOUActualizedDisplacement(final double[] optimalRates,
                                                  final int offset,
                                                  final int pio,
                                                  double branchLength,
                                                  DMatrixRMaj inverseSelectionStrength) {
        DMatrixRMaj displacementOU = wrap(displacements, pio, dimProcess, 1);
        DMatrixRMaj optVal = wrap(optimalRates, offset, dimProcess, 1);
        DMatrixRMaj displacement = new DMatrixRMaj(dimProcess, 1);

        CommonOps_DDRM.mult(inverseSelectionStrength, displacementOU, displacement);
        CommonOps_DDRM.scale(-1.0, displacement);

        CommonOps_DDRM.addEquals(displacement, branchLength, optVal);

        unwrap(displacement, displacements, pio + dimProcess);
    }

    private void computeIOUVarianceBranch(final int sourceOffset,
                                          final int destinationOffset,
                                          double branchLength,
                                          DMatrixRMaj inverseSelectionStrength) {
        DMatrixRMaj actualization = wrap(actualizations, destinationOffset, dimProcess, dimProcess);
        DMatrixRMaj stationaryVariance = wrap(stationaryVariances, sourceOffset, dimProcess, dimProcess);

        DMatrixRMaj invAS = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.mult(inverseSelectionStrength, stationaryVariance, invAS);

        //// Variance YY
        DMatrixRMaj varianceYY = wrap(variances, destinationOffset, dimProcess, dimProcess);

        //// Variance XX
        DMatrixRMaj varianceXX = new DMatrixRMaj(dimProcess, dimProcess);
        // Variance 1
        CommonOps_DDRM.multTransB(invAS, inverseSelectionStrength, varianceXX);
        DMatrixRMaj temp = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.multTransB(varianceXX, actualization, temp);
        CommonOps_DDRM.multAdd(-1.0, actualization, temp, varianceXX);
        // Delta
        DMatrixRMaj delta = new DMatrixRMaj(dimProcess, dimProcess);
        addTrans(invAS, delta);
        // Variance 2
        CommonOps_DDRM.addEquals(varianceXX, branchLength, delta);
        // Variance 3
        DMatrixRMaj temp2 = CommonOps_DDRM.identity(dimProcess);
        CommonOps_DDRM.addEquals(temp2, -1.0, actualization);
        DMatrixRMaj temp3 = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.mult(temp2, inverseSelectionStrength, temp3);
        CommonOps_DDRM.mult(temp3, delta, temp2);
        addTrans(temp2, temp);
        // All
        CommonOps_DDRM.addEquals(varianceXX, -1.0, temp);

        //// Variance XY
        DMatrixRMaj varianceXY = new DMatrixRMaj(dimProcess, dimProcess);
        // Variance 1
        CommonOps_DDRM.multTransB(stationaryVariance, temp3, varianceXY);
        // Variance 2
        CommonOps_DDRM.mult(temp3, stationaryVariance, temp);
        CommonOps_DDRM.multTransB(temp, actualization, temp2);
        // All
        CommonOps_DDRM.addEquals(varianceXY, -1.0, temp2);

        //// Variance YX
        DMatrixRMaj varianceYX = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.transpose(varianceXY, varianceYX);

        blockUnwrap(varianceYY, varianceXX, varianceXY, varianceYX, variances, destinationOffset);
        schurComplementInverse(varianceYY, varianceXX, varianceXY, varianceYX, precisions, destinationOffset);
    }

    private void computeIOUActualization(final int scaledOffset,
                                         DMatrixRMaj inverseSelectionStrength) {
        // YY
        DMatrixRMaj actualizationOU = wrap(actualizations, scaledOffset, dimProcess, dimProcess);

        // XX
        DMatrixRMaj temp = CommonOps_DDRM.identity(dimProcess);
        CommonOps_DDRM.addEquals(temp, -1.0, actualizationOU);
        DMatrixRMaj actualizationIOU = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.mult(inverseSelectionStrength, temp, actualizationIOU);

        // YX and XX
        DMatrixRMaj actualizationYX = new DMatrixRMaj(dimProcess, dimProcess); // zeros
        DMatrixRMaj actualizationXX = CommonOps_DDRM.identity(dimProcess);

        blockUnwrap(actualizationOU, actualizationXX, actualizationIOU, actualizationYX, actualizations, scaledOffset);
    }

    private void addTrans(DMatrixRMaj A, DMatrixRMaj B) {
        CommonOps_DDRM.transpose(A, B);
        CommonOps_DDRM.addEquals(B, A);
    }

    private void blockUnwrap(final DMatrixRMaj YY, final DMatrixRMaj XX, final DMatrixRMaj XY, final DMatrixRMaj YX, final double[] destination, final int offset) {
        for (int i = 0; i < dimProcess; i++) { // Rows
            for (int j = 0; j < dimProcess; j++) {
                destination[offset + i * dimTrait + j] = YY.get(i, j);
                destination[offset + (i + dimProcess) * dimTrait + j + dimProcess] = XX.get(i, j);
            }
            for (int j = 0; j < dimProcess; j++) {
                destination[offset + i * dimTrait + j + dimProcess] = YX.get(i, j);
                destination[offset + (i + dimProcess) * dimTrait + j] = XY.get(i, j);
            }
        }
    }

    private void schurComplementInverse(final DMatrixRMaj A, final DMatrixRMaj D,
                                        final DMatrixRMaj C, final DMatrixRMaj B,
                                        final double[] destination, final int offset) {
        DMatrixRMaj invA = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.invert(A, invA);
        DMatrixRMaj invMatD = getSchurInverseComplement(invA, D, C, B);

        DMatrixRMaj invAB = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.mult(invA, B, invAB);
        DMatrixRMaj invMatB = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.mult(-1.0, invAB, invMatD, invMatB);

        DMatrixRMaj CinvA = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.mult(C, invA, CinvA);
        DMatrixRMaj invMatC = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.mult(-1.0, invMatD, CinvA, invMatC);

        DMatrixRMaj invMatA = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.mult(-1.0, invMatB, CinvA, invMatA);
        CommonOps_DDRM.addEquals(invMatA, invA);

        blockUnwrap(invMatA, invMatD, invMatC, invMatB, destination, offset);
    }

    private DMatrixRMaj getSchurInverseComplement(final DMatrixRMaj invA, final DMatrixRMaj D,
                                        final DMatrixRMaj C, final DMatrixRMaj B) {
        DMatrixRMaj complement = new DMatrixRMaj(dimProcess, dimProcess);
        DMatrixRMaj tmp = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.mult(invA, B, tmp);
        CommonOps_DDRM.mult(-1.0, C, tmp, complement);
        CommonOps_DDRM.addEquals(complement, D);
        CommonOps_DDRM.invert(complement);
        return complement;
    }

    @Override
    public void updateIntegratedOrnsteinUhlenbeckDiffusionMatrices(int precisionIndex, final int[] probabilityIndices,
                                                                   final double[] edgeLengths, final double[] optimalRates,
                                                                   final double[] diagonalStrengthOfSelectionMatrix,
                                                                   final double[] rotation,
                                                                   int updateCount) {

        updateOrnsteinUhlenbeckDiffusionMatrices(precisionIndex, probabilityIndices, edgeLengths, optimalRates,
                diagonalStrengthOfSelectionMatrix, rotation, updateCount);

        if (DEBUG) {
            System.err.println("Matrices (safe with actualized drift, integrated):");
        }

        int matrixTraitSize = dimTrait * dimTrait;
        int matrixProcessSize = dimProcess * dimProcess;
        final int unscaledOffset = matrixProcessSize * precisionIndex;

        final DMatrixRMaj inverseSelectionStrength = getInverseSelectionStrength(diagonalStrengthOfSelectionMatrix, rotation);

        if (TIMING) {
            startTime("drift2");
        }

        int offset = 0;
        for (int up = 0; up < updateCount; ++up) {

            final int pio = dimTrait * probabilityIndices[up];
            final double edgeLength = edgeLengths[up];

            computeIOUActualizedDisplacement(optimalRates, offset, pio, edgeLength, inverseSelectionStrength);
            offset += dimProcess;
        }

        if (TIMING) {
            endTime("drift2");
        }

        if (TIMING) {
            startTime("diffusion2");
        }

        for (int up = 0; up < updateCount; ++up) {

            final int scaledOffset = matrixTraitSize * probabilityIndices[up];
            final double edgeLength = edgeLengths[up];

            computeIOUVarianceBranch(unscaledOffset, scaledOffset, edgeLength, inverseSelectionStrength);
        }

        if (TIMING) {
            endTime("diffusion2");
        }

        if (TIMING) {
            startTime("actualization2");
        }

        for (int up = 0; up < updateCount; ++up) {

            final int scaledOffset = matrixTraitSize * probabilityIndices[up];

            computeIOUActualization(scaledOffset, inverseSelectionStrength);
        }

        if (TIMING) {
            endTime("actualization");
        }

    }

    ///////////////////////////////////////////////////////////////////////////
    /// Tree-traversal functions
    ///////////////////////////////////////////////////////////////////////////

//    @Override
//    public void updatePreOrderPartial(
//            final int kBuffer, // parent
//            final int iBuffer, // node
//            final int iMatrix,
//            final int jBuffer, // sibling
//            final int jMatrix) {
//
//        throw new RuntimeException("Not yet implemented");
//    }

    @Override
    void actualizePrecision(DMatrixRMaj Pjp, DMatrixRMaj QjPjp, int jbo, int jmo, int jdo) {
        final DMatrixRMaj Qdj = wrap(actualizations, jmo, dimTrait, dimTrait);
        scalePrecision(Qdj, Pjp, QjPjp, Pjp);
    }

    @Override
    void actualizeVariance(DMatrixRMaj Vip, int ibo, int imo, int ido) {
        final DMatrixRMaj Qdi = wrap(actualizations, imo, dimTrait, dimTrait);
        final DMatrixRMaj QiVip = matrixQdiPip;
        scaleVariance(Qdi, Vip, QiVip, Vip);
    }

    @Override
    void scaleAndDriftMean(int ibo, int imo, int ido) {
        final DMatrixRMaj Qdi = wrap(actualizations, imo, dimTrait, dimTrait);
        final DMatrixRMaj ni = wrap(preOrderPartials, ibo, dimTrait, 1);
        final DMatrixRMaj niacc = matrixNiacc;
        CommonOps_DDRM.mult(Qdi, ni, niacc);
        unwrap(niacc, preOrderPartials, ibo);

        for (int g = 0; g < dimTrait; ++g) {
            preOrderPartials[ibo + g] += displacements[ido + g];
        }

    }

    @Override
    void computePartialPrecision(int ido, int jdo, int imo, int jmo,
                                 DMatrixRMaj Pip, DMatrixRMaj Pjp, DMatrixRMaj Pk) {

        final DMatrixRMaj Qdi = wrap(actualizations, imo, dimTrait, dimTrait);
        final DMatrixRMaj Qdj = wrap(actualizations, jmo, dimTrait, dimTrait);

        final DMatrixRMaj QdiPip = matrixQdiPip;
        final DMatrixRMaj QdiPipQdi = matrix0;
        scalePrecision(Qdi, Pip, QdiPip, QdiPipQdi);

        final DMatrixRMaj QdjPjpQdj = matrix1;
        final DMatrixRMaj QdjPjp = matrixQdjPjp;
        scalePrecision(Qdj, Pjp, QdjPjp, QdjPjpQdj);

        CommonOps_DDRM.add(QdiPipQdi, QdjPjpQdj, Pk);

//        forceSymmetric(Pk);

        if (DEBUG) {
            System.err.println("Qdi: " + Qdi);
            System.err.println("\tQdiPip: " + QdiPip);
            System.err.println("\tQdiPipQdi: " + QdiPipQdi);
            System.err.println("\tQdj: " + Qdj);
            System.err.println("\tQdjPjp: " + QdjPjp);
            System.err.println("\tQdjPjpQdj: " + QdjPjpQdj);
        }
    }

    private void scalePrecision(DMatrixRMaj Q, DMatrixRMaj P,
                                DMatrixRMaj QtP, DMatrixRMaj QtPQ) {
        CommonOps_DDRM.multTransA(Q, P, QtP);
//        symmetricMult(Q, P, QtPQ);
        CommonOps_DDRM.mult(QtP, Q, QtPQ);
        forceSymmetric(QtPQ);
    }

    private void scaleVariance(DMatrixRMaj Q, DMatrixRMaj P,
                               DMatrixRMaj QtP, DMatrixRMaj QtPQ) {
        CommonOps_DDRM.mult(Q, P, QtP);
        CommonOps_DDRM.multTransB(QtP, Q, QtPQ);
    }

    @Override
    void computeWeightedSum(final double[] ipartial,
                            final double[] jpartial,
                            final int dimTrait,
                            final double[] out) {
        weightedSum(ipartial, 0, matrixQdiPip, jpartial, 0, matrixQdjPjp, dimTrait, out);
    }

    private double[] actualizations;
    private DMatrixRMaj matrixQdiPip;
    private DMatrixRMaj matrixQdjPjp;
    private DMatrixRMaj matrixNiacc;
    private final boolean isActualizationSymmetric;
}
