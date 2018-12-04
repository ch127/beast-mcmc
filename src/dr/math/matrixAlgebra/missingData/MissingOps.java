package dr.math.matrixAlgebra.missingData;

import dr.inference.model.MatrixParameterInterface;
import dr.math.matrixAlgebra.*;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.SingularOps_DDRM;
import org.ejml.dense.row.decomposition.lu.LUDecompositionAlt_DDRM;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.dense.row.factory.LinearSolverFactory_DDRM;
import org.ejml.dense.row.linsol.lu.LinearSolverLu_DDRM;
import org.ejml.dense.row.misc.UnrolledDeterminantFromMinor_DDRM;
import org.ejml.dense.row.misc.UnrolledInverseFromMinor_DDRM;
import org.ejml.interfaces.decomposition.SingularValueDecomposition_F64;
import org.ejml.interfaces.linsol.LinearSolver;

import java.util.Arrays;

import static dr.math.matrixAlgebra.missingData.InversionResult.Code.*;
import static dr.util.EuclideanToInfiniteNormUnitBallTransform.projection;

/**
 * @author Marc A. Suchard
 */
public class MissingOps {

    public static DMatrixRMaj wrap(final double[] source, final int offset,
                                      final int numRows, final int numCols) {
        double[] buffer = new double[numRows * numCols];
        return wrap(source, offset, numRows, numCols, buffer);
    }

    public static DMatrixRMaj wrap(final double[] source, final int offset,
                                      final int numRows, final int numCols,
                                      final double[] buffer) {
        System.arraycopy(source, offset, buffer, 0, numRows * numCols);
        return DMatrixRMaj.wrap(numRows, numCols, buffer);
    }


    public static DMatrixRMaj wrapSymmetric(final double[] source, final int offset,
                                               final int numRows, final int numCols) {

        DMatrixRMaj S = wrap(source, offset, numRows, numCols);
        forceSymmetric(S);
        return S;
    }

    public static DMatrixRMaj wrap(MatrixParameterInterface A) {
        return wrap(A.getParameterValues(), 0, A.getRowDimension(), A.getColumnDimension());
    }

    public static DMatrixRMaj wrapDiagonal(final double[] source, final int offset,
                                              final int dim) {
        double[] buffer = new double[dim * dim];
        return wrapDiagonal(source, offset, dim, buffer);
    }

    public static DMatrixRMaj wrapDiagonal(final double[] source, final int offset,
                                              final int dim,
                                              final double[] buffer) {
        for (int i = 0; i < dim; ++i) {
            buffer[i * dim + i] = source[i];
        }
        return DMatrixRMaj.wrap(dim, dim, buffer);
    }

    public static DMatrixRMaj wrapDiagonalInverse(final double[] source, final int offset,
                                                     final int dim) {
        double[] buffer = new double[dim * dim];
        return wrapDiagonalInverse(source, offset, dim, buffer);
    }

    public static DMatrixRMaj wrapDiagonalInverse(final double[] source, final int offset,
                                                     final int dim,
                                                     final double[] buffer) {
        for (int i = 0; i < dim; ++i) {
            buffer[i * dim + i] = 1 / source[i];
        }
        return DMatrixRMaj.wrap(dim, dim, buffer);
    }

    public static DMatrixRMaj wrapSpherical(final double[] source, final int offset,
                                               final int dim) {
        double[] buffer = new double[dim * dim];
        return wrapSpherical(source, offset, dim, buffer);
    }

    public static DMatrixRMaj wrapSpherical(final double[] source, final int offset,
                                               final int dim,
                                               final double[] buffer) {
        fillSpherical(source, offset, dim, buffer);
        DMatrixRMaj res = DMatrixRMaj.wrap(dim, dim, buffer);
        CommonOps_DDRM.transpose(res); // Column major.
        return res;
    }

    private static void fillSpherical(final double[] source, final int offset,
                               final int dim, final double[] buffer) {
        for (int i = 0; i < dim; i++) {
            System.arraycopy(source, offset + i * (dim - 1),
                    buffer, i * dim, dim - 1);
            buffer[(i + 1) * dim - 1] = projection(source, offset + i * (dim - 1), dim - 1);
        }
    }

    public static DMatrixRMaj copy(ReadableMatrix source) {
        final int len = source.getDim();
        double[] buffer = new double[len];
        for (int i = 0; i < len; ++i) {
            buffer[i] = source.get(i);
        }
        return DMatrixRMaj.wrap(source.getMinorDim(), source.getMajorDim(), buffer);
    }

    public static void copy(DMatrixRMaj source, WritableMatrix destination) {
        final int len = destination.getDim();
        for (int i = 0; i < len; ++i) {
            destination.set(i, source.get(i));
        }
    }

    public static void gatherRowsAndColumns(final DMatrixRMaj source, final DMatrixRMaj destination,
                                                      final int[] rowIndices, final int[] colIndices) {

        final int rowLength = rowIndices.length;
        final int colLength = colIndices.length;
        final double[] out = destination.getData();

        int index = 0;
        for (int i = 0; i < rowLength; ++i) {
            final int rowIndex = rowIndices[i];
            for (int j = 0; j < colLength; ++j) {
                out[index] = source.unsafe_get(rowIndex, colIndices[j]);
                ++index;
            }
        }
    }

    public static void scatterRowsAndColumns(final DMatrixRMaj source, final DMatrixRMaj destination,
                                             final int[] rowIdices, final int[] colIndices, final boolean clear) {
        if (clear) {
            Arrays.fill(destination.getData(), 0.0);
        }

        final int rowLength = rowIdices.length;
        final int colLength = colIndices.length;
        final double[] in = source.getData();

        int index = 0;
        for (int i = 0; i < rowLength; ++i) {
            final int rowIndex = rowIdices[i];
            for (int j = 0; j < colLength; ++j) {
                destination.unsafe_set(rowIndex, colIndices[j], in[index]);
                ++index;
            }
        }
    }

    public static void unwrap(final DMatrixRMaj source, final double[] destination, final int offset) {
        System.arraycopy(source.getData(), 0, destination, offset, source.getNumElements());
    }

    public static void unwrapIdentity(final double[] destination, final int offset, final int dim) {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < i; j++) {
                destination[offset + i * dim + j] = 0.0;
            }
            destination[offset + i * dim + i] = 1.0;
            for (int j = i + 1; j < dim; j++) {
                destination[offset + i * dim + j] = 0.0;
            }
        }
    }

    public static void blockUnwrap(final DMatrixRMaj block, final double[] destination,
                                   final int destinationOffset,
                                   final int offsetRow, final int offsetCol,
                                   final int nCols) {
        for (int i = 0; i < block.getNumRows(); i++) { // Rows
            for (int j = 0; j < block.getNumCols(); j++) {
                destination[destinationOffset + (i + offsetRow) * nCols + j + offsetCol] = block.get(i, j);
            }
        }
    }

    public static boolean anyDiagonalInfinities(DMatrixRMaj source) {
        boolean anyInfinities = false;
        for (int i = 0; i < source.getNumCols() && !anyInfinities; ++i) {
            if (Double.isInfinite(source.unsafe_get(i, i))) {
                anyInfinities = true;
            }
        }
        return anyInfinities;
    }

    public static boolean allFiniteDiagonals(DMatrixRMaj source) {
        boolean allFinite = true;

        final int length = source.getNumCols();
        for (int i = 0; i < length; ++i) {
            allFinite &= !Double.isInfinite(source.unsafe_get(i, i));
        }
        return allFinite;
    }

    public static int countFiniteDiagonals(DMatrixRMaj source) {
        final int length = source.getNumCols();

        int count = 0;
        for (int i = 0; i < length; ++i) {
            final double d = source.unsafe_get(i, i);
            if (!Double.isInfinite(d)) {
                ++count;
            }
        }
        return count;
    }

    public static int countZeroDiagonals(DMatrixRMaj source) {
        final int length = source.getNumCols();

        int count = 0;
        for (int i = 0; i < length; ++i) {
            final double d = source.unsafe_get(i, i);
            if (d == 0.0) {
                ++count;
            }
        }
        return count;
    }

    public static boolean allZeroDiagonals(DMatrixRMaj source) {
        final int length = source.getNumCols();

        for (int i = 0; i < length; ++i) {
            if (source.unsafe_get(i,i) != 0.0) {
                return false;
            }
        }
        return true;
    }

    public static void getFiniteDiagonalIndices(final DMatrixRMaj source, final int[] indices) {
        final int length = source.getNumCols();

        int index = 0;
        for (int i = 0; i < length; ++i) {
            final double d = source.unsafe_get(i, i);
            if (!Double.isInfinite(d)) {
                indices[index] = i;
                ++index;
            }
        }
    }

    public static int countFiniteNonZeroDiagonals(ReadableMatrix source) {
        final int length = source.getMajorDim();

        int count = 0;
        for (int i = 0; i < length; ++i) {
            final double d = source.get(i, i);
            if (!Double.isInfinite(d) && d != 0.0) {
                ++count;
            }
        }
        return count;
    }

    public static int countFiniteNonZeroDiagonals(DMatrixRMaj source) {
        final int length = source.getNumCols();

        int count = 0;
        for (int i = 0; i < length; ++i) {
            final double d = source.unsafe_get(i, i);
            if (!Double.isInfinite(d) && d != 0.0) {
                ++count;
            }
        }
        return count;
    }

    public static void getFiniteNonZeroDiagonalIndices(final DMatrixRMaj source, final int[] indices) {
        final int length = source.getNumCols();

        int index = 0;
        for (int i = 0; i < length; ++i) {
            final double d = source.unsafe_get(i, i);
            if (!Double.isInfinite(d) && d != 0.0) {
                indices[index] = i;
                ++index;
            }
        }
    }

    public static void addToDiagonal(DMatrixRMaj source, double increment) {
        final int width = source.getNumRows();
        for (int i = 0; i < width; ++i) {
            source.unsafe_set(i,i, source.unsafe_get(i, i) + increment);
        }
    }

    public static double det(DMatrixRMaj mat) {
        int numCol = mat.getNumCols();
        int numRow = mat.getNumRows();
        if(numCol != numRow) {
            throw new IllegalArgumentException("Must be a square matrix.");
        } else if(numCol <= 6) {
            return numCol >= 2? UnrolledDeterminantFromMinor_DDRM.det(mat):mat.get(0);
        } else {
            LUDecompositionAlt_DDRM alg = new LUDecompositionAlt_DDRM();
            if(alg.inputModified()) {
                mat = mat.copy();
            }

            return !alg.decompose(mat)?0.0D:alg.computeDeterminant().real;
        }
    }

    public static double invertAndGetDeterminant(DMatrixRMaj mat, DMatrixRMaj result, boolean log) {

        final int numCol = mat.getNumCols();
        final int numRow = mat.getNumRows();
        if (numCol != numRow) {
            throw new IllegalArgumentException("Must be a square matrix.");
        }

        if (numCol <= 5) {

            if (numCol >= 2) {
                UnrolledInverseFromMinor_DDRM.inv(mat, result);
            } else {
                result.set(0, 1.0D / mat.get(0));
            }

            double det = numCol >= 2 ?
                    UnrolledDeterminantFromMinor_DDRM.det(mat) :
                    mat.get(0);
            return log ? Math.log(det) : det;

        } else {

            LUDecompositionAlt_DDRM alg = new LUDecompositionAlt_DDRM();
            LinearSolverLu_DDRM solver = new LinearSolverLu_DDRM(alg);
            if (solver.modifiesA()) {
                mat = mat.copy();
            }

            if (!solver.setA(mat)) {
                return Double.NaN;
            }

            solver.invert(result);

            return log ? computeLogDeterminant(alg) : alg.computeDeterminant().real;

        }
    }

    private static double computeLogDeterminant(LUDecompositionAlt_DDRM alg) {
        int n = alg.getLU().getNumCols();
        if (n != alg.getLU().getNumRows()) {
            throw new IllegalArgumentException("Must be a square matrix.");
        } else {
            double logDet = 0;
            double[] dataLU = alg.getLU().getData();
            for(int i = 0; i < n * n; i += n + 1) {
                logDet += Math.log(Math.abs(dataLU[i]));
            }

            return logDet;
        }
    }

    public static InversionResult safeDeterminant(DMatrixRMaj source, boolean invert) {
        final int finiteCount = countFiniteNonZeroDiagonals(source);

        InversionResult result;

        if (finiteCount == 0) {
            result = new InversionResult(NOT_OBSERVED, 0, 0);
        } else {
//            LinearSolver<DMatrixRMaj> solver = LinearSolverFactory_DDRM.pseudoInverse(true);
//            solver.setA(source);
//
//            SingularValueDecomposition_F64<DMatrixRMaj> svd = solver.getDecomposition();
//            double[] values = svd.getSingularValues();
//
//            if (values == null) {
//                throw new RuntimeException("Unable to perform SVD");
//            }

            SingularValueDecomposition_F64<DMatrixRMaj> svd = DecompositionFactory_DDRM.svd(source.getNumRows(), source.getNumCols(), false, false, false);
            if (!svd.decompose(source)) {
                if (SingularOps_DDRM.rank(svd) == 0) return new InversionResult(NOT_OBSERVED, 0, 0);
                throw new RuntimeException("SVD decomposition failed");
            }
            double[] values = svd.getSingularValues();

            int dim = 0;
            double logDet = 0;
            for (int i = 0; i < values.length; i++) {
                final double lambda = values[i];
                if (lambda > 0.0) {
                    logDet += Math.log(lambda);
                    ++dim;
                }
            }

            if (!invert) {
                logDet = -logDet;
            }

            result = new InversionResult(dim == source.getNumCols() ? FULLY_OBSERVED : PARTIALLY_OBSERVED, dim, logDet, true);
        }

        return result;
    }

    public static InversionResult safeSolve(DMatrixRMaj A,
                                            WrappedVector b,
                                            WrappedVector x,
                                            boolean getDeterminat) {
        final int dim = b.getDim();

        assert(A.getNumRows() == dim && A.getNumCols() == dim);

        final DMatrixRMaj B = wrap(b.getBuffer(), b.getOffset(), dim, 1);
        final DMatrixRMaj X = new DMatrixRMaj(dim, 1);

        InversionResult ir = safeSolve(A, B, X, getDeterminat);


        for (int row = 0; row < dim; ++row) {
            x.set(row, X.unsafe_get(row, 0));
        }

        return ir;
    }

    public static InversionResult safeSolve(DMatrixRMaj A, DMatrixRMaj B, DMatrixRMaj X, boolean getDeterminant) {

        final int finiteCount = countFiniteNonZeroDiagonals(A);

        InversionResult result;
        if (finiteCount == 0) {
            Arrays.fill(X.getData(), 0);
            result = new InversionResult(NOT_OBSERVED, 0, 0);
        } else {

            LinearSolver<DMatrixRMaj,DMatrixRMaj> solver = LinearSolverFactory_DDRM.pseudoInverse(true);
            solver.setA(A);
            solver.solve(B, X);

            int dim = 0;
            double logDet = 0;

            if (getDeterminant) {
//                SingularValueDecomposition_F64<DMatrixRMaj> svd = solver.getDecomposition();
//                double[] values = svd.getSingularValues();

                SingularValueDecomposition_F64<DMatrixRMaj> svd = DecompositionFactory_DDRM.svd(A.getNumRows(), A.getNumCols(), false, false, false);
                if (!svd.decompose(A)) {
                    if (SingularOps_DDRM.rank(svd) == 0) return new InversionResult(NOT_OBSERVED, 0, 0);
                    throw new RuntimeException("SVD decomposition failed");
                }
                double[] values = svd.getSingularValues();

//                double eps = SingularOps_DDRM.singularThreshold(svd);

                for (int i = 0; i < values.length; ++i) {
                    final double lambda = values[i];
                    if (lambda > 0.0) {
                        logDet += Math.log(lambda);
                        ++dim;
                    }
                }
            }

            result = new InversionResult(dim == A.getNumCols() ? FULLY_OBSERVED : PARTIALLY_OBSERVED, dim, logDet, true);
        }

        return result;
    }

    public static void safeSolveSymmPosDef(DMatrixRMaj A,
                                           WrappedVector b,
                                           WrappedVector x) {
        final int dim = b.getDim();

        assert (A.getNumRows() == dim && A.getNumCols() == dim);

        final DMatrixRMaj B = wrap(b.getBuffer(), b.getOffset(), dim, 1);
        final DMatrixRMaj X = new DMatrixRMaj(dim, 1);

        safeSolveSymmPosDef(A, B, X);


        for (int row = 0; row < dim; ++row) {
            x.set(row, X.unsafe_get(row, 0));
        }
    }

    public static void safeSolveSymmPosDef(DMatrixRMaj A, DMatrixRMaj B, DMatrixRMaj X) {

        final int finiteCount = countFiniteNonZeroDiagonals(A);

        InversionResult result;
        if (finiteCount == 0) {
            Arrays.fill(X.getData(), 0);
        } else {
            LinearSolver<DMatrixRMaj,DMatrixRMaj> solver = LinearSolverFactory_DDRM.symmPosDef(A.getNumCols());
            DMatrixRMaj Abis = new DMatrixRMaj(A);
            if(solver.setA(Abis)) {
                solver.solve(B, X);
            } else {
                LinearSolver<DMatrixRMaj,DMatrixRMaj> solverSVD = LinearSolverFactory_DDRM.pseudoInverse(true);
                solverSVD.setA(A);
                solverSVD.solve(B, X);
            }
        }
    }

    public static InversionResult safeInvert(ReadableMatrix source, WritableMatrix destination, boolean getDeterminant) {

        final int dim = source.getMajorDim();
        final int finiteCount = countFiniteNonZeroDiagonals(source);
        double logDet = 0;

        if (finiteCount == dim) {

            DMatrixRMaj result = new DMatrixRMaj(dim, dim);
            DMatrixRMaj copyOfSource = copy(source);
            if (getDeterminant) {
                logDet = invertAndGetDeterminant(copyOfSource, result, true);
            } else {
                CommonOps_DDRM.invertSPD(copyOfSource, result);
            }

            copy(result, destination);

            return new InversionResult(FULLY_OBSERVED, dim, logDet, true);
        }

        return null;
    }

    public static InversionResult safeInvert(DMatrixRMaj source, DMatrixRMaj destination, boolean getDeterminant) {

        final int dim = source.getNumCols();
        final int finiteCount = countFiniteNonZeroDiagonals(source);
        double logDet = 0;

        if (finiteCount == dim) {
            if (getDeterminant) {
                logDet = invertAndGetDeterminant(source, destination, true);
            } else {
                CommonOps_DDRM.invertSPD(source, destination);
            }
            return new InversionResult(FULLY_OBSERVED, dim, logDet, true);
        } else {
            if (finiteCount == 0) {
                Arrays.fill(destination.getData(), 0);
                return new InversionResult(NOT_OBSERVED, 0, 0);
            } else {
                final int[] finiteIndices = new int[finiteCount];
                getFiniteNonZeroDiagonalIndices(source, finiteIndices);

                final DMatrixRMaj subSource = new DMatrixRMaj(finiteCount, finiteCount);
                gatherRowsAndColumns(source, subSource, finiteIndices, finiteIndices);

                final DMatrixRMaj inverseSubSource = new DMatrixRMaj(finiteCount, finiteCount);
                if (getDeterminant) {
                    logDet = invertAndGetDeterminant(subSource, inverseSubSource, true);
                } else {
                    CommonOps_DDRM.invertSPD(subSource, inverseSubSource);
                }

                scatterRowsAndColumns(inverseSubSource, destination, finiteIndices, finiteIndices, true);

                return new InversionResult(PARTIALLY_OBSERVED, finiteCount, logDet, true);
            }
        }
    }

    public static void matrixVectorMultiple(final DMatrixRMaj A,
                                       final WrappedVector x,
                                       final WrappedVector y,
                                       final int dim) {
        if (buffer.length < dim) {
            buffer = new double[dim];
        }

        for (int row = 0; row < dim; ++row) {
            double sum = 0.0;
            for (int col = 0; col < dim; ++col) {
                sum += A.unsafe_get(row, col) * x.get(col);
            }
            buffer[row] = sum;
        }

        for (int col = 0; col < dim; ++col) {
            y.set(col, buffer[col]);
        }
    }

    private static double[] buffer = new double[16];

    public static void safeWeightedAverage(final WrappedVector mi,
                                           final DMatrixRMaj Pi,
                                           final WrappedVector mj,
                                           final DMatrixRMaj Pj,
                                           final WrappedVector mk,
                                           final DMatrixRMaj Vk,
                                           final int dimTrait) {
//        countZeroDiagonals(Vk);
        final double[] tmp = new double[dimTrait];
        for (int g = 0; g < dimTrait; ++g) {
            double sum = 0.0;
            boolean iInf = Double.isInfinite(Pi.unsafe_get(g,g));
            boolean jInf = Double.isInfinite(Pj.unsafe_get(g,g));
            if (iInf && jInf) {
                throw new IllegalArgumentException("Both precision matrices are infinite in dimension " + g);
            } else if (iInf) {
                sum = mi.get(g);
            } else if (jInf) {
                sum = mj.get(g);
            } else {
                for (int h = 0; h < dimTrait; ++h) {
                    sum += Pi.unsafe_get(g, h) * mi.get(h);
                    sum += Pj.unsafe_get(g, h) * mj.get(h);
                }
            }

            tmp[g] = sum;
        }

        for (int g = 0; g < dimTrait; ++g) {
            double sum = 0.0;
            if (Vk.unsafe_get(g, g) == 0.0) {
                sum = tmp[g];
            } else {
                for (int h = 0; h < dimTrait; ++h) {
                    sum += Vk.unsafe_get(g, h) * tmp[h];
                }
            }
            mk.set(g, sum);
        }
    }

    public static void weightedAverage(final ReadableVector mi,
                                       final DMatrixRMaj Pi,
                                       final ReadableVector mj,
                                       final DMatrixRMaj Pj,
                                       final WritableVector mk,
                                       final DMatrixRMaj Vk,
                                       final int dimTrait) {
        final double[] tmp = new double[dimTrait];
        for (int g = 0; g < dimTrait; ++g) {
            double sum = 0.0;
            for (int h = 0; h < dimTrait; ++h) {
                sum += Pi.unsafe_get(g, h) * mi.get(h);
                sum += Pj.unsafe_get(g, h) * mj.get(h);
            }
            tmp[g] = sum;
        }
        for (int g = 0; g < dimTrait; ++g) {
            double sum = 0.0;
            for (int h = 0; h < dimTrait; ++h) {
                sum += Vk.unsafe_get(g, h) * tmp[h];
            }
            mk.set(g, sum);
        }
    }

    public static void weightedAverage(final ReadableVector mi,
                                       final ReadableMatrix Pi,
                                       final ReadableVector mj,
                                       final ReadableMatrix Pj,
                                       final WritableVector mk,
                                       final ReadableMatrix Vk,
                                       final int dimTrait) {
        final double[] tmp = new double[dimTrait];
        for (int g = 0; g < dimTrait; ++g) {
            double sum = 0.0;
            for (int h = 0; h < dimTrait; ++h) {
                sum += Pi.get(g, h) * mi.get(h);
                sum += Pj.get(g, h) * mj.get(h);
            }
            tmp[g] = sum;
        }
        for (int g = 0; g < dimTrait; ++g) {
            double sum = 0.0;
            for (int h = 0; h < dimTrait; ++h) {
                sum += Vk.get(g, h) * tmp[h];
            }
            mk.set(g, sum);
        }
    }

    public static void weightedAverage(final double[] ipartial,
                                       final int ibo,
                                       final DMatrixRMaj Pi,
                                       final double[] jpartial,
                                       final int jbo,
                                       final DMatrixRMaj Pj,
                                       final double[] kpartial,
                                       final int kbo,
                                       final DMatrixRMaj Vk,
                                       final int dimTrait) {
        final double[] tmp = new double[dimTrait];
        weightedAverage(ipartial, ibo, Pi, jpartial, jbo, Pj, kpartial, kbo, Vk, dimTrait, tmp);
    }

    public static void weightedSum(final double[] ipartial,
                                       final int ibo,
                                       final DMatrixRMaj Pi,
                                       final double[] jpartial,
                                       final int jbo,
                                       final DMatrixRMaj Pj,
                                       final int dimTrait,
                                       final double[] out) {
        for (int g = 0; g < dimTrait; ++g) {
            double sum = 0.0;
            for (int h = 0; h < dimTrait; ++h) {
                sum += Pi.unsafe_get(g, h) * ipartial[ibo + h];
                sum += Pj.unsafe_get(g, h) * jpartial[jbo + h];
            }
            out[g] = sum;
        }
    }

    public static void weightedSumActualized(final double[] ipartial,
                                             final int ibo,
                                             final DMatrixRMaj Pi,
                                             final double[] iactualization,
                                             final int ido,
                                             final double[] jpartial,
                                             final int jbo,
                                             final DMatrixRMaj Pj,
                                             final double[] jactualization,
                                             final int jdo,
                                             final int dimTrait,
                                             final double[] out) {
        for (int g = 0; g < dimTrait; ++g) {
            double sum = 0.0;
            for (int h = 0; h < dimTrait; ++h) {
                sum += iactualization[ido + g] * Pi.unsafe_get(g, h) * ipartial[ibo + h];
                sum += jactualization[jdo + g] * Pj.unsafe_get(g, h) * jpartial[jbo + h];
            }
            out[g] = sum;
        }
    }

    public static void weightedAverage(final double[] ipartial,
                                       final int ibo,
                                       final DMatrixRMaj Pi,
                                       final double[] jpartial,
                                       final int jbo,
                                       final DMatrixRMaj Pj,
                                       final double[] kpartial,
                                       final int kbo,
                                       final DMatrixRMaj Vk,
                                       final int dimTrait,
                                       final double[] tmp) {
        weightedSum(ipartial, ibo, Pi, jpartial, jbo, Pj, dimTrait, tmp);
        for (int g = 0; g < dimTrait; ++g) {
            double sum = 0.0;
            for (int h = 0; h < dimTrait; ++h) {
                sum += Vk.unsafe_get(g, h) * tmp[h];
            }
            kpartial[kbo + g] = sum;
        }
    }

    public static double weightedInnerProduct(final double[] partials,
                                              final int bo,
                                              final DMatrixRMaj P,
                                              final int dimTrait) {
        double SS = 0;

        // vector-matrix-vector
        for (int g = 0; g < dimTrait; ++g) {
            final double ig = partials[bo + g];
            for (int h = 0; h < dimTrait; ++h) {
                final double ih = partials[bo + h];
                SS += ig * P.unsafe_get(g, h) * ih;
            }
        }

        return SS;
    }

    public static double weightedInnerProductOfDifferences(final double[] source1,
                                                           final int source1Offset,
                                                           final double[] source2,
                                                           final int source2Offset,
                                                           final DMatrixRMaj P,
                                                           final int dimTrait) {
        double SS = 0;
        for (int g = 0; g < dimTrait; ++g) {
            final double gDifference = source1[source1Offset + g] - source2[source2Offset + g];

            for (int h = 0; h < dimTrait; ++h) {
                final double hDifference = source1[source1Offset + h] - source2[source2Offset + h];

                SS += gDifference * P.unsafe_get(g, h) * hDifference;
            }
        }

        return SS;
    }



    public static double weightedThreeInnerProduct(final double[] ipartials,
                                                   final int ibo,
                                                   final DMatrixRMaj Pip,
                                                   final double[] jpartials,
                                                   final int jbo,
                                                   final DMatrixRMaj Pjp,
                                                   final double[] kpartials,
                                                   final int kbo,
                                                   final DMatrixRMaj Pk,
                                                   final int dimTrait) {

        // TODO Is it better to split into 3 separate calls to weightedInnerProduct?

        double SSi = 0;
        double SSj = 0;
        double SSk = 0;

        // vector-matrix-vector TODO in parallel
        for (int g = 0; g < dimTrait; ++g) {
            final double ig = ipartials[ibo + g];
            final double jg = jpartials[jbo + g];
            final double kg = kpartials[kbo + g];

            for (int h = 0; h < dimTrait; ++h) {
                final double ih = ipartials[ibo + h];
                final double jh = jpartials[jbo + h];
                final double kh = kpartials[kbo + h];

                SSi += ig * Pip.unsafe_get(g, h) * ih;
                SSj += jg * Pjp.unsafe_get(g, h) * jh;
                SSk += kg * Pk .unsafe_get(g, h) * kh;
            }
        }

        return SSi + SSj - SSk;
    }


    public static void add(ReadableMatrix p1,
                           ReadableMatrix p2,
                           WritableMatrix p12) {

        assert (p1.getDim() == p2.getDim());
        assert (p1.getDim() == p12.getDim());

        final int dim = p12.getDim();

        for (int i = 0; i < dim; ++i) {
            p12.set(i, p1.get(i) + p2.get(i));
        }
    }

    public static void forceSymmetric(DMatrixRMaj P) {
        DMatrixRMaj Ptrans = new DMatrixRMaj(P);
        CommonOps_DDRM.transpose(P, Ptrans);
        CommonOps_DDRM.addEquals(P, Ptrans);
        CommonOps_DDRM.scale(0.5, P);
    }

    public static void symmetricMult(DMatrixRMaj Q, DMatrixRMaj P, DMatrixRMaj QtPQ) {
        int dimTrait = Q.getNumCols();
        assert dimTrait == Q.getNumRows() && dimTrait == P.getNumCols() && dimTrait == P.getNumRows();
        for (int i = 0; i < dimTrait; i++) {
            for (int j = i; j < dimTrait; j++) {
                double val = 0;
                for (int k = 0; k < dimTrait; k++) {
                    for (int r = 0; r < dimTrait; r++) {
                        val += P.unsafe_get(k, r) * Q.unsafe_get(k, i) * Q.unsafe_get(r, j);
                    }
                }
                QtPQ.unsafe_set(i, j, val);
                QtPQ.unsafe_set(j, i, val);
            }
        }
    }
}
