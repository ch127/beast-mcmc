package dr.evomodel.treedatalikelihood.preorder;

import dr.evomodel.treedatalikelihood.continuous.cdi.PrecisionType;
import dr.math.matrixAlgebra.missingData.MissingOps;
import org.ejml.data.DMatrixRMaj;

import static dr.math.matrixAlgebra.missingData.MissingOps.safeInvert;

/**
 * @author Marc A. Suchard
 */
public class NormalSufficientStatistics {

    private final DMatrixRMaj mean;
    private final DMatrixRMaj precision;

    private DMatrixRMaj variance = null;

    NormalSufficientStatistics(double[] buffer,
                                      int index,
                                      int dim,
                                      DMatrixRMaj Pd,
                                      PrecisionType precisionType) {

        int partialOffset = (dim + precisionType.getMatrixLength(dim)) * index;
        this.mean = MissingOps.wrap(buffer, partialOffset, dim, 1);
        this.precision = DMatrixRMaj.wrap(dim, dim,
                precisionType.getScaledPrecision(buffer, partialOffset, Pd.data, dim));

    }

    @SuppressWarnings("unused")
    NormalSufficientStatistics(double[] mean,
                                      double[] precision,
                                      int index,
                                      int dim,
                                      DMatrixRMaj Pd,
                                      PrecisionType precisionType) {

        int meanOffset = dim * index;
        this.mean = MissingOps.wrap(mean, meanOffset, dim, 1);

        int precisionOffset = (dim * dim) * index;
//        this.precision = new DMatrixRMaj(dim, dim);
        this.precision = MissingOps.wrap(precision, precisionOffset, dim, dim);
//                DMatrixRMaj.wrap(dim, dim,
//                        precisionType.getScaledPrecision(precision, precisionOffset, Pd.data, dim));

    }

    @SuppressWarnings("unused")
    public NormalSufficientStatistics(DMatrixRMaj mean,
                                      DMatrixRMaj precision) {
        this.mean = mean;
        this.precision = precision;
    }

    public NormalSufficientStatistics(DMatrixRMaj mean, DMatrixRMaj precision, DMatrixRMaj variance) {
        this.mean = mean;
        this.precision = precision;
        this.variance = variance;
    }

    public double getMean(int row) {
        return mean.get(row);
    }

    public double getPrecision(int row, int col) {
        return precision.unsafe_get(row, col);
    }

    public double getVariance(int row, int col) {
        if (variance == null) {
            variance = new DMatrixRMaj(precision.numRows, precision.numCols);
            safeInvert(precision, variance, false);
        }

        return variance.unsafe_get(row, col);
    }

    @Deprecated
    public DMatrixRMaj getRawPrecision() { return precision; }

    @Deprecated
    public DMatrixRMaj getRawMean() { return mean; }

    @Deprecated
    public DMatrixRMaj getRawVariance() {
        if (variance == null) { // TODO Code duplication
            variance = new DMatrixRMaj(precision.numRows, precision.numCols);
            safeInvert(precision, variance, false);
        }

        return variance;
    }

    public String toString() {
        return mean + " " + precision;
    }

    String toVectorizedString() {
        StringBuilder sb = new StringBuilder();
        sb. append(toVectorizedString(mean.getData())).append(" ").append(toVectorizedString(precision.getData()));
        if (variance != null) {
            sb.append(" ").append(toVectorizedString(variance.getData()));
        }
        return sb.toString();
    }

    public static String toVectorizedString(DMatrixRMaj matrix) {
        return toVectorizedString(matrix.getData());
    }

    private static String toVectorizedString(double[] vector) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < vector.length - 1; ++i) {
            sb.append(vector[i]).append(" ");
        }
        sb.append(vector[vector.length - 1]);
        return sb.toString();
    }

}
