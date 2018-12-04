package dr.evomodel.treedatalikelihood.preorder;

import dr.math.matrixAlgebra.WrappedVector;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import static dr.math.matrixAlgebra.missingData.MissingOps.gatherRowsAndColumns;

/**
 * @author Marc A. Suchard
 */
public class ConditionalVarianceAndTransform2 {

    /**
     * For partially observed tips: (y_1, y_2)^t \sim N(\mu, \Sigma) where
     * <p>
     * \mu = (\mu_1, \mu_2)^t
     * \Sigma = ((\Sigma_{11}, \Sigma_{12}), (\Sigma_{21}, \Sigma_{22})^t
     * <p>
     * then  y_1 | y_2 \sim N (\bar{\mu}, \bar{\Sigma}), where
     * <p>
     * \bar{\mu} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(y_2 - \mu_2), and
     * \bar{\Sigma} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^1\Sigma{21}
     */

    final private DMatrixRMaj sBar;
    final private DMatrixRMaj affineTransform;

    private final int[] missingIndices;
    private final int[] notMissingIndices;
    private final double[] tempStorage;

    private final int numMissing;
    private final int numNotMissing;

    private static final boolean DEBUG = false;

    private double[][] cholesky = null;
    private DMatrixRMaj sBarInv = null;

    ConditionalVarianceAndTransform2(final DMatrixRMaj variance,
                                            final int[] missingIndices, final int[] notMissingIndices) {

        assert (missingIndices.length + notMissingIndices.length == variance.getNumRows());
        assert (missingIndices.length + notMissingIndices.length == variance.getNumCols());

        this.missingIndices = missingIndices;
        this.notMissingIndices = notMissingIndices;

        if (DEBUG) {
            System.err.println("variance:\n" + variance);
        }

        DMatrixRMaj S22 = new DMatrixRMaj(notMissingIndices.length, notMissingIndices.length);
        gatherRowsAndColumns(variance, S22, notMissingIndices, notMissingIndices);

        if (DEBUG) {
            System.err.println("S22:\n" + S22);
        }

        DMatrixRMaj S22Inv = new DMatrixRMaj(notMissingIndices.length, notMissingIndices.length);
        CommonOps_DDRM.invert(S22, S22Inv);

        if (DEBUG) {
            System.err.println("S22Inv:\n" + S22Inv);
        }

        DMatrixRMaj S12 = new DMatrixRMaj(missingIndices.length, notMissingIndices.length);
        gatherRowsAndColumns(variance, S12, missingIndices, notMissingIndices);

        if (DEBUG) {
            System.err.println("S12:\n" + S12);
        }

        DMatrixRMaj S12S22Inv = new DMatrixRMaj(missingIndices.length, notMissingIndices.length);
        CommonOps_DDRM.mult(S12, S22Inv, S12S22Inv);

        if (DEBUG) {
            System.err.println("S12S22Inv:\n" + S12S22Inv);
        }

        DMatrixRMaj S12S22InvS21 = new DMatrixRMaj(missingIndices.length, missingIndices.length);
        CommonOps_DDRM.multTransB(S12S22Inv, S12, S12S22InvS21);

        if (DEBUG) {
            System.err.println("S12S22InvS21:\n" + S12S22InvS21);
        }

        sBar = new DMatrixRMaj(missingIndices.length, missingIndices.length);
        gatherRowsAndColumns(variance, sBar, missingIndices, missingIndices);
        CommonOps_DDRM.subtract(sBar, S12S22InvS21, sBar);


        if (DEBUG) {
            System.err.println("sBar:\n" + sBar);
        }


        this.affineTransform = S12S22Inv;
        this.tempStorage = new double[missingIndices.length];

        this.numMissing = missingIndices.length;
        this.numNotMissing = notMissingIndices.length;

    }

    public WrappedVector getConditionalMean(final double[] y, final int offsetY,
                                     final double[] mu, final int offsetMu) {

        double[] muBar = new double[numMissing];

        double[] shift = new double[numNotMissing];
        for (int i = 0; i < numNotMissing; ++i) {
            final int notI = notMissingIndices[i];
            shift[i] = y[offsetY + notI] - mu[offsetMu + notI];
        }

        for (int i = 0; i < numMissing; ++i) {
            double delta = 0.0;
            for (int k = 0; k < numNotMissing; ++k) {
                delta += affineTransform.unsafe_get(i, k) * shift[k];
            }

            muBar[i] = mu[offsetMu + missingIndices[i]] + delta;
        }

        return new WrappedVector.Raw(muBar, 0, numMissing);
    }

//    void scatterResult(final double[] source, final int offsetSource,
//                       final double[] destination, final int offsetDestination) {
//        for (int i = 0; i < numMissing; ++i) {
//            destination[offsetDestination + missingIndices[i]] = source[offsetSource + i];
//        }
//    }

    final double[][] getConditionalCholesky() {
        if (cholesky == null) {
            this.cholesky = ProcessSimulationDelegate.AbstractContinuousTraitDelegate.getCholeskyOfVariance(sBar.data, missingIndices.length);
        }
        return cholesky;
    }

//    public final DMatrixRMaj getAffineTransform() {
//        return affineTransform;
//    }

    final DMatrixRMaj getConditionalVariance() {
        return sBar;
    }

    final DMatrixRMaj getConditionalPrecision() {
        if (sBarInv == null) {
            sBarInv = new DMatrixRMaj(numMissing, numMissing);
            CommonOps_DDRM.invert(sBar, sBarInv);
        }
        return sBarInv;
    }

    final double[] getTemporaryStorage() {
        return tempStorage;
    }
}
