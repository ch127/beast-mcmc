package dr.evomodel.treedatalikelihood.preorder;

import dr.evolution.tree.Tree;
import dr.evomodel.continuous.MultivariateDiffusionModel;
import dr.evomodel.treedatalikelihood.continuous.*;
import dr.math.distributions.MultivariateNormalDistribution;
import dr.math.matrixAlgebra.ReadableVector;
import dr.math.matrixAlgebra.WrappedMatrix;
import dr.math.matrixAlgebra.WrappedVector;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import static dr.math.matrixAlgebra.missingData.MissingOps.*;

/**
 * @author Marc A. Suchard
 * @author Paul Bastide
 */
public class MultivariateConditionalOnTipsRealizedDelegate extends ConditionalOnTipsRealizedDelegate {

    private static final boolean DEBUG = false;

    final private PartiallyMissingInformation missingInformation;

    public MultivariateConditionalOnTipsRealizedDelegate(String name, Tree tree,
                                                         MultivariateDiffusionModel diffusionModel,
                                                         ContinuousTraitPartialsProvider dataModel,
                                                         ConjugateRootTraitPrior rootPrior,
                                                         ContinuousRateTransformation rateTransformation,
                                                         ContinuousDataLikelihoodDelegate likelihoodDelegate) {
        super(name, tree, diffusionModel, dataModel, rootPrior, rateTransformation, likelihoodDelegate);
        missingInformation = new PartiallyMissingInformation(tree, dataModel);
    }

    @Override
    protected void simulateTraitForRoot(final int offsetSample, final int offsetPartial) {

        // TODO Bad programming -- should not need to know about internal layout
        // Layout, offset, dim
        // trait, 0, dT
        // precision, dT, dT * dT
        // variance, dT + dT * dT, dT * dT
        // scalar, dT + 2 * dT * dT, 1

        // Integrate out against prior
        final DMatrixRMaj rootPrec = wrap(partialNodeBuffer, offsetPartial + dimTrait, dimTrait, dimTrait);
        final DMatrixRMaj priorPrec = new DMatrixRMaj(dimTrait, dimTrait);
        CommonOps_DDRM.mult(Pd, wrap(partialPriorBuffer, offsetPartial + dimTrait, dimTrait, dimTrait), priorPrec);

        final DMatrixRMaj totalPrec = new DMatrixRMaj(dimTrait, dimTrait);
        CommonOps_DDRM.add(rootPrec, priorPrec, totalPrec);

        final DMatrixRMaj totalVar = new DMatrixRMaj(dimTrait, dimTrait);
        safeInvert(totalPrec, totalVar, false);

        final double[] tmp = new double[dimTrait];
        final double[] mean = new double[dimTrait];

        for (int g = 0; g < dimTrait; ++g) {
            double sum = 0.0;
            for (int h = 0; h < dimTrait; ++h) {
                sum += rootPrec.unsafe_get(g, h) * partialNodeBuffer[offsetPartial + h];
                sum += priorPrec.unsafe_get(g, h) * partialPriorBuffer[offsetPartial + h];
            }
            tmp[g] = sum;
        }
        for (int g = 0; g < dimTrait; ++g) {
            double sum = 0.0;
            for (int h = 0; h < dimTrait; ++h) {
                sum += totalVar.unsafe_get(g, h) * tmp[h];
            }
            mean[g] = sum;
        }

        final double[][] cholesky = getCholeskyOfVariance(totalVar.getData(), dimTrait);

        MultivariateNormalDistribution.nextMultivariateNormalCholesky(
                mean, 0, // input mean
                cholesky, 1.0, // input variance
                sample, offsetSample, // output sample
                tmpEpsilon);

        if (DEBUG) {
            System.err.println("Attempt to simulate root");
            System.err.println("Root  mean: " + new WrappedVector.Raw(partialNodeBuffer, offsetPartial, dimTrait));
            System.err.println("Root  prec: " + rootPrec);
            System.err.println("Prior mean: " + new WrappedVector.Raw(partialPriorBuffer, offsetPartial, dimTrait));
            System.err.println("Prior prec: " + priorPrec);
            System.err.println("Total prec: " + totalPrec);
            System.err.println("Total  var: " + totalVar);


            System.err.println("draw mean: " + new WrappedVector.Raw(mean, 0, dimTrait));
            System.err.println("sample: " + new WrappedVector.Raw(sample, offsetSample, dimTrait));
        }
    }

//    boolean extremeValue(final DMatrixRMaj x) {
//        return extremeValue(new WrappedVector.Raw(x.getData(), 0, x.getNumElements()));
//    }
//
//    boolean extremeValue(final WrappedVector x) {
//        boolean valid = true;
//        for (int i = 0; i < x.getDim() && (valid); ++i) {
//            if (Double.isNaN(x.get(i)) || Math.abs(x.get(i)) > 1E2) {
//                valid = false;
//            }
//        }
//        return !valid;
//    }

    @Override
    protected void simulateTraitForNode(final int nodeIndex,
                                        final int traitIndex,
                                        final int offsetSample,
                                        final int offsetParent,
                                        final int offsetPartial,
                                        final int isExternal,
                                        final double branchPrecision) {

        if (isExternal == 1) { // Is external
            simulateTraitForExternalNode(nodeIndex, traitIndex, offsetSample, offsetParent, offsetPartial, branchPrecision);
        } else {
            simulateTraitForInternalNode(offsetSample, offsetParent, offsetPartial, branchPrecision);
        }
    }

//    // TODO Delegate to approach process
//    private WrappedVector getBranchSpecificMeanForChild(final WrappedVector parent) {
//        return parent;
//    }
//

    private final static boolean NEW_TIP_WITH_NO_DATA = true; //TODO: What does this do ?

    private void simulateTraitForExternalNode(final int nodeIndex,
                                              final int traitIndex,
                                              final int offsetSample,
                                              final int offsetParent,
                                              final int offsetPartial,
                                              final double branchPrecision) {

        final DMatrixRMaj P0 = wrap(partialNodeBuffer, offsetPartial + dimTrait, dimTrait, dimTrait);
        final int missingCount = countFiniteDiagonals(P0);

        if (missingCount == 0) { // Completely observed

            System.arraycopy(partialNodeBuffer, offsetPartial, sample, offsetSample, dimTrait);

        } else {

            final int zeroCount = countZeroDiagonals(P0);
            if (zeroCount == dimTrait) { //  All missing completely at random
                //TODO: This is N(X_pa(j), l_j V_root). Why not N(X_pa(j), V_branch) ?

                final double sqrtScale = Math.sqrt(1.0 / branchPrecision);

                if (NEW_TIP_WITH_NO_DATA) {

//                    final ReadableVector parentSample = new WrappedVector.Raw(sample, offsetParent, dimTrait);

                    final ReadableVector M;
                    M = getMeanBranch(offsetParent);
//                    if (hasNoDrift) {
//                        M = parentSample;
//                    } else {
//                        M = getMeanWithDrift(parentSample,
//                                new WrappedVector.Raw(displacementBuffer, 0, dimTrait));
//                    }

                    MultivariateNormalDistribution.nextMultivariateNormalCholesky(
                            M, // input mean
                            new WrappedMatrix.ArrayOfArray(cholesky), sqrtScale, // input variance
                            new WrappedVector.Raw(sample, offsetSample, dimTrait),
                            tmpEpsilon);

                } else {

                    // TODO Drift?
                    assert (!likelihoodDelegate.getDiffusionProcessDelegate().hasDrift());

                    MultivariateNormalDistribution.nextMultivariateNormalCholesky(
                            sample, offsetParent, // input mean
                            cholesky, sqrtScale, // input variance
                            sample, offsetSample, // output sample
                            tmpEpsilon);
                }

            } else {

                if (missingCount == dimTrait) { // All missing, but not completely at random

                    simulateTraitForInternalNode(offsetSample, offsetParent, offsetPartial, branchPrecision);

                } else { // Partially observed

                    System.arraycopy(partialNodeBuffer, offsetPartial, sample, offsetSample, dimTrait); // copy observed values

                    final PartiallyMissingInformation.HashedIntArray indices = missingInformation.getMissingIndices(nodeIndex, traitIndex);
                    final int[] observed = indices.getComplement();
                    final int[] missing = indices.getArray();

                    final DMatrixRMaj V1 = getVarianceBranch(branchPrecision);
//                    final DMatrixRMaj V1 = new DMatrixRMaj(dimTrait, dimTrait);
//                    CommonOps_DDRM.scale(1.0 / branchPrecision, Vd, V1);

                    ConditionalVarianceAndTransform2 transform =
                            new ConditionalVarianceAndTransform2(
                                    V1, missing, observed
                            ); // TODO Cache (via delegated function)

                    final DMatrixRMaj cP0 = new DMatrixRMaj(missing.length, missing.length);
                    gatherRowsAndColumns(P0, cP0, missing, missing);

                    final WrappedVector cM2 = transform.getConditionalMean(
                            partialNodeBuffer, offsetPartial, // Tip value
                            sample, offsetParent); // Parent value

                    final DMatrixRMaj cP1 = transform.getConditionalPrecision();

                    final DMatrixRMaj cP2 = new DMatrixRMaj(missing.length, missing.length);
                    final DMatrixRMaj cV2 = new DMatrixRMaj(missing.length, missing.length);
                    CommonOps_DDRM.add(cP0, cP1, cP2); //TODO: Shouldn't P0 = 0 always in this situation ?

                    safeInvert(cP2, cV2, false);

                    // TODO Drift?
//                    assert (!likelihoodDelegate.getDiffusionProcessDelegate().hasDrift());

                    if (NEW_CHOLESKY) {
                        DMatrixRMaj cC2 = getCholeskyOfVariance(cV2, missing.length);

                        MultivariateNormalDistribution.nextMultivariateNormalCholesky(
                                cM2, // input mean
                                new WrappedMatrix.WrappedDenseMatrix(cC2), 1.0, // input variance
                                new WrappedVector.Indexed(sample, offsetSample, missing, missing.length), // output sample
                                tmpEpsilon);
                    } else {
                        double[][] cC2 = getCholeskyOfVariance(cV2.getData(), missing.length);


                        MultivariateNormalDistribution.nextMultivariateNormalCholesky(
                                cM2, // input mean
                                new WrappedMatrix.ArrayOfArray(cC2), 1.0, // input variance
                                new WrappedVector.Indexed(sample, offsetSample, missing, missing.length), // output sample
                                tmpEpsilon);
                    }

                    if (DEBUG) {
                        final WrappedVector M0 = new WrappedVector.Raw(partialNodeBuffer, offsetPartial, dimTrait);

                        final WrappedVector M1 = new WrappedVector.Raw(sample, offsetParent, dimTrait);
                        final DMatrixRMaj P1 = new DMatrixRMaj(dimTrait, dimTrait);
                        CommonOps_DDRM.scale(branchPrecision, Pd, P1);

                        final WrappedVector newSample = new WrappedVector.Raw(sample, offsetSample, dimTrait);

                        System.err.println("sT F E N");
                        System.err.println("M0: " + M0);
                        System.err.println("P0: " + P0);
                        System.err.println("");
                        System.err.println("M1: " + M1);
                        System.err.println("P1: " + P1);
                        System.err.println("");
                        System.err.println("cP0: " + cP0);
                        System.err.println("cM2: " + cM2);
                        System.err.println("cP1: " + cP1);
                        System.err.println("cP2: " + cP2);
                        System.err.println("cV2: " + cV2);
//                        System.err.println("cC2: " + new Matrix(cC2));
                        System.err.println("SS: " + newSample);
                        System.err.println("");
                    }
                }
            }
        }
    }

    private final static boolean NEW_CHOLESKY = false;

    ReadableVector getMeanBranch(int offsetParent) {
        // Get parent value
        final double[] parentSample = new double[dimTrait];
        System.arraycopy(sample, offsetParent, parentSample, 0, dimTrait);

        // Get expectation
        final double[] expectation = new double[dimTrait];
        cdi.getBranchExpectation(actualizationBuffer, parentSample, displacementBuffer, expectation);

        return new WrappedVector.Raw(expectation, 0, dimTrait);
    }

//    private ReadableVector getMeanWithDrift(final ReadableVector mean,
//                                                 final ReadableVector drift) {
//        return new ReadableVector.Sum(mean, drift);
//    }

//    private ReadableVector getMeanWithDrift(double[] mean, int offsetMean, double[] drift, int dim) {
//        for (int i = 0;i < dim; ++i) {
//            tmpDrift[i] = mean[offsetMean + i] + drift[i];
//        }
//        return new WrappedVector.Raw(tmpDrift, 0, dimTrait);
//    }

    private void simulateTraitForInternalNode(final int offsetSample,
                                              final int offsetParent,
                                              final int offsetPartial,
                                              final double branchPrecision) {

        if (!Double.isInfinite(branchPrecision)) {
            // Here we simulate X_j | X_pa(j), Y

            final WrappedVector M0 = new WrappedVector.Raw(partialNodeBuffer, offsetPartial, dimTrait);
            final DMatrixRMaj P0 = wrap(partialNodeBuffer, offsetPartial + dimTrait, dimTrait, dimTrait);

//            final ReadableVector parentSample = new WrappedVector.Raw(sample, offsetParent, dimTrait);

            final ReadableVector M1;
            final DMatrixRMaj P1;

            M1 = getMeanBranch(offsetParent);
            P1 = getPrecisionBranch(branchPrecision);

//            if (hasNoDrift) {
//                M1 = parentSample; // new WrappedVector.Raw(sample, offsetParent, dimTrait);
//                P1 = new DMatrixRMaj(dimTrait, dimTrait);
//                CommonOps_DDRM.scale(branchPrecision, Pd, P1);
//            } else {
//                M1 = getMeanWithDrift(parentSample,
//                        new WrappedVector.Raw(displacementBuffer, 0, dimTrait)); //getMeanWithDrift(sample, offsetParent, displacementBuffer, dimTrait);
//                P1 = DMatrixRMaj.wrap(dimTrait, dimTrait, precisionBuffer);
//            }

            final WrappedVector M2 = new WrappedVector.Raw(tmpMean, 0, dimTrait);
            final DMatrixRMaj P2 = new DMatrixRMaj(dimTrait, dimTrait);
            final DMatrixRMaj V2 = new DMatrixRMaj(dimTrait, dimTrait);

            CommonOps_DDRM.add(P0, P1, P2);
            safeInvert(P2, V2, false);
            weightedAverage(M0, P0, M1, P1, M2, V2, dimTrait);

            final WrappedMatrix C2;
            if (NEW_CHOLESKY) {
                DMatrixRMaj tC2 = getCholeskyOfVariance(V2, dimTrait);
                C2 = new WrappedMatrix.WrappedDenseMatrix(tC2);

                MultivariateNormalDistribution.nextMultivariateNormalCholesky(
                        M2, // input mean
                        C2, 1.0, // input variance
                        new WrappedVector.Raw(sample, offsetSample, dimTrait),
                        tmpEpsilon);

            } else {
                double[][] tC2 = getCholeskyOfVariance(V2.getData(), dimTrait);
                C2 = new WrappedMatrix.ArrayOfArray(tC2);

                MultivariateNormalDistribution.nextMultivariateNormalCholesky(
                        M2, // input mean
                        C2, 1.0, // input variance
                        new WrappedVector.Raw(sample, offsetSample, dimTrait),
                        tmpEpsilon);
            }

            if (DEBUG) {
                System.err.println("sT F I N");
                System.err.println("M0: " + M0);
                System.err.println("P0: " + P0);
                System.err.println("M1: " + M1);
                System.err.println("P1: " + P1);
                System.err.println("M2: " + M2);
                System.err.println("V2: " + V2);
                System.err.println("C2: " + C2);
                System.err.println("SS: " + new WrappedVector.Raw(sample, offsetSample, dimTrait));
                System.err.println("");

                if (!check(M2))  {
                    System.exit(-1);
                }
            }

        } else {

            System.arraycopy(sample, offsetParent, sample, offsetSample, dimTrait);

            if (DEBUG) {
                System.err.println("sT F I N infinite branch precision");
                System.err.println("SS: " + new WrappedVector.Raw(sample, offsetSample, dimTrait));
            }
        }
    }

    private boolean check(ReadableVector m2) {
        for (int i = 0; i < m2.getDim(); ++i) {
            if (Double.isNaN(m2.get(i))) {
                return false;
            }
        }
        return true;
    }

    DMatrixRMaj getPrecisionBranch(double branchPrecision){
        if (!hasDrift) {
            DMatrixRMaj P1 = new DMatrixRMaj(dimTrait, dimTrait);
            CommonOps_DDRM.scale(branchPrecision, Pd, P1);
            return P1;
        } else {
            return DMatrixRMaj.wrap(dimTrait, dimTrait, precisionBuffer);
        }
    }

    DMatrixRMaj getVarianceBranch(double branchPrecision){
        if (!hasDrift) {
            final DMatrixRMaj V1 = new DMatrixRMaj(dimTrait, dimTrait);
            CommonOps_DDRM.scale(1.0 / branchPrecision, Vd, V1);
            return V1;
        } else {
            DMatrixRMaj P = getPrecisionBranch(branchPrecision);
            DMatrixRMaj V = new DMatrixRMaj(dimTrait, dimTrait);
            CommonOps_DDRM.invert(P, V);
            return V;
        }
    }
}
