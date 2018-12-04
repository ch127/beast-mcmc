package dr.evomodel.treedatalikelihood.continuous.cdi;

import dr.math.matrixAlgebra.WrappedVector;
import dr.math.matrixAlgebra.missingData.InversionResult;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import static dr.math.matrixAlgebra.missingData.InversionResult.Code.NOT_OBSERVED;
import static dr.math.matrixAlgebra.missingData.MissingOps.*;

/**
 * @author Marc A. Suchard
 */

public class SafeMultivariateIntegrator extends MultivariateIntegrator {

    private static boolean DEBUG = false;

    public SafeMultivariateIntegrator(PrecisionType precisionType, int numTraits, int dimTrait, int dimProcess,
                                      int bufferCount, int diffusionCount) {
        super(precisionType, numTraits, dimTrait, dimProcess, bufferCount, diffusionCount);

        allocateStorage();

        System.err.println("Trying SafeMultivariateIntegrator");
    }

    private void allocateStorage() {

        precisions = new double[dimTrait * dimTrait * bufferCount];
        variances = new double[dimTrait * dimTrait * bufferCount];

        vectorDelta = new double[dimTrait];

        matrixQjPjp = new DMatrixRMaj(dimTrait, dimTrait);

        partialsDimData = new int[bufferCount];
    }

    private static final boolean TIMING = false;

    @Override
    public void getBranchPrecision(int bufferIndex, double[] precision) {

        if (bufferIndex == -1) {
            throw new RuntimeException("Not yet implemented");
        }

        assert (precision != null);
        assert (precision.length >= dimTrait * dimTrait);

        System.arraycopy(precisions, bufferIndex * dimTrait * dimTrait,
                precision, 0, dimTrait * dimTrait);
    }

    @Override
    public void getRootPrecision(int priorBufferIndex, double[] precision) {

        assert (precision != null);
        assert (precision.length >= dimTrait * dimTrait);

        int priorOffset = dimPartial * priorBufferIndex;

        System.arraycopy(partials, priorOffset + dimTrait,
                precision, 0, dimTrait * dimTrait);
    }

    private int getEffectiveDimension(int iBuffer) {
        return partialsDimData[iBuffer];
    }

    private void setEffectiveDimension(int iBuffer, int effDim) {
        partialsDimData[iBuffer] = effDim;
    }

    @Override
    public void setPostOrderPartial(int bufferIndex, final double[] partial) {
        super.setPostOrderPartial(bufferIndex, partial);
        int effDim = 0;
        for (int i = 0; i < dimTrait; i++) {
            if (partial[dimTrait + i * (dimTrait + 1)] != 0) ++effDim;
        }
        partialsDimData[bufferIndex] = effDim;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Setting variances, displacement and actualization vectors
    ///////////////////////////////////////////////////////////////////////////

    public void updateBrownianDiffusionMatrices(int precisionIndex, final int[] probabilityIndices,
                                                final double[] edgeLengths, final double[] driftRates,
                                                int updateCount) {

        super.updateBrownianDiffusionMatrices(precisionIndex, probabilityIndices, edgeLengths, driftRates, updateCount);

        assert (diffusions != null);
        assert (probabilityIndices.length >= updateCount);
        assert (edgeLengths.length >= updateCount);

        if (DEBUG) {
            System.err.println("Matrices (safe):");
        }

        final int matrixSize = dimProcess * dimProcess;
        final int unscaledOffset = matrixSize * precisionIndex;

        if (TIMING) {
            startTime("diffusion");
        }

        for (int up = 0; up < updateCount; ++up) {

            if (DEBUG) {
                System.err.println("\t" + probabilityIndices[up] + " <- " + edgeLengths[up]);
            }

            final double edgeLength = edgeLengths[up];

            final int scaledOffset = matrixSize * probabilityIndices[up];

            scale(diffusions, unscaledOffset, 1.0 / edgeLength, precisions, scaledOffset, matrixSize);
            scale(inverseDiffusions, unscaledOffset, edgeLength, variances, scaledOffset, matrixSize); // TODO Only if necessary
        }

        if (TIMING) {
            endTime("diffusion");
        }
    }

    static void scale(final double[] source,
                      final int sourceOffset,
                      final double scale,
                      final double[] destination,
                      final int destinationOffset,
                      final int length) {
        for (int i = 0; i < length; ++i) {
            destination[destinationOffset + i] = scale * source[sourceOffset + i];
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Tree-traversal functions
    ///////////////////////////////////////////////////////////////////////////

    @Override
    public void updatePreOrderPartial(
            final int kBuffer, // parent
            final int iBuffer, // node
            final int iMatrix,
            final int jBuffer, // sibling
            final int jMatrix) {

        // Determine buffer offsets
        int kbo = dimPartial * kBuffer;
        int ibo = dimPartial * iBuffer;
        int jbo = dimPartial * jBuffer;

        // Determine matrix offsets
        final int imo = dimTrait * dimTrait * iMatrix;
        final int jmo = dimTrait * dimTrait * jMatrix;

        // Determine diagonal matrix offsets
        final int ido = dimTrait * iMatrix;
        final int jdo = dimTrait * jMatrix;

        // Read variance increments along descendant branches of k
        final DMatrixRMaj Vdi = wrap(variances, imo, dimTrait, dimTrait);
        final DMatrixRMaj Vdj = wrap(variances, jmo, dimTrait, dimTrait);

//        final DMatrixRMaj Pdi = wrap(precisions, imo, dimTrait, dimTrait); // TODO Only if needed
        final DMatrixRMaj Pdj = wrap(precisions, jmo, dimTrait, dimTrait); // TODO Only if needed

//        final DMatrixRMaj Vd = wrap(inverseDiffusions, precisionOffset, dimTrait, dimTrait);

        if (DEBUG) {
            System.err.println("updatePreOrderPartial for node " + iBuffer);
            System.err.println("\tVdj: " + Vdj);
            System.err.println("\tVdi: " + Vdi);
        }

        // For each trait // TODO in parallel
        for (int trait = 0; trait < numTraits; ++trait) {

            // A. Get current precision of k and j
            final DMatrixRMaj Pk = wrap(preOrderPartials, kbo + dimTrait, dimTrait, dimTrait);
//            final DMatrixRMaj Pj = wrap(partials, jbo + dimTrait, dimTrait, dimTrait);

//            final DMatrixRMaj Vk = wrap(preOrderPartials, kbo + dimTrait + dimTrait * dimTrait, dimTrait, dimTrait);
//            final DMatrixRMaj Vj = wrap(partials, jbo + dimTrait + dimTrait * dimTrait, dimTrait, dimTrait);

            // B. Inflate variance along sibling branch using matrix inversion
            final DMatrixRMaj Vjp = matrix0;
            final DMatrixRMaj Pjp = matrixPjp;
            increaseVariances(jbo, jBuffer, Vdj, Pdj, Pjp, false);

            // Actualize
            final DMatrixRMaj QjPjp = matrixQjPjp;
            actualizePrecision(Pjp, QjPjp, jbo, jmo, jdo);

            // C. Compute prePartial mean
            final DMatrixRMaj Pip = matrixPip;
            CommonOps_DDRM.add(Pk, Pjp, Pip);

            final DMatrixRMaj Vip = matrix1;
            safeInvert(Pip, Vip, false);

            final double[] delta = vectorDelta;
            computeDelta(jbo, jdo, delta);

            final double[] tmp = vector0;
            weightedAverage(preOrderPartials, kbo, Pk,
                    delta, 0, QjPjp,
                    preOrderPartials, ibo, Vip,
                    dimTrait, tmp);

            scaleAndDriftMean(ibo, imo, ido);

            // C. Inflate variance along node branch
            final DMatrixRMaj Vi = Vip;
            actualizeVariance(Vip, ibo, imo, ido);
            inflateBranch(Vdi, Vip, Vi);

            final DMatrixRMaj Pi = matrixPk;
            safeInvert(Vi, Pi, false);

            // X. Store precision results for node
            unwrap(Pi, preOrderPartials, ibo + dimTrait);
            unwrap(Vi, preOrderPartials, ibo + dimTrait + dimTrait * dimTrait);

            if (DEBUG) {
                System.err.println("trait: " + trait);
                System.err.println("pM: " + new WrappedVector.Raw(preOrderPartials, kbo, dimTrait));
                System.err.println("pP: " + Pk);
                System.err.println("sM: " + new WrappedVector.Raw(partials, jbo, dimTrait));
                DMatrixRMaj Pj = wrap(partials, ibo + dimTrait, dimTrait, dimTrait);
                DMatrixRMaj Vj = new DMatrixRMaj(dimTrait, dimTrait);
                CommonOps_DDRM.invert(Pj, Vj);
                System.err.println("sP: " + Vj);
                System.err.println("sP: " + Pj);
                System.err.println("sVp: " + Vjp);
                System.err.println("sPp: " + Pjp);
                System.err.println("Pip: " + Pip);
                System.err.println("QiPip: " + QjPjp);
                System.err.println("cM: " + new WrappedVector.Raw(preOrderPartials, ibo, dimTrait));
                System.err.println("cV: " + Vi);
            }

            // Get ready for next trait
            kbo += dimPartialForTrait;
            ibo += dimPartialForTrait;
            jbo += dimPartialForTrait;
        }
    }

    private void inflateBranch(DMatrixRMaj Vj, DMatrixRMaj Vdj, DMatrixRMaj Vjp) {
        CommonOps_DDRM.add(Vj, Vdj, Vjp);
    }

    void actualizePrecision(DMatrixRMaj P, DMatrixRMaj QP, int jbo, int jmo, int jdo) {
        CommonOps_DDRM.scale(1.0, P, QP);
    }

    void actualizeVariance(DMatrixRMaj V, int ibo, int imo, int ido) {
        // Do nothing
    }

    void scaleAndDriftMean(int ibo, int imo, int ido) {
        // Do nothing
    }

    void computeDelta(int jbo, int jdo, double[] delta) {
        System.arraycopy(partials, jbo, delta, 0, dimTrait);
    }

    @Override
    protected void updatePartial(
            final int kBuffer,
            final int iBuffer,
            final int iMatrix,
            final int jBuffer,
            final int jMatrix,
            final boolean incrementOuterProducts
    ) {

        if (incrementOuterProducts) {
            throw new RuntimeException("Outer-products are not supported.");
        }

        if (TIMING) {
            startTime("total");
        }

        // Determine buffer offsets
        int kbo = dimPartial * kBuffer;
        int ibo = dimPartial * iBuffer;
        int jbo = dimPartial * jBuffer;

        // Determine matrix offsets
        final int imo = dimTrait * dimTrait * iMatrix;
        final int jmo = dimTrait * dimTrait * jMatrix;

        // Determine diagonal matrix offsets
        final int ido = dimTrait * iMatrix;
        final int jdo = dimTrait * jMatrix;

        // Read variance increments along descendant branches of k
        final DMatrixRMaj Vdi = wrapSymmetric(variances, imo, dimTrait, dimTrait);
        final DMatrixRMaj Vdj = wrapSymmetric(variances, jmo, dimTrait, dimTrait);

        final DMatrixRMaj Pdi = wrapSymmetric(precisions, imo, dimTrait, dimTrait); // TODO Only if needed
        final DMatrixRMaj Pdj = wrapSymmetric(precisions, jmo, dimTrait, dimTrait); // TODO Only if needed

        if (DEBUG) {
            System.err.println("variance diffusion: " + wrap(inverseDiffusions, precisionOffset, dimProcess, dimProcess));
            System.err.println("precisionOffset = " + precisionOffset);
            System.err.println("\tVdi: " + Vdi);
            System.err.println("\tVdj: " + Vdj);
        }

        // For each trait // TODO in parallel
        for (int trait = 0; trait < numTraits; ++trait) {

            // Layout, offset, dim
            // trait, 0, dT
            // precision, dT, dT * dT
            // variance, dT + dT * dT, dT * dT
            // scalar, dT + 2 * dT * dT, 1

            // Increase variance along the branches i -> k and j -> k

            final DMatrixRMaj Pip = matrixPip;
            final DMatrixRMaj Pjp = matrixPjp;


            InversionResult ci = increaseVariances(ibo, iBuffer, Vdi, Pdi, Pip, true);
            InversionResult cj = increaseVariances(jbo, jBuffer, Vdj, Pdj, Pjp, true);

            if (TIMING) {
                endTime("peel2");
                startTime("peel3");
            }

            // Compute partial mean and precision at node k

            // A. Partial precision and variance (for later use) using one matrix inversion
            final DMatrixRMaj Pk = matrixPk;
            computePartialPrecision(ido, jdo, imo, jmo, Pip, Pjp, Pk);

            if (TIMING) {
                endTime("peel3");
            }

            // B. Partial mean
            partialMean(ibo, jbo, kbo, ido, jdo);

            if (TIMING) {
                startTime("peel5");
            }

            // C. Store precision
            unwrap(Pk, partials, kbo + dimTrait);

            if (TIMING) {
                endTime("peel5");
            }

            if (DEBUG) {
                final DMatrixRMaj Pi = wrap(partials, ibo + dimTrait, dimTrait, dimTrait);
                final DMatrixRMaj Pj = wrap(partials, jbo + dimTrait, dimTrait, dimTrait);
                reportMeansAndPrecisions(trait, ibo, jbo, kbo, Pi, Pj, Pk);
            }

            // Computer remainder at node k
            double remainder = 0.0;

            if (DEBUG) {
                reportInversions(ci, cj, Pip, Pjp);
            }

            if (TIMING) {
                startTime("remain");
            }

            if (!(ci.getReturnCode() == NOT_OBSERVED || cj.getReturnCode() == NOT_OBSERVED)) {

                // Inner products
                double SS = computeSS(ibo, Pip, jbo, Pjp, kbo, Pk, dimTrait);

                remainder += -0.5 * SS;

                if (DEBUG) {
                    System.err.println("\t\t\tSS = " + (SS));
                }
            } // End if remainder

            int dimensionChange = getEffectiveDimension(iBuffer) + getEffectiveDimension(jBuffer);

//            int dimensionChange = ci.getEffectiveDimension() + cj.getEffectiveDimension()
//                    - ck.getEffectiveDimension();
//
//            setEffectiveDimension(kBuffer, ck.getEffectiveDimension());

            remainder += -dimensionChange * LOG_SQRT_2_PI;

            double deti = 0;
            double detj = 0;
//            double detk = 0;
            if (!(ci.getReturnCode() == NOT_OBSERVED)) {
                deti = ci.getLogDeterminant(); // TODO: for OU, use det(exp(M)) = exp(tr(M)) ? (Qdi = exp(-A l_i))
            }
            if (!(cj.getReturnCode() == NOT_OBSERVED)) {
                detj = cj.getLogDeterminant();
            }
//            if (!(ck.getReturnCode() == NOT_OBSERVED)) {
//                detk = ck.getLogDeterminant();
//            }
            remainder += -0.5 * (deti + detj); // + detk);

            // TODO Can get SSi + SSj - SSk from inner product w.r.t Pt (see outer-products below)?

            if (DEBUG) {
                System.err.println("\t\t\tdeti = " + ci.getLogDeterminant());
                System.err.println("\t\t\tdetj = " + cj.getLogDeterminant());
//                System.err.println("\t\t\tdetk = " + ck.getLogDeterminant());
                System.err.println("\t\tremainder: " + remainder);
            }

            if (TIMING) {
                endTime("remain");
            }

            // Accumulate remainder up tree and store

            remainders[kBuffer * numTraits + trait] = remainder
                    + remainders[iBuffer * numTraits + trait] + remainders[jBuffer * numTraits + trait];

            // Get ready for next trait
            kbo += dimPartialForTrait;
            ibo += dimPartialForTrait;
            jbo += dimPartialForTrait;
        }

        if (TIMING) {
            endTime("total");
        }
    }

    private void reportInversions(InversionResult ci, InversionResult cj,
                                  DMatrixRMaj Pip, DMatrixRMaj Pjp) {
        System.err.println("i status: " + ci);
        System.err.println("j status: " + cj);
//        System.err.println("k status: " + ck);
        System.err.println("Pip: " + Pip);
        System.err.println("Pjp: " + Pjp);
    }

    private InversionResult increaseVariances(int ibo,
                                              int iBuffer,
                                              final DMatrixRMaj Vdi,
                                              final DMatrixRMaj Pdi,
                                              final DMatrixRMaj Pip,
                                              final boolean getDeterminant) {

        if (TIMING) {
            startTime("peel1");
        }

        // A. Get current precision of i and j
        final DMatrixRMaj Pi = wrapSymmetric(partials, ibo + dimTrait, dimTrait, dimTrait);

        if (TIMING) {
            endTime("peel1");
            startTime("peel2");
        }

        // B. Integrate along branch using two matrix inversions

        final boolean useVariancei = anyDiagonalInfinities(Pi);
        InversionResult ci = null;

        if (useVariancei) {

            final DMatrixRMaj Vip = matrix0;
            final DMatrixRMaj Vi = wrapSymmetric(partials, ibo + dimTrait + dimTrait * dimTrait, dimTrait, dimTrait);
//                CommonOps_DDRM.add(Vi, vi, Vd, Vip);  // TODO Fix
            CommonOps_DDRM.add(Vi, Vdi, Vip);
            assert !allZeroOrInfinite(Vip) :  "Zero-length branch on data is not allowed.";
            ci = safeInvert(Vip, Pip, getDeterminant);

        } else {

            final DMatrixRMaj tmp1 = matrix0;
            CommonOps_DDRM.add(Pi, Pdi, tmp1);
            final DMatrixRMaj tmp2 = new DMatrixRMaj(dimTrait, dimTrait);
            safeInvert(tmp1, tmp2, false);
            CommonOps_DDRM.mult(tmp2, Pi, tmp1);
            idMinusA(tmp1);
            if (getDeterminant && getEffectiveDimension(iBuffer) == 0) ci = safeDeterminant(tmp1, false);
            CommonOps_DDRM.mult(Pi, tmp1, Pip);
            forceSymmetric(Pip);
            if (getDeterminant && getEffectiveDimension(iBuffer) > 0) ci = safeDeterminant(Pip, false);
        }

        if (TIMING) {
            endTime("peel2");
        }

        return ci;
    }

    private static void idMinusA(DMatrixRMaj A) {
        CommonOps_DDRM.scale(-1.0, A);
        for (int i = 0; i < A.numCols; i++) {
            A.set(i, i, 1.0 + A.get(i, i));
        }
    }

    private static boolean allZeroOrInfinite(DMatrixRMaj M) {
        for (int i = 0; i < M.getNumElements(); i++) {
            if (Double.isFinite(M.get(i)) && M.get(i) != 0.0) return false;
        }
        return true;
    }

    void computePartialPrecision(int ido, int jdo, int imo, int jmo,
                                 DMatrixRMaj Pip, DMatrixRMaj Pjp, DMatrixRMaj Pk) {
        CommonOps_DDRM.add(Pip, Pjp, Pk);
    }

    void partialMean(int ibo, int jbo, int kbo,
                     int ido, int jdo) {
        if (TIMING) {
            startTime("peel4");
        }

        final double[] tmp = vector0;
        weightedSum(partials, ibo, matrixPip, partials, jbo, matrixPjp, dimTrait, tmp);


        final WrappedVector kPartials = new WrappedVector.Raw(partials, kbo, dimTrait);
        final WrappedVector wrapTmp = new WrappedVector.Raw(tmp, 0, dimTrait);

        safeSolve(matrixPk, wrapTmp, kPartials, false);

        if (TIMING) {
            endTime("peel4");
            startTime("peel5");
        }
//        return ck;
    }


//    private final Map<String, Long> startTimes = new HashMap<String, Long>();
//
//    private void startTime(String key) {
//        startTimes.put(key, System.nanoTime());
//    }
//
//    private void endTime(String key) {
//        long start = startTimes.get(key);
//
//        Long total = times.get(key);
//        if (total == null) {
//            total = new Long(0);
//        }
//
//        long run = total + (System.nanoTime() - start);
//        times.put(key, run);
//
////            System.err.println("run = " + run);
////            System.exit(-1);
//    }

//        private void incrementTiming(long start, long end, String key) {
//            Long total = times.get(key);
//
//            System.err.println(start + " " + end + " " + key);
//            System.exit(-1);
//            if (total == null) {
//                total = new Long(0);
//                times.put(key, total);
//            }
//            total += (end - start);
////            times.put(key, total);
//        }

    @Override
    public void calculateRootLogLikelihood(int rootBufferIndex, int priorBufferIndex, final double[] logLikelihoods,
                                           boolean incrementOuterProducts, boolean isIntegratedProcess) {
        assert (logLikelihoods.length == numTraits);

        assert (!incrementOuterProducts);

        if (DEBUG) {
            System.err.println("Root calculation for " + rootBufferIndex);
            System.err.println("Prior buffer index is " + priorBufferIndex);
        }

        int rootOffset = dimPartial * rootBufferIndex;
        int priorOffset = dimPartial * priorBufferIndex;

        final DMatrixRMaj Pd = wrap(diffusions, precisionOffset, dimProcess, dimProcess);
//        final DMatrixRMaj Vd = wrap(inverseDiffusions, precisionOffset, dimTrait, dimTrait);

        // TODO For each trait in parallel
        for (int trait = 0; trait < numTraits; ++trait) {

            final DMatrixRMaj PPrior = wrap(partials, priorOffset + dimTrait, dimTrait, dimTrait);
            final DMatrixRMaj VPrior = wrap(partials, priorOffset + dimTrait + dimTrait * dimTrait, dimTrait, dimTrait);


            // TODO Block below is for the conjugate prior ONLY
            {

                if (!isIntegratedProcess) {
                    final DMatrixRMaj PTmp = new DMatrixRMaj(dimTrait, dimTrait);
                    CommonOps_DDRM.mult(Pd, PPrior, PTmp);
                    PPrior.set(PTmp); // TODO What does this do?
                } else {
                    DMatrixRMaj Pdbis = new DMatrixRMaj(dimTrait, dimTrait);
                    blockUnwrap(Pd, Pdbis.data, 0, 0, 0, dimTrait);
                    blockUnwrap(Pd, Pdbis.data, dimProcess, dimProcess, 0, dimTrait);

                    final DMatrixRMaj PTmp = new DMatrixRMaj(dimTrait, dimTrait);
                    CommonOps_DDRM.mult(Pdbis, PPrior, PTmp);
                    PPrior.set(PTmp);
                }
            }

            final DMatrixRMaj VTotal = new DMatrixRMaj(dimTrait, dimTrait);

            final DMatrixRMaj PTotal = new DMatrixRMaj(dimTrait, dimTrait);
            CommonOps_DDRM.invert(VTotal, PTotal);  // TODO Does this do anything?

            InversionResult ctot = increaseVariances(rootOffset, rootBufferIndex, VPrior, PPrior, PTotal, true);

            double SS = weightedInnerProductOfDifferences(
                    partials, rootOffset,
                    partials, priorOffset,
                    PTotal, dimTrait);

            double dettot = (ctot.getReturnCode() == NOT_OBSERVED) ? 0 : ctot.getLogDeterminant();

            final double logLike =
//                    - ctot.getEffectiveDimension() * LOG_SQRT_2_PI
//                    - 0.5 * Math.log(CommonOps_DDRM.det(VTotal))
//                    + 0.5 * Math.log(CommonOps_DDRM.det(PTotal))
                    - 0.5 * dettot
                    - 0.5 * SS;

            final double remainder = remainders[rootBufferIndex * numTraits + trait];
            logLikelihoods[trait] = logLike + remainder;

//            if (incrementOuterProducts) {
//
//                assert false : "Should not get here";
//
////                int opo = dimTrait * dimTrait * trait;
////                int opd = precisionOffset;
////
////                double rootScalar = partials[rootOffset + dimTrait + 2 * dimTrait * dimTrait];
////                final double priorScalar = partials[priorOffset + dimTrait];
////
////                if (!Double.isInfinite(priorScalar)) {
////                    rootScalar = rootScalar * priorScalar / (rootScalar + priorScalar);
////                }
////
////                for (int g = 0; g < dimTrait; ++g) {
////                    final double gDifference = partials[rootOffset + g] - partials[priorOffset + g];
////
////                    for (int h = 0; h < dimTrait; ++h) {
////                        final double hDifference = partials[rootOffset + h] - partials[priorOffset + h];
////
////                        outerProducts[opo] += gDifference * hDifference
//////                                    * PTotal.unsafe_get(g, h) / diffusions[opd];
////                                * rootScalar;
////                        ++opo;
////                        ++opd;
////                    }
////                }
////
////                degreesOfFreedom[trait] += 1; // increment degrees-of-freedom
//            }

            if (DEBUG) {
                System.err.print("mean:");
                for (int g = 0; g < dimTrait; ++g) {
                    System.err.print(" " + partials[rootOffset + g]);
                }
                System.err.println("");
                System.err.println("PRoot: " + wrap(partials, rootOffset + dimTrait, dimTrait, dimTrait));
                System.err.println("PPrior: " + PPrior);
                System.err.println("PTotal: " + PTotal);
                System.err.println("\n SS:" + SS);
                System.err.println("det:" + dettot);
                System.err.println("remainder:" + remainder);
                System.err.println("likelihood" + (logLike + remainder));

//                if (incrementOuterProducts) {
//                    System.err.println("Outer-products:" + wrap(outerProducts, dimTrait * dimTrait * trait, dimTrait, dimTrait));
//                }
            }

            rootOffset += dimPartialForTrait;
            priorOffset += dimPartialForTrait;
        }

        if (DEBUG) {
            System.err.println("End");
        }
    }

//    private InversionResult computeBranchAdjustedPrecision(final double[] partials,
//                                                           final int bo,
//                                                           final DMatrixRMaj P,
//                                                           final DMatrixRMaj Pd,
//                                                           final DMatrixRMaj Vd,
//                                                           final double v,
//                                                           final DMatrixRMaj Pp) {
//        InversionResult c;
//        if (anyDiagonalInfinities(P)) {
//            // Inflate variance
//            final DMatrixRMaj Vp = matrix0;
//            final DMatrixRMaj Vi = wrap(partials, bo + dimTrait + dimTrait * dimTrait, dimTrait, dimTrait);
//
//            CommonOps_DDRM.add(Vi, v, Vd, Vp);
//            c = safeInvert(Vp, Pp, true);
//        } else {
//            // Deflate precision
//            final DMatrixRMaj PPlusPd = matrix0;
//            CommonOps_DDRM.add(P, 1.0 / v, Pd, PPlusPd);
//
//            final DMatrixRMaj PPlusPdInv = new DMatrixRMaj(dimTrait, dimTrait);
//            safeInvert(PPlusPd, PPlusPdInv, false);
//
//            CommonOps_DDRM.mult(PPlusPdInv, P, Pp);
//            CommonOps_DDRM.mult(P, Pp, PPlusPdInv);
//            CommonOps_DDRM.add(P, -1, PPlusPdInv, Pp);
//            c = safeDeterminant(Pp, false);
//        }
//
//        return c;
//    }

    double computeSS(final int ibo,
                     final DMatrixRMaj Pip,
                     final int jbo,
                     final DMatrixRMaj Pjp,
                     final int kbo,
                     final DMatrixRMaj Pk,
                     final int dimTrait) {
        return weightedThreeInnerProduct(partials, ibo, Pip,
                partials, jbo, Pjp,
                partials, kbo, Pk,
                dimTrait);
    }

    private DMatrixRMaj matrixQjPjp;
    private double[] vectorDelta;
    private int[] partialsDimData;
}
