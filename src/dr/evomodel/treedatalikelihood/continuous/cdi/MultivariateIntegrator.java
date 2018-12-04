package dr.evomodel.treedatalikelihood.continuous.cdi;

import dr.evomodel.treedatalikelihood.preorder.BranchSufficientStatistics;
import dr.math.matrixAlgebra.WrappedVector;
import dr.math.matrixAlgebra.missingData.InversionResult;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import java.util.HashMap;
import java.util.Map;

import static dr.math.matrixAlgebra.missingData.InversionResult.Code.NOT_OBSERVED;
import static dr.math.matrixAlgebra.missingData.MissingOps.*;

/**
 * @author Marc A. Suchard
 */

public class MultivariateIntegrator extends ContinuousDiffusionIntegrator.Basic {

    private static boolean DEBUG = false;

    public MultivariateIntegrator(PrecisionType precisionType, int numTraits, int dimTrait, int dimProcess,
                                  int bufferCount, int diffusionCount) {
        super(precisionType, numTraits, dimTrait, dimProcess, bufferCount, diffusionCount);

        assert precisionType == PrecisionType.FULL;

        allocateStorage();

        if (TIMING) {
            times = new HashMap<String, Long>();
        } else {
            times = null;
        }
    }

    @Override
    public String getReport() {

        StringBuilder sb = new StringBuilder();

        if (TIMING) {
            sb.append("\nTIMING:");
            for (String key : times.keySet()) {
                String value = String.format("%4.3e", (double) times.get(key));
                sb.append("\n").append(key).append("\t\t").append(value);
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    private static final boolean TIMING = false;

    private final Map<String, Long> times;

    DMatrixRMaj matrix0;
    DMatrixRMaj matrix1;
    DMatrixRMaj matrixPip;
    DMatrixRMaj matrixPjp;
    DMatrixRMaj matrixPk;
    private DMatrixRMaj matrix5;
    private DMatrixRMaj matrix6;

    double[] vector0;

    private void allocateStorage() {
        inverseDiffusions = new double[dimProcess * dimProcess * diffusionCount];

        vector0 = new double[dimTrait];
        matrix0 = new DMatrixRMaj(dimTrait, dimTrait);
        matrix1 = new DMatrixRMaj(dimTrait, dimTrait);
        matrixPip = new DMatrixRMaj(dimTrait, dimTrait);
        matrixPjp = new DMatrixRMaj(dimTrait, dimTrait);
        matrixPk = new DMatrixRMaj(dimTrait, dimTrait);
        matrix5 = new DMatrixRMaj(dimTrait, dimTrait);
        matrix6 = new DMatrixRMaj(dimTrait, dimTrait);
    }

    @Override
    public void setDiffusionPrecision(int precisionIndex, final double[] matrix, double logDeterminant) {
        super.setDiffusionPrecision(precisionIndex, matrix, logDeterminant);

        assert (inverseDiffusions != null);

        final int offset = dimProcess * dimProcess * precisionIndex;
        DMatrixRMaj precision = wrap(diffusions, offset, dimProcess, dimProcess);
        DMatrixRMaj variance = new DMatrixRMaj(dimProcess, dimProcess);
        CommonOps_DDRM.invert(precision, variance);
        unwrap(variance, inverseDiffusions, offset);

        if (DEBUG) {
            System.err.println("At precision index: " + precisionIndex);
            System.err.println("precision: " + precision);
            System.err.println("variance : " + variance);
        }
    }

//    @Override
//    public boolean requireDataAugmentationForOuterProducts() {
//        return true;
//    }

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
        final int imo = dimMatrix * iMatrix;
        final int jmo = dimMatrix * jMatrix;

        // Read variance increments along descendant branches of k
        final double vi = branchLengths[imo];
        final double vj = branchLengths[jmo];

        final DMatrixRMaj Vd = wrap(inverseDiffusions, precisionOffset, dimTrait, dimTrait);

        if (DEBUG) {
            System.err.println("updatePreOrderPartial for node " + iBuffer);
//                System.err.println("variance diffusion: " + Vd);
            System.err.println("\tvi: " + vi + " vj: " + vj);
//                System.err.println("precisionOffset = " + precisionOffset);
        }

        // For each trait // TODO in parallel
        for (int trait = 0; trait < numTraits; ++trait) {

            // A. Get current precision of k and j
            final DMatrixRMaj Pk = wrap(preOrderPartials, kbo + dimTrait, dimTrait, dimTrait);
//                final DMatrixRMaj Pj = wrap(partials, jbo + dimTrait, dimTrait, dimTrait);

//                final DMatrixRMaj Vk = wrap(prePartials, kbo + dimTrait + dimTrait * dimTrait, dimTrait, dimTrait);
            final DMatrixRMaj Vj = wrap(partials, jbo + dimTrait + dimTrait * dimTrait, dimTrait, dimTrait);

            if (allZeroDiagonals(Vj)) {

                final DMatrixRMaj Pj = wrap(partials, jbo + dimTrait, dimTrait, dimTrait);

                assert (!allZeroDiagonals(Pj));

                safeInvert(Pj, Vj, false);
            }

            // B. Inflate variance along sibling branch using matrix inversion
//                final DMatrixRMaj Vjp = new DMatrixRMaj(dimTrait, dimTrait);
            final DMatrixRMaj Vjp = matrix1;
            CommonOps_DDRM.add(Vj, vj, Vd, Vjp);

//                final DMatrixRMaj Pjp = new DMatrixRMaj(dimTrait, dimTrait);
            final DMatrixRMaj Pjp = matrixPjp;
            safeInvert(Vjp, Pjp, false);

//                final DMatrixRMaj Pip = new DMatrixRMaj(dimTrait, dimTrait);
            final DMatrixRMaj Pip = matrixPip;
            CommonOps_DDRM.add(Pk, Pjp, Pip);

//                final DMatrixRMaj Vip = new DMatrixRMaj(dimTrait, dimTrait);
            final DMatrixRMaj Vip = matrix0;
            safeInvert(Pip, Vip, false);

            // C. Compute prePartial mean
//                final double[] tmp = new double[dimTrait];
            // TODO Rewrite using weightedAverage()
            final double[] tmp = vector0;
            for (int g = 0; g < dimTrait; ++g) {
                double sum = 0.0;
                for (int h = 0; h < dimTrait; ++h) {
                    sum += Pk.unsafe_get(g, h) * preOrderPartials[kbo + h]; // Read parent
                    sum += Pjp.unsafe_get(g, h) * partials[jbo + h];   // Read sibling
                }
                tmp[g] = sum;
            }
            for (int g = 0; g < dimTrait; ++g) {
                double sum = 0.0;
                for (int h = 0; h < dimTrait; ++h) {
                    sum += Vip.unsafe_get(g, h) * tmp[h];
                }
                preOrderPartials[ibo + g] = sum; // Write node
//                preBranchPartials[ibo +g] = sum; // TODO Only when necessary
            }

            // C. Inflate variance along node branch
            @SuppressWarnings("redundant")
            final DMatrixRMaj Vi = Vip;
            CommonOps_DDRM.add(vi, Vd, Vip, Vi);

//                final DMatrixRMaj Pi = new DMatrixRMaj(dimTrait, dimTrait);
            final DMatrixRMaj Pi = matrixPk;
            safeInvert(Vi, Pi, false);

            // X. Store precision results for node
            unwrap(Pi, preOrderPartials, ibo + dimTrait);
            unwrap(Vi, preOrderPartials, ibo + dimTrait + dimTrait * dimTrait);
//            unwrap(Pip, preBranchPartials, ibo + dimTrait); // TODO Only when necessary

            if (DEBUG) {
                System.err.println("trait: " + trait);
                System.err.println("pM: " + new WrappedVector.Raw(preOrderPartials, kbo, dimTrait));
                System.err.println("pP: " + Pk);
                System.err.println("sM: " + new WrappedVector.Raw(partials, jbo, dimTrait));
                System.err.println("sV: " + Vj);
                System.err.println("sVp: " + Vjp);
                System.err.println("sPp: " + Pjp);
                System.err.println("Pip: " + Pip);
                System.err.println("cM: " + new WrappedVector.Raw(preOrderPartials, ibo, dimTrait));
                System.err.println("cV: " + Vi);
            }

            // Get ready for next trait
            kbo += dimPartialForTrait;
            ibo += dimPartialForTrait;
            jbo += dimPartialForTrait;
        }
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
        final int imo = dimMatrix * iMatrix;
        final int jmo = dimMatrix * jMatrix;

        // Read variance increments along descendant branches of k
        final double vi = branchLengths[imo];
        final double vj = branchLengths[jmo];

        final DMatrixRMaj Vd = wrap(inverseDiffusions, precisionOffset, dimTrait, dimTrait);

        if (DEBUG) {
            System.err.println("variance diffusion: " + Vd);
            System.err.println("\tvi: " + vi + " vj: " + vj);
            System.err.println("precisionOffset = " + precisionOffset);
        }

        // For each trait // TODO in parallel
        for (int trait = 0; trait < numTraits; ++trait) {

            // Layout, offset, dim
            // trait, 0, dT
            // precision, dT, dT * dT
            // variance, dT + dT * dT, dT * dT
            // scalar, dT + 2 * dT * dT, 1

            if (TIMING) {
                startTime("peel1");
            }

            // Increase variance along the branches i -> k and j -> k

            // A. Get current precision of i and j
            final double lpi = partials[ibo + dimTrait + 2 * dimTrait * dimTrait];
            final double lpj = partials[jbo + dimTrait + 2 * dimTrait * dimTrait];

            final DMatrixRMaj Pi = wrap(partials, ibo + dimTrait, dimTrait, dimTrait);
            final DMatrixRMaj Pj = wrap(partials, jbo + dimTrait, dimTrait, dimTrait);

            final DMatrixRMaj Vi = wrap(partials, ibo + dimTrait + dimTrait * dimTrait, dimTrait, dimTrait);
            final DMatrixRMaj Vj = wrap(partials, jbo + dimTrait + dimTrait * dimTrait, dimTrait, dimTrait);

            if (TIMING) {
                endTime("peel1");
                startTime("peel2");
            }

            // B. Integrate along branch using two matrix inversions
            @SuppressWarnings("SpellCheckingInspection")
            final double lpip = Double.isInfinite(lpi) ?
                    1.0 / vi : lpi / (1.0 + lpi * vi);
            @SuppressWarnings("SpellCheckingInspection")
            final double lpjp = Double.isInfinite(lpj) ?
                    1.0 / vj : lpj / (1.0 + lpj * vj);

//                final DMatrixRMaj Vip = new DMatrixRMaj(dimTrait, dimTrait);
//                final DMatrixRMaj Vjp = new DMatrixRMaj(dimTrait, dimTrait);
            final DMatrixRMaj Vip = matrix0;
            final DMatrixRMaj Vjp = matrix1;

            CommonOps_DDRM.add(Vi, vi, Vd, Vip);
            CommonOps_DDRM.add(Vj, vj, Vd, Vjp);

            if (TIMING) {
                endTime("peel2");
                startTime("peel2a");
            }

//                final DMatrixRMaj Pip = new DMatrixRMaj(dimTrait, dimTrait);
//                final DMatrixRMaj Pjp = new DMatrixRMaj(dimTrait, dimTrait);
            final DMatrixRMaj Pip = matrixPip;
            final DMatrixRMaj Pjp = matrixPjp;

            InversionResult ci = safeInvert(Vip, Pip, true);
            InversionResult cj = safeInvert(Vjp, Pjp, true);

            if (TIMING) {
                endTime("peel2a");
                startTime("peel3");
            }

            // Compute partial mean and precision at node k

            // A. Partial precision and variance (for later use) using one matrix inversion
            final double lpk = lpip + lpjp;

//                final DMatrixRMaj Pk = new DMatrixRMaj(dimTrait, dimTrait);
            final DMatrixRMaj Pk = matrixPk;

            CommonOps_DDRM.add(Pip, Pjp, Pk);

//                final DMatrixRMaj Vk = new DMatrixRMaj(dimTrait, dimTrait);
            final DMatrixRMaj Vk = matrix5;
            InversionResult ck = safeInvert(Pk, Vk, true);

            // B. Partial mean
//                for (int g = 0; g < dimTrait; ++g) {
//                    partials[kbo + g] = (pip * partials[ibo + g] + pjp * partials[jbo + g]) / pk;
//                }

            if (TIMING) {
                endTime("peel3");
                startTime("peel4");
            }

//                final double[] tmp = new double[dimTrait];
//            final double[] tmp = vector0;
//            for (int g = 0; g < dimTrait; ++g) {
//                double sum = 0.0;
//                for (int h = 0; h < dimTrait; ++h) {
//                    sum += Pip.unsafe_get(g, h) * partials[ibo + h];
//                    sum += Pjp.unsafe_get(g, h) * partials[jbo + h];
//                }
//                tmp[g] = sum;
//            }
//            for (int g = 0; g < dimTrait; ++g) {
//                double sum = 0.0;
//                for (int h = 0; h < dimTrait; ++h) {
//                    sum += Vk.unsafe_get(g, h) * tmp[h];
//                }
//                partials[kbo + g] = sum;
//            }

            weightedAverage(
                    partials, ibo, Pip,
                    partials, jbo, Pjp,
                    partials, kbo, Vk,
                    dimTrait, vector0);

            if (TIMING) {
                endTime("peel4");
                startTime("peel5");
            }

            // C. Store precision
            partials[kbo + dimTrait + 2 * dimTrait * dimTrait] = lpk;

            unwrap(Pk, partials, kbo + dimTrait);
            unwrap(Vk, partials, kbo + dimTrait + dimTrait * dimTrait);

            if (TIMING) {
                endTime("peel5");
            }

            if (DEBUG) {
                reportMeansAndPrecisions(trait, ibo, jbo, kbo, Pi, Pj, Pk);
//                System.err.println("\tTrait: " + trait);
//                System.err.println("Pi: " + Pi);
//                System.err.println("Pj: " + Pj);
//                System.err.println("Pk: " + Pk);
//                System.err.print("\t\tMean i:");
//                for (int e = 0; e < dimTrait; ++e) {
//                    System.err.print(" " + partials[ibo + e]);
//                }
//                System.err.print("\t\tMean j:");
//                for (int e = 0; e < dimTrait; ++e) {
//                    System.err.print(" " + partials[jbo + e]);
//                }
//                System.err.print("\t\tMean k:");
//                for (int e = 0; e < dimTrait; ++e) {
//                    System.err.print(" " + partials[kbo + e]);
//                }
//                System.err.println("");
            }

            // Computer remainder at node k
            double remainder = 0.0;

            if (DEBUG) {
                System.err.println("i status: " + ci);
                System.err.println("j status: " + cj);
                System.err.println("k status: " + ck);
                System.err.println("Pip: " + Pip);
                System.err.println("Vip: " + Vip);
                System.err.println("Pjp: " + Pjp);
                System.err.println("Vjp: " + Vjp);
            }

            if (!(ci.getReturnCode() == NOT_OBSERVED || cj.getReturnCode() == NOT_OBSERVED)) {

                if (TIMING) {
                    startTime("remain");
                }

                // Inner products
//                double SSk = 0;
//                double SSj = 0;
//                double SSi = 0;
//
//                // vector-matrix-vector TODO in parallel
//                for (int g = 0; g < dimTrait; ++g) {
//                    final double ig = partials[ibo + g];
//                    final double jg = partials[jbo + g];
//                    final double kg = partials[kbo + g];
//
//                    for (int h = 0; h < dimTrait; ++h) {
//                        final double ih = partials[ibo + h];
//                        final double jh = partials[jbo + h];
//                        final double kh = partials[kbo + h];
//
//                        SSi += ig * Pip.unsafe_get(g, h) * ih;
//                        SSj += jg * Pjp.unsafe_get(g, h) * jh;
//                        SSk += kg * Pk .unsafe_get(g, h) * kh;
//                    }
//                }
//
//                double SS = SSi + SSj - SSk;

                double SS = weightedThreeInnerProduct(
                        partials, ibo, Pip,
                        partials, jbo, Pjp,
                        partials, kbo, Pk,
                        dimTrait);

//                    final DMatrixRMaj Vt = new DMatrixRMaj(dimTrait, dimTrait);
                final DMatrixRMaj Vt = matrix6;
                CommonOps_DDRM.add(Vip, Vjp, Vt);

                if (DEBUG) {
                    System.err.println("Vt: " + Vt);
                }

                int dimensionChange = ci.getEffectiveDimension() + cj.getEffectiveDimension()
                        - ck.getEffectiveDimension();

                remainder += -dimensionChange * LOG_SQRT_2_PI - 0.5 *
//                            (Math.log(CommonOps_DDRM.det(Vip)) + Math.log(CommonOps_DDRM.det(Vjp)) - Math.log(CommonOps_DDRM.det(Vk)))
                        (Math.log(ci.getDeterminant()) + Math.log(cj.getDeterminant()) + Math.log(ck.getDeterminant()))
                        - 0.5 * SS;

                // TODO Can get SSi + SSj - SSk from inner product w.r.t Pt (see outer-products below)?

                if (DEBUG) {
//                    System.err.println("\t\t\tSSi = " + (SSi));
//                    System.err.println("\t\t\tSSj = " + (SSj));
//                    System.err.println("\t\t\tSSk = " + (SSk));
                    System.err.println("\t\t\tSS = " + (SS));
                    System.err.println("\t\t\tdetI = " + Math.log(ci.getDeterminant()));
                    System.err.println("\t\t\tdetJ = " + Math.log(cj.getDeterminant()));
                    System.err.println("\t\t\tdetK = " + Math.log(ck.getDeterminant()));
                    System.err.println("\t\tremainder: " + remainder);
                }

                if (TIMING) {
                    endTime("remain");
                }


//                if (incrementOuterProducts) {
//
//                    assert (false);
//
//                    final DMatrixRMaj Pt = new DMatrixRMaj(dimTrait, dimTrait);
//                    safeInvert(Vt, Pt, false);
//
//                    int opo = dimTrait * dimTrait * trait;
////                    int opd = precisionOffset;
//
//                    for (int g = 0; g < dimTrait; ++g) {
//                        final double ig = partials[ibo + g];
//                        final double jg = partials[jbo + g];
//
//                        for (int h = 0; h < dimTrait; ++h) {
//                            final double ih = partials[ibo + h];
//                            final double jh = partials[jbo + h];
//
//                            outerProducts[opo] += (ig - jg) * (ih - jh)
////                                        * Pt.unsafe_get(g, h)
////                                        * Pk.unsafe_get(g, h)
//
////                                        / diffusions[opd];
//                                    // * pip * pjp / (pip + pjp);
//                                    * lPip * lPjp / (lPip + lPjp);
//                            ++opo;
////                            ++opd;
//                        }
//                    }
//
//                    if (DEBUG) {
//                        System.err.println("Outer-products:" + wrap(outerProducts, dimTrait * dimTrait * trait, dimTrait, dimTrait));
//                    }
//
//                    degreesOfFreedom[trait] += 1; // increment degrees-of-freedom
//                }
            } // End if remainder

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

    void reportMeansAndPrecisions(final int trait,
                                  final int ibo,
                                  final int jbo,
                                  final int kbo,
                                  final DMatrixRMaj Pi,
                                  final DMatrixRMaj Pj,
                                  final DMatrixRMaj Pk) {
        System.err.println("\ttrait: " + trait);
        System.err.println("Pi: " + Pi);
        System.err.println("Pj: " + Pj);
        System.err.println("Pk: " + Pk);
        System.err.print("\t\tmean i:");
        for (int e = 0; e < dimTrait; ++e) {
            System.err.print(" " + partials[ibo + e]);
        }
        System.err.print("\t\tmean j:");
        for (int e = 0; e < dimTrait; ++e) {
            System.err.print(" " + partials[jbo + e]);
        }
        System.err.print("\t\tmean k:");
        for (int e = 0; e < dimTrait; ++e) {
            System.err.print(" " + partials[kbo + e]);
        }
        System.err.println("");
    }

//    double weightedInnerProduct(final int ibo,
//                                final int jbo,
//                                final int kbo,
//                                final DMatrixRMaj Pip,
//                                final DMatrixRMaj Pjp,
//                                final DMatrixRMaj Pk) {
//
//        double SSi = 0;
//        double SSj = 0;
//        double SSk = 0;
//
//        // vector-matrix-vector TODO in parallel
//        for (int g = 0; g < dimTrait; ++g) {
//            final double ig = partials[ibo + g];
//            final double jg = partials[jbo + g];
//            final double kg = partials[kbo + g];
//
//            for (int h = 0; h < dimTrait; ++h) {
//                final double ih = partials[ibo + h];
//                final double jh = partials[jbo + h];
//                final double kh = partials[kbo + h];
//
//                SSi += ig * Pip.unsafe_get(g, h) * ih;
//                SSj += jg * Pjp.unsafe_get(g, h) * jh;
//                SSk += kg * Pk .unsafe_get(g, h) * kh;
//            }
//        }
//
//        return SSi + SSj - SSk;
//    }

    private final Map<String, Long> startTimes = new HashMap<String, Long>();

    void startTime(String key) {
        startTimes.put(key, System.nanoTime());
    }

    void endTime(String key) {
        long start = startTimes.get(key);

        Long total = times.get(key);
        if (total == null) {
            total = 0L;
        }

        long run = total + (System.nanoTime() - start);
        times.put(key, run);
    }

    @Override
    public void calculatePreOrderRoot(int priorBufferIndex, int rootNodeIndex) {

        super.calculatePreOrderRoot(priorBufferIndex, rootNodeIndex);

        final DMatrixRMaj Pd = wrap(diffusions, precisionOffset, dimTrait, dimTrait);
        final DMatrixRMaj Vd = wrap(inverseDiffusions, precisionOffset, dimTrait, dimTrait);

        int rootOffset = dimPartial * rootNodeIndex;

        // TODO For each trait in parallel
        for (int trait = 0; trait < numTraits; ++trait) {

            @SuppressWarnings("SpellCheckingInspection")
            final DMatrixRMaj Proot = wrap(preOrderPartials, rootOffset + dimTrait, dimTrait, dimTrait);
            @SuppressWarnings("SpellCheckingInspection")
            final DMatrixRMaj Vroot = wrap(preOrderPartials, rootOffset + dimTrait + dimTrait * dimTrait, dimTrait, dimTrait);

            // TODO Block below is for the conjugate prior ONLY
            {
                final DMatrixRMaj tmp = matrix0;

                CommonOps_DDRM.mult(Pd, Proot, tmp);
                unwrap(tmp, preOrderPartials, rootOffset + dimTrait);

                CommonOps_DDRM.mult(Vd, Vroot, tmp);
                unwrap(tmp, preOrderPartials, rootOffset + dimTrait + dimTrait * dimTrait);
            }
            rootOffset += dimPartialForTrait;
        }
    }

    @Override
    public void calculateRootLogLikelihood(int rootBufferIndex, int priorBufferIndex, final double[] logLikelihoods,
                                           boolean incrementOuterProducts, boolean isIntegratedProcess) {
        assert(logLikelihoods.length == numTraits);

        assert (!incrementOuterProducts);
        assert(!isIntegratedProcess);

        if (DEBUG) {
            System.err.println("Root calculation for " + rootBufferIndex);
            System.err.println("Prior buffer index is " + priorBufferIndex);
        }

        int rootOffset = dimPartial * rootBufferIndex;
        int priorOffset = dimPartial * priorBufferIndex;

        final DMatrixRMaj Vd = wrap(inverseDiffusions, precisionOffset, dimTrait, dimTrait);

        // TODO For each trait in parallel
        for (int trait = 0; trait < numTraits; ++trait) {

            final DMatrixRMaj PRoot = wrap(partials, rootOffset + dimTrait, dimTrait, dimTrait);
            final DMatrixRMaj PPrior = wrap(partials, priorOffset + dimTrait, dimTrait, dimTrait);

            final DMatrixRMaj VRoot = wrap(partials, rootOffset + dimTrait + dimTrait * dimTrait, dimTrait, dimTrait);
            final DMatrixRMaj VPrior = wrap(partials, priorOffset + dimTrait + dimTrait * dimTrait, dimTrait, dimTrait);

            // TODO Block below is for the conjugate prior ONLY
            {
                final DMatrixRMaj VTmp = new DMatrixRMaj(dimTrait, dimTrait);
                CommonOps_DDRM.mult(Vd, VPrior, VTmp);
                VPrior.set(VTmp);
            }

            final DMatrixRMaj VTotal = new DMatrixRMaj(dimTrait, dimTrait);
            CommonOps_DDRM.add(VRoot, VPrior, VTotal);

            final DMatrixRMaj PTotal = new DMatrixRMaj(dimTrait, dimTrait);
            CommonOps_DDRM.invert(VTotal, PTotal);  // TODO Can return determinant at same time to avoid extra QR decomposition

//            double SS = 0;
//            for (int g = 0; g < dimTrait; ++g) {
//                final double gDifference = partials[rootOffset + g] - partials[priorOffset + g];
//
//                for (int h = 0; h < dimTrait; ++h) {
//                    final double hDifference = partials[rootOffset + h] - partials[priorOffset + h];
//
//                    SS += gDifference * PTotal.unsafe_get(g, h) * hDifference;
//                }
//            }
            double SS = weightedInnerProductOfDifferences(
                    partials, rootOffset,
                    partials, priorOffset,
                    PTotal, dimTrait);

            final double logLike = -dimTrait * LOG_SQRT_2_PI - 0.5 * Math.log(CommonOps_DDRM.det(VTotal)) - 0.5 * SS;

            final double remainder = remainders[rootBufferIndex * numTraits + trait];
            logLikelihoods[trait] = logLike + remainder;

//            if (incrementOuterProducts) {
//
//                assert (false);
//
//                int opo = dimTrait * dimTrait * trait;
//                int opd = precisionOffset;
//
//                double rootScalar = partials[rootOffset + dimTrait + 2 * dimTrait * dimTrait];
//                final double priorScalar = partials[priorOffset + dimTrait];
//
//                if (!Double.isInfinite(priorScalar)) {
//                    rootScalar = rootScalar * priorScalar / (rootScalar + priorScalar);
//                }
//
//                for (int g = 0; g < dimTrait; ++g) {
//                    final double gDifference = partials[rootOffset + g] - partials[priorOffset + g];
//
//                    for (int h = 0; h < dimTrait; ++h) {
//                        final double hDifference = partials[rootOffset + h] - partials[priorOffset + h];
//
//                        outerProducts[opo] += gDifference * hDifference * rootScalar;
//                        ++opo;
//                        ++opd;
//                    }
//                }
//
//                degreesOfFreedom[trait] += 1; // increment degrees-of-freedom
//            }

            if (DEBUG) {
                System.err.print("mean:");
                for (int g = 0; g < dimTrait; ++g) {
                    System.err.print(" " + partials[rootOffset + g]);
                }
                System.err.println("");
                System.err.println("P  root: " + PRoot);
                System.err.println("V  root: " + VRoot);
                System.err.println("P prior: " + PPrior);
                System.err.println("V prior: " + VPrior);
                System.err.println("P total: " + PTotal);
                System.err.println("V total: " + VTotal);
                System.err.println("\t" + logLike + " " + (logLike + remainder));
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

    ///////////////////////////////////////////////////////////////////////////
    /// Derivation Functions
    ///////////////////////////////////////////////////////////////////////////

    public void getPrecisionPreOrderDerivative(BranchSufficientStatistics statistics, DMatrixRMaj gradient) {

        final DMatrixRMaj Pi = statistics.getParent().getRawPrecision();
        final DMatrixRMaj Vdi = statistics.getBranch().getRawVariance();

        DMatrixRMaj VdPi = matrix0;
        DMatrixRMaj temp = matrix1;

        CommonOps_DDRM.mult(Vdi, Pi, VdPi);
        CommonOps_DDRM.mult(gradient, VdPi, temp);
        CommonOps_DDRM.multTransA(VdPi, temp, gradient);
    }

    double[] inverseDiffusions;
}
