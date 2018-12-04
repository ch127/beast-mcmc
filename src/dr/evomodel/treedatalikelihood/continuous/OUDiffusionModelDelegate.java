/*
 * AbstractOUDiffusionModelDelegate.java
 *
 * Copyright (c) 2002-2016 Alexei Drummond, Andrew Rambaut and Marc Suchard
 *
 * This file is part of BEAST.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership and licensing.
 *
 * BEAST is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 *  BEAST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAST; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA  02110-1301  USA
 */

package dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.tree.Tree;
import dr.evomodel.branchratemodel.BranchRateModel;
import dr.evomodel.continuous.MultivariateDiffusionModel;
import dr.evomodel.continuous.MultivariateElasticModel;
import dr.evomodel.treedatalikelihood.continuous.cdi.ContinuousDiffusionIntegrator;
import dr.inference.model.Model;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import java.util.List;

import static dr.math.matrixAlgebra.missingData.MissingOps.wrap;

/**
 * A simple OU diffusion model delegate with branch-specific drift and constant diffusion
 *
 * @author Marc A. Suchard
 * @author Paul Bastide
 * @version $Id$
 */
public class OUDiffusionModelDelegate extends AbstractDriftDiffusionModelDelegate {

    // Here, branchRateModels represents optimal values

    private MultivariateElasticModel elasticModel;

    public OUDiffusionModelDelegate(Tree tree,
                                    MultivariateDiffusionModel diffusionModel,
                                    List<BranchRateModel> branchRateModels,
                                    MultivariateElasticModel elasticModel) {
        this(tree, diffusionModel, branchRateModels, elasticModel, 0);
    }

    private OUDiffusionModelDelegate(Tree tree,
                                     MultivariateDiffusionModel diffusionModel,
                                     List<BranchRateModel> branchRateModels,
                                     MultivariateElasticModel elasticModel,
                                     int partitionNumber) {
        super(tree, diffusionModel, branchRateModels, partitionNumber);
        this.elasticModel = elasticModel;
        addModel(elasticModel);
    }

    @Override
    public boolean hasDrift() {
        return true;
    }

    @Override
    public boolean hasActualization() {
        return true;
    }

    @Override
    public boolean hasDiagonalActualization() {
        return elasticModel.isDiagonal();
    }

    public boolean isSymmetric() {
        return elasticModel.isSymmetric();
    }

    public double[][] getStrengthOfSelection() {
        return elasticModel.getStrengthOfSelectionMatrix();
    }

    public double[] getEigenValuesStrengthOfSelection() {
        return elasticModel.getEigenValuesStrengthOfSelection();
    }

    public double[] getEigenVectorsStrengthOfSelection() {
        return elasticModel.getEigenVectorsStrengthOfSelection();
    }

    @Override
    public void setDiffusionModels(ContinuousDiffusionIntegrator cdi, boolean flip) {
        super.setDiffusionModels(cdi, flip);

        cdi.setDiffusionStationaryVariance(getEigenBufferOffsetIndex(0),
                getEigenValuesStrengthOfSelection(), getEigenVectorsStrengthOfSelection());
    }

    @Override
    public void updateDiffusionMatrices(ContinuousDiffusionIntegrator cdi, int[] branchIndices, double[] edgeLengths,
                                        int updateCount, boolean flip) {

        int[] probabilityIndices = new int[updateCount];

        for (int i = 0; i < updateCount; i++) {
            if (flip) {
                flipMatrixBufferOffset(branchIndices[i]);
            }
            probabilityIndices[i] = getMatrixBufferOffsetIndex(branchIndices[i]);
        }

        cdi.updateOrnsteinUhlenbeckDiffusionMatrices(
                getEigenBufferOffsetIndex(0),
                probabilityIndices,
                edgeLengths,
                getDriftRates(branchIndices, updateCount),
                getEigenValuesStrengthOfSelection(),
                getEigenVectorsStrengthOfSelection(),
                updateCount);
    }

    @Override
    public void getGradientPrecision(double scalar, DMatrixRMaj gradient) {
        throw new RuntimeException("not yet implemented");
    }

    @Override
    protected void handleModelChangedEvent(Model model, Object object, int index) {
        if (model == elasticModel) {
            fireModelChanged(model);
        } else {
            super.handleModelChangedEvent(model, object, index);
        }
    }

    @Override
    public double[][] getJointVariance(final double priorSampleSize,
                                       final double[][] treeVariance, final double[][] treeSharedLengths,
                                       final double[][] traitVariance) {
        if (hasDiagonalActualization()) {
            return getJointVarianceDiagonal(priorSampleSize, treeVariance, treeSharedLengths, traitVariance);
        }
        return getJointVarianceFull(priorSampleSize, treeVariance, treeSharedLengths, traitVariance);
    }

    private double[][] getJointVarianceFull(final double priorSampleSize,
                                            final double[][] treeVariance, final double[][] treeSharedLengths,
                                            final double[][] traitVariance) {

        double[] eigVals = this.getEigenValuesStrengthOfSelection();
        DMatrixRMaj V = wrap(this.getEigenVectorsStrengthOfSelection(), 0, dim, dim);
        DMatrixRMaj Vinv = new DMatrixRMaj(dim, dim);
        CommonOps_DDRM.invert(V, Vinv);

        DMatrixRMaj transTraitVariance = new DMatrixRMaj(traitVariance);

        DMatrixRMaj tmp = new DMatrixRMaj(dim, dim);
        CommonOps_DDRM.mult(Vinv, transTraitVariance, tmp);
        CommonOps_DDRM.multTransB(tmp, Vinv, transTraitVariance);

        // inverse of eigenvalues
        double[][] invEigVals = new double[dim][dim];
        for (int p = 0; p < dim; ++p) {
            for (int q = 0; q < dim; ++q) {
                invEigVals[p][q] = 1 / (eigVals[p] + eigVals[q]);
            }
        }

        // Computation of matrix
        int ntaxa = tree.getExternalNodeCount();
        double ti;
        double tj;
        double tij;
        double ep;
        double eq;
        DMatrixRMaj varTemp = new DMatrixRMaj(dim, dim);
        double[][] jointVariance = new double[dim * ntaxa][dim * ntaxa];
        for (int i = 0; i < ntaxa; ++i) {
            for (int j = 0; j < ntaxa; ++j) {
                ti = treeSharedLengths[i][i];
                tj = treeSharedLengths[j][j];
                tij = treeSharedLengths[i][j];
                for (int p = 0; p < dim; ++p) {
                    for (int q = 0; q < dim; ++q) {
                        ep = eigVals[p];
                        eq = eigVals[q];
                        varTemp.set(p, q, Math.exp(-ep * ti) * Math.exp(-eq * tj) * (invEigVals[p][q] * (Math.exp((ep + eq) * tij) - 1) + 1 / priorSampleSize) * transTraitVariance.get(p, q));
                    }
                }
                CommonOps_DDRM.mult(V, varTemp, tmp);
                CommonOps_DDRM.multTransB(tmp, V, varTemp);
                for (int p = 0; p < dim; ++p) {
                    for (int q = 0; q < dim; ++q) {
                        jointVariance[i * dim + p][j * dim + q] = varTemp.get(p, q);
                    }
                }
            }
        }
        return jointVariance;
    }

    private double[][] getJointVarianceDiagonal(final double priorSampleSize,
                                                final double[][] treeVariance, final double[][] treeSharedLengths,
                                                final double[][] traitVariance) {

        // Eigen of strength of selection matrix
        double[] eigVals = this.getEigenValuesStrengthOfSelection();
        int ntaxa = tree.getExternalNodeCount();
        double ti;
        double tj;
        double tij;
        double ep;
        double eq;
        double var;
        DMatrixRMaj varTemp = new DMatrixRMaj(dim, dim);
        double[][] jointVariance = new double[dim * ntaxa][dim * ntaxa];
        for (int i = 0; i < ntaxa; ++i) {
            for (int j = 0; j < ntaxa; ++j) {
                ti = treeSharedLengths[i][i];
                tj = treeSharedLengths[j][j];
                tij = treeSharedLengths[i][j];
                for (int p = 0; p < dim; ++p) {
                    for (int q = 0; q < dim; ++q) {
                        ep = eigVals[p];
                        eq = eigVals[q];
                        if (ep + eq == 0.0) {
                            var = (tij + 1 / priorSampleSize) * traitVariance[p][q];
                        } else {
                            var = Math.exp(-ep * ti) * Math.exp(-eq * tj) * ((Math.exp((ep + eq) * tij) - 1) / (ep + eq) + 1 / priorSampleSize) * traitVariance[p][q];
                        }
                        varTemp.set(p, q, var);
                    }
                }
                for (int p = 0; p < dim; ++p) {
                    for (int q = 0; q < dim; ++q) {
                        jointVariance[i * dim + p][j * dim + q] = varTemp.get(p, q);
                    }
                }
            }
        }
        return jointVariance;
    }
}