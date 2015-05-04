package dr.app.beagle.evomodel.branchmodel.lineagespecific;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import beagle.Beagle;
import beagle.BeagleFactory;

import dr.app.beagle.evomodel.branchmodel.BranchModel;
import dr.app.beagle.evomodel.branchmodel.EpochBranchModel;
import dr.app.beagle.evomodel.branchmodel.HomogeneousBranchModel;
import dr.app.beagle.evomodel.sitemodel.GammaSiteRateModel;
import dr.app.beagle.evomodel.sitemodel.SiteRateModel;
import dr.app.beagle.evomodel.substmodel.FrequencyModel;
import dr.app.beagle.evomodel.substmodel.HKY;
import dr.app.beagle.evomodel.substmodel.SubstitutionModel;
import dr.app.beagle.evomodel.treelikelihood.BeagleTreeLikelihood;
import dr.app.beagle.evomodel.treelikelihood.BufferIndexHelper;
import dr.app.beagle.evomodel.treelikelihood.PartialsRescalingScheme;
import dr.app.beagle.evomodel.treelikelihood.SubstitutionModelDelegate;
import dr.app.beagle.tools.BeagleSequenceSimulator;
import dr.app.beagle.tools.Partition;
import dr.evolution.alignment.Alignment;
import dr.evolution.alignment.PatternList;
import dr.evolution.datatype.DataType;
import dr.evolution.datatype.Nucleotides;
import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evomodel.branchratemodel.BranchRateModel;
import dr.evomodel.branchratemodel.CountableBranchCategoryProvider;
import dr.evomodel.branchratemodel.StrictClockBranchRates;
import dr.evomodel.tree.TreeModel;
import dr.inference.loggers.LogColumn;
import dr.inference.loggers.NumberColumn;
import dr.inference.model.CompoundLikelihood;
import dr.inference.model.CompoundModel;
import dr.inference.model.Likelihood;
import dr.inference.model.Model;
import dr.inference.model.Parameter;
import dr.inference.model.Likelihood.Abstract;
import dr.math.MathUtils;

@SuppressWarnings("serial")
public class BeagleBranchLikelihood implements Likelihood {

	// Constructor fields
	private PatternList patternList;
	private TreeModel treeModel;
	private BranchModel branchModel;
	private SiteRateModel siteRateModel;
	private FrequencyModel freqModel;
	private BranchRateModel branchRateModel;

	// Likelihood fields
	private String id = null;
	private boolean used = true;

	
	// Beagle fields
	private Beagle beagle;

	
	
	
	public BeagleBranchLikelihood(
			PatternList patternList, //
			TreeModel treeModel, //
			BranchModel branchModel, //
			SiteRateModel siteRateModel,
			FrequencyModel freqModel, //
			BranchRateModel branchRateModel //
			) {
		
		
		this.patternList = patternList;
		this.treeModel = treeModel;
		this.branchModel = branchModel;
		this.siteRateModel = siteRateModel;
		this.freqModel = freqModel;
		this.branchRateModel = branchRateModel;
		
		this.loadBeagleInstance();
		
//		CompoundLikelihood cl =	(CompoundLikelihood)likelihoods.get(0);
//		cl.
		
		
	}//END: Constructor
	
	
	// //////////////
	// ---PUBLIC---//
	// //////////////
	
	@Override
	public double getLogLikelihood() {
		double loglikelihood = 0;
		
		
		
		
		
		
		
		
		
		return loglikelihood;
	}//END: getLogLikelihood

	
	public double getBranchLoglikelihood(int i) {
		double loglikelihood = 0;
		
		
		
		
		
		
		
		
		int[] parentBufferIndices = null;
		int[] childBufferIndices = null;
		int[] probabilityIndices = null;
		int[] categoryWeightsIndices = null;
		int[] stateFrequenciesIndices = null;
		int[] cumulativeScaleIndices = null;
		int count = 1;
		double[] outSumLogLikelihood = null;		
		
		beagle.calculateEdgeLogLikelihoods(
				parentBufferIndices, //
				childBufferIndices, // 
				probabilityIndices, // 
				null, // int[] firstDerivativeIndices
				null, // int[] secondDerivativeIndices 
				categoryWeightsIndices, // 
				stateFrequenciesIndices, // 
				cumulativeScaleIndices, // 
				count, // 
				outSumLogLikelihood, // 
				null, // int[] outSumFirstDerivative, // 
				null //int[]  outSumSecondDerivative //
		);
		
		
		return loglikelihood;
	}//END: getBranchLoglikelihood
	
	// ///////////////
	// ---PRIVATE---//
	// ///////////////

	private void loadBeagleInstance() {

		SubstitutionModelDelegate	substitutionModelDelegate = new SubstitutionModelDelegate(treeModel,
				branchModel);
		
		DataType dataType = freqModel.getDataType();
		
		int partitionSiteCount = patternList.getPatternCount();
		
		int nodeCount = treeModel.getNodeCount();
		BufferIndexHelper matrixBufferHelper = new BufferIndexHelper(nodeCount, 0);

		int tipCount = treeModel.getExternalNodeCount();
		int internalNodeCount = treeModel.getInternalNodeCount();

		BufferIndexHelper partialBufferHelper = new BufferIndexHelper(nodeCount, tipCount);
		BufferIndexHelper scaleBufferHelper = new BufferIndexHelper(internalNodeCount + 1, 0);

		int compactPartialsCount = tipCount;
		int stateCount = dataType.getStateCount();
		int patternCount = partitionSiteCount;
		int siteRateCategoryCount = siteRateModel.getCategoryCount();
		
		int[] resourceList = new int[] { 0 };
		long preferenceFlags = 0;
		long requirementFlags = 0;

		beagle = BeagleFactory.loadBeagleInstance(tipCount, //
				partialBufferHelper.getBufferCount(), //
				compactPartialsCount, //
				stateCount, //
				patternCount, //
				substitutionModelDelegate.getEigenBufferCount(), //
				substitutionModelDelegate.getMatrixBufferCount(), //
				siteRateCategoryCount, //
				scaleBufferHelper.getBufferCount(), //
				resourceList, //
				preferenceFlags, //
				requirementFlags);
		
	}// END: loadBeagleInstance
	
	// /////////////////
	// ---INHERITED---//
	// /////////////////

	@Override
	public LogColumn[] getColumns() {
		return new dr.inference.loggers.LogColumn[] { new LikelihoodColumn(
				getId() == null ? "likelihood" : getId()) };
	}

	@Override
	public String getId() {
		return this.id;
	}

	@Override
	public void setId(String id) {
		this.id = id;
	}

	@Override
	public Model getModel() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void makeDirty() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String prettyName() {
		return Abstract.getPrettyName(this);
	}

	@Override
	public boolean isUsed() {
		return used;
	}

	@Override
	public void setUsed() {
		used = true;
	}

	@Override
	public boolean evaluateEarly() {
		return false;
	}

	// ///////////////////////
	// ---PRIVATE CLASSES---//
	// ///////////////////////

	private class LikelihoodColumn extends NumberColumn {

		public LikelihoodColumn(String label) {
			super(label);
		}// END: Constructor

		public double getDoubleValue() {
			return getLogLikelihood();
		}
		
	}// END: LikelihoodColumn class

	// ////////////
	// ---TEST---//
	// ////////////
	
	  public static void main(String[] args) {
		
		  try {
		  
          MathUtils.setSeed(666);

          int sequenceLength = 1000;
          ArrayList<Partition> partitionsList = new ArrayList<Partition>();

          // create tree
          NewickImporter importer = new NewickImporter(
                  "((SimSeq1:22.0,SimSeq2:22.0):12.0,(SimSeq3:23.1,SimSeq4:23.1):10.899999999999999);");
          Tree tree = importer.importTree(null);
          TreeModel treeModel = new TreeModel(tree);

          // create Frequency Model
          Parameter freqs = new Parameter.Default(new double[]{0.25, 0.25,
                  0.25, 0.25});
          FrequencyModel freqModel = new FrequencyModel(Nucleotides.INSTANCE,
                  freqs);

          // create branch model
          Parameter kappa1 = new Parameter.Default(1, 1);

          HKY hky1 = new HKY(kappa1, freqModel);

        BranchModel homogeneousBranchModel = new HomogeneousBranchModel(hky1);

          List<SubstitutionModel> substitutionModels = new ArrayList<SubstitutionModel>();
          substitutionModels.add(hky1);
          List<FrequencyModel> freqModels = new ArrayList<FrequencyModel>();
          freqModels.add(freqModel);

          // create branch rate model
          Parameter rate = new Parameter.Default(1, 0.001);
          BranchRateModel branchRateModel = new StrictClockBranchRates(rate);

          // create site model
          GammaSiteRateModel siteRateModel = new GammaSiteRateModel(
                  "siteModel");


          // create partition
          Partition partition1 = new Partition(treeModel, //
        		  homogeneousBranchModel,//
                  siteRateModel, //
                  branchRateModel, //
                  freqModel, //
                  0, // from
                  sequenceLength - 1, // to
                  1 // every
          );

          partitionsList.add(partition1);

          // feed to sequence simulator and generate data
          BeagleSequenceSimulator simulator = new BeagleSequenceSimulator(partitionsList
//          		, sequenceLength
          );
          
          Alignment alignment = simulator.simulate(false, false);
		  
		  
          BeagleTreeLikelihood nbtl = new BeagleTreeLikelihood(alignment, treeModel, homogeneousBranchModel, siteRateModel, branchRateModel, null, false, PartialsRescalingScheme.DEFAULT);

          System.out.println("BTL(homogeneous) = " + nbtl.getLogLikelihood());
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
	        } catch (Exception e) {
	            e.printStackTrace();
	            System.exit(-1);
	        } // END: try-catch block
		  
		  
	  }//END: main
	
}// END: class
