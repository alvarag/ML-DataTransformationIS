/*
 * MLeNN.java
 * Copyright (C) 2016 Burgos University, Burgos, Spain 
 * @author Álvar Arnaiz-González
 *     
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *     
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *     
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package meka.filters.multilabel;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;
import java.util.Vector;

import meka.core.F;
import meka.core.MLUtils;
import mulan.data.MultiLabelInstances;

import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.filters.SimpleBatchFilter;

/**
 * Wrapper for the original code of Heuristic ML undersampling. 
 * Presented by: Charte, F., Rivera, A. J., del Jesus, M. J., & 
 * Herrera, F. (2014, September). MLeNN: a first approach to
 * heuristic multilabel undersampling. In International Conference on
 * Intelligent Data Engineering and Automated Learning (pp. 1-9). 
 * Springer International Publishing.
 * <p>
 * Original code: <url>http://simidat.ujaen.es/papers/mlenn</url>
 * <p>
 * Valid options are:
 * <p>
 * hamming threshold <br>
 * delta value <br>
 * number of nearest neighbours <br>
 * 
 * @author Álvar Arnaiz-González
 * @version 20161006
 */
public class MLeNN extends SimpleBatchFilter {

	private static final long serialVersionUID = -223478926896879027L;

	/**
	 * Number of nearest neighbours.
	 */
	protected int m_K = 3;

	/**
	 * Hamming threshold.
	 */
	protected float m_HammingThreshold = 0.75f;

	/**
	 * Delta.
	 */
	protected double m_Delta = 0.5;

	public void setK(int k) {
		m_K = k;
	}

	public int getK() {

		return m_K;
	}

	public void setThreshold(float t) {
		m_HammingThreshold = t;
	}

	public float getThreshold() {

		return m_HammingThreshold;
	}

	public void setDelta(double d) {
		m_Delta = d;
	}

	public double getDelta() {

		return m_Delta;
	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {
		ArrayList<Option> options = new ArrayList<Option>(3);

		options.add(new Option("\tNumber of nearest neighbour.", "K", 0, "-K"));

		options.add(new Option("\tHamming threshold.", "H", 0, "-H"));

		options.add(new Option("\tDelta.", "D", 0, "-D"));

		Enumeration<Option> enu = super.listOptions();

		while (enu.hasMoreElements())
			options.add(enu.nextElement());

		return Collections.enumeration(options);
	}

	/**
	 * Parses a given list of options.
	 * 
	 * @param options
	 *            the list of options as an array of strings.
	 * @throws Exception
	 *             if an option is not supported.
	 */
	public void setOptions(String[] options) throws Exception {
		String tmpStr;

		tmpStr = Utils.getOption('K', options);

		if (tmpStr.length() > 0)
			setK(Integer.parseInt(tmpStr));
		else
			setK(3);

		tmpStr = Utils.getOption('H', options);

		if (tmpStr.length() > 0)
			setThreshold(Float.parseFloat(tmpStr));
		else
			setThreshold(0.75f);

		tmpStr = Utils.getOption('D', options);

		if (tmpStr.length() > 0)
			setDelta(Double.parseDouble(tmpStr));
		else
			setDelta(0.5);

		super.setOptions(options);
	}

	/**
	 * Gets the current settings of the Classifier.
	 * 
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		String[] options = super.getOptions();

		for (int i = 0; i < options.length; i++)
			result.add(options[i]);

		result.add("-K");
		result.add("" + getK());

		result.add("-H");
		result.add("" + getThreshold());

		result.add("-D");
		result.add("" + getDelta());

		return result.toArray(new String[result.size()]);
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {

		return new Instances(inputFormat, 0);
	}

	@Override
	public String globalInfo() {

		return "MLeNN: A First Approach to Heuristic Multilabel Undersampling.";
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		Instances instancesTemplate, filtered,
		            result = new Instances (instances, instances.numInstances());
		MLSBag myBag;
		
		// Only change first batch of data
		if (isFirstBatchDone())
			return new Instances(instances);
		
		Random r = instances.getRandomNumberGenerator(0);
		String name = "temp_" + MLUtils.getDatasetName(instances) + "_" + r.nextLong() + ".arff";
		
		if (m_Debug)
			System.out.println("Using temporary file: "+name);
		
		int L = instances.classIndex();

		// rename attributes, because MULAN doesn't deal well with hypens etc
		for(int i = L; i < instances.numAttributes(); i++) {
			instances.renameAttribute(i,"a_"+i);
		}
		
		BufferedWriter writer = new BufferedWriter(new FileWriter(name));
		instancesTemplate = F.meka2mulan(new Instances(instances), L);
		writer.write(instancesTemplate.toString());
		writer.flush();
		writer.close();
		MultiLabelInstances train = new MultiLabelInstances(name, L);
		
		try {
			new File(name).delete();
		} catch(Exception e) {
			System.err.println("[Error] Failed to delete temporary file: "+ name +
			                     ". You may want to delete it manually.");
		}

		instancesTemplate = new Instances(train.getDataSet(),0);
		
		myBag = new MLSBag(train, m_Delta);
		new MlEnn(myBag, m_HammingThreshold, m_K).applyMethod();
		
		// Convert from mulan to meka.
		filtered = F.mulan2meka(myBag.getMlDS().getDataSet(), L);
		
		for (int i = 0; i < filtered.numInstances(); i++) 
			result.add(filtered.instance(i));
		
		return result;
	}

	/**
	 * @author Chartre F.
	 */
	private class MLSBag {
		private MultiLabelInstances mlDS;
		private Double parameterDelta;
		private List<Integer> majorClasses;
		private List<Integer> minorClasses;
		private double[] labelFrequencies;
		private HashMap<Double, Integer> freqIndex;
		private double mean;
		private double std;
		private double meanIR;
		private double[] labelIR;
		private int maxCounter;
		private int minCounter;

		public MultiLabelInstances getMlDS() {
			return this.mlDS;
		}

		public void setMlDS(MultiLabelInstances mlDS) {
			this.mlDS = mlDS;
		}

		public int getMaxCounter() {
			return this.maxCounter;
		}

		public double getMeanIR() {
			return this.meanIR;
		}

//		public Map<String, Object> getParameter() {
//			return this.parameterDelta;
//		}

		public double[] getLabelFreq() {
			return this.labelFrequencies;
		}

		public HashMap<Double, Integer> getFreqIndex() {
			return this.freqIndex;
		}

		public double getMean() {
			return this.mean;
		}

		public double getStd() {
			return this.std;
		}

		public List<Integer> getMajorClasses() {
			return this.majorClasses;
		}

		public List<Integer> getMinorClasses() {
			return this.minorClasses;
		}

		public MLSBag(MultiLabelInstances mlDS, double delta) {
			this.mlDS = mlDS;
			this.parameterDelta = delta;
			getLabelFrequencies();
		}

		public final void getLabelFrequencies() {
			int nLabels = this.mlDS.getNumLabels();
			int[] counters = new int[nLabels];
			int numInstances = this.mlDS.getNumInstances();
			this.labelFrequencies = new double[nLabels];

			counters = getCounters(this.mlDS.getDataSet());

			double sum = 0.0D;
			double squaredSum = 0.0D;

			this.freqIndex = new HashMap();
			for (int index = 0; index < nLabels; index++) {
				this.labelFrequencies[index] = (counters[index] * 100.0F / numInstances);

				this.freqIndex.put(
						Double.valueOf(this.labelFrequencies[index]),
						Integer.valueOf(index));

				sum += this.labelFrequencies[index];
				squaredSum += this.labelFrequencies[index]
						* this.labelFrequencies[index];
			}
			this.mean = (sum / nLabels);
			this.std = Math.sqrt(squaredSum / nLabels - this.mean * this.mean);

//			getMayLabels(MLeNN.MLSUtils.uniformIRSet());
//			getMinLabels(MLeNN.MLSUtils.uniformIRSet());
			getMayLabels(true);
			getMinLabels(true);
		}

		public List<Integer> getMayLabels() {
			List<Integer> indexes = new ArrayList();
			if (((Double) this.parameterDelta.doubleValue()) < 0.0D) {
				this.parameterDelta = Double.valueOf(0.5D);
			}
			
			double limit = this.mean
					+ ((Double) this.parameterDelta.doubleValue())
					* this.std;
			for (int index = 0; index < this.labelFrequencies.length; index++) {
				if (this.labelFrequencies[index] >= limit) {
					indexes.add(Integer.valueOf(index));
				}
			}
			this.majorClasses = indexes;
			return indexes;
		}

		public List<Integer> getMayLabels(boolean IR) {
			if (!IR) {
				return getMayLabels();
			}
			List<Integer> indexes = new ArrayList();
			for (int index = 0; index < this.labelFrequencies.length; index++) {
				if (this.labelIR[index] < this.meanIR) {
					indexes.add(Integer.valueOf(index));
				}
			}
			this.majorClasses = indexes;
			return indexes;
		}

		public List<Integer> getMinLabels(boolean IR) {
			List<Integer> indexes = new ArrayList();
			for (int index = 0; index < this.labelFrequencies.length; index++) {
				if (this.labelIR[index] > this.meanIR) {
					indexes.add(Integer.valueOf(index));
				}
			}
			this.minorClasses = indexes;
			return indexes;
		}

		public boolean isMinority(Instance aInstance) {
			return isInGroup(aInstance, this.minorClasses);
		}

		public boolean isMajority(Instance aInstance) {
			return isInGroup(aInstance, this.majorClasses);
		}

		public boolean isInGroup(Instance aInstance, List<Integer> aGroup) {
			for (Integer index : aGroup) {
				if (aInstance.value(this.mlDS.getLabelIndices()[index
						.intValue()]) == 1.0D) {
					return true;
				}
			}
			return false;
		}

		public Integer[] getLabelsUnderMean(double[] frequencies) {
			double sum = 0.0D;

			List<Integer> listIndexes = new ArrayList();
			for (double f : frequencies) {
				sum += f;
			}
			long mean = Math.round(sum / frequencies.length);
			for (int index = 0; index < frequencies.length; index++) {
				if (Math.round(frequencies[index]) < mean) {
					listIndexes.add(Integer.valueOf(index));
				}
			}
			return (Integer[]) listIndexes.toArray(new Integer[listIndexes
					.size()]);
		}

		public int[] getCounters(Instances ds) {
			int[] counters = new int[this.mlDS.getNumLabels()];
			for (Instance aInstance : ds) {
				addCounters(aInstance, counters);
			}
			this.maxCounter = (this.minCounter = counters[0]);
			for (int c : counters) {
				this.maxCounter = (c > this.maxCounter ? c : this.maxCounter);
				this.minCounter = (c < this.minCounter ? c : this.minCounter);
			}
			double sumIRs = 0.0D;
			this.labelIR = new double[this.mlDS.getNumLabels()];
			double minIR;
			double maxIR = minIR = counters[0] > 0 ? this.maxCounter
					/ counters[0] : 0.0D;
			for (int ix = 0; ix < this.mlDS.getNumLabels(); ix++) {
				if (counters[ix] > 0) {
					this.labelIR[ix] = (this.maxCounter / counters[ix]);
					sumIRs += this.labelIR[ix];

					maxIR = this.labelIR[ix] > maxIR ? this.labelIR[ix] : maxIR;
					minIR = this.labelIR[ix] < minIR ? this.labelIR[ix] : minIR;
				} else {
					this.labelIR[ix] = 1.0D;
				}
			}
			this.meanIR = (sumIRs / this.mlDS.getNumLabels());

			sumIRs = 0.0D;
			for (int ix = 0; ix < this.mlDS.getNumLabels(); ix++) {
				sumIRs += (this.labelIR[ix] - this.meanIR)
						* (this.labelIR[ix] - this.meanIR);
			}
			sumIRs /= (this.mlDS.getNumLabels() - 1);

			return counters;
		}

		public void addCounters(Instance aInstance, int[] counters) {
			int[] labelIndices = this.mlDS.getLabelIndices();
			for (int index = 0; index < labelIndices.length; index++) {
				int tmp21_19 = index;
				int[] tmp21_18 = counters;
				tmp21_18[tmp21_19] = ((int) (tmp21_18[tmp21_19] + aInstance
						.value(labelIndices[index])));
			}
		}

		public double[] getLabelFrequencies(int[] counters, int numInstances) {
			int nLabels = this.mlDS.getNumLabels();
			double[] frequencies = new double[nLabels];
			for (int index = 0; index < nLabels; index++) {
				frequencies[index] = (counters[index] * 100.0F / numInstances);
			}
			return frequencies;
		}

		public Instances getMinBag() {
			Instances imlDS = this.mlDS.getDataSet();
			Instances bag = new Instances(this.mlDS.clone().getDataSet());
			bag.delete();
			for (int index = 0; index < imlDS.numInstances(); index++) {
				Instance aInstance = imlDS.get(index);
				if (isMinority(aInstance)) {
					bag.add(aInstance);
					imlDS.delete(index);
					index--;
				}
			}
			return bag;
		}

		public Instances getBagOfLabel(int labelIndex) {
			Instances imlDS = this.mlDS.getDataSet();
			Instances bag = new Instances(this.mlDS.clone().getDataSet());
			bag.delete();
			for (int index = 0; index < imlDS.numInstances(); index++) {
				Instance aInstance = imlDS.get(index);
				if (aInstance.value(this.mlDS.getLabelIndices()[labelIndex]) == 1.0D) {
					bag.add(aInstance);
					imlDS.delete(index);
					index--;
				}
			}
			return bag;
		}

		public int getNumInstancesOfLabel(int labelIndex) {
			int[] counters = getCounters(this.mlDS.getDataSet());

			return counters[labelIndex];
		}

		public int getMeanInstancesPerLabel() {
			int[] counters = getCounters(this.mlDS.getDataSet());

			double sum = 0.0D;
			for (int count : counters) {
				sum += count;
			}
			return (int) (sum / (counters.length - this.majorClasses.size()));
		}

		public int[] getMayLabels(double[] labelFrequencies) {
			return getLabels(labelFrequencies, true);
		}

		public int[] getMinLabels(double[] labelFrequencies) {
			return getLabels(labelFrequencies, false);
		}

		private int[] getLabels(double[] labelFrequencies, boolean reverse) {
			int numLabels = labelFrequencies.length * 5.0D / 100.0D < 1.0D ? 1
			                 : (int) (labelFrequencies.length * 5.0D / 100.0D);
			int[] labelIndices = new int[numLabels];
			ArrayList<Double> freqClone = new ArrayList();
			for (int index = 0; index < labelFrequencies.length; index++) {
				freqClone.add(Double.valueOf(labelFrequencies[index]));
			}
			Collections.sort(freqClone, Collections.reverseOrder());
			if (!reverse) {
				Collections.reverse(freqClone);
			}
			Double[] temp = (Double[]) freqClone.toArray(new Double[freqClone.size()]);
			for (int index = 0; index < numLabels; index++) {
				double element = temp[index].doubleValue();
				for (int index2 = 0; index2 < freqClone.size(); index2++) {
					if (labelFrequencies[index2] == element) {
						labelIndices[index] = index2;
					}
				}
			}
			return labelIndices;
		}

		public int[] getOrderedMajorityIndexes() {
			Map<Double, Integer> freqIndex = getFreqIndex();
			TreeSet<Double> keys = new TreeSet(freqIndex.keySet());
			int[] indexes = new int[getMajorClasses().size()];
			int index = 0;
			for (Double key : keys) {
				if (getMajorClasses().contains(freqIndex.get(key))) {
					indexes[(index++)] = ((Integer) freqIndex.get(key))
							.intValue();
				}
			}
			return indexes;
		}
	}

	/**
	 * @author Chartre F.
	 */
	private class MlEnn {
		private MLeNN.MLSBag myBag;
		private Instances mliDS;
		protected int nearestNeighbors = 3;
		protected int minDiffs = 2;
		protected double distanceThreshold = 0.75D;

		public MlEnn(MLeNN.MLSBag bag, double HT, int nearestNeighbors) {
			this.myBag = bag;
			this.mliDS = this.myBag.getMlDS().getDataSet();
			this.nearestNeighbors = nearestNeighbors;
			this.distanceThreshold = HT;
			this.minDiffs = (nearestNeighbors / 2 + 1);
		}

		public int getNearestNeighbors() {
			return this.nearestNeighbors;
		}

		public void setNearestNeighbors(int nearestNeighbors) {
			this.nearestNeighbors = nearestNeighbors;
		}

		public void applyMethod() {
			List<Instance> instanceToDelete = new ArrayList();

			double[] hammingDistance = new double[0];
			int minorityInstances = 0;
			int origInstances = this.mliDS.numInstances();
			for (int indexOfInstance = 0; indexOfInstance < this.mliDS
					.numInstances(); indexOfInstance++) {
				if (!this.myBag.isMinority(this.mliDS.get(indexOfInstance))) {
					int[] NN = getNN(indexOfInstance);
					hammingDistance = getHammingDistance(NN, indexOfInstance);

					int numDiffs = 0;
					for (int i = 0; i < hammingDistance.length; i++) {
						if (hammingDistance[i] > this.distanceThreshold) {
							numDiffs++;
						}
					}
					if (numDiffs > this.minDiffs) {
						instanceToDelete.add(this.mliDS.get(indexOfInstance));
						this.mliDS.remove(this.mliDS.get(indexOfInstance--));
					}
				} else {
					minorityInstances++;
				}
			}

//			System.out.println("Samples to delete:" + instanceToDelete.size()
//					+ "/" + origInstances + " (" + minorityInstances + ")");
		}

		private double[] getHammingDistance(int[] NN, int indexOfInstance) {
			double[] distance = new double[getNearestNeighbors()];
			for (int i = 0; i < NN.length; i++) {
				distance[i] = getHammingDistance(this.mliDS.get(i),
						this.mliDS.get(indexOfInstance));
			}
			return distance;
		}

		private double getHammingDistance(Instance instance1, Instance instance2) {
			int numLabels = this.myBag.getMlDS().getNumLabels();
			int[] labelSet1 = getLabelSet(instance1);
			int[] labelSet2 = getLabelSet(instance2);
			double diff = 0.0D;
			double activeLabels = 0.0D;
			for (int i = 0; i < numLabels; i++) {
				if (labelSet1[i] != labelSet2[i]) {
					diff += 1.0D;
				}
				activeLabels += labelSet1[i] + labelSet2[i];
			}
			return diff / activeLabels;
		}

		private int[] getLabelSet(Instance aInstance) {
			int numLabels = this.myBag.getMlDS().getNumLabels();
			int[] labelIndices = this.myBag.getMlDS().getLabelIndices();
			for (int i = 0; i < numLabels; i++) {
				labelIndices[i] = (aInstance.stringValue(labelIndices[i])
						.equals("1") ? 1 : 0);
			}
			return labelIndices;
		}

		private int[] getNN(int indexOfInstance) {
			EuclideanDistance distance = new EuclideanDistance(this.mliDS);
			int[] NN = new int[getNearestNeighbors()];
			List distances = new LinkedList();
			for (int i = 0; i < this.mliDS.numInstances(); i++) {
				if (i != indexOfInstance) {
					double aDistance = distance.distance(this.mliDS.get(i),
							this.mliDS.get(indexOfInstance));
					distances.add(new Object[] { Double.valueOf(aDistance),
							Integer.valueOf(i) });
				}
			}
			Collections.sort(distances, new Comparator() {
				public int compare(Object o1, Object o2) {
					double distance1 = ((Double) ((Object[]) (Object[]) o1)[0])
							.doubleValue();
					double distance2 = ((Double) ((Object[]) (Object[]) o2)[0])
							.doubleValue();
					return Double.compare(distance1, distance2);
				}
			});
			
			if (distances.size() == 0)
				System.err.println ("ERROR!");
			
			for (int i = 0; i < getNearestNeighbors(); i++) {
				NN[i] = ((Integer) ((Object[]) (Object[]) distances.get(i))[1])
						.intValue();
			}
			return NN;
		}

		private int[] getVdmNN(int indexOfInstance) {
			List distanceToInstance = new LinkedList();
			int[] NN = new int[getNearestNeighbors()];

			Instance instanceI = this.mliDS.instance(indexOfInstance);

			int index = getRandomActiveLabelFrom(instanceI);
			Map vdmMap = getVdmMap(index);
			for (int j = 0; j < this.mliDS.numInstances(); j++) {
				Instance instanceJ = this.mliDS.instance(j);
				if (indexOfInstance != j) {
					double distance = 0.0D;
					Enumeration attrEnum = this.mliDS.enumerateAttributes();
					while (attrEnum.hasMoreElements()) {
						Attribute attr = (Attribute) attrEnum.nextElement();
						if (attr.index() < this.myBag.getMlDS()
								.getLabelIndices()[0]) {
							double iVal = instanceI.value(attr);
							double jVal = instanceJ.value(attr);
							if (attr.isNumeric()) {
								distance += Math.pow(iVal - jVal, 2.0D);
							} else {
								distance += ((double[][]) (double[][]) vdmMap
										.get(attr))[((int) iVal)][((int) jVal)];
							}
						}
					}
					distance = Math.pow(distance, 0.5D);
					distanceToInstance.add(new Object[] {
							Double.valueOf(distance), Integer.valueOf(j) });
				}
			}
			Collections.sort(distanceToInstance, new Comparator() {
				public int compare(Object o1, Object o2) {
					double distance1 = ((Double) ((Object[]) (Object[]) o1)[0])
							.doubleValue();
					double distance2 = ((Double) ((Object[]) (Object[]) o2)[0])
							.doubleValue();
					return Double.compare(distance1, distance2);
				}
			});
			for (int i = 0; i < getNearestNeighbors(); i++) {
				NN[i] = ((Integer) ((Object[]) (Object[]) distanceToInstance
						.get(i))[1]).intValue();
			}
			return NN;
		}

		private int getRandomActiveLabelFrom(Instance iInstance) {
			List<Integer> indexes = new ArrayList();
			int numLabels = this.myBag.getMlDS().getNumLabels();
			int[] labelIndices = this.myBag.getMlDS().getLabelIndices();
			for (int i = 0; i < numLabels; i++) {
				if (iInstance.stringValue(labelIndices[i]).equals("1")) {
					indexes.add(Integer.valueOf(i));
				}
			}
			Random rnd = new Random();
			int idx = rnd.nextInt(indexes.size());

			return ((Integer) indexes.get(idx)).intValue();
		}

		private Map getVdmMap(int index) {
			Map vdmMap = new HashMap();

			Enumeration attrEnum = this.mliDS.enumerateAttributes();

			Attribute labelAttr = this.mliDS.attribute(this.myBag.getMlDS()
					.getLabelIndices()[index]);
			while (attrEnum.hasMoreElements()) {
				Attribute attr = (Attribute) attrEnum.nextElement();
				if ((attr.index() < this.myBag.getMlDS().getLabelIndices()[0])
						&& ((attr.isNominal()) || (attr.isString()))) {
					double[][] vdm = new double[attr.numValues()][attr
							.numValues()];
					vdmMap.put(attr, vdm);
					int[] featureValueCounts = new int[attr.numValues()];
					int[][] featureValueCountsByClass = new int[2][attr
							.numValues()];
					Enumeration instanceEnum = this.mliDS.enumerateInstances();
					while (instanceEnum.hasMoreElements()) {
						Instance instance = (Instance) instanceEnum
								.nextElement();
						int value = (int) instance.value(attr);
						int classValue = (int) instance.value(labelAttr);
						featureValueCounts[value] += 1;
						featureValueCountsByClass[classValue][value] += 1;
					}
					for (int valueIndex1 = 0; valueIndex1 < attr.numValues(); valueIndex1++) {
						for (int valueIndex2 = 0; valueIndex2 < attr
								.numValues(); valueIndex2++) {
							double sum = 0.0D;
							for (int classValueIndex = 0; classValueIndex < 2; classValueIndex++) {
								double c1i = featureValueCountsByClass[classValueIndex][valueIndex1];
								double c2i = featureValueCountsByClass[classValueIndex][valueIndex2];
								double c1 = featureValueCounts[valueIndex1];
								double c2 = featureValueCounts[valueIndex2];
								double term1 = c1i / c1;
								double term2 = c2i / c2;
								sum += Math.abs(term1 - term2);
							}
							vdm[valueIndex1][valueIndex2] = sum;
						}
					}
				}
			}
			return vdmMap;
		}
	}
}
