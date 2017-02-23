/*
 * BRIS.java
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import meka.classifiers.multilabel.Evaluation;
import meka.classifiers.multilabel.MLkNN;
import meka.classifiers.multilabel.MultiLabelClassifier;
import meka.core.MLUtils;
import meka.core.Result;

import weka.core.Instances;
import weka.core.Option;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.filters.SimpleBatchFilter;

/**
 * Instance selection for ML by means of binary relevance.<br>
 * The threshold is computed using a votation method like DIS.
 * <p>
 * Valid options are:
 * <p>
 * number of nearest neighbours <br>
 * alpha for fitness function <br>
 * percentage of instances for error computation (in fitness function) <br>
 * 
 * @author Álvar Arnaiz-González
 * @version 20160929
 */
public abstract class BRIS extends SimpleBatchFilter implements Randomizable {

	private static final long serialVersionUID = -3952573991629758138L;

	/**
	 * The seed value for randomizing the data.
	 */
	protected int m_Seed = 1;
	
	/**
	 * Alpha of fitness function.
	 */
	protected double m_Alpha = 0.95;
	
	/**
	 * Proportion of instances for error computation [0-1].
	 */
	protected double m_PropInstErr = 0.1;
	
	@Override
	public void setSeed(int s) {
		m_Seed = s;
	}

	@Override
	public int getSeed() {
		return m_Seed;
	}

	public String seedTipText() {
		return "The seed value for randomizing the data.";
	}

	public void setAlpha (double a) {
		m_Alpha = a;
	}

	public double getAlpha () {
		
		return m_Alpha;
	}

	public void setPropInstErr (double e) {
		m_PropInstErr = e;
	}

	public double getPropInstErr () {
		
		return m_PropInstErr;
	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {
		ArrayList<Option> options = new ArrayList<Option>(3);

		options.add(new Option("\tAlpha value.", "A", 0, "-A"));

		options.add(new Option("\tProportion of instances for error computation.", "E", 0, "-E"));

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

		tmpStr = Utils.getOption('A', options);

		if (tmpStr.length() > 0)
			setAlpha(Double.parseDouble(tmpStr));
		else
			setAlpha(0.75);

		tmpStr = Utils.getOption('E', options);

		if (tmpStr.length() > 0)
			setPropInstErr(Double.parseDouble(tmpStr));
		else
			setPropInstErr(0.1);

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

		result.add("-A");
		result.add("" + getAlpha());

		result.add("-E");
		result.add("" + getPropInstErr());

		return result.toArray(new String[result.size()]);
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
		
		return new Instances(inputFormat, 0);
	}

	@Override
	public String globalInfo() {
		
		return "Instance selection by using binary relevance (with voting).";
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		Instances result = new Instances(instances, instances.numInstances());
		int[] remove = new int[instances.numInstances()];
		int threshold;
		
		// Only change first batch of data
		if (isFirstBatchDone())
			return new Instances(instances);
		
		// Init with zeroes
		for (int i = 0; i < remove.length; i++)
			remove[i] = 0;
		
		// Compute votes
		computeVotes(instances, remove);
		
		if (getDebug())
			System.out.println("Final votes: " + Arrays.toString(remove));
		
		// Compute the best threshold
		threshold = computeThreshold (instances, remove);
		
		// Add the instances to result.
		for (int i = 1; i < instances.numInstances(); i++)
			if (remove[i] < threshold)
				result.add(instances.instance(i));

		return result;
	}
	
	/**
	 * Computes the votes using binary relevance ENN.
	 * 
	 * @param instances Instances to filter.
	 * @param remove Array with votes for removal.
	 * @param numLabels Number of labels
	 * @throws Exception If something goes wrong.
	 */
	protected void computeVotes(Instances instances, int[] remove) 
	                        throws Exception {
		int numLabels = instances.classIndex();
		
		// BR
		for(int j = 0; j < numLabels; j++) {
			//Select only class attribute 'j'
			Instances instances_j = MLUtils.keepAttributesAt(new Instances(instances),
			                                                  new int[]{j},numLabels);
			instances_j.setClassIndex(0);

			// Generate the dataset
			Instances tmp = new Instances(instances_j);
		
			// Apply IS
			applyIS (tmp, remove);
		}
	}
	
	/**
	 * Applies the IS method. Should be overwritten.
	 *  
	 * @param instances Insances to filter.
	 * @param remove Vector with votes for removal.
	 */
	protected abstract void applyIS (Instances instances, int[] remove) throws Exception;
	
	/**
	 * Computes the best threshold according to alpha value.
	 * 
	 * @param instances Original data set. 
	 * @param remove Vector with votes of every single instance.
	 * @return The threshold chosen.
	 */
	private int computeThreshold (Instances instances, int[] remove) {
		Instances trainSet, testSet;
		double error, memory, tmpFitness, minFitness = Double.MAX_VALUE;
		int numLabels = instances.classIndex();
		int fitness = 0;
		
		testSet = getRandomSubset(instances, m_PropInstErr);

		// Compute the fitness value for v in the interval [0, #classes].
		for (int i = 1; i <= numLabels + 1; i++) {
			// Instances with votes <= i
			trainSet = getInstancesUnderVotes(instances, remove, i);
			
			// The number of k for MLkNN is, by default, 10.
			if (trainSet.numInstances() > 10) {
				// Error
				error = computeHammingLoss(trainSet, testSet);
				
				// Memory
				memory = (double)trainSet.numInstances() / instances.numInstances();
				
				// f(v) = alpha * error(v) + (1-alpha) * m(v)
				tmpFitness = (m_Alpha * error) + ((1d - m_Alpha) * memory);
				
				if (getDebug())
					System.out.println("Mem: " + memory + ", HammLoss: " + error + " => Fitness for " + i + ": " + tmpFitness);
				
				// Select the minimum fitness value
				if (tmpFitness < minFitness) {
					minFitness = tmpFitness;
					fitness = i;
				}
			}
			
			// Don't try further comparisons because we have the whole set
			if (trainSet.numInstances() == instances.numInstances())
				return fitness;
		}
		
		return fitness;
	}

	/**
	 * Returns a random subset of instances.
	 * 
	 * @param instances Original set of instances.
	 * @param prop Proportion of instances to select [0-1].
	 * @return Random selected subset.
	 */
	private Instances getRandomSubset (Instances instances, double prop) {
		Instances subset, shuffleInst;
		int num = (int)(instances.numInstances() * prop);
		
		// At least retain instance
		if (num < 1)
			num = 1;
		
		subset = new Instances(instances, num);
		shuffleInst = new Instances(instances);
		shuffleInst.randomize(instances.getRandomNumberGenerator(m_Seed));
		
		for (int i = 0; i < num; i++)
			subset.add(shuffleInst.get(i));
		
		return subset;
	}

	/**
	 * Returns a subset of instances which have lower votes in remove array than 
	 * threshold.
	 * 
	 * @param instances Original data set.
	 * @param remove Votes array for removal.
	 * @param threshold Threshold.
	 * @return Subset of instances with lower votes' number than threshold. 
	 */
	private Instances getInstancesUnderVotes (Instances instances, int[] remove, 
	                                                int threshold) {
		Instances outInstances = new Instances(instances, instances.numInstances());
		
		for (int i = 0; i < remove.length; i++)
			if (remove[i] < threshold)
				outInstances.add(instances.instance(i));
		
		return outInstances;
	}

	/**
	 * Computes the hamming loss by training a MLkNN classifier with train
	 * and testing it with test.
	 * 
	 * @param train Training set.
	 * @param test Testing set.
	 * @return Hamming loss value.
	 */
	private double computeHammingLoss(Instances train, Instances test) {
		MultiLabelClassifier classifier = new MLkNN();
		Result res = null;
		
		try {
			res = Evaluation.evaluateModel(classifier, train, test, "PCut1", "3");
		} 
		catch (Exception e) {
			System.err.println("Failed to evaluate dataset '" + train.relationName() 
			                 + "' with classifier: " + Utils.toCommandLine(classifier));
			System.err.println(e.toString());
		}

		return ((Double)res.getMeasurement("Hamming loss"));
	}

}
