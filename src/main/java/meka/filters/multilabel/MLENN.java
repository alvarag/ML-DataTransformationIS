/*
 * MLENN.java
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
import java.util.HashSet;
import java.util.TreeSet;

import meka.core.MLEvalUtils;
import meka.core.Metrics;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.filters.SimpleBatchFilter;

/**
 * Editing multi-labeled data using the kNN rule. Presented by:
 * Kanj, S., Abdallah, F., Den&#339;ux, T., & Tout, K. (2016). 
 * Editing training data for multi-label classification with the 
 * k-nearest neighbor rule. Pattern Analysis and Applications, 
 * 19(1), 145-161. 
 * <p>
 * Valid options are:
 * <p>
 * threshold for selecting or discarding an instance <br>
 * nearest neighbours used in the editing algorithm <br>
 * 
 * @author Álvar Arnaiz-González
 * @version 20160929
 */
public class MLENN extends SimpleBatchFilter {

	private static final long serialVersionUID = -27845533234035212L;

	/** 
	 * The classifier. 
	 */
	protected Classifier m_Classifier = new meka.classifiers.multilabel.MLkNN();

	/**
	 * Number of nearest neighbours.
	 */
	protected int m_K = 10;
	
	/**
	 * Threshold for removing.
	 */
	protected double m_Threshold = 0.99;
	
	public void setThreshold (double t) {
		m_Threshold = t;
	}

	public double getThreshold () {
		
		return m_Threshold;
	}

	public void setK (int k) {
		m_K = k;
	}

	public int getK () {
		
		return m_K;
	}
	
	/**
	 * Set the base learner.
	 * 
	 * @param newClassifier the classifier to use.
	 */
	public void setClassifier(Classifier newClassifier) {

		m_Classifier = newClassifier;
	}

	/**
	 * Get the classifier used as the base learner.
	 * 
	 * @return the classifier used as the classifier
	 */
	public Classifier getClassifier() {

		return m_Classifier;
	}
	
	/**
	 * String describing default classifier.
	 */
	protected String defaultClassifierString() {

		return "meka.classifiers.multilabel.MLkNN";
	}

	/**
	 * String describing options for default classifier.
	 */
	protected String[] defaultClassifierOptions() {

		return new String[0];
	}

	/**
	 * Gets the classifier specification string, which contains the class name
	 * of the classifier and any options to the classifier
	 * 
	 * @return the classifier string
	 */
	protected String getClassifierSpec() {

		Classifier c = getClassifier();
		return c.getClass().getName() + " " + 
		        Utils.joinOptions(((OptionHandler) c).getOptions());
	}
	  
	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {
		ArrayList<Option> options = new ArrayList<Option>(4);

		options.add(new Option("\tFull class name of classifier to use, followed\n"
		           + "\tby classifier options.\n"
		           + "\teg: \"meka.classifiers.multilabel.MLkNN\"",
		           "W", 1, "-W <classifier specification>"));

		if (getClassifier() instanceof OptionHandler) {
			options.add(new Option("", "", 0, "\nOptions specific to classifier "
		                             + getClassifier().getClass().getName() + ":"));
			options.addAll(Collections.list(((OptionHandler) getClassifier()).listOptions()));
		}

		options.add(new Option("\tNumber of nearest neighbour.", "K", 0, "-K"));

		options.add(new Option("\tThreshold for removing (% of initial Hloss).", "T", 0, "-T"));

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
			setK(10);

		tmpStr = Utils.getOption('T', options);

		if (tmpStr.length() > 0)
			setThreshold(Double.parseDouble(tmpStr));
		else
			setThreshold(0.15);

		String classifierName = Utils.getOption('W', options);

		if (classifierName.length() > 0) {
			setClassifier(AbstractClassifier.forName(classifierName, null));
			setClassifier(AbstractClassifier.forName(classifierName,
					Utils.partitionOptions(options)));
		} else {
			setClassifier(AbstractClassifier.forName(defaultClassifierString(), null));
			String[] classifierOptions = Utils.partitionOptions(options);
			if (classifierOptions.length > 0) {
				setClassifier(AbstractClassifier.forName(defaultClassifierString(), 
				                                           classifierOptions));
			} else {
				setClassifier(AbstractClassifier.forName(defaultClassifierString(), 
				                                           defaultClassifierOptions()));
			}
		}
		
		super.setOptions(options);
	}

	/**
	 * Gets the current settings of the Classifier.
	 * 
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String[] getOptions() {
		ArrayList<String> result = new ArrayList<String>();
		String[] options = super.getOptions();
		String[] classifierOptions;

		for (int i = 0; i < options.length; i++)
			result.add(options[i]);

		result.add("-K");
		result.add("" + getK());

		result.add("-T");
		result.add("" + getThreshold());

		result.add("-W");
		result.add(getClassifier().getClass().getName());

		classifierOptions = ((OptionHandler) m_Classifier).getOptions();
		
		if (classifierOptions.length > 0) {
			result.add("--");
			Collections.addAll(result, classifierOptions);
		}

		return result.toArray(new String[result.size()]);
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
		
		return new Instances(inputFormat, 0);
	}

	@Override
	public String globalInfo() {
		
		return "Editing multi-labeled data using the k-NN rule.";
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		Instances result;
		Double[] uniqueSortHammLoss, indivHammLoss;
		double meanHammLoss, threshold = -1;

		// Only change first batch of data
		if (isFirstBatchDone())
			return new Instances(instances);

		result = new Instances(instances);
		removeDuplicateInstances(result);

		// Main loop of the algorithm
		do {
			// Individual hamming loss.
			indivHammLoss = computeHammingLoss(result);
	
			// Mean hamming loss.
			meanHammLoss = computeMean(indivHammLoss);
			
			// Compute once the threshold
			if (threshold == -1) {
				threshold = meanHammLoss * m_Threshold;
				if (getDebug())
					System.out.println ("Threshold: " + threshold);
			}
			
			// Stop the algorithm
			if (meanHammLoss < threshold)
				return result;
			
			// Find uniques and sort.
			uniqueSortHammLoss = findUniquesAndSort(indivHammLoss);
		
			if (getDebug())
				System.out.println ("Hamm loss: " + Arrays.toString(uniqueSortHammLoss));

			// Stop the algorithm if deletion process would delete all instances
			if (uniqueSortHammLoss.length == 1)
				return result;
			
			for (int j = result.numInstances() - 1; j >= 0; j--) {
				if (indivHammLoss[j] >= uniqueSortHammLoss[0])
					result.remove(j);
			}
		// Stop if there are only two unique values or the size of the data set
		// is lower than one.
		} while (result.size() > 1 && uniqueSortHammLoss.length > 2);

		return result;
	}
	
	/**
	 * Computes the hamming loss for each instance.
	 * 
	 * @param instances Instances.
	 * @return Double's array with the hamming loss of each instance.
	 * @throws Exception If something wrong occurs.
	 */
	private Double[] computeHammingLoss (Instances instances) throws Exception {
		ArrayList<double[]> predictions;
		NearestNeighbourSearch nnSearch;
		Instances nn;
		int[] y = new int[instances.classIndex()], 
		      yPred = new int[instances.classIndex()];
		Double[] hLoss;
		double predThreshold;
		
		nnSearch = new LinearNNSearch(instances);
		((LinearNNSearch)nnSearch).setSkipIdentical(true);
		
		predictions = new ArrayList<>(instances.numInstances());
		
		// Accumulate the predictions for each instance
		for (int i = 0; i < instances.numInstances(); i++) {
			nn = nnSearch.kNearestNeighbours(instances.instance(i), m_K + 1);
			predictions.add(getPrediction (instances.instance(i), nn, 
			                                nnSearch.getDistanceFunction()));
		}
		
		// Compute the threshold for predictions.
		predThreshold = Double.parseDouble(MLEvalUtils.getThreshold(predictions, 
		                                    instances, "PCut1"));
		
		hLoss = new Double[instances.numInstances()];
		
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < instances.classIndex(); j++) {
				y[j] = (int)instances.get(i).value(j);
				
				if (predictions.get(i)[j] >= predThreshold)
					yPred[j] = 1;
				else
					yPred[j] = 0;
			}
			
			hLoss[i] = Metrics.L_Hamming(y, yPred);
		}
		
		return hLoss;
	}
	
	/**
	 * Computes the mean of the vector.
	 * 
	 * @param vDouble Vector.
	 * @return Mean of the vector.
	 */
	private double computeMean (Double[] vDouble) {
		double mean = 0;
		
		for (double d : vDouble)
			mean += d;
		
		return mean/ vDouble.length;
	}
	
	/**
	 * Removes repeated values on the array and sort it. 
	 * 
	 * @param origArray Array to be processed.
	 * @return Sorted array without repeated values.
	 */
	private Double[] findUniquesAndSort (Double[] origArray) {
		Double[] resultArray; 
		
		// Unique
		resultArray = new HashSet<Double>(Arrays.asList(origArray)).toArray(new Double[0]);
		
		// Sort descent
		Arrays.sort(resultArray, Collections.reverseOrder());
		
		return resultArray;
	}
	
	/**
	 * Returns the classes prediction for target.
	 * 
	 * @param target Instance.
	 * @param neighbours Target's neighbours.
	 * @param distance Distance function to use.
	 * @return an array containing the estimated membership probabilities 
	 *         of the test instance in each class or the numeric prediction
	 * @throws Exception Exception launched by the classifier.
	 */
	private double[] getPrediction (Instance target, Instances neighbours, 
	                                  DistanceFunction distance) throws Exception {
		double[] distr;
		
		m_Classifier.buildClassifier(neighbours);
		
		distr = m_Classifier.distributionForInstance(target);
		
		return distr;
	}

	/**
	 * Removes instances duplicated.
	 * 
	 * @param instances Instances with duplicates.
	 */
	public static void removeDuplicateInstances(Instances instances) {
		TreeSet<Instance> hashSet = new TreeSet<Instance>(
		                              new InstanceComparator(true));
		int i = 0;

		// Recorrer todas las instancias;
		while (i < instances.numInstances())
			// Si la instancia ya existe la eliminamos, sino pasa a la
			// siguiente.
			if (hashSet.add(instances.instance(i)))
				i++;
			else
				instances.delete(i);
	}

}
