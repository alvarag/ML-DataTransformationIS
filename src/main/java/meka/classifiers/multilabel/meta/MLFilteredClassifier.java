/*
 * MLFilteredClassifier.java
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
package meka.classifiers.multilabel.meta;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.SingleClassifierEnhancer;
import weka.core.Attribute;
import weka.core.BatchPredictor;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.WekaException;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import meka.classifiers.multilabel.MultiLabelClassifier;

/**
<!-- globalinfo-start -->
* Class for running an arbitrary ML classifier on data that has been passed through 
* an arbitrary filter. Like the classifier, the structure of the filter is based 
* exclusively on the training data and test instances will be processed by the filter 
* without changing their structure.
* <p/>
<!-- globalinfo-end -->
*
<!-- options-start -->
* Valid options are: <p/>
* 
* <pre> -F &lt;filter specification&gt;
*  Full class name of filter to use, followed
*  by filter options.
*   
<!-- options-end -->
*
* @author Álvar Arnaiz-González (alvarag@ubu.es)
 * @version 20160713
*/
public class MLFilteredClassifier extends SingleClassifierEnhancer 
                                  implements MultiLabelClassifier {

	private static final long serialVersionUID = -1797855022888626411L;

	/** 
	 * The filter 
	 */
	protected Filter m_Filter = new meka.filters.multilabel.BRENN();

	/** 
	 * The instance structure of the filtered instances
	 */
	protected Instances m_FilteredInstances;
	
	/**
	 * Execution time of the filtering step. 
	 */
	protected double m_FilteringTime;

	/**
	 * Compression: 1 - (#_filter / #_total)
	 */
	protected double m_Compression;

	/**
	 * Returns a string describing this classifier
	 * 
	 * @return a description of the classifier suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Class for running an arbitrary classifier on data that has been passed "
				+ "through an arbitrary filter. Like the classifier, the structure of the filter "
				+ "is based exclusively on the training data and test instances will be processed "
				+ "by the filter without changing their structure.";
	}

	/**
	 * String describing default classifier.
	 * 
	 * @return the default classifier classname
	 */
	protected String defaultClassifierString() {

		return "meka.classifiers.multilabel.MLkNN";
	}

	/**
	 * String describing default filter.
	 */
	protected String defaultFilterString() {

		return "meka.filters.multilabel.BRENN";
	}

	/**
	 * Default constructor.
	 */
	public MLFilteredClassifier() {
		m_Classifier = new meka.classifiers.multilabel.MLkNN();
		m_Filter = new meka.filters.multilabel.BRENN();
	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(1);
		newVector.addElement(new Option("\tFull class name of filter to use, followed\n"
		                     + "\tby filter options.\n"
		                     + "\teg: \"weka.filters.unsupervised.attribute.Remove -V -R 1,2\"",
		                     "F", 1, "-F <filter specification>"));

		if (getFilter() instanceof OptionHandler) {
			newVector.addElement(new Option("", "", 0, "\nOptions specific to filter "
			                                 + getFilter().getClass().getName() + ":"));
			newVector.addAll(Collections.list(((OptionHandler) getFilter()).
			                                     listOptions()));
		}

		newVector.addAll(Collections.list(super.listOptions()));

		newVector.addAll(Collections.list(((OptionHandler) m_Classifier).
		                              listOptions()));

		return newVector.elements();
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 * 
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 * 
	 * <pre>
	 * -F &lt;filter specification&gt;
	 *  Full class name of filter to use, followed
	 *  by filter options.
	 * </pre>
	 * 
	 * <pre>
	 * -D
	 *  If set, classifier is run in debug mode and
	 *  may output additional info to the console
	 * </pre>
	 * 
	 * <pre>
	 * -W
	 *  Full name of base classifier.
	 * </pre>
	 * 
	 * <pre>
	 * Options specific to the classifier:
	 * </pre>
	 * 
	 * <!-- options-end -->
	 * 
	 * @param options
	 *            the list of options as an array of strings
	 * @throws Exception
	 *             if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {
		String filterString = Utils.getOption('F', options);

		if (filterString.length() <= 0) {
			filterString = defaultFilterString();
		}

		String[] filterSpec = Utils.splitOptions(filterString);

		if (filterSpec.length == 0) {
			throw new IllegalArgumentException(
					"Invalid filter specification string");
		}

		String filterName = filterSpec[0];

		filterSpec[0] = "";

		setFilter((Filter) Utils.forName(Filter.class, filterName, filterSpec));

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of the Classifier.
	 * 
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String[] getOptions() {
		Vector<String> options = new Vector<String>();

		options.add("-F");
		options.add("" + getFilterSpec());

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[0]);
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String filterTipText() {
		
		return "The filter to be used.";
	}

	/**
	 * Sets the filter
	 * 
	 * @param filter the filter with all options set.
	 */
	public void setFilter(Filter filter) {

		m_Filter = filter;
	}

	/**
	 * Gets the filter used.
	 * 
	 * @return the filter
	 */
	public Filter getFilter() {

		return m_Filter;
	}

	/**
	 * Gets the filter specification string, which contains the class name of
	 * the filter and any options to the filter
	 * 
	 * @return the filter string.
	 */
	protected String getFilterSpec() {
		Filter c = getFilter();
		
		if (c instanceof OptionHandler) {
			return c.getClass().getName() + " "
			        + Utils.joinOptions(((OptionHandler) c).getOptions());
		}
		
		return c.getClass().getName();
	}

	/**
	 * Returns default capabilities of the classifier.
	 * 
	 * @return the capabilities of this classifier
	 */
	public Capabilities getCapabilities() {
		Capabilities result;

		if (getFilter() == null)
			result = super.getCapabilities();
		else
			result = getFilter().getCapabilities();

		// the filtered classifier always needs a class
		result.disable(Capability.NO_CLASS);

		// set dependencies
		for (Capability cap : Capability.values())
			result.enableDependency(cap);

		return result;
	}

	/**
	 * Sets up the filter and runs checks.
	 * 
	 * @return filtered data
	 */
	protected Instances setUp(Instances data) throws Exception {
		long tBefore, tAfter;
		int numBefore, numAfter;
		
		if (m_Classifier == null)
			throw new Exception("No base classifiers have been set!");

		getCapabilities().testWithFail(data);

		numBefore = data.numInstances();

		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

		m_Filter.setInputFormat(data); // filter capabilities are checked here

		tBefore = System.currentTimeMillis();

		data = Filter.useFilter(data, m_Filter);
		
		tAfter = System.currentTimeMillis();
		numAfter = data.numInstances();
		
		m_FilteringTime = (tAfter - tBefore) / 1000.0;
		m_Compression = 1 - (double) numAfter  / numBefore;

		// can classifier handle the data?
		testCapabilities(getClassifier(), data);

		m_FilteredInstances = data.stringFreeStructure();
		
		return data;
	}
	
	/**
	 * TestCapabilities. Make sure the training data is suitable.
	 * 
	 * @param D the data
	 */
	private void testCapabilities(MultiLabelClassifier classifier, Instances D) 
	                                throws Exception {
		// get the classifier's capabilities, enable all class attributes and do
		// the usual test
		Capabilities cap = classifier.getCapabilities();
		cap.enableAllClasses();

		// get the capabilities again, test class attributes individually
		int L = D.classIndex();
		
		for (int j = 0; j < L; j++) {
			Attribute c = D.attribute(j);
			cap.testWithFail(c, true);
		}
	}

	/**
	 * Build the classifier on the filtered data.
	 * 
	 * @param data
	 *            the training data
	 * @throws Exception
	 *             if the classifier could not be built successfully
	 */
	public void buildClassifier(Instances data) throws Exception {

		m_Classifier.buildClassifier(setUp(data));
	}

	/**
	 * Filters the instance so that it can subsequently be classified.
	 */
	protected Instance filterInstance(Instance instance) throws Exception {
		if (m_Filter.numPendingOutput() > 0)
			throw new Exception("Filter output queue not empty!");
		
		if (!m_Filter.input(instance)) {
			if (!m_Filter.mayRemoveInstanceAfterFirstBatchDone()) {
				throw new Exception("Filter didn't make the test instance"
				                     + " immediately available!");
			} else {
				m_Filter.batchFinished();
				return null;
			}
		}
		
		m_Filter.batchFinished();

		return m_Filter.output();
	}

	/**
	 * Classifies a given instance after filtering.
	 * 
	 * @param instance
	 *            the instance to be classified
	 * @return the class distribution for the given instance
	 * @throws Exception if instance could not be classified successfully
	 */
	public double[] distributionForInstance(Instance instance) throws Exception {
		Instance newInstance = filterInstance(instance);
		
		if (newInstance == null) {
			// filter has consumed the instance (e.g. RemoveWithValues
			// may do this). We will indicate no prediction for this
			// instance
			double[] unclassified = null;
			
			if (instance.classAttribute().isNumeric()) {
				unclassified = new double[1];
				unclassified[0] = Utils.missingValue();
			} else {
				// all zeros
				unclassified = new double[instance.classAttribute().numValues()];
			}
			return unclassified;
		} else {
			return m_Classifier.distributionForInstance(newInstance);
		}
	}

	/**
	 * Batch scoring method. Calls the appropriate method for the base learner
	 * if it implements BatchPredictor. Otherwise it simply calls the
	 * distributionForInstance() method repeatedly.
	 * 
	 * @param insts the instances to get predictions for
	 * @return an array of probability distributions, one for each instance
	 * @throws Exception if a problem occurs
	 */
	public double[][] distributionsForInstances(Instances insts)
			throws Exception {

		if (getClassifier() instanceof BatchPredictor) {
			Instances filteredInsts = Filter.useFilter(insts, m_Filter);
			if (filteredInsts.numInstances() != insts.numInstances()) {
				throw new WekaException("FilteredClassifier: filter has returned " +
				                          "more/less instances than required.");
			}
			return ((BatchPredictor) getClassifier())
			         .distributionsForInstances(filteredInsts);
		} else {
			double[][] result = new double[insts.numInstances()][insts.numClasses()];
			for (int i = 0; i < insts.numInstances(); i++) {
				result[i] = distributionForInstance(insts.instance(i));
			}
			
			return result;
		}
	}

	/**
	 * Output a representation of this classifier
	 * 
	 * @return a representation of this classifier
	 */
	public String toString() {
		if (m_FilteredInstances == null)
			return "FilteredClassifier: No model built yet.";

		String result = "FilteredClassifier using " + getClassifierSpec()
		                 + " on data filtered through " + getFilterSpec()
		                 + "\n\nFiltered Header\n" + m_FilteredInstances.toString()
		                 + "\n\nClassifier Model\n" + m_Classifier.toString();
		
		return result;
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 100 $");
	}

	@Override
	public String getModel() {
		
		return m_Filter + "\n" + m_Classifier;
	}
	
	/**
	 * Get the classifier used as the base learner.<br>
	 * It forces to use a multilabel classifier.
	 * 
	 * @return the classifier used as the classifier
	 */
	public MultiLabelClassifier getClassifier() {

		return (MultiLabelClassifier)m_Classifier;
	}

	/**
	 * Returns the execution time of the filtering stage.
	 *  
	 * @return the execution time.
	 */
	public double getFilteringTime () {
		
		return m_FilteringTime;
	}

	/**
	 * Returns the compression achieved by the filter.
	 * 
	 * @return the compression.
	 */
	public double getCompression () {
		
		return m_Compression;
	}

}
