/*
 * HDLSBo.java
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
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.filters.Filter;

/**
 * LSBo instance selection for ML by means of hamming distance threshold.
 * <p>
 * Valid options are:
 * <p>
 * threshold for local set<br>
 * 
 * @author Álvar Arnaiz-González
 * @version 20160705
 */
public class HDLSBo extends HDLSSm {

	private static final long serialVersionUID = -1580882344732139729L;

	/**
	 * Threshold for removing.
	 */
	protected double m_ThresholdLSB = 0.15;
	
	public void setThresholdLSB (double t) {
		m_ThresholdLSB = t;
	}

	public double getThresholdLSB () {
		
		return m_ThresholdLSB;
	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {
		ArrayList<Option> options = new ArrayList<Option>(2);

		options.add(new Option("\tThreshold for removing.", "B", 0, "-B"));

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

		tmpStr = Utils.getOption('B', options);

		if (tmpStr.length() > 0)
			setThresholdLSB(Double.parseDouble(tmpStr));
		else
			setThresholdLSB(0.15);

		super.setOptions(options);
	}

	/**
	 * Gets the current settings of the Filter.
	 * 
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		String[] options = super.getOptions();

		for (int i = 0; i < options.length; i++)
			result.add(options[i]);

		result.add("-B");
		result.add("" + getThresholdLSB());

		return result.toArray(new String[result.size()]);
	}

	@Override
	public String globalInfo() {
		
		return "LSBo instance selection by means of hamming distance.";
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		Instances filtered, solution = new Instances (instances, 
		                                   instances.numInstances() / 2);
		ArrayList<Integer> alreadySelectedIndexes  = new ArrayList<Integer>();
		ArrayList<Integer>[] localSets;
		HDLSSm hdlssm = new HDLSSm();
		int[] nearestEnemies = new int[instances.numInstances()];
		boolean add;
		int currInst;

		// Only change first batch of data
		if (isFirstBatchDone())
			return new Instances(instances);
		
		// Apply HDLSSm
		hdlssm.setThreshold(m_ThresholdLSS);
		hdlssm.setInputFormat(instances);
		filtered = Filter.useFilter(instances, hdlssm);
		
		// Order instances according to local sets' cardinality.
		localSets = new ArrayList[filtered.numInstances()];
		int[] orderedInstances = new int[filtered.numInstances()];
		
		computeLocalSets(filtered, localSets, nearestEnemies, m_ThresholdLSB);
		
		orderInstances(filtered, localSets, orderedInstances);
		
		// Selection process
		for (int i = 0; i < filtered.numInstances(); i++) {
			currInst = orderedInstances[i];
			add = true;
			
			for (int j = 0; add && j < localSets[currInst].size(); j++)
				for (int k = 0; add && k < alreadySelectedIndexes.size(); k++)
					if (localSets[currInst].get(j) == alreadySelectedIndexes.get(k))
						add = false;
			
			// Add the instance if the intersection between its LS 
			// and the solution data set is empty.
			if (add) {
				alreadySelectedIndexes.add(currInst);
				solution.add(filtered.instance(currInst));
			}
		}
		
		return solution;
	}

	protected void orderInstances (Instances instances, ArrayList<Integer>[] localSets, int[] orderedInstances) {
		double[] arrayToSort = new double[instances.numInstances()];
		double indexOfInstances[] = new double[instances.numInstances()];
		
		for (int i = 0; i < instances.numInstances(); i++)
			indexOfInstances[i] = i;
		
		// Compute the cardinality of each LS.
		for (int i = 0; i < localSets.length; i++)
			arrayToSort[i] = localSets[i].size();
		
		NearestNeighbourSearch.quickSort(arrayToSort, indexOfInstances, 0, arrayToSort.length - 1);
		
		for (int i = 0; i < indexOfInstances.length ; i++)
			orderedInstances[i] = (int)indexOfInstances[i];
	}
}
