/*
 * HDLSSm.java
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

import meka.core.Metrics;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NormalizableDistance;
import weka.core.Option;
import weka.core.Utils;
import weka.filters.SimpleBatchFilter;

/**
 * LSSm instance selection for ML by means of hamming distance threshold.
 * <p>
 * Valid options are:
 * <p>
 * threshold for local set<br>
 * 
 * @author Álvar Arnaiz-González
 * @version 20160705
 */
public class HDLSSm extends SimpleBatchFilter {
	
	private static final long serialVersionUID = 494665828580458658L;
	
	/**
	 * Threshold for removing.
	 */
	protected double m_ThresholdLSS = 0.15;
	
	public void setThreshold (double t) {
		m_ThresholdLSS = t;
	}

	public double getThreshold () {
		
		return m_ThresholdLSS;
	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {
		ArrayList<Option> options = new ArrayList<Option>(2);

		options.add(new Option("\tThreshold for removing.", "T", 0, "-T"));

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

		tmpStr = Utils.getOption('T', options);

		if (tmpStr.length() > 0)
			setThreshold(Double.parseDouble(tmpStr));
		else
			setThreshold(0.15);

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

		result.add("-T");
		result.add("" + getThreshold());

		return result.toArray(new String[result.size()]);
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
		
		return new Instances(inputFormat, 0);
	}

	@Override
	public String globalInfo() {
		
		return "LSSm instance selection by means of hamming distance.";
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		Instances result = new Instances(instances, instances.numInstances());
		ArrayList<Integer>[] localSets = new ArrayList[instances.numInstances()];
		int[] nearestEnemies = new int[instances.numInstances()];
		int u, h;
		
		// Only change first batch of data
		if (isFirstBatchDone())
			return new Instances(instances);
		
		// Compute local sets.
		computeLocalSets(instances, localSets, nearestEnemies, m_ThresholdLSS);
		
		for (int i = 0; i < instances.numInstances(); i++) {
			u = h = 0;
			
			// Compute u(e).
			for (ArrayList<Integer> localset : localSets)
				for (int indexLocalSet : localset)
					if (indexLocalSet == i)
						u++;
			
			// Compute h(e).
			for (int ne : nearestEnemies)
				if (ne == i)
					h++;
			
			if (getDebug()) 
				System.out.println("u/h: " + u + "/" + h);
			
			// If u(e) >= h(e) add the instance to the solution.
			if (u >= h)
				result.add(instances.instance(i));
		}
		
		return result;
	}
	
	/**
	 * Computes the local sets of the trainSet's instances.
	 * 
	 * @param trainSet Data set.
	 * @param localSets Local sets of trainSet's instances.
	 * @param enemies Index of the nearest enemy of each instance.
	 * @param threshold Threshold for local set computations.
	 * @throws Exception If distance computation fails. 
	 */
	protected void computeLocalSets (Instances trainSet, ArrayList<Integer>[] localSets, int[] enemies, double threshold) throws Exception {
		DistanceFunction distanceFunction;
		Instance instI, instJ;
		double[] distances;
		int[] classI, classJ;
		double distNearEnemy, hamm;
		
		// Don't normalize the distances.
		distanceFunction = new EuclideanDistance(trainSet);
		((NormalizableDistance)distanceFunction).setDontNormalize(true);
		
		for (int i = 0; i < trainSet.numInstances(); i++) {
			localSets[i] = new ArrayList<Integer>();
			distances = new double[trainSet.numInstances()];
			distNearEnemy = Double.MAX_VALUE;
			
			instI = trainSet.instance(i);
			classI = getClassValues(instI);
			
			for (int j = 0; j < trainSet.numInstances(); j++) {
				if (i != j) {
					instJ = trainSet.instance(j);
					classJ = getClassValues(instJ);
					
					distances[j] = distanceFunction.distance(instI, instJ);
					
					hamm = Metrics.L_Hamming(classI, classJ);
					
					if (hamm > threshold && distances[j] < distNearEnemy) {
						distNearEnemy = distances[j];
						enemies[i] = j;
					}
				}
			}

			// Compute the LS of instI.
			for (int j = 0; j < distances.length; j++)
				if (i != j)
					if (distances[j] < distNearEnemy)
						localSets[i].add(j);
		}
	}
	
	/**
	 * Return the class value of a ML instance.
	 * 
	 * @param inst Instance.
	 * @return Array with class' values.
	 */
	protected int[] getClassValues (Instance inst) {
		int[] instClass = new int[inst.classIndex()];
		
		for (int i = 0; i < inst.classIndex(); i++)
			instClass[i] = (int)inst.value(i);
		
		return instClass;
	}

}
