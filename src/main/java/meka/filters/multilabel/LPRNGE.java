/*
 * LPRNGE.java
 * Copyright (C) 2017 Burgos University, Burgos, Spain 
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

import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

/**
 * RNGE instance selection for ML by means of local powerset.<br>
 * <p>
 * Valid options are:
 * <p>
 * whether or not use the 2nd order graph <br>
 * 
 * @author Álvar Arnaiz-González
 * @version 20171226
 */
public class LPRNGE extends LPIS {

	private static final long serialVersionUID = 1974167341692127687L;

	/**
	 * Graph order to use.
	 */
	protected int m_FirstOrder = 1;

	public String kTipText() {
		return "Use the first or second order graph.";
	}

	public void setFirstOrder(int o) {
		m_FirstOrder = o;
	}

	public int getFirstOrder() {

		return m_FirstOrder;
	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {
		ArrayList<Option> options = new ArrayList<Option>(1);

		options.add(new Option("\tUse the first or second order graph.", "O", 0, "-O"));

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

		tmpStr = Utils.getOption('O', options);

		if (tmpStr.length() > 0)
			setFirstOrder(Integer.parseInt(tmpStr));
		else
			setFirstOrder(1);

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

		result.add("-O");
		result.add("" + getFirstOrder());

		return result.toArray(new String[result.size()]);
	}

	@Override
	public String globalInfo() {

		return "RNGE instance selection by using local powerset.";
	}

	@Override
	protected boolean[] applyIS(Instances instances) throws Exception {
		DistanceFunction distance = new EuclideanDistance(instances);
		Instances nn;
		boolean[][] graph = new boolean[instances.numInstances()][instances.numInstances()];
		boolean[] remove = new boolean[instances.numInstances()];
		double min, dist;
		int rel;

		for (int i = 0; i < instances.numInstances(); i++)
			remove[i] = false;

		// Build the Relative Neighbourhood Graph (RNG)
		for (int i = 0; i < instances.numInstances(); i++) {
			min = Double.MAX_VALUE;
			rel = -1;

			for (int j = i; j < instances.numInstances(); j++) {
				if (i != j) {
					dist = distance.distance(instances.get(i), instances.get(j));

					if (dist < min) {
						min = dist;
						rel = j;
					}
				}
			}

			if (rel != -1) {
				graph[i][rel] = true;
				graph[rel][i] = true;
			}
		}

		// Discard instances according to the algorithm.
		for (int i = 0; i < instances.numInstances(); i++) {
			// Empty set of instances, capacity is not taken into account.
			nn = new Instances(instances, 10);

			for (int j = 0; j < instances.numInstances(); j++)
				if (graph[i][j])
					nn.add(instances.get(j));

			if (BRENN.isMisclassified(instances.instance(i), nn)) {
				// 1st order
				if (m_FirstOrder == 1) {
					remove[i] = true;
				} else {
					// 2nd order
					for (int j = 0; j < instances.numInstances(); j++)
						if (graph[i][j] && instances.get(i).classValue() == instances.get(j).classValue())
							for (int k = 0; k < instances.numInstances(); k++)
								if (graph[j][k])
									nn.add(instances.get(k));

					if (BRENN.isMisclassified(instances.instance(i), nn)) {
						remove[i] = true;
					}
				}
			}
		}

		return remove;
	}
}
