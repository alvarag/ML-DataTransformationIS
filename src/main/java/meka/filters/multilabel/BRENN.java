/*
 * BRENN.java
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

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

/**
 * ENN instance selection for ML by means of binary relevance.<br>
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
public class BRENN extends BRIS {

	private static final long serialVersionUID = 1974167341692127687L;

	/**
	 * Number of nearest neighbours.
	 */
	protected int m_K = 3;

	public String kTipText() {
		return "Number of nearest neighbours.";
	}

	public void setK (int k) {
		m_K = k;
	}

	public int getK () {
		
		return m_K;
	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {
		ArrayList<Option> options = new ArrayList<Option>(1);

		options.add(new Option("\tNumber of nearest neighbours.", "K", 0, "-K"));

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

		return result.toArray(new String[result.size()]);
	}

	@Override
	public String globalInfo() {
		
		return "ENN instance selection by using binary relevance (with voting).";
	}

	@Override
	protected void applyIS(Instances instances, int[] remove) throws Exception {
		NearestNeighbourSearch nnSearch = new LinearNNSearch(instances);
		Instances nn;
		
		for (int i = instances.numInstances() - 1; i >= 0; i--) {
			nn = nnSearch.kNearestNeighbours(instances.instance(i), m_K);
			
			if (isMisclassified (instances.instance(i), nn)) {
				remove[i]++;
				instances.remove(i);
				nnSearch = new LinearNNSearch(instances);
			}
		}
	}

	/**
	 * Returns whether or not the target instance is misclassified by the
	 * instances of nn.
	 * 
	 * @param target Instance to check.
	 * @param nn Instances (neighbours) of target.
	 * @return True if target is misclassified using nn, false otherwise.
	 */
	protected static boolean isMisclassified (Instance target, Instances nn) {
		int[] nnClass = new int[target.numClasses()];
		int max = 0;
		int pred = -1;
		
		for (Instance inst : nn)
			nnClass[(int)inst.classValue()]++;
		
		for (int i = 0; i < nnClass.length; i++) {
			if (nnClass[i] > max) {
				pred = i;
				max = nnClass[i];
			}
		}
		
		if (target.classValue() == pred)
			return false;
		
		return true;
	}
}
