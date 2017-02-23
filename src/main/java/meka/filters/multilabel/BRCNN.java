/*
 * BRCNN.java
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

import java.util.Vector;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

/**
 * CNN instance selection for ML by means of binary relevance.<br>
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
public class BRCNN extends BRIS {

	private static final long serialVersionUID = 1974167341692127687L;

	@Override
	public String globalInfo() {
		
		return "CNN instance selection by using binary relevance (with voting).";
	}

	@Override
	protected void applyIS(Instances instances, int[] remove) throws Exception {
		NearestNeighbourSearch nnSearch;
		Instances reducedSet = new Instances(instances, instances.numInstances() / 10);
		Instance inst;
		Vector<Double> classSelected = new Vector<Double>(instances.classAttribute().numValues());
		boolean[] selected = new boolean[instances.numInstances()];
		
		for (int i = 0; i < remove.length; i++)
			selected[i] = false;
		
		// Starts with an instance of each class.
		for (int i = 0; i < instances.numInstances(); i++) {
			inst = instances.instance(i);
			
			// If any instance of the current's class has been already selected
			if (!classSelected.contains(inst.classValue())) {
				selected[i] = true;
				reducedSet.add(inst);
				classSelected.add(inst.classValue());
			}
			
			// Stop if all classes have been already selected
			if (classSelected.size() == instances.classAttribute().numValues())
				i = instances.numInstances();
		}
		
		// Init the NN search.
		nnSearch = new LinearNNSearch(reducedSet);
		
		// Run CNN.
		for (int i = 0; i < instances.numInstances(); i++) {
			if (!selected[i]) {
				inst = instances.instance(i);
				if (nnSearch.nearestNeighbour(inst).classValue() != inst.classValue()) {
					selected[i] = true;
					reducedSet.add(inst);
					i = 0;
					nnSearch = new LinearNNSearch(reducedSet);
				}
			}
		}
		
		// Accumulate votes in remove.
		for (int i = 0; i < instances.numInstances(); i++)
			if (!selected[i])
				remove[i]++;
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
