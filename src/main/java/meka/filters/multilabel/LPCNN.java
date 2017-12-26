/*
 * LPCNN.java
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

import java.util.Vector;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

/**
 * CNN instance selection for ML by means of label powerset.<br>
 * <p>
 * 
 * @author Álvar Arnaiz-González
 * @version 20171226
 */
public class LPCNN extends LPIS {

	private static final long serialVersionUID = -9154050780637146555L;

	@Override
	public String globalInfo() {

		return "CNN instance selection by using local powerset.";
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) throws Exception {

		return new Instances(inputFormat, 0);
	}

	protected boolean[] applyIS(Instances instances) throws Exception {
		NearestNeighbourSearch nnSearch;
		Instances reducedSet = new Instances(instances, instances.numInstances() / 10);
		Instance inst;
		Vector<Double> classSelected = new Vector<Double>(instances.classAttribute().numValues());
		boolean[] selected = new boolean[instances.numInstances()];
		boolean[] remove = new boolean[instances.numInstances()];

		for (int i = 0; i < instances.numInstances(); i++)
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
		for (int i = 0; i < instances.numInstances(); i++) {
			if (!selected[i])
				remove[i] = true;
			else
				remove[i] = false;
		}

		return remove;
	}
}
