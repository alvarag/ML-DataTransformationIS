/*
 * LPIS.java
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

import meka.core.PSUtils;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

/**
 * Instance selection for ML by means of label powerset.<br>
 * The code of LP is from LC.java of Jesse Read.
 * <p>
 * 
 * @author Álvar Arnaiz-González
 * @version 20171226
 */
public abstract class LPIS extends SimpleBatchFilter {

	private static final long serialVersionUID = -9154050711637146555L;

	@Override
	public String globalInfo() {

		return "Instance selection by using local powerset.";
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) throws Exception {

		return new Instances(inputFormat, 0);
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		Instances filtered = new Instances(instances);
		boolean remove[];

		// Only change first batch of data
		if (isFirstBatchDone())
			return new Instances(instances);

		int L = instances.classIndex();

		// Transform Instances
		if (getDebug())
			System.out.print("Transforming Instances ...");

		Instances instancesSingleLbl = PSUtils.LCTransformation(instances, L);

		if (getDebug())
			System.out.print("Filtering data set (" + "K = " + instancesSingleLbl.attribute(0).numValues() + ", N = "
					+ instancesSingleLbl.numInstances() + "), ...");

		// Apply instance selection algorithm over single label data set.
		remove = applyIS(instancesSingleLbl);

		// Remove undesired instances.
		for (int i = instances.numInstances() - 1; i >= 0; i--)
			if (remove[i])
				filtered.remove(i);
		
		if (filtered.numInstances() == 0) {
			System.err.println("All instances have been removed, selected the first one instead");
			filtered.add(instances.get(0));
		}

		if (getDebug())
			System.out.println("Done, final size: " + filtered.numInstances());

		return filtered;
	}

	protected abstract boolean[] applyIS(Instances instances) throws Exception;
}
