/*
 * LPLSS.java
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

import weka.core.Instances;

/**
 * LSS instance selection for ML by means of local powerset.<br>
 * <p>
 * 
 * @author Álvar Arnaiz-González
 * @version 20171226
 */
public class LPLSS extends LPIS {

	private static final long serialVersionUID = -8947084609767341124L;

	@Override
	public String globalInfo() {

		return "LSS instance selection by using local powerset.";
	}

	@Override
	protected boolean[] applyIS(Instances instances) throws Exception {
		ArrayList<Integer>[] mLocalSets = new ArrayList[instances.numInstances()];
		int[] mNearestEnemies = new int[instances.numInstances()];
		boolean[] remove = new boolean[instances.numInstances()];
		int u = 0, h = 0;

		for (int i = 0; i < instances.numInstances(); i++)
			remove[i] = false;

		// Sort the tmp set according to the distance to their nearest enemy.
		BRLSS.computeLocalSets(instances, mLocalSets, mNearestEnemies);

		// Computes u(e).
		for (int i = 0; i < instances.numInstances(); i++) {
			for (ArrayList<Integer> localset : mLocalSets)
				for (int indexLocalSet : localset)
					if (indexLocalSet == i)
						u++;

			// Computes h(e).
			for (int ne : mNearestEnemies)
				if (ne == i)
					h++;

			// If u(e) >= h(e) add i-th instance to the solution.
			if (u < h)
				remove[i] = true;
		}

		return remove;
	}
}
