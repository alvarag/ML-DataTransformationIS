/*
 * BRLSS.java
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

import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NormalizableDistance;

/**
 * LSS instance selection for ML by means of binary relevance.<br>
 * The threshold is computed using a votation method like DIS.
 * <p>
 * Valid options are:
 * <p>
 * alpha for fitness function <br>
 * percentage of instances for error computation (in fitness function) <br>
 * 
 * @author Álvar Arnaiz-González
 * @version 20161031
 */
public class BRLSS extends BRIS {

	private static final long serialVersionUID = -8947084609767341924L;

	@Override
	public String globalInfo() {
		
		return "LSS instance selection by using binary relevance (with voting).";
	}

	@Override
	protected void applyIS(Instances instances, int[] remove) throws Exception {
		int u = 0, h = 0;
		ArrayList<Integer>[] mLocalSets = new ArrayList[instances.numInstances()];
		int[] mNearestEnemies = new int[instances.numInstances()];
		
		// Sort the tmp set according to the distance to their nearest enemy.
		computeLocalSets(instances, mLocalSets, mNearestEnemies);
			
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
				remove[i]++;
		}
	}

	/**
	 * Computes localsets.
	 * 
	 * @param trainSet Training set.
	 * @param localSets Local sets.
	 * @param enemies Array with the enemyies' indexes.
	 */
	protected static void computeLocalSets (Instances trainSet, ArrayList<Integer>[] localSets, int[] enemies) {
		DistanceFunction distanceFunction;
		Instance instI, instJ;
		double[] distances;
		double distNearEnemy;
		
		// DON'T normalize distances.
		distanceFunction = new EuclideanDistance(trainSet);
		((NormalizableDistance)distanceFunction).setDontNormalize(true);
		
		for (int i = 0; i < trainSet.numInstances(); i++) {
			localSets[i] = new ArrayList<Integer>();
			distances = new double[trainSet.numInstances()];
			distNearEnemy = Double.MAX_VALUE;
			
			instI = trainSet.instance(i);
			
			// Compute the distance between instI and the others.
			for (int j = 0; j < trainSet.numInstances(); j++) {
				if (i != j) {
					instJ = trainSet.instance(j);
					distances[j] = distanceFunction.distance(instI, instJ);
					
					if (instI.classValue() != instJ.classValue() && 
					     distances[j] < distNearEnemy) {
						distNearEnemy = distances[j];
						enemies[i] = j;
					}
				}
			}

			// Compute the localset of instI.
			for (int j = 0; j < distances.length; j++)
				if (i != j)
					if (distances[j] < distNearEnemy)
						localSets[i].add(j);
		}
	}
}
