/*
 * MLNormalize.java
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
package meka.filters.unsupervised.attribute;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.Utils;
import weka.filters.unsupervised.attribute.Normalize;

/**
 * <!-- globalinfo-start --> Normalizes all numeric values in the given ML dataset
 * (apart from the class attribute, if set). The resulting values are by default
 * in [0,1] for the data used to compute the normalization intervals. But with
 * the scale and translation parameters one can change that, e.g., with scale =
 * 2.0 and translation = -1.0 you get values in the range [-1,+1].
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -unset-class-temporarily
 *  Unsets the class index temporarily before the filter is
 *  applied to the data.
 *  (default: no)
 * </pre>
 * 
 * <pre>
 * -S &lt;num&gt;
 *  The scaling factor for the output range.
 *  (default: 1.0)
 * </pre>
 * 
 * <pre>
 * -T &lt;num&gt;
 *  The translation of the output range.
 *  (default: 0.0)
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @author Álvar Arnaiz González (alvarag at ubu dot es)
 * @version $Revision: 10215 $
 */
public class MLNormalize extends Normalize {

	private static final long serialVersionUID = 6842399446112595228L;

	/**
	 * Signify that this batch of input to the filter is finished. If the filter
	 * requires all instances prior to filtering, output() may now be called to
	 * retrieve the filtered instances.
	 * 
	 * @return true if there are instances pending output
	 * @throws Exception
	 *             if an error occurs
	 * @throws IllegalStateException
	 *             if no input structure has been defined
	 */
	@Override
	public boolean batchFinished() throws Exception {
		if (getInputFormat() == null) {
			throw new IllegalStateException("No input instance format defined");
		}

		if (m_MinArray == null) {
			Instances input = getInputFormat();
			// Compute minimums and maximums
			m_MinArray = new double[input.numAttributes()];
			m_MaxArray = new double[input.numAttributes()];
			for (int i = 0; i < input.numAttributes(); i++) {
				m_MinArray[i] = Double.NaN;
			}

			for (int j = 0; j < input.numInstances(); j++) {
				double[] value = input.instance(j).toDoubleArray();
				for (int i = 0; i < input.numAttributes(); i++) {
					if (input.attribute(i).isNumeric()
							&& (input.classIndex() <= i)) {
						if (!Utils.isMissingValue(value[i])) {
							if (Double.isNaN(m_MinArray[i])) {
								m_MinArray[i] = m_MaxArray[i] = value[i];
							} else {
								if (value[i] < m_MinArray[i]) {
									m_MinArray[i] = value[i];
								}
								if (value[i] > m_MaxArray[i]) {
									m_MaxArray[i] = value[i];
								}
							}
						}
					}
				}
			}

			// Convert pending input instances
			for (int i = 0; i < input.numInstances(); i++) {
				convertInstance(input.instance(i));
			}
		}
		// Free memory
		flushInput();

		m_NewBatch = true;
		return (numPendingOutput() != 0);
	}

	/**
	 * Convert a single instance over. The converted instance is added to the
	 * end of the output queue.
	 * 
	 * @param instance
	 *            the instance to convert
	 * @throws Exception
	 *             if conversion fails
	 */
	protected void convertInstance(Instance instance) throws Exception {
		Instance inst = null;
		if (instance instanceof SparseInstance) {
			double[] newVals = new double[instance.numAttributes()];
			int[] newIndices = new int[instance.numAttributes()];
			double[] vals = instance.toDoubleArray();
			int ind = 0;
			for (int j = 0; j < instance.numAttributes(); j++) {
				double value;
				if (instance.attribute(j).isNumeric()
						&& (!Utils.isMissingValue(vals[j]))
						&& (getInputFormat().classIndex() <= j)) {
					if (Double.isNaN(m_MinArray[j])
							|| (m_MaxArray[j] == m_MinArray[j])) {
						value = 0;
					} else {
						value = (vals[j] - m_MinArray[j])
								/ (m_MaxArray[j] - m_MinArray[j]) * m_Scale
								+ m_Translation;
						if (Double.isNaN(value)) {
							throw new Exception("A NaN value was generated "
									+ "while normalizing "
									+ instance.attribute(j).name());
						}
					}
					if (value != 0.0) {
						newVals[ind] = value;
						newIndices[ind] = j;
						ind++;
					}
				} else {
					value = vals[j];
					if (value != 0.0) {
						newVals[ind] = value;
						newIndices[ind] = j;
						ind++;
					}
				}
			}
			double[] tempVals = new double[ind];
			int[] tempInd = new int[ind];
			System.arraycopy(newVals, 0, tempVals, 0, ind);
			System.arraycopy(newIndices, 0, tempInd, 0, ind);
			inst = new SparseInstance(instance.weight(), tempVals, tempInd,
					instance.numAttributes());
		} else {
			double[] vals = instance.toDoubleArray();
			for (int j = 0; j < getInputFormat().numAttributes(); j++) {
				if (instance.attribute(j).isNumeric()
						&& (!Utils.isMissingValue(vals[j]))
						&& (getInputFormat().classIndex() <= j)) {
					if (Double.isNaN(m_MinArray[j])
							|| (m_MaxArray[j] == m_MinArray[j])) {
						vals[j] = 0;
					} else {
						vals[j] = (vals[j] - m_MinArray[j])
								/ (m_MaxArray[j] - m_MinArray[j]) * m_Scale
								+ m_Translation;
						if (Double.isNaN(vals[j])) {
							throw new Exception("A NaN value was generated "
									+ "while normalizing "
									+ instance.attribute(j).name());
						}
					}
				}
			}
			inst = new DenseInstance(instance.weight(), vals);
		}
		inst.setDataset(instance.dataset());
		push(inst);
	}

}
