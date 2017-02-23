/*
 * MLkNN.java
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
package meka.classifiers.multilabel;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;
import meka.classifiers.multilabel.AbstractMultiLabelClassifier;
import meka.classifiers.multilabel.MultiLabelClassifier;
import meka.core.F;
import meka.core.MLUtils;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.lazy.IBLR_ML;
import mulan.data.MultiLabelInstances;

/**
<!-- globalinfo-start -->
* Wrapper for ML KNN classifiers of MULAN.
* <p/>
<!-- globalinfo-end -->
*
<!-- options-start -->
* Valid options are: <p/>
* 
* <pre> -T &lt;selected classifier of MULAN&gt;
*  MLkNN or IBLR_ML.
*   
* <pre> -K &lt;number of nearest neighbours&gt;
*  number of nearest neighbours to use
*   
<!-- options-end -->
*
* @author Álvar Arnaiz-González (alvarag@ubu.es)
* @version $Revision: 100 $
*/
public class MLkNN extends AbstractMultiLabelClassifier implements MultiLabelClassifier {

	private static final long serialVersionUID = 8119923629191799192L;

	protected MultiLabelLearner m_MULAN;

	protected Instances m_InstancesTemplate;

    private boolean isDebug = false;

	protected int m_K = 10;
	
	public static final int TYPE_MLkNN = 0;
	
	public static final int TYPE_IBLR = 1;
	
	public static final Tag[] TAGS_TYPE = {new Tag (TYPE_MLkNN, "MLkNN"),
	                                       new Tag (TYPE_IBLR, "IBLR_ML")};

	protected int mType = TYPE_MLkNN;
	
	public int getNumOfNearestNeighbour () {
		
		return m_K;
	}
	
	public void setNumOfNearestNeighbour (int nn) {
		m_K = nn;
	}
	
	public String numOfNearestNeighbourTipText () {
		
		return "Nearest neighbour number to used.";
	}

    public void setDebug(boolean debug) {
        isDebug = debug;
    }

    public boolean getDebug() {
    	
        return isDebug;
    }
    
    public String isDebugTipText () {
    	
    	return "Whether debugging is turned on.";
    }

    public void setType (SelectedTag value) {
		if (value.getTags() == TAGS_TYPE)
			mType = value.getSelectedTag().getID();
	}

	public SelectedTag getType () {
		
		return new SelectedTag(mType, TAGS_TYPE);
	}
	
	public String typeTipText () {
		
		return "Multilabel kNN algorithm from MULAN.";
	}

	public String[] getOptions () {
		Vector<String> result = new Vector<String>();
		
		result.add("-T");
		result.add("" + mType);
		
		result.add("-K");
		result.add("" + getNumOfNearestNeighbour());
		
		return result.toArray(new String[result.size()]); 
	}

	public Enumeration<Option> listOptions () {
		Vector<Option> newVector = new Vector<Option>();

		newVector.addElement(new Option("\tSpecifies the number of nearest neighbours\n" +
		                                "\t(default 1)", "K", 1, "-K <num>"));
		
		newVector.addElement(new Option("\tSet type of solver (default: 1)\n"+
		                                "\t\t 0 = MLkNN\n"+
		                                "\t\t 1 = IBLR-ML\n",
		                                "T", 1, "-T <int>"));
		
		return newVector.elements();
	}

	public void setOptions (String[] options) throws Exception {
		String numStr = Utils.getOption('K', options);
		String tmpStr = Utils.getOption('T', options);
		
		if (numStr.length() != 0)
			setNumOfNearestNeighbour(Integer.parseInt(numStr));
		else
	    	setNumOfNearestNeighbour(10);
	    
	    if (tmpStr.length() != 0)
	    	setType(new SelectedTag(Integer.parseInt(tmpStr), TAGS_TYPE));
	    else
	    	setType(new SelectedTag(TYPE_MLkNN, TAGS_TYPE));
	}

	@Override
	public void buildClassifier(Instances trainingSet) throws Exception {
	  	testCapabilities(trainingSet);
	  	
		long before = System.currentTimeMillis();
		if (getDebug()) System.err.print(" moving target attributes to the beginning ... ");

		Random r = trainingSet.getRandomNumberGenerator(System.currentTimeMillis());
		String name = "temp_"+MLUtils.getDatasetName(trainingSet)+"_"+r.nextLong()+".arff";
		System.err.println("Using temporary file: "+name);
		int L = trainingSet.classIndex();

		// rename attributes, because MULAN doesn't deal well with hypens etc
		for(int i = L; i < trainingSet.numAttributes(); i++) {
			trainingSet.renameAttribute(i,"a_"+i);
		}
		BufferedWriter writer = new BufferedWriter(new FileWriter(name));
		m_InstancesTemplate = F.meka2mulan(new Instances(trainingSet),L);
		writer.write(m_InstancesTemplate.toString());
		writer.flush();
		writer.close();
		MultiLabelInstances train = new MultiLabelInstances(name,L); 
		try {
			new File(name).delete();
		} catch(Exception e) {
			System.err.println("[Error] Failed to delete temporary file: "+name+". You may want to delete it manually.");
		}

		if (getDebug()) System.out.println(" done ");
		long after = System.currentTimeMillis();

		if (getDebug())
			System.out.println("[Note] Discount "+((after - before)/1000.0)+ " seconds from this build time");

		m_InstancesTemplate = new Instances(train.getDataSet(),0);

		if (mType == TYPE_MLkNN)
			m_MULAN = new mulan.classifier.lazy.MLkNN(getNumOfNearestNeighbour(), 1.0);
		else if (mType == TYPE_IBLR)
			m_MULAN = new IBLR_ML(getNumOfNearestNeighbour());
		
		m_MULAN.setDebug(getDebug());
		m_MULAN.build(train);
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		int L = instance.classIndex();
		
		Instance x = F.meka2mulan((Instance)instance.copy(),L); 
		
		x.setDataset(m_InstancesTemplate);
		
		double y[] = m_MULAN.makePrediction(x).getConfidences();
		
		return y;
	}

	/**
	 * Output a representation of this classifier
	 * 
	 * @return a representation of this classifier
	 */
	public String toString() {
		if (m_MULAN == null)
			return "FilteredClassifier: No model built yet.";

		return "ML kNN wrapper using " + m_MULAN.toString();
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 100 $");
	}

	@Override
	public String getModel() {
		
		return m_MULAN.toString();
	}

}
