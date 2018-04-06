package meka.filters.multilabel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import meka.core.A;
import meka.core.OptionUtils;
import meka.core.SuperLabelUtils;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;

/**
 * Instance selection for ML by means of RAkEL.<br>
 * The threshold is computed using a voting method like DIS.
 * <p>
 * Valid options are:
 * <p>
 * number of nearest neighbours <br>
 * alpha for fitness function <br>
 * percentage of instances for error computation (in fitness function) <br>
 * number of labels in each partition <br>
 * number of subsets <br>
 * pruning values for frequent/infrequent labels <br>
 * 
 * @author Álvar Arnaiz-González
 * @version 20180402
 */
public class RAkELIS extends BRIS {

	private static final long serialVersionUID = 8623949588377551176L;

	/**
	 * Instance selection algorithm to use.
	 */
	protected LPIS m_IS = new LPENN();

	/**
	 * The number of labels in each partition.
	 */
	protected int m_K = 3;

	/**
	 * the number of subsets
	 */
	protected int m_M = 10;

	protected int m_N = 0;

	protected int m_P = 0;

	/**
	 * Computes the votes using RAkEL-based IS.
	 * 
	 * @param instances Instances to filter.
	 * @param remove Array with votes for removal.
	 * @throws Exception If something goes wrong.
	 */
	protected void computeVotes(Instances instances, int[] remove) throws Exception {
		Random random = new Random(m_Seed);
		int L = instances.classIndex();
		int kMap[][];

		// Note: a slightly roundabout way of doing it:
		int num = (int) Math.ceil(L / m_K);
		kMap = SuperLabelUtils.generatePartition(A.make_sequence(L), num, random, true);
		m_M = kMap.length;

		if (getDebug())
			System.out.println("Building " + m_M + " models of " + m_K + " partitions:");

		for (int i = 0; i < m_M; i++) {
			if (getDebug())
				System.out.println("\tpartitioning model " + (i + 1) + "/" + m_M + ": " 
				                   + Arrays.toString(kMap[i]) + ", P=" + m_P + ", N=" + m_N);
			
			// Performs the selection of the desired instances and label powerset.
			Instances D_i = SuperLabelUtils.makePartitionDataset(instances, kMap[i], m_P, m_N);

			if (getDebug())
				System.out.println("\tbuilding model " + (i + 1) + "/" + m_M + ": "
				                    + Arrays.toString(kMap[i]));
			
			applyIS(D_i, remove);
		}
	}

	@Override
	protected void applyIS(Instances instances, int[] remove) throws Exception {
		// Apply IS algorithm on the data set
		boolean[] rem = m_IS.applyIS(instances);

		// Accumulate the removal votes.
		for (int i = 0; i < rem.length; i++)
			if (rem[i])
				remove[i]++;

	}

	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>();

		result.addElement(new Option("\tSets M (default 10): the number of subsets", "M", 1, "-M <num>"));
		
		result.addElement(new Option(
				"\tSets the pruning value, defining an infrequent labelset as one which "
						+ "occurs <= P times in the data (P = 0 defaults to LC).\n\tdefault: " + m_P + "\t(LC)",
				"P", 1, "-P <value>"));

		result.addElement(new Option("\tSets the (maximum) number of frequent labelsets to subsample from the "
				+ "infrequent labelsets.\n\tdefault: " + m_N + "\t(none)\n\tn\tN = n\n\t-n"
				+ "\tN = n, or 0 if LCard(D) >= 2\n\tn-m\tN = random(n,m)", "N", 1, "-N <value>"));

		result.addElement(new Option("\t" + kTipText(), "k", 1, "-k <num>"));

		result.addElement(new Option("\tFull name of base instance selector.\n" + "\t(default: " + defaultISString()
				+ ((defaultISOptions().length > 0) ? " with options " + Utils.joinOptions(defaultISOptions()) + ")"
						: ")"),
				"W", 1, "-W"));

		result.addAll(Collections.list(super.listOptions()));

		result.addElement(new Option("", "", 0, "\nOptions specific to classifier " + m_IS.getClass().getName() + ":"));

		result.addAll(Collections.list(((OptionHandler) m_IS).listOptions()));

		return OptionUtils.toEnumeration(result);
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		String tmpStr;
		String instSelecName = Utils.getOption('I', options);

		if (instSelecName.length() > 0) {
			setIS((Filter) Utils.forName(LPIS.class, instSelecName, null));
			setIS((Filter) Utils.forName(LPIS.class, instSelecName, Utils.partitionOptions(options)));
		} else {
			setIS((Filter) Utils.forName(LPIS.class, defaultISString(), null));
			String[] classifierOptions = Utils.partitionOptions(options);

			if (classifierOptions.length > 0) {
				setIS((Filter) Utils.forName(LPIS.class, defaultISString(), classifierOptions));
			} else {
				setIS((Filter) Utils.forName(LPIS.class, defaultISString(), defaultISOptions()));
			}
		}

		setM(OptionUtils.parse(options, 'M', 10));
		
		tmpStr = Utils.getOption('P', options);
		if (tmpStr.length() != 0)
			setP(parseValue(tmpStr));
		else
			setP(parseValue("0"));

		tmpStr = Utils.getOption('N', options);
		if (tmpStr.length() != 0)
			setN(parseValue(tmpStr));
		else
			setN(parseValue("0"));

		setK(OptionUtils.parse(options, 'k', 3));

		super.setOptions(options);
	}

	private int parseValue(String s) {
		int i = s.indexOf('-');

		Random m_R = new Random(m_Seed);

		if (i > 0 && i < s.length()) {
			int lo = Integer.parseInt(s.substring(0, i));
			int hi = Integer.parseInt(s.substring(i + 1, s.length()));

			return lo + m_R.nextInt(hi - lo + 1);
		}

		return Integer.parseInt(s);
	}

	@Override
	public String[] getOptions() {
		List<String> result = new ArrayList<String>();

		OptionUtils.add(result, 'M', getM());
		OptionUtils.add(result, 'P', getP());
		OptionUtils.add(result, 'N', getN());
		OptionUtils.add(result, 'k', getK());
		
		OptionUtils.add(result, super.getOptions());

		result.add("-I");
		result.add(getIS().getClass().getName());

		String[] filterOptions = ((OptionHandler) m_IS).getOptions();
		if (filterOptions.length > 0) {
			result.add("--");
			Collections.addAll(result, filterOptions);
		}

		return OptionUtils.toArray(result);
	}

	/**
	 * Get the k parameter (the size of each partition).
	 */
	public int getK() {
		return m_K;
	}

	/**
	 * Sets the k parameter (the size of each partition)
	 */
	public void setK(int k) {
		m_K = k;
	}

	public String kTipText() {
		return "The number of labels in each partition -- should be "
				+ "1 <= k < (L/2) where L is the total number of labels.";
	}

	/**
	 * Gets the pruning value P.
	 * 
	 * @return pruning value P.
	 */
	public int getP() {
		return m_P;
	}

	/**
	 * Sets the pruning value P, defining an infrequent labelset as one which
	 * occurs less than P times in the data (P = 0 defaults to LC).
	 * 
	 * @param p pruning value.
	 */
	public void setP(int p) {
		m_P = p;
	}

	public String pTipText() {
		return "The pruning value P, defining an infrequent labelset as one which "
				+ "occurs less than P times in the data (P = 0 defaults to LC).";
	}

	/**
	 * Gets the M parameter (the number of subsets).
	 * 
	 * @return the number of subsets
	 */
	public int getM() {
		return m_M;
	}

	/**
	 * Sets the M parameter (the number of subsets).
	 * 
	 * @param m the number of subsets
	 */
	public void setM(int m) {
		m_M = m;
	}

	public String mTipText() {
		return "The number of subsets to draw (which together form an ensemble)";
	}

	/**
	 * Gets the subsampling value N.
	 * 
	 * @return the (maximum) number of frequent labelsets to subsample 
	 *          from the infrequent labelsets.
	 */
	public int getN() {
		return m_N;
	}

	/**
	 * Sets the subsampling value N.
	 * 
	 * @param n the (maximum) number of frequent labelsets to subsample 
	 *          from the infrequent labelsets.
	 */
	public void setN(int n) {
		m_N = n;
	}

	public String nTipText() {
		return "The subsampling value N, the (maximum) number of frequent labelsets "
				+ "to subsample from the infrequent labelsets.";
	}

	/**
	 * String describing default IS.
	 * 
	 * @return BRENN.
	 */
	protected String defaultISString() {

		return "meka.filters.multilabel.BRENN";
	}

	/**
	 * String describing options for default IS.
	 * 
	 * @return empty
	 */
	protected String[] defaultISOptions() {

		return new String[0];
	}

	/**
	 * Set the base IS.
	 *
	 * @param newIS the filter to use.
	 */
	public void setIS(Filter newIS) {

		m_IS = (LPIS) newIS;
	}

	/**
	 * Get the filter used as the base learner.
	 *
	 * @return the IS used as the filter
	 */
	public Filter getIS() {

		return m_IS;
	}

	/**
	 * Gets the classifier specification string, which contains the class name of
	 * the classifier and any options to the classifier
	 *
	 * @return the classifier string
	 */
	protected String getISSpec() {
		Filter c = getIS();

		return c.getClass().getName() + " " + Utils.joinOptions(((OptionHandler) c).getOptions());
	}

	public String isTipText() {
		return "The base instance selection algorithm to be used.";
	}

}
