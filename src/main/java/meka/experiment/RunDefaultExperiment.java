/*
 * RunDefaultExperiment.java
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
package meka.experiment;

import meka.events.LogListener;
import meka.experiment.events.ExecutionStageEvent;

/**
 * Runs experiments in command-line interface.
 * <p>
 * 
 * @author Álvar Arnaiz-González
 * @version 20160621
 */
public class RunDefaultExperiment extends DefaultExperiment {

	private static final long serialVersionUID = -1102576778931135005L;

	public static void main(String[] args) {
		Experiment exp = null;

		try {
			exp = new RunDefaultExperiment();
			exp.setOptions(args);
			System.out.println("Initializing...");
			exp.initialize();
			System.out.println("Running...");
			exp.run();
			System.out.println("Finishing...");
			exp.finish();
			System.out.println("Finished!");
		} catch (Exception ex) {
			System.err.println(ex.getMessage());
			ex.printStackTrace();
		}
	}

	/**
	 * Initializes the experiment. Avoid the problem with one thread.
	 *
	 * @return          null if successfully initialized, otherwise error message
	 */
	public String initialize() {
		String      result;

		debug("pre: init");
		m_Initializing = true;
		m_Running      = false;
		m_Stopping     = false;

		notifyExecutionStageListeners(ExecutionStageEvent.Stage.INITIALIZING);

		ExperimentUtils.ensureThreadSafety(this);

		for (LogListener l: getLogListeners()) {
			m_DatasetProvider.addLogListener(l);
			m_StatisticsHandler.addLogListener(l);
			m_Evaluator.addLogListener(l);
		}

		m_Statistics.clear();
		result = handleError(m_DatasetProvider, m_DatasetProvider.initialize());
		if (result == null)
			result = handleError(m_StatisticsHandler, m_StatisticsHandler.initialize());

		if (result != null)
			log(result);

		m_Initializing = false;
		debug("post: init");

		return result;
	}

}
