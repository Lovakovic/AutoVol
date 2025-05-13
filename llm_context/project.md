**Project Title:** AutoVol

**Core Objective:** To develop an LLM-powered agent (`AutoVol`) capable of assisting digital forensics analysts in performing RAM dump analysis using the Volatility 3 framework. The goal is to create a tool that is both innovative ("impressive") and practically useful, automating parts of the analysis workflow while maintaining transparency and reproducibility.

**Core Concept:**
The agent interacts with the user via a Command Line Interface (CLI). The user provides a RAM dump file and initial context (e.g., suspicion of malware, type of activity to investigate). The agent, orchestrated by LangGraph, then interacts with an LLM (Gemini 2.5 Pro) to:
1.  Determine the appropriate OS profile for the dump.
2.  Based on the user's context and the results of previous commands, reason about the next logical Volatility 3 plugin to execute.
3.  Formulate the precise Volatility 3 command (plugin name + arguments).
4.  Execute the command via a secure wrapper (`volatility_runner.py`).
5.  Receive the output from Volatility.
6.  Present the LLM's reasoning, the executed command, and a (potentially truncated) view of the output to the user.
7.  Feed the output back to the LLM for analysis and determination of the next step.
8.  Repeat steps 2-7 until the analysis based on the initial context is deemed complete by the LLM or stopped by the user.
9.  Generate a final Markdown report detailing the entire analysis process (reasoning, commands, outputs) for reproducibility and review.

**Architecture & Technology Stack:**
*   **Language:** Python 3.12
*   **Agent Framework:** LangGraph
*   **LLM:** Gemini 2.5 Pro Preview 05-06 (via `langchain-google-vertexai` library)
*   **Volatility:** Volatility 3 (executed as a subprocess)
*   **Interface:** CLI (using `Typer`)
*   **State Management:** In-memory (within LangGraph state)
*   **Configuration:** `.env` file for API keys and Volatility path (`ANTHROPIC_API_KEY`, `VOLATILITY3_PATH`)
*   **Reporting:** Markdown file generation
*   **Key Components:**
    *   `main.py`: CLI entry point.
    *   `agent.py`: LangGraph definition (AppState, nodes, edges).
    *   `volatility_runner.py`: Contains functions to execute Volatility (`execute_volatility_plugin`), detect profiles (`detect_profile`), and defines the LangChain tool (`volatility_runner_tool`) for the LLM.
    *   `reports/`: Directory for output reports.
