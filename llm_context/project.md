**Project Title:** AutoVol

**Core Objective:** To develop an LLM-powered agent () capable of assisting digital forensics analysts in performing RAM dump analysis using the Volatility 3 framework. The goal is to create a tool that is both innovative and practically useful, automating parts of the analysis workflow while maintaining transparency and reproducibility.

**Core Concept:**
The agent interacts with the user via a Command Line Interface (CLI), typically run within a Docker container. The user provides a RAM dump file (mounted into the container) and initial context. The agent, orchestrated by LangGraph, then interacts with an LLM (Gemini 2.5 Pro) to:
1.  Determine the appropriate OS profile for the dump.
2.  Based on the user's context and the results of previous commands, reason about the next logical Volatility 3 plugin, Python script, or workspace file listing to execute.
3.  Formulate the precise Volatility 3 command (plugin name + arguments), Python code, or file listing parameters.
4.  Execute the command/code via secure wrappers (, , ). Volatility and Python scripts operate within a session-specific  directory where files can be created (e.g., by Volatility plugins like ) and then processed (e.g., by Python scripts).
5.  Receive the output from the executed tool.
6.  Present the LLM's reasoning, the executed command/code, and a (potentially truncated) view of the output to the user, with an option for human review and modification of proposed actions.
7.  Feed the output back to the LLM for analysis and determination of the next step.
8.  Repeat steps 2-7 until the analysis based on the initial context is deemed complete by the LLM or stopped by the user.
9.  Generate a final Markdown report detailing the entire analysis process (reasoning, commands, outputs, links to files in  and ) for reproducibility and review.

**Architecture & Technology Stack:**
*   **Language:** Python 3.12
*   **Agent Framework:** LangGraph
*   **LLM:** Gemini 2.5 Pro Preview 05-06 (via  library)
*   **Volatility:** Volatility 3 (v2.26.0 as per Docker setup, executed as a subprocess via  CLI tool installed in the container).
*   **Interface:** CLI (using )
*   **Containerization:** Docker and Docker Compose for environment and dependency management.
*   **State Management:** In-memory (within LangGraph state for a session).
*   **Configuration (primarily for Docker environment):**
    *   Google Cloud credentials () mounted into the container.
    *   Paths for RAM dump input and report/workspace output managed via Docker volumes.
    *    file can still be used for local development outside Docker, but  is less critical as the system defaults to using  from PATH.
*   **Reporting:** Markdown file generation (summary and detailed steps).
*   **Key Components:**
    *   : CLI entry point.
    *   : LangGraph definition (AppState, nodes, edges), tool bindings.
    *   : Contains functions to execute Volatility plugins (using  from PATH), detect profiles. Defines the .
    *   : Executes Python code within the session workspace. Defines the .
    *   : Provides file system utilities for the agent within the session workspace (e.g., ).
    *   : Contains the system prompt for the LLM.
    *   : Defines the Docker image, including Volatility 3 (v2.26.0) and all dependencies.
    *   : Simplifies building and running the Docker container with necessary volume mounts.
    *   : Python dependencies for AutoVol.
    *   : Host directory (mounted into container) for output reports. Each session creates a subdirectory like  containing:
        *   
        *   
        *   : Directory for full stdout/stderr of Volatility commands.
        *   : Directory for files dumped by Volatility plugins or created/modified by Python scripts.
        *   : Directory for Python scripts saved for user review before execution.

