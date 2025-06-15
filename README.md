# AutoVol: LLM-Powered RAM Forensics Agent

AutoVol is a Master's thesis project developing an intelligent agent capable of performing RAM dump analysis using the Volatility 3 framework, guided by Large Language Models (LLMs). The agent, run via a Docker container, interacts with the user through a Command Line Interface (CLI). It takes an initial forensic context, then autonomously (or with user verification) decides which Volatility plugins to run or Python scripts to execute, interprets their output, manages files in a dedicated session workspace, and generates comprehensive reports.

## Core Features

* **Automated Volatility 3 Execution:** Runs Volatility plugins based on LLM reasoning.
* **Python Scripting:** Enables the LLM to generate and execute Python scripts for data processing, analysis, and file manipulation within a secure session workspace.
* **File Workspace Management:** Provides a dedicated directory for each analysis session where Volatility can output files (e.g., process dumps, extracted files) and Python scripts can operate on them. Includes a tool for the LLM to list workspace contents.
* **Intelligent Analysis Flow:** LLM-driven sequential analysis based on initial context and tool outputs.
* **Transparent Reasoning:** Exposes the LLM's thinking process for each step.
* **Human-in-the-Loop:** Allows user review, modification, or denial of proposed actions (Volatility commands or Python scripts).
* **OS Profile Detection:** Automatically attempts to determine the RAM dump's OS profile.
* **Reproducible Reports:** Generates detailed Markdown reports (summary and step-by-step) outlining the analysis, commands, reasoning, and links to generated files.
* **Containerized Environment:** Uses Docker and Docker Compose for easy setup, dependency management, and consistent execution.

## Tech Stack

* **Programming Language:** Python 3.12
* **RAM Forensics Framework:** [Volatility 3](https://github.com/volatilityfoundation/volatility) (v2.26.0 used in Docker image)
* **LLM Agent Framework:** [LangGraph](https://python.langchain.com/docs/langgraph)
* **LLM:** Google Gemini 2.5 Pro (via `langchain-google-vertexai`)
* **CLI Framework:** [Typer](https://typer.tiangolo.com/)
* **Containerization:** Docker, Docker Compose
* **Reporting:** Markdown

## Setup and Running with Docker (Recommended)

Using Docker is the recommended way to run AutoVol, as it handles all dependencies, including Volatility 3.

**Prerequisites:**

* Docker and Docker Compose installed.
* A Google Cloud Platform (GCP) project with Vertex AI API enabled.
* A GCP service account key JSON file (`vertex_key.json`) with permissions for Vertex AI.
* A memory dump file you want to analyze.

**Steps:**

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url> # Or the path to where you have AutoVol
   cd AutoVol
   ```

2. **Prepare Directories and Files:**

   * Create a directory for your memory dumps, e.g., `sample_dumps/` inside the `AutoVol` project directory.

     ```bash
     mkdir -p sample_dumps
     cp /path/to/your/memory_dump.mem sample_dumps/test_dump.mem
     ```
   * Create a directory for your Google Cloud key, e.g., `gcloud_keys/` inside the `AutoVol` project directory.

     ```bash
     mkdir -p gcloud_keys
     cp /path/to/your/vertex_key.json gcloud_keys/vertex_key.json
     ```
   * The `docker-compose.yml` is pre-configured to use these paths (`./sample_dumps/test_dump.mem` and `./gcloud_keys/vertex_key.json`). Adjust the `docker-compose.yml` if you use different names or locations.

3. **Build the Docker Image:**

   ```bash
   docker compose build
   ```

4. **Run AutoVol:**
   To run AutoVol execute:

   ```bash
   docker compose run autovol
   ```

   **Important:** You must use `docker compose run` instead of `docker compose up` to ensure the container runs interactively and allows you access to the CLI and interact with the agent. The `docker compose up` command is typically used for services that run in the background or continuously.

   This will start the analysis with the dump and context specified in the `command:` section of `docker-compose.yml`.

   To run with a different dump or context, you can override the command:

   ```bash
   # Ensure the dump file is correctly mapped in docker-compose.yml volumes first
   docker compose run --rm autovol /mnt/dump.mem --context "Investigate for specific malware signature XYZ"
   ```

   * `autovol` is the service name in `docker-compose.yml`.
   * `/mnt/dump.mem` is the path *inside the container* where the dump is mounted.
   * `--rm` automatically removes the container when it exits.

5. **Output:**

   * Reports and any files generated in the session workspace will be saved to the `./autovol_output` directory on your host machine (this path is mapped from `/app/reports` inside the container).

## Local Development Setup (Alternative, more complex)

If you prefer to run locally without Docker (not recommended for full dependency management):

1. **Python:** Version 3.12 or higher.

2. **Volatility 3 (v2.26.0 or compatible):**

   * Cloned and set up. Its `vol` command should ideally be on your system PATH, or you can set the `VOLATILITY3_PATH` environment variable to point directly to the `vol.py` script (though the application now primarily relies on `vol` being on PATH).

3. **Google Cloud Authentication:** As per Docker prerequisites.

4. **Create Virtual Environment & Install Dependencies:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate # or .venv\\Scripts\\activate on Windows
   pip install -r requirements.txt
   ```

5. **Set Environment Variables (Optional, if not using a `.env` file):**

   * `GOOGLE_APPLICATION_CREDENTIALS=path/to/your/vertex_key.json`

6. **Run:**

   ```bash
   python -m autovol.main /path/to/your/memory_dump.mem --context "Your forensic query"
   ```

## Disclaimer

AutoVol is an experimental project developed for a Master's thesis. It is intended for educational and research purposes. While it aims for forensic soundness, always verify findings with established manual procedures and tools. **Do not rely solely on AutoVol for critical investigations without thorough validation.**
