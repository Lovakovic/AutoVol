# AutoVol: LLM-Powered RAM Forensics Agent

AutoVol is a Master's thesis project aiming to develop an intelligent agent capable of performing RAM dump analysis using the Volatility 3 framework, guided by Large Language Models (LLMs). The agent interacts with the user via a Command Line Interface (CLI), takes initial forensic context, and then autonomously (or with user verification) decides which Volatility plugins to run, interprets their output, and generates a comprehensive report.

## Project Goals

*   **Automate RAM Analysis:** Leverage LLMs to streamline parts of the RAM forensics workflow.
*   **Intelligent Plugin Selection:** Enable the LLM to choose relevant Volatility 3 plugins based on the investigation context.
*   **Transparent Reasoning:** Expose the LLM's reasoning process to the user for verification and trust.
*   **Reproducible Reports:** Generate detailed Markdown reports outlining the analysis steps, commands executed, and findings.
*   **CLI-First:** Provide a practical and efficient command-line interface for forensic analysts.

## Features (Planned)

*   Automated execution of Volatility 3 plugins.
*   LLM-driven sequential analysis based on initial context and plugin outputs.
*   Automatic detection of the RAM dump's OS profile.
*   Display of LLM reasoning for each step.
*   User interaction for providing initial suspicion/context.
*   Generation of detailed Markdown reports.
*   Management of analysis sessions (in-memory for now).

## Tech Stack

*   **Programming Language:** Python 3.12+
*   **RAM Forensics Framework:** [Volatility 3](https://github.com/volatilityfoundation/volatility)
*   **LLM Agent Framework:** [LangGraph](https://python.langchain.com/docs/langgraph)
*   **LLM:** Google Gemini Pro (via Vertex AI)
*   **LLM Orchestration:** [LangChain](https://python.langchain.com/docs/get_started/introduction) (specifically `langchain-google-vertexai`)
*   **CLI Framework:** [Typer](https://typer.tiangolo.com/)
*   **Reporting:** Markdown

## Prerequisites

1.  **Python:** Version 3.12 or higher.
2.  **Volatility 3:**
    *   Cloned from the official repository: `git clone https://github.com/volatilityfoundation/volatility3.git`
    *   Dependencies installed as per Volatility 3 documentation.
    *   Ensure `vol.py` is executable and preferably accessible in your `PATH` or its location is known.
3.  **Google Cloud Project & Vertex AI:**
    *   A Google Cloud Platform (GCP) project.
    *   Vertex AI API enabled for your project.
    *   Authentication configured for GCP. The most common ways are:
        *   Setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your service account key JSON file.
        *   Running `gcloud auth application-default login` if you're developing locally.
4.  **Environment Variables (Recommended):**
    *   `GOOGLE_CLOUD_PROJECT`: Your GCP Project ID.
    *   `GOOGLE_CLOUD_REGION`: The region for your Vertex AI resources (e.g., `us-central1`).

## Setup

1.  **Clone this repository (if you haven't already):**
    ```bash
    git clone <your-repo-url>
    cd AutoVol
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (We will create `requirements.txt` in the next step).

4.  **Configure Volatility 3 Path (if not in system PATH):**
    You might need to configure the path to your `volatility3/vol.py` script within the `AutoVol` application if it's not globally accessible.

## Basic Usage (Planned)

The primary interaction will be through the CLI:

```bash
python -m autovol.main analyze /path/to/memory.dmp --context "Suspect malware beaconing activity"
```

## Disclaimer

AutoVol is an experimental project developed for a Master's thesis. It is intended for educational and research purposes. While it aims for forensic soundness, always verify findings with established manual procedures and tools. **Do not rely solely on AutoVol for critical investigations without thorough validation.**

