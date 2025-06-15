# Model Switching Guide for AutoVol

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Keys

Add the appropriate API key to your `.env` file:

**For Claude (Anthropic):**
```bash
ANTHROPIC_API_KEY="your-anthropic-api-key"
```

**For OpenAI:**
```bash
OPENAI_API_KEY="your-openai-api-key"
```

**For Gemini (Google Vertex AI):**
- Either set `GOOGLE_APPLICATION_CREDENTIALS` environment variable
- Or place `vertex_key.json` in the project root

### 3. Switch Models in agent.py

In `/autovol/agent.py`, comment/uncomment the desired model configuration:

**To use Claude 3.5 Sonnet:**
```python
# Claude configuration
llm = ChatAnthropic(
  model="claude-sonnet-4-20250514",
  temperature=0.7,
  max_tokens=8000,
)
```

**To use Gemini 2.5 Pro:**
```python
# Gemini configuration
llm = ChatVertexAI(
  model="gemini-2.5-pro-preview-05-06",
  temperature=0.7,
  max_output_tokens=8000,
)
```

**To use GPT-4o:**
```python
# OpenAI configuration
llm = ChatOpenAI(
  model="o4-mini",
  temperature=0.7,
  max_tokens=8000,
)
```
