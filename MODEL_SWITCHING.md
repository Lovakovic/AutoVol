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
  model="claude-3-5-sonnet-20241022",
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
  model="gpt-4o",
  temperature=0.7,
  max_tokens=8000,
)
```

## Testing

After switching models, test with:
```bash
python test_claude.py  # Modify for your chosen model
```

## Notes

- All three models support tool calling, which is required for AutoVol
- Token usage tracking is handled automatically for all models
- Context window limits vary by model:
  - Claude 3.5 Sonnet: 200K tokens
  - Gemini 2.5 Pro: 2M tokens
  - GPT-4o: 128K tokens

## Model-Specific Considerations

### Claude
- Excellent at following complex instructions
- Strong reasoning capabilities
- May include thinking process in metadata

### Gemini
- Very large context window
- Good for processing large amounts of forensic data
- Multimodal capabilities for image analysis

### OpenAI
- Well-tested and stable
- Good balance of speed and capability
- Strong tool use implementation