# 🧠 Lightweight Cognitive Brain - Quick Setup Guide

## ✨ What's Different?

The **lightweight version** uses **ONLY Gemini API** - no heavy ML libraries needed!

### Traditional Version vs Lite Version

| Feature | Traditional | Lite (Gemini-Only) |
|---------|------------|-------------------|
| **Download Size** | 3-4 GB | < 50 MB |
| **Install Time** | 10-15 minutes | 30 seconds |
| **Dependencies** | torch, transformers, sentence-transformers, spacy | Just google-genai + basic utils |
| **Functionality** | 100% | 100% (same capabilities!) |
| **Speed** | Local ML models | Cloud API (fast with good internet) |

## 🚀 Quick Start (3 Steps)

### Step 1: Install Lightweight Dependencies

```bash
cd "C:\Users\USER\PycharmProjects\Brain"
pip install -r requirements_lite.txt
```

This installs only:
- google-genai (Gemini API)
- pydantic, structlog, numpy
- pytest (optional, for testing)

**Total install time: ~30 seconds** ⚡

### Step 2: Get Your Gemini API Key

1. Visit: https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy your key (starts with "AIza...")

### Step 3: Configure Your API Key

Edit `.env` file:

```env
GEMINI_API_KEY=AIzaSy...  # ← Paste your actual API key here
GEMINI_MODEL=gemini-2.0-flash-exp
```

## 🎯 Run the Cognitive Brain

### Option 1: Interactive Mode

```bash
python main_lite.py
```

Then type your queries:
```
You: What would happen if gravity was twice as strong?
Brain: [reasoning and analysis...]
```

### Option 2: Single Query Mode

```bash
python main_lite.py "Explain the relationship between consciousness and attention"
```

## 📋 Example Queries

Try these to test different brain regions:

```bash
# Logical Reasoning
python main_lite.py "If all birds can fly and penguins are birds, can penguins fly?"

# Creative Reasoning
python main_lite.py "Design a city for people who can fly"

# Causal Reasoning
python main_lite.py "What would happen if all insects disappeared?"

# Analogical Reasoning
python main_lite.py "How is the human brain similar to the internet?"
```

## 🧪 Verify Setup

Check if everything is working:

```bash
# Check API key is configured
python check_api_key.py

# Test imports
python -c "from brain_regions.language.comprehension_lite import LanguageComprehensionLite; print('✓ Imports working')"

# Quick test
python main_lite.py "Hello"
```

## 🏗️ Architecture (Lite Version)

The lite version uses these brain regions:

- **Language Processing**
  - LanguageComprehensionLite (Gemini-powered NLP)
  - LanguageProduction (Gemini-powered text generation)
  - SemanticDecoderLite (Gemini embeddings)

- **Memory Systems**
  - WorkingMemory (7±2 item capacity with compression)
  - Hippocampus (episodic memory with replay)

- **Reasoning Modules**
  - LogicalReasoning (deductive/inductive logic)
  - AnalogicalReasoning (cross-domain mapping)
  - CausalReasoning (cause-effect chains)
  - CreativeReasoning (novel solutions)

- **Integration**
  - GlobalWorkspace (consciousness-like integration)

- **Executive Control**
  - PrefrontalCortex (planning and goal management)

## 📊 What's Replaced?

### Before (Heavy Dependencies):
- sentence-transformers → **Gemini Embedding API**
- spacy (NLP) → **Gemini Text Analysis**
- torch/transformers → **Not needed!**

### How It Works:

1. **Text Embeddings**: Instead of downloading a 500MB sentence-transformer model, we use Gemini's `text-embedding-004` model via API

2. **NLP Tasks**: Instead of spacy models, we use Gemini to extract:
   - Tokens, lemmas, POS tags
   - Named entities
   - Dependency relationships

3. **Same Results**: The cognitive architecture works exactly the same - just powered by cloud API instead of local models

## 🔧 Troubleshooting

### Import Errors

If you see import errors about missing modules:

```bash
# Reinstall lite requirements
pip install -r requirements_lite.txt

# Check what's installed
pip list | grep -E "(genai|pydantic|structlog|numpy)"
```

### API Key Issues

```bash
# Verify key is set
python check_api_key.py

# Should show: "API key configured: True"
```

If False, edit `.env` and paste your actual Gemini API key.

### Rate Limiting

If you hit Gemini API rate limits:
- Free tier: 15 requests/minute
- Paid tier: Higher limits

Wait a moment and try again, or upgrade your API key at https://aistudio.google.com/

## 💡 Tips

1. **Internet Required**: Lite version needs internet for Gemini API calls
2. **Caching**: Embeddings are cached to reduce API calls
3. **Cost**: Gemini API has generous free tier, then ~$0.00015 per 1K chars
4. **Speed**: With good internet, often faster than local models!

## 🆚 When to Use Which Version?

### Use Lite Version (main_lite.py) When:
- ✅ Quick setup needed
- ✅ Limited disk space
- ✅ Good internet connection
- ✅ Don't want to manage ML models
- ✅ Want latest Gemini capabilities

### Use Full Version (main.py) When:
- ✅ Offline processing required
- ✅ Maximum privacy (local models)
- ✅ Already have models downloaded
- ✅ Need specific model versions

## 📚 Next Steps

Once you have the lite version running:

1. Try different types of queries
2. Check `examples/` directory for complex scenarios
3. Monitor logs: `INFO` level shows cognitive pipeline
4. Explore the Global Workspace integration
5. Build your own cognitive applications!

## 🤝 Support

- Issues: Check error messages in logs
- API Problems: https://ai.google.dev/gemini-api/docs
- Project Issues: Create GitHub issue

---

**Ready to think? Run `python main_lite.py` and start asking questions! 🧠✨**
