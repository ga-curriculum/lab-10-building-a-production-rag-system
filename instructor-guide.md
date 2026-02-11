# Lab 10 Instructor Guide: Building a Production RAG System

**Duration:** 60 minutes  
**Format:** Hands-on Lab

---

## Lab Overview

Students build a complete RAG system for an HR Q&A assistant. They'll load policy documents, create a vector store, build a LangChain RAG chain, evaluate retrieval quality, and implement production patterns like fallback responses.

---

## Pre-Lab Checklist

### Requirements

- [ ] Python 3.8+
- [ ] OpenAI API key
- [ ] `pip install langchain langchain-openai langchain-community chromadb`
- [ ] Internet connection

### Common Setup Issues

| Issue | Solution |
|-------|----------|
| LangChain import errors | Ensure all langchain packages installed (base + openai + community) |
| ChromaDB SQLite errors | Update SQLite or use `chromadb.Client()` instead of persistent |
| Rate limits | Add delays between API calls |
| Memory issues | Reduce chunk count or use smaller batches |

---

## Time Allocation

| Part | Duration | Focus |
|------|----------|-------|
| Part 0: Setup | 5 min | Environment, sample docs |
| Part 1: Load & Chunk | 10 min | Chunking strategies |
| Part 2: Vector Store | 10 min | ChromaDB, similarity search |
| Part 3: RAG Chain | 15 min | LangChain, custom prompts |
| Part 4: Evaluation | 10 min | Retrieval quality, comparison |
| Part 5: Production | 10 min | Fallback, conversation |

---

## Part-by-Part Guidance

### Part 1: Load and Chunk (10 min)

**Key Points:**
- Show all three chunking strategies
- Compare chunk counts (small=many, large=few)
- Examine individual chunks for quality

**Teaching Moment:**
> "Look at this chunk. Does it make sense on its own? Could you answer a question from just this text? That's the test for good chunking."

**Common Mistakes:**
- Not understanding chunk_overlap purpose
- Choosing chunks too small (fragments) or too large (noise)

---

### Part 2: Vector Store (10 min)

**Key Points:**
- ChromaDB auto-embeds documents
- Similarity search returns most relevant chunks
- Lower distance = more similar

**Teaching Moment:**
When showing search with scores:
> "This score tells us how confident the retrieval is. Low scores mean good matches. This is what we'll use for fallback logic later."

---

### Part 3: RAG Chain (15 min)

**Key Points:**
- Basic chain works but lacks control
- Custom prompt adds citations and fallback language
- `return_source_documents=True` enables verification

**Critical Teaching Moment:**
The custom prompt is where hallucination prevention happens:
> "Notice we explicitly say 'Answer based ONLY on the provided context.' This instruction, combined with temperature=0, constrains the LLM to use our documents."

**Discussion Prompt:**
> "What happens if we remove the 'only' from the prompt?"

*Answer: LLM may add information from training data, causing hallucinations.*

---

### Part 4: Evaluation (10 min)

**Key Points:**
- Ground truth dataset enables measurement
- Retrieval can be evaluated separately from generation
- Chunk size affects retrieval quality

**Teaching Moment:**
> "We're measuring retrieval, not the final answer. If retrieval fails, no LLM can save us. This is why we test retrieval first."

**Expected Results:**
- Medium chunks (500) should perform best
- Small chunks may miss context
- Large chunks may return irrelevant info

---

### Part 5: Production Patterns (10 min)

**Key Points:**
- Fallback prevents confident wrong answers
- Confidence threshold is tunable
- Conversation memory enables follow-ups

**Teaching Moment:**
> "The fallback is actually a feature, not a limitation. 'I don't know, please contact HR' is infinitely better than a wrong answer about vacation policy."

---

## Common Student Questions

### Q: Why not just put all docs in the prompt?
**A:** Token limits and cost. Our policy doc is 3,000 chars. Real knowledge bases are millions. RAG retrieves only what's needed.

### Q: What chunk size should I use in production?
**A:** Start with 500, measure retrieval quality, adjust. There's no universal answer — it depends on your documents.

### Q: How do I know if retrieval or generation is failing?
**A:** Check source_documents. If the right info is there but the answer is wrong → generation problem. If the wrong docs are retrieved → retrieval problem.

### Q: Can I use a different embedding model?
**A:** Yes! Try Cohere, Hugging Face models, or larger OpenAI models. The process is the same.

### Q: How do I update the knowledge base?
**A:** Re-chunk and re-embed the changed documents. ChromaDB supports upsert for updates.

---

## Assessment Criteria

| Criterion | Points | Description |
|-----------|--------|-------------|
| Setup Complete | 10 | Documents created, imports working |
| Chunking Done | 15 | Three strategies compared |
| Vector Store | 15 | ChromaDB created, search working |
| RAG Chain | 25 | Custom prompt, citations working |
| Evaluation | 20 | Retrieval measured, comparison done |
| Fallback | 15 | Confidence-based routing working |

**Total: 100 points**

---

## Wrap-Up Discussion (5 minutes)

After the lab, discuss:

1. **What was the most important thing you learned?**
   - Expected: Chunking matters, retrieval before generation

2. **How would you improve this system for production?**
   - Expected: More documents, better chunking, hybrid search, user feedback

3. **When would you NOT use RAG?**
   - Expected: Small static knowledge, style adaptation, real-time data

---

## Connection to Course

Set expectations:

> "You've now built the retrieval layer (Lesson 9) and the full RAG pipeline (Lesson 10). These patterns power most production AI assistants — from customer support to internal knowledge bases. The next module covers Feature Stores, which manage structured features for ML models."

---

## Post-Lab Notes

- Students who finish early should try stretch goals
- Note common chunking mistakes for future improvements
- Collect feedback on evaluation section (was it clear?)

---

## Resources for Instructors

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [RAGAS Evaluation](https://docs.ragas.io/)
