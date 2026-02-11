# Lab 10: Building a Production RAG System

**Duration:** 60 minutes  
**Format:** Hands-on Lab  
**Tools:** LangChain, ChromaDB, OpenAI

---

## Overview

In this lab, you will build a complete RAG (Retrieval-Augmented Generation) system that:
1. Loads and chunks **real documents**
2. Creates a **vector store** for semantic search
3. Builds a **RAG pipeline** with LangChain
4. **Evaluates and improves** retrieval quality
5. Implements **production patterns** (fallback, citations)

By the end, you'll have a working Q&A system that answers questions accurately with source citations.

---

## Learning Objectives

By the end of this lab, you will be able to:

1. Load and chunk documents for RAG
2. Create and query a vector store
3. Build a LangChain RAG chain
4. Evaluate retrieval quality
5. Optimize chunk size and retrieval parameters
6. Implement production fallback behavior

---

## Scenario

You are building a **knowledge assistant** for RetailPulse's HR team. Employees frequently ask questions about company policies:
- "How many vacation days do I get?"
- "What's the process for expense reimbursement?"
- "Can I work from home?"

Currently, employees email HR and wait 24-48 hours for responses. Your RAG system will provide instant, accurate answers with citations to the source policy documents.

---

## Part 0: Environment Setup (5 minutes)

### Step 1: Install Dependencies

```bash
pip install langchain langchain-openai langchain-community chromadb
```

### Step 2: Set Up API Key

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

### Step 3: Create Project Structure

```bash
mkdir -p ~/rag-lab/data
cd ~/rag-lab
```

### Step 4: Create Sample Documents

Create `data/policies.txt`:

```python
policies_content = """
# RetailPulse Employee Policies Handbook

## 1. Vacation and Time Off Policy

### 1.1 Vacation Days
Full-time employees receive 15 vacation days per year. Vacation days accrue monthly at a rate of 1.25 days per month. New employees begin accruing vacation days immediately upon hire.

### 1.2 Vacation Rollover
Up to 5 unused vacation days may be rolled over to the following calendar year. Days exceeding the 5-day rollover limit will be forfeited on December 31st.

### 1.3 Requesting Time Off
All vacation requests must be submitted through the HR Portal at least 2 weeks in advance. Requests are subject to manager approval based on business needs. Employees will receive confirmation within 3 business days.

### 1.4 Vacation Payout
Upon termination of employment, employees will be paid for accrued but unused vacation days at their current rate of pay.

## 2. Sick Leave Policy

### 2.1 Sick Days Allowance
All employees receive 10 sick days per year. Sick days are available immediately upon hire and do not accrue monthly.

### 2.2 Documentation Requirements
A doctor's note is required for sick leave absences exceeding 3 consecutive business days. Notes should be submitted to HR within 5 days of return to work.

### 2.3 Sick Leave Usage
Sick leave may be used for personal illness, medical appointments, or caring for an immediate family member. Sick leave cannot be used for vacation purposes.

## 3. Remote Work Policy

### 3.1 Eligibility
Employees in eligible roles may work remotely up to 3 days per week with manager approval. Some positions require full-time office presence.

### 3.2 Requirements
Remote workers must have reliable high-speed internet (minimum 25 Mbps). Company VPN must be used for all work-related activities. Employees must be available during core hours (10 AM - 3 PM local time).

### 3.3 Equipment
The company provides laptops for remote work. Employees are responsible for their home office setup including desk, chair, and monitors.

## 4. Expense Reimbursement Policy

### 4.1 Eligible Expenses
Business-related travel, meals during travel, conference fees, and pre-approved equipment purchases are eligible for reimbursement.

### 4.2 Submission Requirements
Expense reports must be submitted within 30 days of the expense. Original receipts are required for all expenses over $25. Digital receipts are acceptable.

### 4.3 Approval Process
Expenses under $500 require manager approval only. Expenses between $500-$2000 require director approval. Expenses over $2000 require VP approval.

### 4.4 Reimbursement Timeline
Approved expenses are reimbursed within 2 weeks via direct deposit to the employee's payroll account.

## 5. Performance Review Policy

### 5.1 Review Cycle
Performance reviews are conducted annually in Q1 (January-March). Mid-year check-ins occur in July.

### 5.2 Self-Assessment
Employees must complete a self-assessment form 2 weeks before their scheduled review meeting. Forms are available on the HR Portal.

### 5.3 Rating Scale
Performance is rated on a 5-point scale: 1 (Does Not Meet Expectations), 2 (Partially Meets), 3 (Meets Expectations), 4 (Exceeds), 5 (Outstanding).

### 5.4 Compensation Adjustments
Salary adjustments based on performance reviews take effect April 1st. Bonus eligibility is determined by annual performance rating.
"""

with open("data/policies.txt", "w") as f:
    f.write(policies_content)

print("Policy document created!")
```

---

## Part 1: Load and Chunk Documents (10 minutes)

### Task 1.1: Load Documents

```python
from langchain_community.document_loaders import TextLoader

# Load the policy document
loader = TextLoader("data/policies.txt")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")
print(f"Total characters: {len(documents[0].page_content)}")
print(f"\nFirst 500 characters:\n{documents[0].page_content[:500]}")
```

### Task 1.2: Experiment with Chunking Strategies

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Strategy 1: Small chunks (300 chars)
small_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " ", ""]
)
small_chunks = small_splitter.split_documents(documents)

# Strategy 2: Medium chunks (500 chars) - recommended
medium_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
medium_chunks = medium_splitter.split_documents(documents)

# Strategy 3: Large chunks (1000 chars)
large_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)
large_chunks = large_splitter.split_documents(documents)

print(f"Small chunks (300): {len(small_chunks)} chunks")
print(f"Medium chunks (500): {len(medium_chunks)} chunks")
print(f"Large chunks (1000): {len(large_chunks)} chunks")

# Examine a chunk
print(f"\nSample medium chunk:\n{medium_chunks[2].page_content}")
```

### Task 1.3: Select Best Chunking Strategy

For this lab, we'll use **medium chunks (500 chars)** as they balance precision and context.

```python
# Use medium chunks
chunks = medium_chunks
print(f"Using {len(chunks)} chunks for RAG")
```

---

## Part 2: Create Vector Store (10 minutes)

### Task 2.1: Initialize ChromaDB

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create persistent vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="hr_policies"
)

print(f"Created vector store with {vector_store._collection.count()} vectors")
```

### Task 2.2: Test Vector Search

```python
# Test similarity search
query = "How many vacation days do employees get?"
results = vector_store.similarity_search(query, k=3)

print(f"Query: {query}\n")
print("Top 3 results:")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content[:200] + "...")
```

### Task 2.3: Test Search with Scores

```python
# Get similarity scores
results_with_scores = vector_store.similarity_search_with_score(query, k=3)

print(f"Query: {query}\n")
print("Results with scores (lower = more similar):")
for doc, score in results_with_scores:
    print(f"\nScore: {score:.4f}")
    print(f"Content: {doc.page_content[:150]}...")
```

---

## Part 3: Build RAG Chain (15 minutes)

### Task 3.1: Create Basic RAG Chain

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Create LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Create basic chain
basic_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Test
result = basic_chain.invoke({"query": "How many vacation days do employees get?"})
print(f"Answer: {result['result']}")
```

### Task 3.2: Add Custom Prompt with Citations

```python
# Custom prompt template
template = """You are an HR assistant for RetailPulse. Answer employee questions based ONLY on the policy documents provided below.

INSTRUCTIONS:
- Answer based ONLY on the provided context
- If the answer is not in the context, say "I don't have that information in the policy documents. Please contact HR at hr@retailpulse.com"
- Always cite the specific policy section (e.g., "According to Section 1.1 Vacation Days...")
- Be concise but complete

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Create chain with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
```

### Task 3.3: Test the RAG System

```python
def ask_hr(question: str):
    """Query the HR assistant and display formatted results."""
    result = qa_chain.invoke({"query": question})
    
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")
    print(f"\nANSWER:\n{result['result']}")
    print(f"\n{'='*60}")
    print("SOURCES:")
    for i, doc in enumerate(result['source_documents']):
        print(f"\n[{i+1}] {doc.page_content[:150]}...")
    print(f"{'='*60}\n")

# Test queries
test_questions = [
    "How many vacation days do full-time employees get?",
    "Can I roll over unused vacation days?",
    "What's the process for getting expenses reimbursed?",
    "How do I request time off?",
    "When are performance reviews conducted?",
]

for question in test_questions[:3]:  # Test first 3
    ask_hr(question)
```

---

## Part 4: Evaluate and Improve (10 minutes)

### Task 4.1: Create Evaluation Dataset

```python
# Ground truth Q&A pairs
eval_dataset = [
    {
        "question": "How many vacation days do employees get?",
        "expected_answer": "15 days per year",
        "source_section": "1.1"
    },
    {
        "question": "How many sick days are provided?",
        "expected_answer": "10 sick days per year",
        "source_section": "2.1"
    },
    {
        "question": "How many days can be rolled over?",
        "expected_answer": "Up to 5 unused vacation days",
        "source_section": "1.2"
    },
    {
        "question": "How long do I have to submit expenses?",
        "expected_answer": "Within 30 days",
        "source_section": "4.2"
    },
    {
        "question": "What's the minimum internet speed for remote work?",
        "expected_answer": "25 Mbps",
        "source_section": "3.2"
    },
]
```

### Task 4.2: Evaluate Retrieval Quality

```python
def evaluate_retrieval(dataset, vector_store, k=3):
    """Evaluate if the correct source section is retrieved."""
    hits = 0
    
    for item in dataset:
        results = vector_store.similarity_search(item["question"], k=k)
        
        # Check if any result contains the expected section
        section_found = any(
            item["source_section"] in doc.page_content 
            for doc in results
        )
        
        if section_found:
            hits += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} Q: {item['question'][:50]}...")
        print(f"   Expected section: {item['source_section']}")
    
    recall = hits / len(dataset)
    print(f"\nRetrieval Recall@{k}: {recall:.1%} ({hits}/{len(dataset)})")
    return recall

# Evaluate
print("Retrieval Evaluation:")
evaluate_retrieval(eval_dataset, vector_store)
```

### Task 4.3: Compare Chunk Sizes

```python
def compare_chunk_sizes(document_path, eval_dataset):
    """Compare retrieval quality across different chunk sizes."""
    
    loader = TextLoader(document_path)
    documents = loader.load()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    results = {}
    
    for chunk_size in [300, 500, 800]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1)
        )
        chunks = splitter.split_documents(documents)
        
        # Create temp vector store
        temp_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"temp_{chunk_size}"
        )
        
        # Evaluate
        hits = 0
        for item in eval_dataset:
            search_results = temp_store.similarity_search(item["question"], k=3)
            if any(item["source_section"] in doc.page_content for doc in search_results):
                hits += 1
        
        recall = hits / len(eval_dataset)
        results[chunk_size] = {
            "num_chunks": len(chunks),
            "recall": recall
        }
        
        print(f"Chunk size {chunk_size}: {len(chunks)} chunks, Recall: {recall:.1%}")
    
    return results

# Compare
print("\nChunk Size Comparison:")
comparison = compare_chunk_sizes("data/policies.txt", eval_dataset)
```

---

## Part 5: Production Patterns (10 minutes)

### Task 5.1: Implement Confidence-Based Fallback

```python
def ask_hr_with_fallback(question: str, confidence_threshold: float = 0.5):
    """
    Query HR assistant with confidence-based fallback.
    Returns fallback message if retrieval confidence is low.
    """
    # Check retrieval quality first
    docs_with_scores = vector_store.similarity_search_with_score(question, k=3)
    
    # Best score (lower = more similar, so we convert)
    best_distance = min(score for _, score in docs_with_scores)
    confidence = max(0, 1 - best_distance)  # Approximate confidence
    
    print(f"Retrieval confidence: {confidence:.2f}")
    
    if confidence < confidence_threshold:
        return {
            "answer": "I don't have enough information to answer this question confidently. Please contact HR directly at hr@retailpulse.com or call extension 4567.",
            "confidence": confidence,
            "sources": [],
            "fallback": True
        }
    
    # Proceed with RAG
    result = qa_chain.invoke({"query": question})
    
    return {
        "answer": result["result"],
        "confidence": confidence,
        "sources": [doc.metadata for doc in result["source_documents"]],
        "fallback": False
    }

# Test with in-scope and out-of-scope questions
test_cases = [
    "How many vacation days do I get?",  # In scope
    "What's the policy on bringing dogs to work?",  # Out of scope
    "Do we have a gym membership benefit?",  # Out of scope
]

for question in test_cases:
    print(f"\nQ: {question}")
    result = ask_hr_with_fallback(question)
    print(f"Fallback: {result['fallback']}")
    print(f"A: {result['answer'][:200]}...")
```

### Task 5.2: Add Conversation Memory (Bonus)

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Create conversational chain
conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# Multi-turn conversation
print("Conversational RAG Demo:")
print("-" * 40)

# Turn 1
response1 = conv_chain.invoke({"question": "How many vacation days do I get?"})
print(f"Q: How many vacation days do I get?")
print(f"A: {response1['answer']}\n")

# Turn 2 (follow-up)
response2 = conv_chain.invoke({"question": "Can I roll them over?"})
print(f"Q: Can I roll them over?")
print(f"A: {response2['answer']}\n")

# Turn 3 (another follow-up)
response3 = conv_chain.invoke({"question": "How do I request time off?"})
print(f"Q: How do I request time off?")
print(f"A: {response3['answer']}")
```

---

## Stretch Goals (Optional)

### Stretch 1: Add Hybrid Search

Combine keyword and semantic search for better results:

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Create BM25 retriever (keyword-based)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# Create ensemble
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever],
    weights=[0.3, 0.7]  # 30% keyword, 70% semantic
)
```

### Stretch 2: Add Source Highlighting

Highlight the exact text that answers the question:

```python
def highlight_answer(question, answer, sources):
    """Highlight the relevant portion of source documents."""
    # Implementation would use string matching or LLM extraction
    pass
```

### Stretch 3: Implement RAGAS Evaluation

Use the RAGAS framework for comprehensive evaluation:

```python
# pip install ragas
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
```

---

## Submission Checklist

Before completing the lab, verify:

- [ ] Documents loaded and chunked
- [ ] Vector store created with embeddings
- [ ] Basic RAG chain working
- [ ] Custom prompt with citations
- [ ] Retrieval evaluation completed
- [ ] Chunk size comparison done
- [ ] Fallback mechanism implemented
- [ ] (Optional) Conversational memory working

---

## Key Concepts Applied

| Concept | Where Applied |
|---------|---------------|
| Document loading | LangChain TextLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | ChromaDB |
| RAG chain | RetrievalQA |
| Custom prompts | PromptTemplate with citations |
| Evaluation | Recall@k measurement |
| Fallback | Confidence-based routing |

---

## Cleanup

```python
import shutil
import os

# Remove vector store
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

# Remove data files
if os.path.exists("data/policies.txt"):
    os.remove("data/policies.txt")

print("Cleanup complete!")
```
