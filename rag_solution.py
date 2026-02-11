"""
Lab 10: Building a Production RAG System - COMPLETE SOLUTION

This script contains the full implementation of the HR Policy RAG system.
Students should work through the lab instructions step-by-step,
but can reference this file for the complete solution.

Usage:
    1. Set OPENAI_API_KEY environment variable
    2. Run: python rag_solution.py
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


# =============================================================================
# SAMPLE DATA
# =============================================================================

POLICIES_CONTENT = """
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


# =============================================================================
# EVALUATION DATASET
# =============================================================================

EVAL_DATASET = [
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


# =============================================================================
# RAG SYSTEM CLASS
# =============================================================================

class HRPolicyRAG:
    """Complete RAG system for HR policy Q&A."""
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.vector_store = None
        self.qa_chain = None
        self.chunks = None
        
    def setup_documents(self, content: str = None):
        """Create and save the policy document."""
        if content is None:
            content = POLICIES_CONTENT
            
        os.makedirs("data", exist_ok=True)
        with open("data/policies.txt", "w") as f:
            f.write(content)
        print("Policy document created: data/policies.txt")
        
    def load_and_chunk(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Load documents and split into chunks."""
        # Load
        loader = TextLoader("data/policies.txt")
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s)")
        
        # Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.chunks = splitter.split_documents(documents)
        print(f"Created {len(self.chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
        
        return self.chunks
    
    def create_vector_store(self):
        """Create ChromaDB vector store."""
        if self.chunks is None:
            raise ValueError("No chunks. Call load_and_chunk() first.")
            
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        self.vector_store = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            persist_directory=self.persist_dir,
            collection_name="hr_policies"
        )
        print(f"Vector store created with {self.vector_store._collection.count()} vectors")
        
        return self.vector_store
    
    def build_chain(self, model: str = "gpt-4", top_k: int = 3):
        """Build the RAG chain with custom prompt."""
        if self.vector_store is None:
            raise ValueError("No vector store. Call create_vector_store() first.")
            
        llm = ChatOpenAI(model=model, temperature=0)
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        
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

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        print(f"RAG chain built with {model}, top_k={top_k}")
        return self.qa_chain
    
    def query(self, question: str) -> dict:
        """Query the RAG system."""
        if self.qa_chain is None:
            raise ValueError("No chain. Call build_chain() first.")
            
        result = self.qa_chain.invoke({"query": question})
        
        return {
            "question": question,
            "answer": result["result"],
            "sources": result["source_documents"]
        }
    
    def query_with_fallback(self, question: str, threshold: float = 0.5) -> dict:
        """Query with confidence-based fallback."""
        # Check retrieval confidence
        docs_with_scores = self.vector_store.similarity_search_with_score(question, k=3)
        best_distance = min(score for _, score in docs_with_scores)
        confidence = max(0, 1 - best_distance)
        
        if confidence < threshold:
            return {
                "question": question,
                "answer": "I don't have enough information to answer this question confidently. Please contact HR directly at hr@retailpulse.com or call extension 4567.",
                "confidence": confidence,
                "sources": [],
                "fallback": True
            }
        
        result = self.query(question)
        result["confidence"] = confidence
        result["fallback"] = False
        
        return result
    
    def evaluate_retrieval(self, dataset: list = None, k: int = 3) -> float:
        """Evaluate retrieval quality."""
        if dataset is None:
            dataset = EVAL_DATASET
            
        hits = 0
        
        print("\nRetrieval Evaluation:")
        print("-" * 50)
        
        for item in dataset:
            results = self.vector_store.similarity_search(item["question"], k=k)
            
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
        print(f"\nRecall@{k}: {recall:.1%} ({hits}/{len(dataset)})")
        
        return recall


def ask_hr(rag: HRPolicyRAG, question: str):
    """Pretty-print query results."""
    result = rag.query(question)
    
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\n{'='*60}")
    print("SOURCES:")
    for i, doc in enumerate(result['sources']):
        print(f"\n[{i+1}] {doc.page_content[:150]}...")
    print(f"{'='*60}\n")


def main():
    """Run the complete RAG pipeline."""
    print("=" * 60)
    print("HR POLICY RAG SYSTEM")
    print("=" * 60)
    
    # Initialize
    rag = HRPolicyRAG()
    
    # Setup documents
    rag.setup_documents()
    
    # Load and chunk
    rag.load_and_chunk(chunk_size=500, chunk_overlap=50)
    
    # Create vector store
    rag.create_vector_store()
    
    # Build chain
    rag.build_chain(model="gpt-4", top_k=3)
    
    # Evaluate retrieval
    rag.evaluate_retrieval()
    
    # Test queries
    print("\n" + "=" * 60)
    print("TESTING QUERIES")
    print("=" * 60)
    
    test_questions = [
        "How many vacation days do full-time employees get?",
        "Can I roll over unused vacation days?",
        "What's the process for getting expenses reimbursed?",
        "When are performance reviews conducted?",
    ]
    
    for question in test_questions:
        ask_hr(rag, question)
    
    # Test fallback
    print("\n" + "=" * 60)
    print("TESTING FALLBACK")
    print("=" * 60)
    
    fallback_test = rag.query_with_fallback("What's the policy on bringing pets to work?")
    print(f"\nQ: What's the policy on bringing pets to work?")
    print(f"Confidence: {fallback_test['confidence']:.2f}")
    print(f"Fallback triggered: {fallback_test['fallback']}")
    print(f"A: {fallback_test['answer']}")
    
    print("\n" + "=" * 60)
    print("RAG SYSTEM DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
