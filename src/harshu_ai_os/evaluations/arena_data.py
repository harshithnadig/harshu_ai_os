"""Synthetic data generator for RAG Evaluation Arena v1."""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

@dataclass
class SyntheticFact:
    id: str
    product: str
    version: str
    topic: str
    text: str

@dataclass
class SyntheticDocument:
    id: str
    text: str
    source: str
    topic: str
    version: str
    fact_spans: List[Tuple[str, int, int]]  # list of (fact_id, start_word_idx, end_word_idx)

@dataclass
class SyntheticQuery:
    id: str
    category: str
    difficulty: str
    question: str
    answerable: bool
    reference_answer: str | None
    reference_fact_ids: List[str]
    expected_facts_to_chunks: Dict[str, List[str]]
    split: str = "dev"

def generate_fact_pool() -> List[SyntheticFact]:
    products = ["Nexus API", "DataSync Pro", "CloudVault", "Sentinel AI"]
    versions = ["v1.0", "v2.0", "v3.1-beta"]
    topics = ["Authentication", "Rate Limiting", "Billing", "Data Backup", "Access Control"]
    
    facts = []
    fact_counter = 1
    for prod in products:
        for ver in versions:
            for top in topics:
                # Fact A
                fa_id = f"fact_{fact_counter:04d}"
                fa_text = f"The {top} component in {prod} {ver} operates independently using a distributed cache."
                facts.append(SyntheticFact(fa_id, prod, ver, top, fa_text))
                fact_counter += 1
                
                # Fact B
                fb_id = f"fact_{fact_counter:04d}"
                fb_text = f"Administrators can configure {top} for {prod} under the {ver} security settings menu."
                facts.append(SyntheticFact(fb_id, prod, ver, top, fb_text))
                fact_counter += 1
                
                # Fact C (Error code)
                fc_id = f"fact_{fact_counter:04d}"
                fc_text = f"If you encounter ERR-{random.randint(1000, 9999)}, the {top} service in {prod} {ver} has timed out."
                facts.append(SyntheticFact(fc_id, prod, ver, top, fc_text))
                fact_counter += 1
    return facts

def generate_documents(seed: int = 42, target_chunks: int = 5000, chunk_size_words: int = 50) -> Tuple[List[SyntheticDocument], List[SyntheticFact]]:
    random.seed(seed)
    facts = generate_fact_pool()
    
    target_docs = max(100, target_chunks // 4)
    docs = []
    
    for i in range(target_docs):
        # Pick a random fact to be the primary topic of this document
        main_fact = random.choice(facts)
        doc_id = f"doc_{i:04d}"
        
        # We will build the document word by word
        words = []
        fact_spans = []
        
        # Helper to add words and track fact
        def inject_fact(fact: SyntheticFact):
            fact_words = fact.text.split()
            start = len(words)
            words.extend(fact_words)
            end = len(words)
            fact_spans.append((fact.id, start, end))
            
        def inject_filler(count: int, topic: str):
            fillers = [
                f"This section discusses various aspects of {topic}.",
                "It is important to understand the configuration before proceeding.",
                "Ensure that you have the appropriate permissions.",
                "The system will automatically log all relevant events for auditing.",
                "You can find more details in the standard deployment guide.",
                "Users must restart the service after applying these changes.",
                "Data is encrypted at rest using industry standard algorithms.",
                "In case of a timeout, the system will automatically retry up to three times."
            ]
            for _ in range(count):
                words.extend(random.choice(fillers).split())
                
        # Build document structure
        inject_filler(random.randint(2, 5), main_fact.topic)
        inject_fact(main_fact)
        inject_filler(random.randint(5, 10), main_fact.topic)
        
        # Inject 1-3 more random facts
        num_extra_facts = random.randint(1, 3)
        for _ in range(num_extra_facts):
            extra_fact = random.choice(facts)
            inject_fact(extra_fact)
            inject_filler(random.randint(2, 5), extra_fact.topic)
            
        text = " ".join(words)
        
        docs.append(SyntheticDocument(
            id=doc_id,
            text=text,
            source=f"{main_fact.product.lower().replace(' ', '_')}_{main_fact.version}_{main_fact.topic.lower().replace(' ', '_')}.txt",
            topic=main_fact.topic,
            version=main_fact.version,
            fact_spans=fact_spans
        ))
        
    return docs, facts

def generate_queries(facts: List[SyntheticFact], seed: int = 42, count: int = 500) -> List[SyntheticQuery]:
    random.seed(seed)
    queries = []
    
    for i in range(count):
        category = random.choice([
            "direct_factual", 
            "exact_keyword", 
            "multi_evidence",
            "unsupported", 
            "vague",
            "version_conflict"
        ])
        
        difficulty = "easy" if category in ["direct_factual", "exact_keyword", "unsupported", "vague"] else "hard"
        
        if category == "unsupported":
            q = SyntheticQuery(
                id=f"q_{i:04d}",
                category=category,
                difficulty=difficulty,
                question=f"How do I integrate {random.choice(['Stripe', 'PayPal', 'Kubernetes', 'Redis'])} with the system?",
                answerable=False,
                reference_answer=None,
                reference_fact_ids=[],
                expected_facts_to_chunks={}
            )
            queries.append(q)
            continue
            
        if category == "vague":
            q = SyntheticQuery(
                id=f"q_{i:04d}",
                category=category,
                difficulty=difficulty,
                question=random.choice(["How does it work?", "Fix the error.", "What is the policy?"]),
                answerable=False,
                reference_answer=None,
                reference_fact_ids=[],
                expected_facts_to_chunks={}
            )
            queries.append(q)
            continue
            
        if category == "multi_evidence":
            fact1 = random.choice(facts)
            fact2 = random.choice([f for f in facts if f.id != fact1.id])
            
            question = f"Compare the {fact1.topic} in {fact1.product} {fact1.version} and the {fact2.topic} in {fact2.product} {fact2.version}."
            answer = "Comparison requires both facts."
            
            q = SyntheticQuery(
                id=f"q_{i:04d}",
                category=category,
                difficulty=difficulty,
                question=question,
                answerable=True,
                reference_answer=answer,
                reference_fact_ids=[fact1.id, fact2.id],
                expected_facts_to_chunks={}
            )
            queries.append(q)
            continue
            
        target_fact = random.choice(facts)
        question = ""
        answer = "Fact details."
        
        if category == "direct_factual":
            question = f"What operates independently in the {target_fact.product} {target_fact.version} {target_fact.topic} component?"
        elif category == "exact_keyword":
            import re
            err_codes = re.findall(r'ERR-\d+', target_fact.text)
            if err_codes:
                question = f"How do I resolve {err_codes[0]}?"
            else:
                question = f"Where can administrators configure {target_fact.topic} for {target_fact.product} {target_fact.version}?"
        elif category == "version_conflict":
            question = f"How does {target_fact.topic} work in {target_fact.product} specifically for {target_fact.version}?"
        else:
            question = f"Tell me about {target_fact.topic} in {target_fact.product} {target_fact.version}."
            
        q = SyntheticQuery(
            id=f"q_{i:04d}",
            category=category,
            difficulty=difficulty,
            question=question,
            answerable=True,
            reference_answer=answer,
            reference_fact_ids=[target_fact.id],
            expected_facts_to_chunks={}
        )
        queries.append(q)
        
    random.shuffle(queries)
    
    # Stratify dev/holdout
    groups = {}
    for q in queries:
        key = (q.category, q.answerable)
        if key not in groups:
            groups[key] = []
        groups[key].append(q)
        
    final_queries = []
    for key, group_queries in groups.items():
        dev_count = int(len(group_queries) * 0.8)
        for i, q in enumerate(group_queries):
            if i < dev_count:
                q.split = "dev"
            else:
                q.split = "holdout"
            final_queries.append(q)
            
    final_queries.sort(key=lambda x: x.id)
    return final_queries

def chunk_document_and_map_queries(docs: List[SyntheticDocument], queries: List[SyntheticQuery], chunk_size: int = 50) -> List[Dict[str, Any]]:
    from harshu_ai_os.rag.ingestion import chunk_text
    
    records = []
    # Map fact_id to a list of chunk_ids that contain it
    fact_to_chunks = {}
    
    for doc in docs:
        chunks = chunk_text(doc.text, chunk_size)
        
        # Create chunk records
        for index, chunk in enumerate(chunks):
            chunk_id = f"{doc.id}-chunk_{index}"
            # Track metadata (doc_id, topic, version) as requested
            records.append({
                "id": chunk_id,
                "text": chunk,
                "source": doc.source,
                "chunk_index": index,
                "doc_id": doc.id,
                "topic": doc.topic,
                "version": doc.version
            })
            
            # Which facts are in this chunk?
            chunk_start_word = index * chunk_size
            chunk_end_word = (index + 1) * chunk_size
            
            for fact_id, fact_start, fact_end in doc.fact_spans:
                # If the chunk and the fact overlap
                if max(chunk_start_word, fact_start) < min(chunk_end_word, fact_end):
                    if fact_id not in fact_to_chunks:
                        fact_to_chunks[fact_id] = []
                    # Append unique chunk IDs (in case of double mapping somehow, though unlikely)
                    if chunk_id not in fact_to_chunks[fact_id]:
                        fact_to_chunks[fact_id].append(chunk_id)
                    
    # Map back to queries
    for q in queries:
        if q.answerable:
            q.expected_facts_to_chunks = {}
            for f_id in q.reference_fact_ids:
                if f_id in fact_to_chunks:
                    q.expected_facts_to_chunks[f_id] = fact_to_chunks[f_id]
                else:
                    q.expected_facts_to_chunks[f_id] = []
                    
    return records
