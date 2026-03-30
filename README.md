# NOVA AI Platform ? Task Walkthrough

This repository contains five tasks that together build a customer?support AI system for the NOVA brand. Each task is self?contained but designed to compose into a full pipeline:

1. Task 1: Prompt engineering + intent classification
2. Task 2: MCP?style tools with audit logging
3. Task 3: Retrieval?Augmented Generation (RAG)
4. Task 4: Brand?voice fine?tuning
5. Task 5: End?to?end orchestration with routing + tracing

Below is a detailed walkthrough of each task, the tools used, and the step?by?step flow.

---

**Repository Structure**

- `data/` ? CSV datasets used across tasks (orders, customers, products, returns, tickets)
- `prompts/` ? System and intent prompts for Task 1
- `task1_prompt_engineering.ipynb` ? Prompt engineering + intent pipeline
- `task2_mcp/` ? Tool server, client, and audit logging
- `task3/` ? RAG module, evaluation report, and notebook
- `task4/` ? Fine?tuning notebook and JSONL datasets
- `task5/` ? Full support platform orchestration + tracing

---

**Task 1 ? Prompt Engineering and Intent Classification**

**Goal**
Build a robust intent classification + escalation + injection defense layer that can decide when to use tools, RAG, or a human handoff.

**Key Files**
- `task1_prompt_engineering.ipynb`
- `prompts/nova_system_prompt_v1.txt`
- `prompts/intent_classifier_prompt.txt`
- `prompts/escalation_policy.txt`

**Tools and Libraries Used**
- `transformers` zero?shot classification pipeline
- `pandas` for test cases and dataset loading
- Regex?based escalation and injection detectors

**Step?by?Step Flow**
1. Load customer/order/product datasets from `data/`.
2. Load prompt assets from `prompts/`.
3. Initialize a zero?shot intent classifier (Hugging Face pipeline).
4. Predict intent from the user query.
5. Detect escalation triggers with regex rules.
6. Detect prompt injection attempts.
7. Route to a response template based on intent and risk.
8. Run a set of test cases and view results.

**How to Run**
1. Open `task1_prompt_engineering.ipynb`.
2. Run all cells top?to?bottom.
3. Inspect `results_df` to validate intent and escalation predictions.

---

**Task 2 ? MCP?Style Tools + Audit Logging**

**Goal**
Expose core customer support tools with a lightweight MCP?like server, plus an audit log for every tool call.

**Key Files**
- `task2_mcp/tools.py` ? Tool implementations + audit logging
- `task2_mcp/server.py` ? Tool registry and execution wrapper
- `task2_mcp/client.py` ? Simple client for testing tools
- `task2_mcp/demo.py` ? Multi?step demo scenario
- `task2_mcp/audit_log.jsonl` ? Append?only tool audit trail

**Tools and Libraries Used**
- `pandas` for structured CSV lookup
- `uuid` for return request IDs
- JSONL logging for audits

**Step?by?Step Flow**
1. `NovaMockDB` loads CSVs from `data/` and normalizes string fields.
2. Tool functions execute business logic:
   - `get_order_status`
   - `create_return_request`
   - `get_customer_profile`
   - `search_product_catalog`
   - `recommend_products`
3. Each tool call appends a record to `audit_log.jsonl`.
4. `NovaMCPServer` exposes a tool registry and a unified `execute()` method.
5. `NovaMCPClient` calls tools and prints input/output.
6. `demo.py` runs a realistic multi?step support scenario.

**How to Run**
1. Run `python task2_mcp/demo.py`.
2. Inspect console output and `task2_mcp/audit_log.jsonl`.

---

**Task 3 ? RAG Pipeline**

**Goal**
Build a retrieval?augmented assistant that searches a small knowledge base of product data and policies.

**Key Files**
- `task3/rag_module.py` ? RAG pipeline implementation
- `task3/task3_rag_pipeline.ipynb` ? Demo + evaluation notebook
- `task3/chroma_db/` ? Persistent Chroma vector store
- `task3/evaluation_report.json` ? Evaluation results output

**Tools and Libraries Used**
- `chromadb` for persistent vector storage
- `sentence-transformers` for embeddings
- `CrossEncoder` for reranking

**Step?by?Step Flow**
1. Load product and returns data from `data/`.
2. Build documents:
   - Product catalog entries
   - Policy snippets (returns, shipping, sizing, safety)
   - FAQ snippets
3. Embed and store documents in Chroma.
4. Retrieve top?K candidates with semantic search.
5. Rerank with a cross?encoder for relevance.
6. Generate a simple grounded answer based on retrieved docs.
7. Evaluate against small query set and save report.

**How to Run**
1. Open `task3/task3_rag_pipeline.ipynb`.
2. Run all cells top?to?bottom.
3. Review `task3/evaluation_report.json` for accuracy details.

---

**Task 4 ? Brand Voice Fine?Tuning**

**Goal**
Fine?tune a small chat model to rewrite responses in the NOVA brand voice using LoRA/QLoRA.

**Key Files**
- `task4/task4_finetune.ipynb`
- `task4/brand_voice_train.jsonl`
- `task4/brand_voice_eval.jsonl`
- `task4/task4_outputs/` ? Adapter outputs and eval results

**Tools and Libraries Used**
- `transformers` for model and tokenizer
- `trl` for `SFTTrainer`
- `peft` for LoRA
- `bitsandbytes` for 4?bit quantization (GPU)

**Step?by?Step Flow**
1. Install training dependencies.
2. Load training and evaluation JSONL datasets.
3. Map each example into a single `text` field for SFT training.
4. Load tokenizer and model.
5. Configure optional 4?bit quantization (GPU only).
6. Apply LoRA adapters to attention projection layers.
7. Train with `SFTTrainer`.
8. Save adapter weights and tokenizer.
9. Generate sample rewrites to validate brand tone.
10. Save eval outputs to JSON.

**How to Run**
1. Open `task4/task4_finetune.ipynb`.
2. Run all cells top?to?bottom.
3. Check `task4/task4_outputs/` for adapters and eval outputs.

---

**Task 5 ? End?to?End Support Platform (Orchestration)**

**Goal**
Route queries through tools, RAG, escalation, and brand voice rewrite using a LangGraph state machine.

**Key Files**
- `task5/task5_nova_platform.py` ? Full orchestration graph
- `task5/task5_demo.py` ? Demo runner
- `task5/nova_traces.json` ? Structured trace log

**Tools and Libraries Used**
- `langgraph` for stateful flow
- Task 2 MCP tools for order/return/recommendation
- Task 3 RAG for knowledge lookup
- Task 4 brand voice (lightweight rewrite placeholder)

**Step?by?Step Flow**
1. Initialize `NovaSupportPlatform`.
2. Router node classifies intent and checks escalation/injection.
3. Route decision:
   - Tools for order/return/recommendation
   - RAG for product/sizing queries
   - Escalation for complaints or sensitive issues
   - Injection guard for prompt attacks
   - Fallback for unknown
4. Build a draft response from tools or RAG output.
5. Apply a brand?voice rewrite layer.
6. Log a full trace to `task5/nova_traces.json`.
7. Return final response + trace metadata.

**How to Run**
1. Run `python task5/task5_demo.py`.
2. Inspect console output and `task5/nova_traces.json`.

---

**End?to?End Flow Summary**

1. Task 1 defines intent/escalation/injection handling rules.
2. Task 2 provides tool access and audit logging.
3. Task 3 provides retrieval and grounded answers.
4. Task 4 provides brand?voice fine?tuning (optional).
5. Task 5 orchestrates everything into a single support pipeline.

---

**Notes and Tips**

- If running locally on CPU, expect Task 4 fine?tuning to be slow. Consider a smaller model or fewer steps.
- Task 3 uses a persistent Chroma DB; delete `task3/chroma_db/` to rebuild from scratch.
- Task 5 currently uses a lightweight rule?based brand?voice rewrite; it can be replaced with the Task 4 fine?tuned adapter when ready.
