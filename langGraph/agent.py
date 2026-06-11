# =============================================================================
# HSL Transport Assistant — LangGraph Agent
# =============================================================================
# Modelled on the Essay Writer pattern from DeepLearning.AI's
# "AI Agents in LangGraph" course.
#
# Prerequisites:
#   - faiss_hsl_index/ folder must exist at ../  (built by langchain/ingest.py)
#   - Ollama running locally with llama3.2 pulled
#
# Run:
#   python3 agent.py
# =============================================================================

import os
from dotenv import load_dotenv
from typing import TypedDict, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

_ = load_dotenv()


# =============================================================================
# 1. MODEL
# =============================================================================

model = ChatOllama(model="llama3.2", temperature=0.1, max_tokens=5000)


# =============================================================================
# 2. FAISS RETRIEVER  (index built by langchain/ingest.py — loaded, not rebuilt)
# =============================================================================

# Path is relative to this file's location inside langgraph/
# Adjust if your folder layout is different.
FAISS_INDEX_PATH = "../faiss_hsl_index"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7},
)


# =============================================================================
# 3. STATE
# =============================================================================

class AgentState(TypedDict):
    task: str           # the user's HSL question
    plan: str           # routing intent summary from the planner
    draft: str          # current answer draft
    critique: str       # factual sufficiency critique from the reflect node
    content: List[str]  # filtered FAISS chunk texts
    revision_number: int
    max_revisions: int


# =============================================================================
# 4. STRUCTURED OUTPUT SCHEMA
# =============================================================================

class Queries(BaseModel):
    queries: List[str]


# =============================================================================
# 5. PROMPTS
# =============================================================================

PLAN_PROMPT = """You are a Semantic Routing Assistant for HSL Transit.
Analyze the user's input question and identify the core transport entity
(e.g., pet policies, line schedules, bike station data, or service alerts)
and any physical constraints (like stop names or zone letters).

Your output must be a clean, summarized statement of intent that will be used
as a dense context vector search string.
Do not write multi-step questions. Do not write an essay outline.
Focus purely on the core operational subject matter."""

WRITER_PROMPT = """
You are the factual HSL Transport Chat Assistant. Your goal is to provide
passengers with clear, direct, and practical answers using ONLY the provided
text blocks.

=== GROUNDING DIRECTIVES ===
* CONTEXT BOUNDARY: Rely strictly on the literal text inside the data blocks.
  If a metric, location, or rule is missing, state
  "Information not available in current records" rather than guessing.
* FARE ZONES: HSL fare zones are exclusively letters (A, B, C, or D).
  Characters like Z, T, or R are Train Line Names — never classify them as zones.
  If a zone letter is not explicitly linked to a station within the text,
  state that zone data is unavailable.
* TRANSPORT MODES: Use exact terminology from the profiles.
  Label "RAIL" as commuter trains, "SUBWAY" as metro, "TRAM" as trams.
  Do not swap or approximate modes.
* CONNECTIVITY: A vehicle only services a station if that station name is
  explicitly written in that line's "Stations Serviced" or
  "All connected stations" text list.

=== USER RESPONSE FORMAT ===
* Speak directly and helpfully to the commuter (e.g., "You do not need...").
* Do not print internal software guidelines or debugging rules.
* Use clean Markdown bullet points with bold text so the passenger can scan
  the answer instantly on a mobile screen.

Context Data Blocks:
--------------------
{content}
--------------------

AGNOSTIC GROUNDING SAFETY RULES:
1. ZERO INFERENCE: If the data blocks contain zero mentions of the primary
   subject matter, output exactly:
   "Information not available in current records."
2. HEADER ISOLATION: Do not extrapolate rules from section headers or index
   lists that have no descriptive paragraphs.
3. CRITICAL ASSUMPTION LAW: Completely isolate what the user asked from what
   the text blocks contain. If the core action or transport mode is missing
   from the blocks, state "Information not available in current records."

=== FINAL FORMAT LAW ===
* Write a unified, conversational paragraph directed at the commuter.
* If you find a direct answer, print that rule clearly.
* Do NOT append "Information not available" to the end of a successful answer.
  Only use that phrase as a standalone response if the chunks contain zero data.
"""

REFLECTION_PROMPT = """You are an agnostic Factual Sufficiency Evaluator for a transit RAG system.
Your sole job is to compare the user's raw question, the retrieved data blocks,
and the generated draft to detect hallucinations or data gaps.

CRITICAL EVALUATION PROTOCOL:
1. Identify the primary subject nouns/topics in the User's Question:
   e.g., if the question is "Can I bring a bike on the bus?",
   the core topics are "bike" and "bus".
2. Do not differentiate singular and plural:
   "pet" is the same as "pets", "train" is the same as "trains".
3. Check the retrieved Data Blocks {content}.
   Are these core topics discussed factually inside the source text?
4. If at least one of the core topics from the question {task} does NOT appear
   in the data blocks, but the generated DRAFT confidently answers the question
   anyway, flag this as a critical "GROUNDING FAILURE".
5. If a GROUNDING FAILURE occurs:
   - Retrieve the core topics from the main user query: {task}
   - Output exactly:
     "RETRY: The retrieved context blocks do not contain factual data regarding
      the core components of the user's question. Clear the current context
      state and execute a broader search query targeting [Insert query Topics]."
6. If the draft accurately maps to the text blocks and fully answers the
   question, output exactly: "NO GROUNDING FAILURE"
7. If the draft is accurate enough without hallucinations, even if it doesn't
   exactly match the requirement, also output exactly: "NO GROUNDING FAILURE"
"""

RESEARCH_PLAN_PROMPT = """You are a database retrieval query expert for HSL transport text handbooks.
Given the user's primary question, generate 2 or 3 highly specific keyword
search variations that target handbook profiles, legal conditions, or system
summaries.

CRITICAL RULE: Every search string MUST contain the primary subject noun of
the user's query (e.g., 'pet', 'dog', 'alert', 'delay', 'stop', 'bike',
'station', 'zone'). Never generate generic transit terms if the user is asking
about a specific policy rule.

Only output the queries — no other text."""

RESEARCH_CRITIQUE_PROMPT = """You are a retrieval specialist for the HSL transport knowledge base.
A fact-checker has reviewed the current answer and identified gaps or
inaccuracies (shown below). Generate a list of short, specific search queries
(max 3) to retrieve the missing information from the HSL knowledge base.
Only generate the queries — no other text."""


# =============================================================================
# 6. NODE FUNCTIONS
# =============================================================================

def plan_node(state: AgentState):
    """Summarises the user query into a compact search-intent statement."""
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state["task"]),
    ]
    response = model.invoke(messages)
    return {"plan": response.content}


def research_plan_node(state: AgentState):
    """Generates keyword queries, retrieves FAISS chunks, and filters with the condenser."""
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state["task"]),
    ])

    # Fresh list — prevents cross-invocation state pollution
    content = []
    raw_docs = []

    for q in queries.queries:
        docs = retriever.invoke(q)
        for doc in docs:
            raw_docs.append(doc.page_content)

    # Condenser: keep only chunks that are relevant to the user's request
    condenser_prompt = (
        "You are an information filtering assistant. "
        "Your job is to determine if the provided text snippet contains direct "
        "factual data needed to answer the user's request. "
        "Respond with exactly 'YES' or 'NO'. Do not explain your choice."
    )
    for snippet in raw_docs:
        verdict = model.invoke([
            SystemMessage(content=condenser_prompt),
            HumanMessage(content=f"User Request: {state['task']}\nText Snippet: {snippet}"),
        ]).content.strip().upper()
        if "YES" in verdict:
            content.append(snippet)

    # Fallback: if condenser filtered everything out, keep top 3 raw chunks
    if not content and raw_docs:
        content = raw_docs[:3]

    return {"content": content}


def generation_node(state: AgentState):
    """Generates or revises the answer draft from filtered context."""
    content = "\n\n".join(state.get("content") or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my retrieval plan:\n\n{state['plan']}"
    )
    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        user_message,
    ]
    if state.get("critique"):
        messages.append(HumanMessage(
            content=f"Critique of previous answer:\n\n{state['critique']}"
        ))
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }


def reflection_node(state: AgentState):
    """Evaluates the draft for grounding failures against the retrieved chunks."""
    source_content = "\n\n".join(state.get("content") or [])
    system_instruction = REFLECTION_PROMPT.format(
        task=state["task"],
        content=source_content,
    )
    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=f"Current Draft Answer to evaluate:\n\n{state['draft']}"),
    ]
    response = model.invoke(messages)
    return {"critique": response.content}


def research_critique_node(state: AgentState):
    """Re-retrieves from FAISS based on the critique's identified gaps."""
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state["critique"]),
    ])
    content = state.get("content") or []
    for q in queries.queries:
        docs = retriever.invoke(q)
        for doc in docs:
            content.append(doc.page_content)
    return {"content": content}


def should_continue(state: AgentState):
    """Routes to END on clean verification, or to research_critique for repair."""
    critique_text = state.get("critique", "").upper()
    if "NO GROUNDING FAILURE" in critique_text:
        return END
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


# =============================================================================
# 7. GRAPH
# =============================================================================

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("planner",           plan_node)
    builder.add_node("research_plan",     research_plan_node)
    builder.add_node("generate",          generation_node)
    builder.add_node("reflect",           reflection_node)
    builder.add_node("research_critique", research_critique_node)

    builder.set_entry_point("planner")

    builder.add_edge("planner",           "research_plan")
    builder.add_edge("research_plan",     "generate")
    builder.add_edge("generate",          "reflect")
    builder.add_edge("research_critique", "generate")

    builder.add_conditional_edges(
        "reflect",
        should_continue,
        {
            END:       END,                   # "NO GROUNDING FAILURE" → done
            "reflect": "research_critique",   # grounding failure → repair loop
        },
    )

    return builder


# =============================================================================
# 8. RUNNER
# =============================================================================

def run(task: str, max_revisions: int = 2):
    """
    Run the agent for a single question and print each node's output.

    Args:
        task: The user's HSL transit question.
        max_revisions: Maximum repair loop iterations before forced exit.
    """
    thread = {"configurable": {"thread_id": "hsl-agent-1"}}

    with SqliteSaver.from_conn_string(":memory:") as memory:
        graph = build_graph().compile(checkpointer=memory)

        for s in graph.stream(
            {
                "task": task,
                "plan": "",
                "draft": "",
                "critique": "",
                "content": [],
                "max_revisions": max_revisions,
                "revision_number": 1,
            },
            thread,
        ):
            for node_name, node_output in s.items():
                print(f"\n{'=' * 60}")
                print(f"Node: {node_name}")
                print("=" * 60)

                if "plan" in node_output:
                    print("PLAN:\n", node_output["plan"])
                elif "content" in node_output:
                    chunks = node_output["content"]
                    print(f"CONTENT ({len(chunks)} chunks retrieved):\n")
                    for idx, chunk in enumerate(chunks, 1):
                        print(f"--- Chunk {idx} ---")
                        print(chunk.strip())
                        print("-" * 20)
                elif "draft" in node_output:
                    print("DRAFT:\n", node_output["draft"])
                elif "critique" in node_output:
                    print("CRITIQUE:\n", node_output["critique"])

        final_state = graph.get_state(thread)
        print("\nFINAL ANSWER\n" + "=" * 60)
        print(final_state.values["draft"])


# =============================================================================
# 9. ENTRY POINT  — change the question here to test different queries
# =============================================================================

if __name__ == "__main__":
    run(
        task="Are there any active service alerts or delays right now?",
        max_revisions=2,
    )
