#!/usr/bin/env python3
"""
HSL Transport Assistant RAG - Panel Dashboard UI

Serves the multi-tab user interface using local Llama 3.1 8B via Ollama and
cross-tab force synchronization for references and raw history metrics.
"""

import os
import time
import panel as pn
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


# Initialize Panel framework extension safely with explicit loading asset tags
pn.extension(loading_indicator=True)

class HSLTransportDashboard:
    def __init__(self):
        # 1. Models setup
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatOllama(model="llama3.1", temperature=0.0)
        
        # 2. Conversational Memory Layer
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
            input_key="input"
        )
        
        # Internal state metrics
        self.last_question = ""
        self.last_sources = "No vector documents retrieved yet."
        
        # 3. Load Local FAISS Vector Store Index
        self.index_dir = "faiss_hsl_index"
        try:
            self.vectorstore = FAISS.load_local(
                self.index_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            self.db_status = "Local FAISS index loaded successfully."
        except Exception as e:
            self.vectorstore = None
            self.retriever = None
            self.db_status = f"Error loading vector store: {str(e)}"

        # 4. HSL Context Prompt
        self.template = """You are a helpful local transport assistant for HSL in Finland.
Answer the user's question utilizing the pre-loaded profiles and the running conversational history provided below.

CRITICAL INSTRUCTIONS:
- STICK STRICTLY TO DATA: Only name specific train lines (like A, E, U, L, or Y) if they are explicitly written in the [Retrieved Transport Context] text blocks.
- NO PLACEHOLDERS: Never invent, guess, or use placeholder letters (like 'Line X') under any circumstances.
- DO NOT display raw numerical Latitude and Longitude coordinates in your final response.
- Use the names of the stations (such as streets, bridges, or transit hubs) to describe where they are located in natural, conversational text.
- If a station profile name or ID matches the neighborhood or station hub queried (e.g., containing 'Pasila', 'Pasilansilta', or 'Tripla'), treat it as a valid nearby location and summarize it clearly for the user.
- Forgive spelling variations or missing accents in Finnish place names.

[Conversation History]
{chat_history}

[Retrieved Transport Context]
{context}

Current Question: {input}
Answer:"""
        
        self.prompt_template = PromptTemplate(
            input_variables=["chat_history", "context", "input"],
            template=self.template
        )

    def execute(self):
        """Processes current text box value, triggers local streaming tokens, and performs a cross-tab refresh."""
        query = txt_input.value
        if not query or query.strip() == "":
            return
            
        self.last_question = query
        
        if not self.retriever:
            conversation_pane.object = "**System Error:** Local FAISS index folder missing or corrupted."
            return

        # Clear text input widget entry immediately to reset UI feel
        txt_input.value = ""

        # Setup quick visual placeholder
        conversation_pane.object = f"🧑 **User:** {query}\n\n🤖 **Agent:** *Thinking...*"

        # A. Execute semantic vector retrieval lookup
        matched_docs = self.retriever.invoke(query)
        
        source_logs = []
        for index, doc in enumerate(matched_docs):
            log_entry = (
                f"### 📍 Reference Chunk {index + 1}\n"
                f"* **Data Classification Type:** `{doc.metadata.get('type', 'Unknown')}`\n"
                f"* **Infrastructure Source:** `{doc.metadata.get('source', 'Unknown')}`\n\n"
                f"```text\n{doc.page_content.strip()}\n```\n---"
            )
            source_logs.append(log_entry)
        self.last_sources = "\n\n".join(source_logs)

        # B. Load multi-turn dialog strings and compile parameters
        context_str = "\n\n".join(d.page_content for d in matched_docs)
        memory_variables = self.memory.load_memory_variables({"input": query})
        history_str = memory_variables.get("chat_history", "")
        
        final_prompt = self.prompt_template.format(
            chat_history=history_str,
            context=context_str,
            input=query
        )

        # C. Loop token streams line-by-line onto the viewport markdown pane
        full_response = ""
        for chunk in self.llm.stream(final_prompt):
            text_chunk = chunk.content if hasattr(chunk, 'content') else str(chunk)
            full_response += text_chunk
            conversation_pane.object = f"🧑 **User:** {query}\n\n🤖 **Agent:** {full_response}"
            
        # D. Append completed sequence back to the LangChain sliding buffer history
        self.memory.save_context({"input": query}, {"output": full_response})
        
        # E.Force background panel panes to evaluate new values immediately
        lquest_pane.object = f"### Last Asked Question:\n> {self.last_question}"
        sources_pane.object = self.last_sources
        history_pane.object = self.get_chats_formatted()

    def get_chats_formatted(self):
        """Formats conversational memory items into clean prose lines."""
        updated_history = self.memory.load_memory_variables({}).get("chat_history", "")
        if not updated_history:
            return "*No conversation history loaded yet.*"
            
        conversation_display = []
        lines = updated_history.split("\n")
        for line in lines:
            if line.startswith("Human:"):
                conversation_display.append(f"🧑 **User:** \n{line[6:]}")
            elif line.startswith("AI:"):
                conversation_display.append(f"🤖 **Agent:** \n{line[3:]}\n")
            else:
                if line.strip():
                    conversation_display.append(line)
        return "\n".join(conversation_display)

    def call_refresh_status(self, clicks):
        if clicks:
            return f"Database Refreshed State Count: {clicks}"
        return self.db_status

    def clr_history(self, event):
        """Flushes conversational state buffers completely."""
        self.memory.clear()
        self.last_question = ""
        self.last_sources = "No vector documents retrieved yet."
        conversation_pane.object = "*Awaiting your transport inquiries...*"
        lquest_pane.object = "### Last Asked Question:\n*No questions submitted in this session yet.*"
        sources_pane.object = "No vector documents retrieved yet."
        history_pane.object = "*No conversation history loaded yet.*"
        txt_update.value = f"🧹 Workspace history reset at {time.strftime('%H:%M:%S')}!"


# =====================================================================
# UI INSTANTIATION & ROUTING
# =====================================================================
backend = HSLTransportDashboard()

# Core Interactive Widgets
txt_input = pn.widgets.TextInput(placeholder='Type your transit question here (e.g. Which trains go to Pasila?)...', width=550)
submit_button = pn.widgets.Button(name="Ask", button_type='success', width=100)

# Declare Static Display Panes
conversation_pane = pn.pane.Markdown("*Awaiting your transport inquiries...*", width=670)
lquest_pane = pn.pane.Markdown("### Last Asked Question:\n*No questions submitted in this session yet.*")
sources_pane = pn.pane.Markdown("No vector documents retrieved yet.", width=650)
history_pane = pn.pane.Markdown("*No conversation history loaded yet.*", width=650)

# Connect UI interactions directly to callback handlers
def handle_submit_event(event):
    backend.execute()

submit_button.on_click(handle_submit_event)

# System Engine Widgets
file_input = pn.widgets.FileInput(accept='.json,.pdf,.docx')
button_load = pn.widgets.Button(name="Load DB", button_type='primary')
button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
txt_update = pn.pane.Markdown("Local offline engine synchronized.")

button_clearhistory.on_click(backend.clr_history)
bound_refresh_label = pn.bind(backend.call_refresh_status, button_load.param.clicks)

# Tabs Setup
tab1 = pn.Column(
    pn.Row(txt_input, submit_button),
    pn.layout.Divider(),
    conversation_pane,
    pn.layout.Divider(),
)

tab2 = pn.Column(
    lquest_pane,
    pn.layout.Divider(),
    pn.panel(sources_pane, height=500, scroll=True),
)

tab3 = pn.Column(
    pn.panel(history_pane, height=500, scroll=True),
)

dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# HSL Transport Assistant RAG Dashboard Workspace')),
    pn.Tabs(
        ('Conversation', tab1),
        ('Database Internal State', tab2),
        ('Raw Chat History', tab3)
    )
)

dashboard.servable()
