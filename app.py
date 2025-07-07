import os
import vertexai
from dotenv import load_dotenv
import streamlit as st

# ADK and Agent Engine Imports
from google.adk.agents import Agent
from vertexai.preview.reasoning_engines import AdkApp

# Import the client library for Vertex AI Search and client options
from google.cloud import discoveryengine_v1 as discoveryengine
from google.api_core.client_options import ClientOptions

# Load environment variables from .env file
load_dotenv()

# --- Import the advanced LangGraph tool ---
from tools import propose_gcp_architecture

# --- CONFIGURATION ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION")
STAGING_BUCKET = os.getenv("STAGING_BUCKET")
MODEL = os.getenv("FAST_MODEL")
DATA_STORE_ID = os.getenv("DATA_STORE_ID")

# --- AGENT AND TOOL DEFINITION ---

@st.cache_resource
def load_agent():
    """Loads the SOMMAS agent and its tools."""
    vertexai.init(
        project=GCP_PROJECT_ID,
        location=GCP_LOCATION,
        staging_bucket=STAGING_BUCKET,
    )

    def search_knowledge_base(query: str) -> str:
        """This tool performs a simple search for factual questions about specific terms or concepts."""
        client_options = ClientOptions(
            api_endpoint="discoveryengine.googleapis.com"
        )
        client = discoveryengine.SearchServiceClient(client_options=client_options)

        serving_config = client.serving_config_path(
            project=GCP_PROJECT_ID,
            location="global",
            data_store=DATA_STORE_ID,
            serving_config="default_config",
        )

        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=5,
        )
        response = client.search(request)
        
        results_str = ""
        for i, result in enumerate(response.results):
            doc = result.document
            content = doc.struct_data.get("content", "No content found.")
            source = doc.struct_data.get("source", "Unknown source.")
            results_str += f"Result {i+1}:\nSource: {source}\nContent: {content}\n\n"
        return results_str if results_str else "No relevant information found."

    # Creates the agent, now giving it BOTH tools.
    agent = Agent(
        model=MODEL,
        # --- The agent now has two tools to choose from ---
        tools=[search_knowledge_base, propose_gcp_architecture],
        name="sommas_agent"
    )

    app = AdkApp(agent=agent)
    return app

# --- Streamlit UI ---

st.title("SOMMAS: The Implementation Strategist")

st.markdown("""
The Society of Mind Multi-Agent System (SOMMAS)

Try asking a simple factual question or a complex architectural query. For example:
- **Simple search:** *Who is Marvin Minsky?*
- **Advanced proposal:** *Propose a scalable architecture for deploying thousands of simple, specialized agents.*
""")

app = load_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a simple question or a complex architectural query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        def response_generator():
            # --- Updated system prompt to guide tool selection ---
            system_prompt = """You are the SOMMAS Implementation Strategist, an expert research assistant and solutions architect.
            - For simple, factual questions about terms or concepts, use the `search_knowledge_base` tool.
            - For complex, high-level, or "how-to" questions that require a plan or architectural proposal, use the `propose_gcp_architecture` tool.
            - Provide clear, synthesized answers based ONLY on the context provided by the tools. Do not use any external knowledge. Cite sources when available."""
            
            full_message = f"SYSTEM PROMPT: {system_prompt}\n\nUSER QUESTION: {prompt}"
            
            for event in app.stream_query(message=full_message, user_id="test_user"):
                if "content" in event and "parts" in event["content"]:
                    for part in event["content"]["parts"]:
                        if "text" in part:
                            yield part["text"]
        
        response = st.write_stream(response_generator)
    st.session_state.messages.append({"role": "assistant", "content": response})