# SCRIBE Project Architecture

## 1. Introduction and Goals

The Scriptural Commentary Reflection Insights and Biblical Exegesis (SCRIBE) project is designed to create a sophisticated AI-powered system capable of interacting with a curated knowledge base. The project is developed in three distinct phases:

1.  **Phase 1: The Knowledge Foundation:** Establishes a private search engine over a set of documents. This involves ingesting raw documents, processing them into a structured format, and indexing them for efficient retrieval using Google Cloud's Vertex AI Search.
2.  **Phase 2: The Conversational Analyst:** Transforms the search engine into a conversational partner. This phase introduces an agent that can understand user queries, perform searches against the knowledge base, and synthesize answers grounded in the retrieved documents.
3.  **Phase 3: The Implementation Strategist:** Elevates the agent to a system capable of deconstructing complex, high-level challenges and synthesizing concrete, actionable implementation plans, particularly for Google Cloud Platform (GCP) architectures. This is achieved by integrating advanced multi-step tools using LangGraph.

The **purpose of this document** is to provide a comprehensive overview of the SCRIBE project's architecture. It details the key components, their interactions, data flows, and the technologies employed to achieve the project's goals. This document is intended for developers, architects, and stakeholders who need to understand the system's design and how its various parts work together.

## 2. High-Level Architecture

The following diagram illustrates the main components of the SCRIBE system and their interactions:

```mermaid
graph TD
    subgraph User Interaction
        User[End User] -->|Query| UI{Streamlit Web UI}
    end

    subgraph Application Layer (app.py, tools.py)
        UI -->|User Request| MainAgent[ADK Agent]
        MainAgent -->|Tool Selection| Router{Tool Router}
        Router -- Simple Query --> SearchTool[search_knowledge_base]
        Router -- Complex Query --> LangGraphTool[propose_gcp_architecture]
        
        subgraph LangGraphTool
            direction LR
            LG_Entry[Entry] --> LG_Search[survey_technologies_node]
            LG_Search --> LG_Synth[synthesize_proposal_node]
            LG_Synth --> LG_Exit[Exit]
        end

        SearchTool -->|Search Query| VertexSearchClient[Vertex AI Search Client]
        LG_Search -->|Search Query| VertexSearchClient

        LG_Synth -->|LLM Call| VertexGenAI[Vertex AI Generative Model]
        MainAgent -->|LLM Call for Orchestration/Response| VertexGenAI
        
        VertexSearchClient -->|Results| SearchTool
        VertexSearchClient -->|Results| LG_Search
        VertexGenAI -->|Synthesized Response| LG_Synth
        VertexGenAI -->|Final Answer| MainAgent

        MainAgent -->|Formatted Response| UI
    end

    subgraph Data Processing & Storage (preprocess.py, GCS, Vertex AI)
        RawDocs[Raw Documents GCS Bucket] -->|Input| PreprocessScript[preprocess.py]
        PreprocessScript -->|processed_corpus.jsonl| ProcessedDocs[Processed Corpus GCS Bucket]
        ProcessedDocs -->|Indexing| VertexSearch[Vertex AI Search Data Store]
        VertexSearchClient -->|Query| VertexSearch
        VertexSearch -->|Search Results| VertexSearchClient
    end

    subgraph Configuration
        EnvFile[.env File] --> MainAgent
        EnvFile --> PreprocessScript
        EnvFile --> SearchTool
        EnvFile --> LangGraphTool
    end
```

## 3. Component Breakdown

This section details each major component of the SCRIBE system.

### 3.1 Data Ingestion and Preprocessing (`preprocess.py`)

*   **Purpose:** To ingest raw documents, process them into a structured format suitable for machine learning, and store them for indexing.
*   **Functionality:**
    *   Reads raw documents (PDF, TXT) from a specified Google Cloud Storage (GCS) bucket (`RAW_DATA_BUCKET`).
    *   Uses LangChain document loaders (`PyPDFLoader`, `TextLoader`) to extract text content.
    *   Applies document-aware chunking using `RecursiveCharacterTextSplitter`. A finer-grained chunking strategy is used for foundational texts, while a standard size is applied to others.
    *   Formats each text chunk into a JSON object containing the `content` and `source` (original filename) metadata.
    *   Aggregates all processed chunks into a single JSON Lines (`.jsonl`) file named `processed_corpus.jsonl`.
    *   Uploads the `processed_corpus.jsonl` to a designated GCS bucket (`PROCESSED_DATA_BUCKET`).
*   **Key Technologies:** Python, LangChain, Google Cloud Storage.
*   **Input:** Raw documents (e.g., `.pdf`, `.txt`) located in the `RAW_DATA_BUCKET` on GCS.
*   **Output:** A single `processed_corpus.jsonl` file in the `PROCESSED_DATA_BUCKET` on GCS.
*   **Key Features:**
    *   **Idempotent Uploads:** Calculates an MD5 hash of the generated corpus and compares it to the existing file in GCS. The upload is skipped if the content has not changed, preventing unnecessary re-indexing by Vertex AI Search.
    *   **Document-Aware Chunking:** Applies different chunking strategies based on the document's importance or type.
    *   **Source Metadata:** Preserves the original filename for each chunk, crucial for citations in the application.

### 3.2 Knowledge Base (Vertex AI Search)

*   **Purpose:** To provide a scalable and efficient search service over the processed document corpus.
*   **Configuration:**
    *   A Vertex AI Search Data Store is created and configured to use the `processed_corpus.jsonl` file from the `PROCESSED_DATA_BUCKET` as its data source.
    *   The schema is typically unstructured, relying on the `content` field for search and `source` for metadata.
*   **Interaction:**
    *   Queried by the `search_knowledge_base` tool in `app.py` for direct factual lookups.
    *   Queried by the `survey_technologies_node` within the `propose_gcp_architecture` LangGraph tool in `tools.py` as the research step.
*   **Key Technologies:** Google Cloud Vertex AI Search.

### 3.3 Application Core (`app.py` and `tools.py`)

This is the heart of the SCRIBE system, handling user interaction, orchestrating AI capabilities, and generating responses.

#### 3.3.1 Streamlit Web Interface (`app.py`)

*   **Purpose:** Provides an interactive chat interface for users to communicate with the SCRIBE agent.
*   **Technology:** Streamlit.
*   **Functionality:**
    *   Displays chat history.
    *   Accepts user input (queries).
    *   Streams responses from the agent back to the user in real-time.

#### 3.3.2 Main Agent (`google.adk.agents.Agent` in `app.py`)

*   **Purpose:** Orchestrates the conversation flow, selects appropriate tools based on user queries, and formulates the final response.
*   **Technology:** Google Agent Development Kit (ADK), Vertex AI Generative Models (e.g., Gemini, specified by `MODEL` environment variable).
*   **Key Features:**
    *   **Tool Integration:** Initialized with two tools: `search_knowledge_base` and `propose_gcp_architecture`.
    *   **System Prompt for Tool Routing:** A detailed system prompt guides the agent on whether to use the simple search tool (for factual questions) or the advanced LangGraph-based tool (for complex architectural proposals).
    *   **Response Synthesis:** Uses its underlying LLM to generate coherent and contextually appropriate responses, incorporating information from the selected tools.

#### 3.3.3 Tool 1: `search_knowledge_base` (in `app.py`)

*   **Purpose:** Provides a direct, efficient way to answer factual questions by searching the knowledge base.
*   **Mechanism:**
    *   Takes a user's query string as input.
    *   Initializes a `discoveryengine.SearchServiceClient`.
    *   Sends the query to the configured Vertex AI Search data store (`DATA_STORE_ID`).
    *   Formats the raw search results (content snippets and source files) into a clean string.
*   **Output:** A string containing formatted search results or a "No relevant information found" message.

#### 3.3.4 Tool 2: `propose_gcp_architecture` (LangGraph in `tools.py`)

*   **Purpose:** Handles complex user requests that require research and the synthesis of a detailed plan or architectural proposal.
*   **Technology:** LangGraph, Vertex AI Search, Vertex AI Generative Models (e.g., Gemini, specified by `MODEL_NAME` which is usually the same as `MODEL`).
*   **Mechanism:** Implemented as a stateful graph with two main nodes:
    1.  **`survey_technologies_node`:**
        *   Receives the user's request from the graph state.
        *   Performs a comprehensive search against the Vertex AI Search data store (similar to `search_knowledge_base` but typically requesting more results).
        *   Populates the `research_results` field in the graph state.
    2.  **`synthesize_proposal_node`:**
        *   Takes the original `user_request` and the `research_results` from the state.
        *   Uses a Vertex AI Generative Model with an expert-level prompt (e.g., "You are an expert Google Cloud solutions architect...") to generate a detailed architectural proposal.
        *   Saves the output to the `final_proposal` field in the graph state.
*   **Output:** A string containing the synthesized architectural proposal.

### 3.4 Configuration (`.env` file)

*   **Purpose:** To store sensitive and environment-specific settings outside the codebase.
*   **Mechanism:** The `dotenv` Python library loads these variables into the application environment at runtime.
*   **Key Variables:**
    *   `GCP_PROJECT_ID`: Your Google Cloud Project ID.
    *   `GCP_LOCATION`: The default GCP region for Vertex AI resources.
    *   `RAW_DATA_BUCKET`: GCS bucket for storing original, unprocessed documents.
    *   `PROCESSED_DATA_BUCKET`: GCS bucket for storing the `processed_corpus.jsonl` file.
    *   `DATA_STORE_ID`: The ID of your Vertex AI Search data store.
    *   `STAGING_BUCKET`: GCS bucket used by Vertex AI for temporary staging of data.
    *   `MODEL` (or `FAST_MODEL`, `MODEL_NAME`): Specifies the Vertex AI Generative Model to be used (e.g., a Gemini model like `gemini-1.5-flash-001`).

## 4. Data Flow

This section describes the sequence of operations and data movement for different types of user interactions.

### 4.1 Data Ingestion and Processing Flow (executed by `preprocess.py`)

1.  **Trigger:** Manual execution of the `preprocess.py` script.
2.  **Read Configuration:** Script loads GCS bucket names and GCP project ID from the `.env` file.
3.  **List Raw Documents:** The script lists all files in the `RAW_DATA_BUCKET`.
4.  **Process Each Document:** For each file:
    a.  Downloads the file to a temporary local directory.
    b.  Detects file type (PDF, TXT).
    c.  Uses LangChain loaders to extract text.
    d.  Chunks the text using `RecursiveCharacterTextSplitter`, applying document-aware strategies.
    e.  Formats each chunk as a JSON object with `content` and `source` (original filename).
5.  **Aggregate Chunks:** All JSON objects are collected.
6.  **Create JSONL Content:** The collected JSON objects are serialized into a single string, with each object on a new line (JSONL format).
7.  **Calculate Hash:** An MD5 hash of the new JSONL content is calculated.
8.  **Check Existing Hash (Idempotency):**
    a.  The script attempts to retrieve metadata (including MD5 hash) of the `processed_corpus.jsonl` file in the `PROCESSED_DATA_BUCKET`.
    b.  If the file exists and its hash matches the new hash, the script exits (no changes).
9.  **Upload to GCS:** If the file doesn't exist or the hash differs, the new `processed_corpus.jsonl` content is uploaded to the `PROCESSED_DATA_BUCKET`, overwriting any existing file.
10. **Vertex AI Search Re-indexing:** (Implicit) Vertex AI Search, if configured for automatic updates from the GCS location, will detect the change in `processed_corpus.jsonl` and re-index the data. The README notes "Vertex AI Search will automatically detect the changes... and re-index your datastore," though the specifics of this (e.g. event-driven vs. scheduled polling by Vertex AI) are managed by the Vertex AI service itself.

### 4.2 Simple Query Flow (using `search_knowledge_base` tool)

1.  **User Input:** User types a factual question into the Streamlit web interface (e.g., "Who is Marvin Minsky?").
2.  **Request to Agent:** The query is sent from the UI to the `MainAgent` in `app.py`.
3.  **Tool Selection:** The `MainAgent`, guided by its system prompt, determines that this is a simple factual question and selects the `search_knowledge_base` tool.
4.  **Execute Tool:** The agent calls the `search_knowledge_base` function with the user's query.
5.  **Query Vertex AI Search:**
    a.  `search_knowledge_base` constructs a search request.
    b.  It sends the request to the Vertex AI Search service, targeting the configured `DATA_STORE_ID`.
6.  **Receive Results:** Vertex AI Search returns relevant document chunks (content and source).
7.  **Format Results:** `search_knowledge_base` formats these results into a readable string.
8.  **Return to Agent:** The formatted string is returned to the `MainAgent`.
9.  **Synthesize Final Answer:** The `MainAgent` uses its underlying LLM to formulate a final answer, incorporating the search results.
10. **Stream Response to UI:** The final answer is streamed back to the Streamlit UI and displayed to the user.

### 4.3 Complex Query Flow (using `propose_gcp_architecture` tool)

1.  **User Input:** User types a complex request into the Streamlit web interface (e.g., "Propose a scalable architecture for deploying thousands of simple, specialized agents.").
2.  **Request to Agent:** The query is sent from the UI to the `MainAgent` in `app.py`.
3.  **Tool Selection:** The `MainAgent`, guided by its system prompt, identifies this as a complex request requiring a proposal and selects the `propose_gcp_architecture` tool.
4.  **Execute Tool (Invoke LangGraph):** The agent calls the `propose_gcp_architecture` function (from `tools.py`) with the user's request. This invokes the compiled LangGraph application.
5.  **LangGraph - Step 1: `survey_technologies_node`:**
    a.  The user's request is passed as input to this node.
    b.  This node queries Vertex AI Search (similar to `search_knowledge_base` but potentially for more comprehensive results, e.g., page_size=10).
    c.  The search results (content snippets and sources) are collected and stored in the `research_results` field of the `GraphState`.
6.  **LangGraph - Step 2: `synthesize_proposal_node`:**
    a.  This node receives the original `user_request` and the `research_results` from the `GraphState`.
    b.  It constructs a detailed prompt, instructing an LLM (Vertex AI Generative Model) to act as a solutions architect and generate a proposal based on the research context and user request.
    c.  The LLM generates the architectural proposal.
    d.  The proposal is stored in the `final_proposal` field of the `GraphState`.
7.  **Return from LangGraph:** The `final_proposal` string is returned from the `propose_gcp_architecture` function to the `MainAgent`.
8.  **Synthesize Final Answer (Optional Refinement):** The `MainAgent` may perform minor formatting or directly use the proposal as the core of its answer.
9.  **Stream Response to UI:** The architectural proposal is streamed back to the Streamlit UI and displayed to the user.

## 5. Deployment

This section outlines how the SCRIBE application is deployed and run, based on the project's current state and documentation.

### 5.1 Current Deployment Model (Local Execution)

*   **Environment Setup:**
    *   A Python virtual environment (e.g., `.venv`) is created to manage dependencies.
    *   Dependencies are installed from `requirements.txt` (generated via `pip-compile requirements.in` and synced with `pip-sync`).
    *   Google Cloud SDK (`gcloud`) is installed and configured for both CLI access and Application Default Credentials (ADC) for the Python scripts.
    *   An `.env` file is created from `.env-example` and populated with necessary GCP project IDs, bucket names, data store ID, and model configurations.
*   **Running the Preprocessing Script:**
    *   The `preprocess.py` script is run manually from the command line after activating the virtual environment:
        ```bash
        python preprocess.py
        ```
*   **Running the Main Application:**
    *   The Streamlit application (`app.py`) is launched from the command line after activating the virtual environment:
        ```bash
        python -m streamlit run app.py
        ```
    *   This starts a local web server, and the application is typically accessed via a web browser at `http://localhost:8501`.

### 5.2 Future Deployment Considerations (as per README)

The project `README.md` mentions potential future enhancements for deployment, which include:

*   **Dockerization:** Packaging the application (including `app.py`, `tools.py`, and dependencies) into a Docker container. This would involve creating a `Dockerfile` (a basic one is present in the repository) to define the image build process.
*   **Google Cloud Run:** Deploying the containerized application as a serverless service on Google Cloud Run. This would make the SCRIBE system accessible as a web service to a wider team or audience without managing underlying infrastructure.
    *   Scripts like `scripts/deploy.sh` and `scripts/run-docker-deploy.sh` in the repository suggest that steps towards Cloud Run deployment have been considered or initiated.
    *   Terraform configurations (`terraform/`) are also present, indicating infrastructure-as-code practices for managing GCP resources, which would be beneficial for a robust Cloud Run deployment (e.g., setting up Artifact Registry for Docker images, IAM permissions, Cloud Run service definitions).

While this document primarily describes the current local execution model, the architecture is designed to be portable to a containerized, cloud-native environment like Google Cloud Run.

## 6. Key Technologies Summary

The SCRIBE project leverages a combination of Python libraries, Google Cloud services, and AI frameworks:

*   **Core Language:**
    *   **Python:** Used for all scripting, application logic, and AI integration.

*   **User Interface:**
    *   **Streamlit:** For creating the interactive web-based chat interface.

*   **Data Processing and Preparation:**
    *   **LangChain:** Used in `preprocess.py` for document loading (e.g., `PyPDFLoader`, `TextLoader`) and text splitting (`RecursiveCharacterTextSplitter`).

*   **AI Orchestration and Tooling:**
    *   **Google Agent Development Kit (ADK) (`google.adk.agents.Agent`):** The core agent framework used in `app.py` for managing conversation flow and tool usage.
    *   **LangGraph:** Used in `tools.py` to define and execute the multi-step `propose_gcp_architecture` tool, enabling stateful, graph-based AI workflows.

*   **Google Cloud Platform (GCP) Services:**
    *   **Vertex AI Search (formerly Discovery Engine):** Provides the powerful search capabilities over the custom corpus. Used as the backend for the knowledge base.
    *   **Vertex AI Generative Models (e.g., Gemini series):** The Large Language Models (LLMs) that power the agent's understanding, tool routing, response synthesis, and the proposal generation in the LangGraph tool.
    *   **Google Cloud Storage (GCS):** Used for storing raw documents, the processed `corpus.jsonl` file, and as a staging area for Vertex AI services.

*   **Configuration Management:**
    *   **`python-dotenv`:** For loading environment variables from a `.env` file.

*   **Development and Deployment (Current & Future):**
    *   **Pip-tools (`pip-compile`, `pip-sync`):** For managing Python dependencies and ensuring reproducible environments.
    *   **Docker (Future):** For containerizing the application for easier deployment and scalability (Dockerfile exists).
    *   **Google Cloud Run (Future):** Targeted serverless platform for deploying the containerized application.
    *   **Terraform (Implied for Cloud Run):** Infrastructure-as-Code tool for managing GCP resources, with configuration files present in the repository.
