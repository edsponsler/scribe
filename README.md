# ASRR Phase 1 Tutorial: The Knowledge Foundation

Welcome! This tutorial provides a comprehensive, step-by-step guide to implementing the "Automated Systematic Retrieval and Review" (ASRR) project in three phases. 

The goal of this phase is to build the ASRR's "mind" by creating a sophisticated, private search engine over a curated set of documents. By the end of this guide, you will have a functional retrieval backend on Google Cloud, meeting the requirements of Phase 2 of the ASRR project. For a copmlete overview fo the ASRR project see: [SOMMAS Wiki](https://github.com/edsponsler/sommas/wiki)

## Prerequisites

Before you begin, ensure you have the following setup:
* **Windows Subsystem for Linux (WSL)** with **Ubuntu** installed and integrated into VS Code. This tutorial assumes all commands are run from the VS Code terminal connected to your WSL Ubuntu environment. [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
* **Visual Studio Code (VS Code)** installed. [Visual Studio Code](https://code.visualstudio.com/download)
* **A Google Cloud Platform (GCP) Account** with a project created and billing enabled. [Google Cloud](https://cloud.google.com/)
* **A GitHub Account**. [GitHub Account](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github) | See this [Guide for setting up Authentication with GitHub](https://github.com/edsponsler/sommas/blob/main/docs/AUTHENTICATING-WITH-GITHUB.md).
* (Optional) **Docker Desktop** installed with WSL. [Docker | WSL 2](https://docs.docker.com/desktop/features/wsl/#download)
* (Optional) **Windows Terminal** installed. [Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/install)

---

### Step 1: Set Up Your Local Project

We will begin by creating a copy of the project template, which contains the necessary scripts and sample documents.

1.  **Create Your Repository:**
    * Navigate to the template repository in your browser: [https://github.com/edsponsler/sommas](https://github.com/edsponsler/sommas)
    * Click the green **"Use this template"** button and select **"Create a new repository"**.
    * Give your new repository a name (e.g., `my-asrr-project`) and click **"Create repository"**.

2.  **Clone Your Repository:**
    * Open VS Code with a terminal connected to WSL Ubuntu.
    * First, ensure Git is installed in your Ubuntu environment:
        ```bash
        sudo apt update && sudo apt install git -y
        ```
    * Clone the repository you just created. Replace `<your-github-username>` and `<your-repo-name>` with your details.
        ```bash
        git clone git@github.com:<your-github-username>/<your-repo-name>.git
        ```
    * Navigate into your new project directory:
        ```bash
        cd <your-repo-name>
        ```

3.  **Create a Python Virtual Environment:**
    * It is a best practice to keep project dependencies isolated. We will create a virtual environment inside your project folder.
        ```bash
        python3 -m venv .venv
        ```
    * Activate the environment. You must do this every time you open a new terminal to work on this project.
        ```bash
        source .venv/bin/activate
        ```
    * Your terminal prompt should now be prefixed with `(.venv)`, indicating the environment is active.

### Step 2: Configure the Google Cloud CLI

The `gcloud` command-line interface (CLI) is essential for interacting with your GCP resources. For this WSL-based setup, a dual installation is required.

1.  **Install the `gcloud` CLI:**
    * **On Windows:** Follow the official instructions to [install the gcloud CLI on Windows](https://cloud.google.com/sdk/docs/install). This is required because it can seamlessly open a browser for authentication.
    * **In WSL Ubuntu:** Follow the official instructions to [install the gcloud CLI on Debian/Ubuntu](https://cloud.google.com/sdk/docs/install#deb).

2.  **Authenticate from WSL:**
    * You need to perform two separate authentications: one for the `gcloud` CLI tool itself, and one for your application code.
    * **For the `gcloud` CLI:** This authorizes you to run `gcloud` commands in the terminal. The `--no-browser` flag is crucial as it prevents WSL from trying (and failing) to open a browser window.
        ```bash
        gcloud auth login --no-browser
        ```
    * This command will output a long `gcloud auth ...` command. **Copy this entire command**.
    * Open a standard **Windows Command Prompt (CMD)**, paste the copied command, and press Enter.
    * This will launch a browser window on your Windows desktop. Complete the login and grant the necessary permissions.
    * **For your Application Code (ADC):** This creates a credential file that your Python script will automatically find and use.
        ```bash
        gcloud auth application-default login --no-browser
        ```
    * You will need to repeat the same copy/paste process into a Windows CMD to complete the browser-based login.

3.  **Set Your Project:**
    * Once authentication is complete, return to your WSL terminal. Tell `gcloud` which project to work on. Replace `your-project-id` with your actual GCP Project ID.
        ```bash
        gcloud config set project your-project-id
        ```

### Step 3: Create Cloud Storage Buckets

[UPDATED: Create Google Cloud resources with terraform. See scripts/]

We need two secure locations in Google Cloud Storage to hold our documents: one for the original raw files and another for the processed, machine-readable data.

1.  **Choose Unique Bucket Names:** Bucket names must be globally unique. A good practice is to append your unique project ID. For example: `asrr-raw-data-your-project-id`.

2.  **Create the Buckets:** Run these commands from your WSL terminal, replacing the placeholder names with your unique names.
    ```bash
    # Create the raw bucket
    gcloud storage buckets create gs://your-unique-raw-bucket-name --project=your-project-id --uniform-bucket-level-access

    # Create the processed bucket
    gcloud storage buckets create gs://your-unique-processed-bucket-name --project=your-project-id --uniform-bucket-level-access
    ```

### Step 4: Run the Preprocessing Pipeline

This step uses the provided Python script to transform our raw documents into a structured format suitable for Vertex AI. We will use `pip-tools` to ensure a consistent and reproducible Python environment.

1.  **Install Python Dependencies:**
    * Ensure your virtual environment is active (`source .venv/bin/activate`).
    * First, install the `pip-tools` package itself:
        ```bash
        pip install pip-tools
        ```
    * Next, use `pip-compile` to generate a `requirements.txt` file from your `requirements.in`. This command resolves and pins all dependencies, creating a complete "lock file" for your environment.
        ```bash
        pip-compile requirements.in
        ```
    * Finally, use `pip-sync` to install all the packages listed in the newly generated `requirements.txt`. `pip-sync` is powerful because it ensures your environment exactly matches the requirements file, adding missing packages and removing any that don't belong.
        ```bash
        pip-sync
        ```

2.  **Upload Raw Corpus to Cloud Storage:**
    * The `proto-corpus` folder in your repository contains the sample documents. Upload them to the `raw` bucket you created.
        ```bash
        gcloud storage cp proto-corpus/* gs://your-unique-raw-bucket-name/
        ```

3.  **Configure and Run the Script:**
    * The script reads its configuration from a local `.env` file. To set this up, create a copy of the example file:
        ```bash
        cp .env-example .env
        ```
    * Open the new `.env` file in VS Code and replace the placeholder values with your actual GCP Project ID and the unique bucket names you created in Step 3.
    * Save the `.env` file. The script will automatically load these settings when you execute it:
        ```bash
        python preprocess.py
        ```
    * The script will download the raw files, perform intelligent chunking, add source metadata, and upload a single `processed_corpus.jsonl` file to your processed bucket.

4.  **Verify the Output:**
    * In the Google Cloud Console, navigate to your processed bucket and confirm that the `processed_corpus.jsonl` file exists.

### Step 5: Build the Search Engine with Vertex AI

With our data prepared, we can now build the search engine itself. We will first create the backend datastore and then create an "App" to act as the frontend interface.

1.  **Create the Datastore:**
    * In the GCP Console, search or navigate to **AI Applications**.
    * Select the **Data Stores** tab from the left menu.
    * Click **+ CREATE NEW DATASTORE**.
    * Configure it with the following settings:
        * **Data Source:** Select **Cloud Storage**.
        * **Import data from Cloud Storage:** Select **Structured data (JSONL)**.
        * **Synchronization frequency** Select **One time**.
        * **Select a folder or a file you want to import** Choose **Folder** tab. Click **Browse**. Select the **row** for your processed bucket (`gs://your-unique-processed-bucket-name`). **Do not click into the bucket.** The path should be to the bucket itself.
        * **Select Continue**
        * **Review schema and assign key properties** The defaults are fine. The Key property fields will be blank.
        * **Select Continue**            
        * **Configure your data store** For **Multi-region** Select **global (Global)**.
        * **Datastore name:** Give it a name like `asrr-corpus-datastore`.
    * Click **Create** and wait 15-30 minutes for the indexing process to complete.

2.  **Create the App and Link the Datastore:**
    * Select the **Apps** tab from the left menu.
    * Click **+ CREATE APP** and select the **Custom search (general)** type.
    * Ensure both **Enterprise edition features** and **Advanced LLM features** are checked.
    * **App name:** Give it a name like `asrr-search-engine`.
    * Fill in your company name.
    * **Location of your app** For **Multi-region** Select **global (Global)**.
    * **Select Continue**
    * Check the box next to the `asrr-corpus-datastore` you just created and click **Create**.

### Step 6: Test Your Search Engine

This final step validates all our work. You will now be able to query your knowledge base.

1.  **Navigate to the Preview Page:**
    * From **AI Application** Select the **Apps** tab, you will now see your `asrr-search-engine` app. Click on it.
    * Select the **Preview** tab.

2.  **Run Test Queries:**
    * Use the search bar to ask questions based on the documents you indexed:
        * `What is a K-line?`
        * `What is the concept of "mindless" agents?`
        * `how is cloud run serverless?`
    * Observe the results. Each result should show a relevant text snippet and the original source file, confirming your entire pipeline is working correctly.

---

## Expanding Your Corpus

To make your ASRR more powerful, you can add more documents to its knowledge base.

* **Find Documents:** You can find scientific and philosophical texts on sites like Google Scholar or public repositories like arXiv.org and PubMed Central. For technical documentation, you can save web pages from official sites as text files.
* **Update Your Corpus:**
    1.  Add the new files (PDF, TXT, etc.) to your local `proto-corpus` folder.
    2.  Re-run the upload command: `gcloud storage cp proto-corpus/* gs://your-unique-raw-bucket-name/`
    3.  Re-run the processing script: `python preprocess.py`
    4.  Vertex AI Search will automatically detect the changes in the `processed_corpus.jsonl` file and re-index your datastore.

You have now successfully completed Phase 1 of the ASRR Project. Congratulations!

---

## ASRR Phase 2: The Conversational Analyst

You've built a powerful, private search engine. In Phase 2, we will transform this search engine into a conversational partner. The objective is to create a true subject matter expert—an AI you can dialogue with to clarify concepts, ask follow-up questions, and receive answers that are synthesized and grounded *only* in your curated documents.

We will use the Vertex AI Agent Builder framework to wrap our search engine in a conversational layer, and Streamlit to create a simple but effective web interface for interaction.

### Step 7: Configure the Conversational App

The `app.py` script, which runs the conversational agent, needs to know the ID of your search app and which Cloud Storage bucket to use for temporary files.

1.  **Find Your Data Store ID:**
    *   In the Google Cloud Console, navigate to **AI Applications**.
    *   Select the **Apps** tab and click on the `asrr-search-engine` app you created in Phase 1.
    *   In the app's menu, select the **Data** tab.
    *   You will see your datastore listed. Copy its **ID** (it will be a long alphanumeric string).

2.  **Update Your `.env` File:**
    *   Open the `.env` file in your project.
    *   You will see placeholders for `DATA_STORE_ID` and `STAGING_BUCKET`.
    *   Paste the **Data Store ID** you just copied as the value for `DATA_STORE_ID`.
    *   For `STAGING_BUCKET`, provide the full path to the staging bucket you created in Step 3 (e.g., `gs://your-unique-staging-bucket-name`).

### Step 8: Understanding the Conversational Analyst (`app.py`)

The `app.py` script orchestrates three main components to bring your conversational analyst to life:

1.  **The Search Tool (`search_knowledge_base`):**
    This Python function is the bridge to the search engine you built in Phase 1. It takes a user's query string as input. It initializes the `discoveryengine` client and points it to your specific data store using the `GCP_PROJECT_ID` and `DATA_STORE_ID` from your `.env` file. It sends the query and formats the raw search results into a clean string, including the content and the source file for each result. This formatted string is what the agent will "read" to find an answer.

2.  **The Agent (`Agent`):**
    The agent is the "brain" of the operation, created using `from google.adk.agents import Agent`. It's initialized with a specific generative model (like Gemini 2.5 Flash) and, most importantly, the `search_knowledge_base` function is passed into its `tools` list. This tells the agent: "When you need to answer a question, you have one tool you can use: this search function." The agent learns to automatically call this tool with a relevant search query whenever it needs information.

3.  **The Web Interface (`Streamlit`):**
    Streamlit is a framework that turns Python scripts into interactive web apps. `app.py` uses it to create the chat interface.
    *   `st.title` sets the page title.
    *   `st.session_state` is used to remember the chat history, so the conversation persists as you interact with the app.
    *   The main loop waits for user input with `st.chat_input`. When you send a message, it's added to the history and displayed on the screen.
    *   The agent's response is streamed back to the UI using `st.write_stream`, providing a real-time, "typing" effect for a better user experience.

### Step 9: Launch and Interact with Your Analyst

With the configuration complete, you can now launch the web application.

1.  **Launch the App:**
    *   In your WSL terminal (with the `.venv` virtual environment active), run the following command:
        ```bash
        python -m streamlit run app.py
        ```

2.  **Interact with the Analyst:**
    *   This command will start a local web server and should automatically open a new tab in your browser. You'll be greeted by the "ASRR: The Conversational Analyst" interface.
    *   Try asking the same questions from the end of Phase 1. Notice the difference: instead of just a list of search results, the agent now provides a synthesized, conversational answer with citations pointing back to the source documents.

---

## Phase 2 Complete

Congratulations! You have successfully completed Phase 2 of the ASRR project. You now have a functional conversational agent that can reason over your private document set, providing grounded, synthesized answers through an interactive web interface. This lays the critical foundation for Phase 3, where we will explore deploying, evaluating, and extending the agent's capabilities.

---

## ASRR Phase 3: The Implementation Strategist

You've built a conversational analyst. In this final phase, we elevate the agent from a mere subject matter expert to a true **Implementation Strategist**. The objective is to empower the agent to deconstruct high-level, complex challenges and synthesize concrete, actionable implementation plans on Google Cloud.

This new capability is powered by **LangGraph**, a library for building stateful, multi-actor applications with LLMs. We use it to define a more sophisticated, multi-step tool that can first perform research and then synthesize a detailed proposal based on its findings.

### How It Works: The `propose_gcp_architecture` Tool

The core of Phase 3 is the new `propose_gcp_architecture` tool, which is defined in the new `tools.py` file. This isn't just a single function call; it's a stateful graph that executes a sequence of steps.

1.  **The State (`GraphState`)**: LangGraph works by passing a "state" object between nodes. Our state is a simple dictionary that tracks the progress of the task:
    *   `user_request`: The original, complex query from the user.
    *   `research_results`: The context gathered from our Vertex AI Search knowledge base.
    *   `final_proposal`: The final, synthesized architectural plan.

2.  **The Nodes (Functions)**: Each step in our workflow is a "node," which is just a Python function that modifies the state.
    *   `survey_technologies_node`: This is the first step. It takes the `user_request` from the state, performs a comprehensive search against our knowledge base, and populates the `research_results` field in the state.
    *   `synthesize_proposal_node`: This node runs after the research is complete. It takes both the original `user_request` and the new `research_results` from the state, feeds them into an expert-level prompt, and uses a generative model to write a detailed architectural proposal. The output is saved to the `final_proposal` field.

3.  **The Graph (`workflow`)**: We define the order of operations by adding nodes and connecting them with "edges". Our graph is a straightforward sequence:
    *   The entry point is `survey_technologies`.
    *   After `survey_technologies` completes, the graph transitions to `synthesize_proposal`.
    *   After `synthesize_proposal` completes, the graph finishes, and the final state (containing the proposal) is returned.

### Integration into the Agent (`app.py`)

The main `app.py` script is updated to integrate this powerful new tool:

*   **Dual Tools**: The agent is now initialized with a list of *two* tools: the simple `search_knowledge_base` function and the new, advanced `propose_gcp_architecture` graph-based tool.
*   **Intelligent Routing**: The system prompt has been enhanced to act as a router. It instructs the agent on how to choose the correct tool based on the user's query.

This routing is the key to the agent's new strategic capability. For example:

*   **Simple Question**: If you ask, `"Who is Minsky?"`, the agent, guided by the prompt, recognizes this as a factual lookup and calls the simple, efficient `search_knowledge_base` tool.
*   **Complex Question**: If you ask, `"Propose a scalable architecture for deploying thousands of simple, specialized agents."`, the agent identifies this as a high-level design task. It then invokes the `propose_gcp_architecture` tool, triggering the entire LangGraph workflow of research followed by synthesis.

### How to Run the Final Application

The method for launching the app remains the same. The new, more powerful agent is now the default.

1.  Ensure your virtual environment is active: `source .venv/bin/activate`
2.  Launch the Streamlit application from the project root folder:
    ```bash
    python -m streamlit run app.py
    ```

---

## Project Complete

This concludes the three-phase implementation of the ASRR project. You have successfully built:

*   **Phase 1**: A robust, private knowledge base using Google Cloud Storage and Vertex AI Search.
*   **Phase 2**: A conversational analyst capable of answering questions grounded in that knowledge base.
*   **Phase 3**: An implementation strategist that uses a multi-step LangGraph agent to research and formulate detailed architectural proposals.

### Future Development

This project serves as a powerful foundation. Here are some ideas for extending its capabilities:

*   **Extend the Corpus**: The single most impactful improvement is to expand the knowledge base. You can add extensive documentation for other cloud providers like AWS and Azure for comparative analysis, ingest more advanced research papers on topics like Multi-Agent Reinforcement Learning (MARL), or add internal best-practice documents.
*   **Advanced Tool Development**: You can build more specialized LangGraph tools as described in the project plan, such as a tool to explicitly compare and contrast frameworks (e.g., Mesa vs. Ray) or a tool to design the "K-line" analogue using a graph database.
*   **Team Deployment**: Package the application using **Docker** and deploy it as a service on **Google Cloud Run**, making it accessible to your entire team.
