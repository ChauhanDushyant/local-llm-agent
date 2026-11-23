# Local LLM Agent with Tool Calling and Streaming

A terminal AI agent written in Python that runs fully on a local machine. It uses the Qwen3 4B model served by Ollama through the OpenAI compatible API, so no paid API key is needed.

## What the agent can do

* Chat in a loop and remember the conversation by sending the full message history on every call
* Call tools: current time, user city and city weather
* Chain tools: for "weather of the city where I live" it first calls the city tool and then the weather tool
* Stream the answer word by word instead of waiting for the full reply
* Show coloured debug logs of every step, tool call and tool result

## Concepts I learned and implemented

1. LLMs are stateless, so memory is the job of the application (message history)
2. Prompt engineering: role, context, task, constraints, examples, output format and prompt injection risks
3. Tool calling with JSON schemas, and why a tool description is only a hint to the model
4. Agent loop with a MAX_STEPS safety limit and a for else fallback message
5. Error handling: tool crashes are caught and sent back to the model as text, and API calls use raise_for_status
6. Streaming: stream=True, delta chunks, flush=True, and rebuilding tool calls that arrive in small pieces, possibly interleaved, using the chunk index
7. Safe interrupt: Ctrl+C stops generation without crashing the program
8. Debug mode through a DEBUG flag in the .env file and a small log utility
9. Modular code: run_tool in tools.py, logging in Utilities
10. Embeddings with nomic embed text (768 dimensions) and cosine similarity using NumPy
11. Semantic search over a small knowledge base of text documents (Code Snippets folder)

## Project structure

```
simple_ai_assistant.py    main agent with streaming and the agent loop
tools.py                  tool functions, tool schemas and run_tool
Utilities/utilities.py    debug log helper
Code Snippets/            small test scripts for streaming, tool streaming, embeddings and search
knowledge_base/           sample company documents used for semantic search
.env.example              sample configuration
```

## How to run

1. Install Ollama and pull the models

```
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

2. Create a virtual environment and install packages

```
python -m venv venv
venv\Scripts\activate
pip install openai python-dotenv requests numpy
```

3. Copy .env.example to .env
4. Make sure Ollama is running, then start the agent

```
python simple_ai_assistant.py
```

Try: "what is the weather of the city where I live". Type exit to quit.

## Limitations and next steps

* The knowledge base search embeds one whole file as one vector, which makes the vector blurry. Next step is chunking
* Embeddings are created again on every start. Next step is saving them to disk
* Search always returns top results even when nothing is relevant. Next step is a score threshold
* The search is not yet connected to the agent. Next step is a full RAG pipeline with a search tool
* The user city tool returns a fixed value for testing
