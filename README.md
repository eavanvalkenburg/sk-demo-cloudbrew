# Semantic Kernel Demo for CloudBrew 2024 - A two-day Microsoft Azure event

This demo shows how to use [Semantic Kernel](https://github.com/microsoft/semantic-kernel) to run a web app using mesop that uses both online models with OpenAI and offline models with Ollama.

## Prerequisites
Copy the .env.example file and rename it to .env. Fill in the values for the following variables:
- OPENAI_API_KEY
- OPENAI_CHAT_MODEL_ID (recommended is GPT-4o)
- OPENAI_EMBEDDING_MODEL_ID (recommended is text-embedding-3-small)
- AZURE_AI_SEARCH_ENDPOINT
- AZURE_AI_SEARCH_API_KEY
- AZURE_AI_SEARCH_INDEX_NAME

You can also adjust the Qdrant settings.


The data loading requires this environment variable:
- GITHUB_TOKEN

### qdrant
Setup qdrant, this can be done in a container locally, or on a remote server. The following command will start a qdrant container on your local machine:
```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

### venv
Create a virtual environment and activate it:
```bash
uv venv --python=3.12
source .venv/bin/activate (or .venv/Scripts/activate on Windows)
uv pip install -r requirements.txt --prerelease=allow
```

## Loading data
To load the data, run the following command, depending on the repo used, this might take a while:
```bash
source .venv/bin/activate (or .venv/Scripts/activate on Windows)
python data_ingestion/main.py
```

You can control which service you want to have indexed, by using the following flags:
```bash
python data/main.py --no-azure # this will only index Qdrant
```
or   
```bash
python data/main.py --no-qdrant # this will only index Azure AI
```

By default, both will be indexed.

To review the data when you have Qdrant running locally you can open: `http://localhost:6333/dashboard` in your browser.

## Running the app
To run the app, simply run the following command:
```bash
source .venv/bin/activate
mesop main.py
``` 

The app will then be available in your browser at `http://localhost:32123/chat`.

When you want to go offline, either set the `MODE` environment variable to `offline` (this won't be picked up until you return mesop) or turn off the network.
If the `MODE` environment is anything other then `offline` or `online`, the app try to actually get a connection (it does a test to the `1.1.1.1` DNS Server by default).
