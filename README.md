# Ragpi

Ragpi is an open-source AI assistant that answers questions using your documentation, GitHub issues, and READMEs. It combines LLMs with intelligent search to provide relevant, documentation-backed answers through a simple API. It supports multiple providers like OpenAI, Ollama, and Deepseek, and has built-in integrations with Discord and Slack. A web widget integration is also available to embed the assistant in your website.

[Documentation](https://docs.ragpi.io) | [API Reference](https://docs.ragpi.io/api)

## Key Features

- 📚 Builds knowledge bases from docs, GitHub issues and READMEs
- 🤖 Agentic RAG system for dynamic document retrieval
- 🔌 Supports OpenAI, Ollama, Deepseek & OpenAI-Compatible models
- 💬 Discord and slack integrations for community support
- 🚀 API-first design with Docker deployment

## Example Workflow

Here's a simple workflow to get started with Ragpi once it's deployed:

### 1. Set up a Source with a Connector

- Use the [`/sources`](https://docs.ragpi.io/api#tag/Sources/operation/create_source_sources_post) endpoint to configure a source with your chosen connector.
- Each connector type has its own configuration parameters.

Example payload using the Sitemap connector:

```json
{
  "name": "example-docs",
  "description": "Documentation for example project. It contains information about configuration, usage, and deployment.",
  "connector": {
    "type": "sitemap",
    "sitemap_url": "https://docs.example.com/sitemap.xml"
  }
}
```

### 2. Monitor Source Synchronization

- After adding a source, documents will be synced automatically. You can monitor the sync process through the [`/tasks`](https://docs.ragpi.io/api#tag/Tasks/operation/get_task_tasks__task_id__get) endpoint.

### 3. Chat with the AI Assistant

- Use the [`/chat`](https://docs.ragpi.io/api#tag/Chat/operation/chat_chat_post) endpoint to query the AI assistant using the configured sources:

  ```json
  {
    "sources": ["example-docs"],
    "messages": [
      { "role": "user", "content": "How do I deploy the example project?" }
    ]
  }
  ```

- You can also interact with the AI assistant through the [Discord](https://docs.ragpi.io/integrations/discord) or [Slack](https://docs.ragpi.io/integrations/slack) integration,
  or by embedding the [Web Widget](https://docs.ragpi.io/integrations/web-widget) in your website.

## Connectors

Ragpi supports the following connectors for building knowledge bases:

- **Documentation Website (Sitemap)**
- **GitHub Issues**
- **GitHub README Files**
- **GitHub PDF Files**
- **REST API Responses**

[Explore connectors →](https://docs.ragpi.io/connectors)

## Providers

Ragpi supports the following LLM providers for generating responses and embeddings:

- **OpenAI** (default)
- **Ollama**
- **Deepseek**
- **OpenAI-compatible APIs**

[Configure providers →](https://docs.ragpi.io/providers/overview)

## Model & Embedding Configuration

### Reasoning models (OpenAI Responses API)

Modern OpenAI reasoning models (e.g. `gpt-5.6-sol`, `gpt-5.6-terra`) can use function
tools together with active reasoning via the Responses API. This path is **opt-in** and
only available when `CHAT_PROVIDER=openai`:

```bash
CHAT_PROVIDER=openai
DEFAULT_CHAT_MODEL=gpt-5.6-sol
CHAT_USE_RESPONSES_API=true
REASONING_EFFORT=medium        # gpt-5.6: none|low|medium|high|xhigh|max (optional)
```

When `CHAT_USE_RESPONSES_API` is off (the default), all providers use the Chat Completions
API exactly as before. `reasoning_effort` may also be set per request. Reasoning continuity
is preserved across the tool-call loop within a single `/chat` request.

> **Privacy:** the Responses path sends `store=true`, i.e. conversation state is retained in
> OpenAI's stored Responses workflow (used for reasoning continuity across tool calls).
> Zero-Data-Retention (`store=false`) is not yet supported.

### Embedding model & dimensions

The embedding model and dimensionality are configured with `EMBEDDING_MODEL` and
`EMBEDDING_DIMENSIONS`. `text-embedding-3-large` is supported at its full 3072 dimensions:

```bash
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072
```

On Postgres, embeddings are always stored as full-precision float32. Above 2000 dimensions
(pgvector's approximate-index limit for the `vector` type) Ragpi automatically indexes a
half-precision (`halfvec`) expression with HNSW and reranks candidates by exact float32
cosine — full-precision ranking with scalable search. This requires the **pgvector server
extension ≥ 0.8.2** (0.8.2 fixed a buffer overflow in parallel HNSW index builds); the
bundled `pgvector/pgvector:pg17` image satisfies it. Redis needs no change. Retrieval
over-fetch is tunable via `EMBEDDING_CANDIDATE_MULTIPLIER` (default 10) and `HNSW_EF_SEARCH`.

Ragpi records a manifest for each store (embedding provider/model/dimensions, storage and
index schema). At startup it validates the configured settings against the manifest and
**fails fast with actionable guidance** on an incompatible change (rather than a cryptic
insert error) — note that a model change requires re-embedding even at the same dimensions.

#### Changing the embedding model / dimensions

Changing the embedding identity (provider, model, or dimensions) requires re-embedding all
documents. There is no automatic data migration. Do **not** use `docker compose down -v`
(it deletes more than embeddings). Instead:

1. Back up the database / Redis data.
2. Stop the API and workers.
3. Remove the document vectors **and** the store manifest, keeping source metadata:
   - **Postgres:** `DROP TABLE <DOCUMENT_STORE_NAMESPACE>;` and delete its row from
     `ragpi_store_manifest` (leave the `source_metadata` table intact).
   - **Redis:** drop the index, delete its `<namespace>:sources:*` keys, and delete the
     `<namespace>:__manifest__` key.
4. Restart with the new `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` (preflight recreates the
   schema/index and writes a new manifest).
5. Re-sync every source (connectors re-fetch and re-embed).
6. Verify document counts and semantic-search results.

If a persistent store is left un-migrated after a dimension/model change, startup fails with
guidance rather than corrupting data. For an existing pre-0.8.2 pgvector extension on the
large path, set `PG_UPDATE_VECTOR_EXTENSION=true` to run `ALTER EXTENSION vector UPDATE` at
startup — note this upgrades the extension for the **entire** database, so back up and
revalidate other pgvector-dependent applications first.

## Integrations

Ragpi supports the following integrations for interacting with the AI assistant:

- [**Discord**](https://docs.ragpi.io/integrations/discord)
- [**Slack**](https://docs.ragpi.io/integrations/slack)
- [**Web Widget**](https://docs.ragpi.io/integrations/web-widget)

## Contributing

Contributions to Ragpi are welcome! Please check out the [contributing guidelines](CONTRIBUTING.md) for more information.
