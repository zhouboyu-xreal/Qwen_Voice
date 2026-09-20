# Agent Memory MCP Server

`memory_mcp_server.py` exposes the existing `MemoryRuntime` through the MCP
stdio transport. The server owns one `SessionDB`, `MemoryNodeManager`, and
`MemoryRuntime` for its entire process lifetime.

## Start

```bash
/Applications/miniconda3/envs/python3_11/bin/python \
  /Users/zhouboyu/Documents/agent_memory/scripts/memory_mcp_server.py
```

The MCP protocol uses stdout. Memory logs, the database, ASR results, and test
reports are resolved under `memory_mcp_server.result_dir` using
`db_name`, `log_name`, `asr_result_dir`, and `report_name`. The server creates
`result_dir` on startup. Log level, queue timeout, runtime/model settings, and
environment references such as `${GLM_API_KEY}` are all read from
`config.yaml`.

Example client configuration:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "/Applications/miniconda3/envs/python3_11/bin/python",
      "args": [
        "/Users/zhouboyu/Documents/agent_memory/scripts/memory_mcp_server.py"
      ]
    }
  }
}
```

## Tools

The MCP server exposes three tools:

- `process_audio_files`: accepts a non-empty `files` array. Each item contains
  an `audio_path` and may override `source_type`, `session_start`, and `tags`.
  The server creates a background audio-processing job and immediately returns
  a `job_id`, avoiding long MCP tool-call idle timeouts.
- `get_audio_processing_job`: accepts a `job_id` and returns the latest job
  status, progress fields, and the final per-file transcription/memory report
  once processing has finished.
- `trigger_memory_recall`: searches the latest committed memory snapshot using
  a `query`, with optional `tags` and `time_end`. The response contains the
  assembled `memory_context`, recall-mode metadata, and timing information.

`process_audio_file`, transcript ingestion, reflection, and pending-queue
operations are internal implementation steps of the `process_audio_files` job;
they are not exposed as separate MCP tools.

The server does not expose internal database methods or the manager's private
fact/state extraction functions. This keeps pending buffering and write
ordering in `MemoryRuntime`.
