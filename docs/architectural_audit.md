# DAAP Real-Time Architectural Audit

Based on the Graphify analysis (`GRAPH_REPORT.md`) and a deep dive into the source code (specifically `ws_handler.py`, `sessions.py`, and `engine.py`), I have identified several critical architectural bugs that will cause the system to fail, hang, or drop connections under real-time production loads.

Here is a breakdown of the structural flaws and their optimal, industry-standard solutions.

---

## 1. WebSocket Head-of-Line Blocking (The "Deadlock" Bug)

**The Bug:**
In `ws_handler.py`, the WebSocket read loop (`await websocket.receive_text()`) is inextricably tangled with execution logic. 
- When `_run_pending_topology_direct()` is triggered, it `await`s the entire execution of the DAG (which could take minutes).
- When `_run_agent_with_question_pump()` is waiting for the agent to plan (`else` block), it uses `asyncio.wait` with a timeout, **completely pausing** `receive_text()`.
- **Impact:** While the LLM is thinking or agents are executing, the server cannot receive any WebSocket messages. If a user tries to send a `/cancel` command, close their browser, or if the load balancer sends a ping, the message sits unread in the TCP buffer. This leads to broken pipes, un-cancellable runaway LLM costs, and terrible UX.

**How Big Tech Solves This (The Actor / Mailbox Pattern):**
Companies like Slack, Discord, and OpenAI (Realtime API) strictly separate I/O from execution using the **Full-Duplex Actor Pattern**. 
- **The Open Source Solution:** Use standard `asyncio.Queue` (or Redis Pub/Sub for multi-instance) as a mailbox. 
- **Implementation:**
  1. The WebSocket handler should have exactly two background tasks: a `reader_task` (infinitely looping `receive_text` -> pushing to a Queue) and a `writer_task`.
  2. The LLM Agent and Topology Execution must be pushed to a separate worker task that reads from that Queue.
  3. If "cancel" is pulled from the queue, the worker task is cancelled immediately.

## 2. Missing Cancellation Tokens for DAG Execution

**The Bug:**
In `engine.py`, `execute_topology` walks the DAG and awaits `run_execution_step`. There is absolutely no mechanism to abort this execution midway if the user cancels or disconnects. The `ws_handler` might catch a disconnect, but the LLM calls and tool executions in `engine.py` will keep running until the DAG completes, burning API credits (runaway execution).

**How Big Tech Solves This (Context Cancellation):**
Golang popularized the `context.Context` pattern (used heavily by Kubernetes and Google internal systems), and Python handles this via `asyncio.Task.cancel()`.
- **The Open Source Solution:** 
  Store the active execution as an `asyncio.Task` on the `Session` object. When a cancellation event is received by the independent WebSocket reader, call `session.active_task.cancel()`. 
  Inside `engine.py`, catch `asyncio.CancelledError`, gracefully clean up resources, and immediately halt the DAG traversal.

## 3. Synchronous I/O Blocking the Async Event Loop

**The Bug:**
In `sessions.py` (`_on_node_complete` callback), when a node finishes, the system calls `daap_memory.remember_node_output(...)`. If this memory integration (which wraps SQLite or Mem0) performs synchronous disk I/O or network requests, it completely blocks the Python Async Event Loop. Because Node execution relies on `asyncio`, blocking the loop prevents other parallel nodes from progressing, artificially spiking latency.

**How Big Tech Solves This (Telemetry Offloading):**
High-throughput systems never do inline synchronous metrics/memory writes. AWS and Netflix use Fire-and-Forget telemetry pipelines.
- **The Open Source Solution:** 
  Use Python's `asyncio.to_thread()` or `loop.run_in_executor()` for synchronous DB writes. Even better, push the `NodeResult` to a background `asyncio.Queue` dedicated to memory/telemetry processing, completely removing disk I/O from the critical path of the topology execution.

## 4. Hardcoded, Non-Jittered Retry Logic for 429s

**The Bug:**
In `engine.py`, the rate-limit handler (`_is_rate_limit_error`) catches 429s and immediately does `await asyncio.sleep(1)` before retrying, with a hardcoded `circuit_breaker_threshold` of 2. If 10 parallel nodes hit a rate limit simultaneously, they will all sleep exactly 1 second and retry simultaneously (the Thundering Herd problem), guaranteeing a second 429 and opening the circuit breaker immediately.

**How Big Tech Solves This (Exponential Backoff with Jitter):**
Amazon specifically documented the "Exponential Backoff and Jitter" pattern to solve this exact microservice failure mode.
- **The Open Source Solution:** 
  Implement the `tenacity` library, or write a custom backoff that uses `sleep_time = min(cap, base * 2 ** attempt) + random.uniform(0, 1)`. This desynchronizes parallel nodes and radically improves API recovery rates without blowing the circuit breaker.

---

### Graphify Context Alignment
As identified in Graphify's `GRAPH_REPORT.md`:
- **Community 23 (WebSocket Handler)** and **Community 14 (Session Store)** are tightly coupled. The graph highlighted the "Master Agent Session Turn Pipeline", which structurally forces the connection lifecycle to depend directly on the execution pipeline. Decoupling these two communities via an async queue will resolve the architectural bottleneck.
