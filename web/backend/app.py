"""FastAPI backend for the public web UI - one WebSocket endpoint that
streams a live review of a GitHub PR back to the browser.

Reuses main.run_review and cli.py's resolve/clone/parse helpers verbatim -
no pipeline logic lives here, only the WS protocol and per-visitor key
handling. Every visitor brings their own GitHub token AND their own choice
of LLM provider(s) - any OpenAI-compatible endpoint, not just Gemini/Groq
(sent once in the init message as base_url+api_key+model per role); nothing
is logged or persisted server-side, so there is no shared secret here for a
stranger to drain.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cli  # noqa: E402
from llm import ProviderConfig  # noqa: E402
from main import run_review  # noqa: E402

# Same stage names/labels/order tui.py already established for run_review's
# on_progress callback, duplicated here as plain dicts rather than importing
# tui.py itself - a stateless API server has no reason to pull in a whole
# terminal-app framework (textual) just to reuse two literals.
# "resolving"/"cloning" happen before run_review is even called (same as
# tui.py's do_review does them directly, not via on_progress).
STAGE_LABELS = {
    "resolving": "Resolving PR",
    "cloning": "Cloning",
    "size_gate": "Size gate",
    "diff_fetched": "Diff fetched",
    "semgrep_done": "Semgrep scan",
    "a1_pass_done": "AI review pass",
    "a2_done": "A2 cluster + vote",
    "a3_done": "A3 adversarial validate",
    "posted": "Post",
}
ALL_STAGES = ["resolving", "cloning", "size_gate", "diff_fetched", "semgrep_done"] + ["a1_pass_done"] * 4 + [
    "a2_done", "a3_done", "posted",
]

RUN_SEMAPHORE = asyncio.Semaphore(3)  # self-preservation for a free-tier host, not a per-user quota

app = FastAPI(title="oss-bugbot web")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tightened to the deployed frontend origin at deploy time, see README
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health() -> dict:
    return {"status": "ok", "stages": ALL_STAGES}


_DRAIN_SENTINEL = object()


async def _run_with_progress(websocket: WebSocket, coro_factory):
    """on_progress fires SYNCHRONOUSLY from inside run_review's own event
    loop (not a thread), so it can't just `await websocket.send_json(...)`
    directly. The first version of this used `loop.create_task(...)` per
    event (fire-and-forget) - live testing caught that those tasks race
    against the handler's own completion and can be dropped or reordered
    before the socket closes. A single queue with one dedicated drainer
    coroutine guarantees in-order, fully-flushed delivery instead: on_progress
    only ever does a non-blocking put, the drainer is the sole writer to the
    socket, and we explicitly await it to drain before moving on.
    """
    queue: asyncio.Queue = asyncio.Queue()

    def on_progress(stage: str, detail: str) -> None:
        queue.put_nowait((stage, detail))

    async def drain() -> None:
        while True:
            item = await queue.get()
            if item is _DRAIN_SENTINEL:
                return
            stage, detail = item
            try:
                await websocket.send_json({
                    "type": "stage", "stage": stage,
                    "label": STAGE_LABELS.get(stage, stage),
                    "detail": detail, "status": "done",
                })
            except Exception:
                pass  # visitor disconnected mid-run; the review still finishes, result just isn't delivered

    drain_task = asyncio.create_task(drain())
    try:
        result = await coro_factory(on_progress)
    finally:
        await queue.put(_DRAIN_SENTINEL)
        await drain_task
    return result


@app.websocket("/ws/review")
async def ws_review(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        init = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    url = (init.get("url") or "").strip()
    github_token = init.get("github_token") or ""
    post = bool(init.get("post", False))

    def _config(d: dict) -> ProviderConfig:
        return ProviderConfig(base_url=d.get("base_url", ""), api_key=d.get("api_key", ""), model=d.get("model", ""))

    a1_config = _config(init.get("a1") or {})
    a2_config = _config(init.get("a2") or {})
    a3_primary_config = _config(init.get("a3_primary") or {})
    a3_fallback_config = _config(init.get("a3_fallback") or {})

    try:
        owner, repo, pr_number = cli.parse_github_url(url)
    except ValueError as exc:
        await websocket.send_json({"type": "error", "stage": "resolving", "message": str(exc)})
        await websocket.close()
        return

    async with RUN_SEMAPHORE:
        try:
            if pr_number is None:
                await websocket.send_json({
                    "type": "stage", "stage": "resolving",
                    "label": "Looking up open pull requests", "detail": "", "status": "running",
                })
                try:
                    prs = await cli.list_open_prs(owner, repo, github_token)
                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    await websocket.send_json({"type": "error", "stage": "resolving", "message": str(exc)})
                    await websocket.close()
                    return
                if not prs:
                    await websocket.send_json({
                        "type": "error", "stage": "resolving",
                        "message": f"{owner}/{repo} has no open pull requests right now.",
                    })
                    await websocket.close()
                    return
                pr_number = prs[0]["number"]

            await websocket.send_json({
                "type": "stage", "stage": "resolving",
                "label": "Resolving PR", "detail": "", "status": "running",
            })
            try:
                info = await cli.resolve_pr_info(owner, repo, pr_number, github_token)
            except httpx.HTTPStatusError as exc:
                await websocket.send_json({
                    "type": "error", "stage": "resolving",
                    "message": f"could not resolve PR ({exc.response.status_code})",
                })
                await websocket.close()
                return
            except (httpx.RequestError, ValueError) as exc:
                await websocket.send_json({"type": "error", "stage": "resolving", "message": str(exc)})
                await websocket.close()
                return
            await websocket.send_json({
                "type": "stage", "stage": "resolving",
                "label": "Resolving PR", "detail": f"{info['head_ref']} @ {info['head_sha'][:7]}", "status": "done",
            })

            with TemporaryDirectory(prefix="oss-bugbot-web-") as tmp:
                checkout = Path(tmp)
                await websocket.send_json({
                    "type": "stage", "stage": "cloning",
                    "label": "Cloning", "detail": "", "status": "running",
                })
                try:
                    await asyncio.to_thread(cli.clone_pr_branch, info["head_clone_url"], info["head_ref"], checkout)
                except subprocess.CalledProcessError as exc:
                    await websocket.send_json({
                        "type": "error", "stage": "cloning", "message": f"clone failed: {exc.stderr}",
                    })
                    await websocket.close()
                    return
                except subprocess.TimeoutExpired:
                    await websocket.send_json({
                        "type": "error", "stage": "cloning",
                        "message": f"clone exceeded {cli.CLONE_TIMEOUT_SECONDS}s - this repo/branch is too large "
                                   f"for the web demo, try a smaller PR.",
                    })
                    await websocket.close()
                    return
                except FileNotFoundError:
                    await websocket.send_json({
                        "type": "error", "stage": "cloning", "message": "git executable not found on the server.",
                    })
                    await websocket.close()
                    return
                await websocket.send_json({
                    "type": "stage", "stage": "cloning", "label": "Cloning", "detail": "", "status": "done",
                })

                result = await _run_with_progress(
                    websocket,
                    lambda on_progress: run_review(
                        owner, repo, pr_number, info["head_sha"], github_token,
                        a1_config, a2_config, a3_primary_config, a3_fallback_config, checkout,
                        post=post, on_progress=on_progress,
                    ),
                )

            await websocket.send_json({"type": "result", "data": result})
        except WebSocketDisconnect:
            return
        finally:
            try:
                await websocket.close()
            except Exception:
                pass
