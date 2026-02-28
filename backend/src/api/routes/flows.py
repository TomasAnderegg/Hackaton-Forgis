"""REST endpoints for flow management."""

import json
import logging
import os
from typing import Any, Optional

from openai import AsyncAzureOpenAI

logger = logging.getLogger(__name__)

# ── LLM client (lazy-init, shared) ─────────────────────────────────────────

_openai_client: Optional[AsyncAzureOpenAI] = None


def _get_openai_client() -> AsyncAzureOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        )
    return _openai_client


# ── System prompt for flow generation ──────────────────────────────────────

_FLOW_SYSTEM_PROMPT = """You are an expert robot automation engineer. Generate a valid JSON flow definition for a DOBOT Nova 5 industrial robot arm based on the user's natural language description.

## Available Skills

### Robot Skills (executor: "robot")
- **move_joint** — Move robot to joint positions
  params: {"target_joints_deg": [j0,j1,j2,j3,j4,j5]}  or "{{variable_name}}"
  timeout_ms: 30000

- **move_linear** — Move robot TCP in Cartesian space
  params: {"target_pose": [x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]}
  timeout_ms: 30000

- **set_tool_output** — Control the pneumatic gripper (dual-solenoid)
  Close: {"index": 1, "status": 1}
  Open:  {"index": 2, "status": 1}
  timeout_ms: 3000

### Camera Skills (executor: "camera")
- **get_bounding_box** — YOLO object detection
  params: {"object_class": "box", "confidence_threshold": 0.5}
  timeout_ms: 10000

- **get_label** — GPT-4V OCR / label reading
  params: {"prompt": "Read the label text", "use_bbox": true}
  timeout_ms: 30000

- **start_streaming** — Start live camera feed
  params: {"fps": 15}

- **stop_streaming** — Stop camera feed
  params: {}

### I/O Skills (executor: "io_robot")
- **io_set_digital_output** — Set a digital output pin
  params: {"pin": 4, "value": true}

## Flow JSON Schema

```json
{
  "id": "unique_snake_case_id",
  "name": "Human Readable Name",
  "initial_state": "first_state_name",
  "loop": false,
  "variables": {
    "midair_joints": [152.345, -17.610, 87.397, 50.315, 2.243, 10.032],
    "pick_joints":   [152.345, -32.610, 72.397, 50.315, 2.243, 10.032],
    "place_joints":  [80.000,  -25.000, 80.000, 45.000, 2.000, 10.000]
  },
  "states": [
    {
      "name": "state_name",
      "steps": [
        {
          "id": "unique_step_id",
          "skill": "skill_name",
          "executor": "robot",
          "params": {},
          "timeout_ms": 30000,
          "error_handling": {"strategy": "stop", "max_retries": 3, "retry_delay_ms": 1000}
        }
      ]
    }
  ],
  "transitions": [
    {"type": "sequential", "from_state": "state_a", "to_state": "state_b"}
  ]
}
```

## Rules
1. Use variable references like "{{variable_name}}" for joint arrays stored in variables
2. All step IDs must be UNIQUE across the entire flow
3. All transition states must exist in the states list
4. The initial_state must exist in the states list
5. For pick & place: always add a safe midair state between pick and place
6. Use "retry" error_handling for vision/detection steps, "stop" for motion steps
7. Default joint placeholders (calibration needed at runtime):
   - midair: [152.345, -17.610, 87.397, 50.315, 2.243, 10.032]
   - pick:   [152.345, -32.610, 72.397, 50.315, 2.243, 10.032]
   - place:  [80.000,  -25.000, 80.000, 45.000, 2.000, 10.000]
8. Generate a unique id using lowercase letters, digits and underscores only

Return ONLY the raw JSON object. No markdown fences, no explanation, no extra text."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from flow import FlowSchema, FlowStatusResponse

router = APIRouter(prefix="/api/flows", tags=["flows"])

# FlowManager will be injected via app state
_flow_manager = None


def set_flow_manager(manager) -> None:
    """Set the flow manager instance (called during app initialization)."""
    global _flow_manager
    _flow_manager = manager


def get_manager():
    """Get the flow manager, raising if not initialized."""
    if _flow_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Flow manager not initialized",
        )
    return _flow_manager


# --- Request/Response Models ---


class FlowListResponse(BaseModel):
    """Response for listing flows."""

    flows: list[str]


class FlowCreateResponse(BaseModel):
    """Response for creating/updating a flow."""

    success: bool
    message: str
    flow_id: Optional[str] = None


class FlowStartResponse(BaseModel):
    """Response for starting a flow."""

    success: bool
    message: str


class FlowAbortResponse(BaseModel):
    """Response for aborting a flow."""

    success: bool
    message: str


class FlowGenerateRequest(BaseModel):
    """Request for generating a flow from prompt."""

    prompt: str


class FlowStep(BaseModel):
    """Step within a state (aligned with backend naming)."""

    id: str
    skill: str
    executor: str
    params: Optional[dict[str, Any]] = None


class FlowNode(BaseModel):
    """Node in the frontend flow format (aligned with backend naming)."""

    id: str
    type: str  # "state", "start", "end"
    label: str
    steps: Optional[list[FlowStep]] = None  # For state nodes
    position: dict[str, float]
    style: Optional[dict[str, Any]] = None  # For sizing


class FlowEdge(BaseModel):
    """Edge in the frontend flow format."""

    id: str
    source: str
    target: str
    type: str = "transitionEdge"
    data: Optional[dict[str, Any]] = None


class FlowGenerateResponse(BaseModel):
    """Response with frontend-compatible flow format."""

    id: str
    name: str
    loop: bool = False
    nodes: list[FlowNode]
    edges: list[FlowEdge]


def convert_backend_to_frontend(flow: FlowSchema) -> FlowGenerateResponse:
    """
    Convert backend flow format to frontend node/edge format.

    Backend: states with steps, transitions between states
    Frontend: start node, state nodes (containing steps), end node, edges

    Uses actual transitions from the flow definition.
    """
    nodes: list[FlowNode] = []
    edges: list[FlowEdge] = []

    # Positions are set to (0,0) — the frontend's layoutFlow() computes real positions.

    # Add start node
    start_node_id = "start"
    nodes.append(FlowNode(
        id=start_node_id,
        type="start",
        label="Start",
        position={"x": 0, "y": 0},
    ))

    # Convert each state to a node with steps inside
    for state in flow.states:
        node_id = state.name

        steps = [
            FlowStep(
                id=step.id,
                skill=step.skill,
                executor=step.executor,
                params=step.params,
            )
            for step in state.steps
        ]

        nodes.append(FlowNode(
            id=node_id,
            type="state",
            label=state.name,
            steps=steps,
            position={"x": 0, "y": 0},
        ))

    # Add end node
    end_node_id = "end"
    nodes.append(FlowNode(
        id=end_node_id,
        type="end",
        label="End",
        position={"x": 0, "y": 0},
    ))

    # Edge from start to initial state
    edges.append(FlowEdge(
        id=f"e_{start_node_id}_{flow.initial_state}",
        source=start_node_id,
        target=flow.initial_state,
    ))

    # Convert actual transitions to edges
    for i, t in enumerate(flow.transitions):
        edge_data: dict[str, Any] = {"transitionType": t.type}
        if t.condition:
            edge_data["condition"] = t.condition

        edges.append(FlowEdge(
            id=f"e_{t.from_state}_{t.to_state}_{i}",
            source=t.from_state,
            target=t.to_state,
            data=edge_data,
        ))

    # Find terminal states (no outgoing transitions)
    states_with_outgoing = {t.from_state for t in flow.transitions}
    terminal_states = [s.name for s in flow.states if s.name not in states_with_outgoing]

    # Add loop-back and/or end edges for terminal states
    for state_name in terminal_states:
        if flow.loop:
            edges.append(FlowEdge(
                id=f"e_loop_{state_name}_{flow.initial_state}",
                source=state_name,
                target=flow.initial_state,
                data={"isLoop": True},
            ))
        edges.append(FlowEdge(
            id=f"e_{state_name}_{end_node_id}",
            source=state_name,
            target=end_node_id,
        ))

    return FlowGenerateResponse(
        id=flow.id,
        name=flow.name,
        loop=flow.loop,
        nodes=nodes,
        edges=edges,
    )


# --- Endpoints ---


@router.get("", response_model=FlowListResponse)
async def list_flows():
    """List all available flows."""
    manager = get_manager()
    return FlowListResponse(flows=manager.list_flows())


@router.get("/status", response_model=FlowStatusResponse)
async def get_status():
    """Get current execution status."""
    manager = get_manager()
    return manager.get_status()


@router.get("/{flow_id}", response_model=FlowSchema)
async def get_flow(flow_id: str):
    """Get a flow definition by ID."""
    manager = get_manager()
    flow = manager.get_flow(flow_id)
    if flow is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Flow '{flow_id}' not found",
        )
    return flow


@router.post("", response_model=FlowCreateResponse)
async def create_flow(flow: FlowSchema):
    """Create or update a flow."""
    manager = get_manager()
    success, error = manager.save_flow(flow)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error or "Failed to save flow",
        )
    return FlowCreateResponse(
        success=True,
        message=f"Flow '{flow.id}' saved",
        flow_id=flow.id,
    )


@router.delete("/{flow_id}")
async def delete_flow(flow_id: str):
    """Delete a flow."""
    manager = get_manager()
    if not manager.delete_flow(flow_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Flow '{flow_id}' not found",
        )
    return {"success": True, "message": f"Flow '{flow_id}' deleted"}


@router.post("/{flow_id}/start", response_model=FlowStartResponse)
async def start_flow(flow_id: str):
    """Start executing a flow."""
    manager = get_manager()
    success, message = await manager.start_flow(flow_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT
            if "already running" in message.lower()
            else status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    return FlowStartResponse(success=True, message=message)


@router.post("/abort", response_model=FlowAbortResponse)
async def abort_flow():
    """Abort the currently running flow."""
    manager = get_manager()
    success, message = await manager.abort_flow()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    return FlowAbortResponse(success=True, message=message)


class FlowFinishResponse(BaseModel):
    """Response for finishing a flow."""

    success: bool
    message: str


class FlowPauseResponse(BaseModel):
    """Response for pausing a flow."""

    success: bool
    message: str


class FlowResumeResponse(BaseModel):
    """Response for resuming a flow."""

    success: bool
    message: str


@router.post("/finish", response_model=FlowFinishResponse)
async def finish_flow():
    """Request graceful finish — complete current loop cycle then stop."""
    manager = get_manager()
    success, message = await manager.finish_flow()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    return FlowFinishResponse(success=True, message=message)


@router.post("/pause", response_model=FlowPauseResponse)
async def pause_flow():
    """Pause the currently running flow."""
    manager = get_manager()
    success, message = await manager.pause_flow()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    return FlowPauseResponse(success=True, message=message)


@router.post("/resume", response_model=FlowResumeResponse)
async def resume_flow():
    """Resume a paused flow."""
    manager = get_manager()
    success, message = await manager.resume_flow()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    return FlowResumeResponse(success=True, message=message)


# Fallback flow when LLM generation fails.
_DEFAULT_FLOW_ID = "dobot_test_pick"


@router.post("/generate", response_model=FlowGenerateResponse)
async def generate_flow(request: FlowGenerateRequest):
    """
    Generate a robot flow from a natural language prompt using Azure OpenAI.

    Calls gpt-4o with a structured system prompt describing all available
    skills and the flow JSON schema. Falls back to the default flow if the
    LLM call or validation fails.
    """
    logger.info("generate_flow called with prompt: %r", request.prompt)
    manager = get_manager()

    azure_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    if azure_key and azure_endpoint:
        try:
            client = _get_openai_client()
            deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

            response = await client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": _FLOW_SYSTEM_PROMPT},
                    {"role": "user", "content": request.prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=4096,
                temperature=0.2,
            )

            raw = response.choices[0].message.content or ""
            logger.info("LLM raw response (first 300 chars): %s", raw[:300])

            flow_data = json.loads(raw)
            flow = FlowSchema(**flow_data)

            valid, err = flow.validate_flow()
            if not valid:
                logger.error("LLM flow validation failed: %s — raw: %s", err, raw[:500])
                raise ValueError(err)

            # Persist so the user can start it immediately
            manager.save_flow(flow)
            logger.info("LLM-generated flow saved: %s", flow.id)

            return convert_backend_to_frontend(flow)

        except Exception as exc:
            logger.warning(
                "Flow generation failed (%s: %s) — falling back to default flow",
                type(exc).__name__, exc,
            )

    else:
        logger.warning(
            "AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT not set — "
            "falling back to default flow '%s'",
            _DEFAULT_FLOW_ID,
        )

    # Fallback: return the hardcoded default flow
    flow = manager.get_flow(_DEFAULT_FLOW_ID)
    if flow is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Default flow '{_DEFAULT_FLOW_ID}' not found",
        )
    return convert_backend_to_frontend(flow)
