"""
Ruflo Swarm Orchestration Blocks for AutoGPT Platform

Implements multi-agent swarm coordination inspired by the Ruflo v3.5 framework:
https://github.com/ruvnet/ruflo

Features:
- Specialized agent roles: architect, coder, reviewer, tester, security-auditor
- Swarm topologies: hierarchical, mesh, ring, star
- Consensus mechanisms: majority-vote, synthesis, first-valid
- Task complexity routing: simple, medium, complex
"""

import json
import uuid
from collections import Counter
from enum import Enum

from backend.blocks._base import (
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.llm import (
    DEFAULT_LLM_MODEL,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    AIBlockBase,
    AICredentials,
    AICredentialsField,
    LlmModel,
    llm_call,
)
from backend.data.model import APIKeyCredentials
from backend.data.model import NodeExecutionStats, SchemaField


class SwarmTopology(str, Enum):
    HIERARCHICAL = "hierarchical"
    MESH = "mesh"
    RING = "ring"
    STAR = "star"


class ConsensusStrategy(str, Enum):
    SYNTHESIS = "synthesis"
    MAJORITY_VOTE = "majority_vote"
    FIRST_VALID = "first_valid"


class AgentRole(str, Enum):
    ARCHITECT = "architect"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    SECURITY_AUDITOR = "security_auditor"


AGENT_SYSTEM_PROMPTS: dict[AgentRole, str] = {
    AgentRole.ARCHITECT: (
        "You are a software architect agent. Analyze the given task and provide "
        "high-level design decisions, architectural patterns, component structure, "
        "and interface definitions. Focus on scalability, maintainability, and "
        "separation of concerns. Be concise and specific."
    ),
    AgentRole.CODER: (
        "You are a senior software engineer agent. Given the task, produce clean, "
        "idiomatic, production-ready code with proper error handling. Follow best "
        "practices for the relevant language/framework. Be concise and precise."
    ),
    AgentRole.REVIEWER: (
        "You are a code review agent. Evaluate the task from a quality perspective: "
        "identify potential bugs, code smells, performance issues, and suggest "
        "improvements. Focus on correctness, readability, and maintainability."
    ),
    AgentRole.TESTER: (
        "You are a QA engineering agent. Design test cases and testing strategies "
        "for the given task. Include unit tests, edge cases, integration scenarios, "
        "and validation criteria. Think about what could go wrong."
    ),
    AgentRole.SECURITY_AUDITOR: (
        "You are a security engineer agent. Analyze the task for security "
        "vulnerabilities, OWASP risks, injection attacks, authentication issues, "
        "and data exposure. Provide specific mitigations and secure coding guidance."
    ),
}

TOPOLOGY_ROLE_SETS: dict[SwarmTopology, list[AgentRole]] = {
    SwarmTopology.HIERARCHICAL: [
        AgentRole.ARCHITECT,
        AgentRole.CODER,
        AgentRole.REVIEWER,
        AgentRole.TESTER,
    ],
    SwarmTopology.MESH: [
        AgentRole.ARCHITECT,
        AgentRole.CODER,
        AgentRole.REVIEWER,
        AgentRole.TESTER,
        AgentRole.SECURITY_AUDITOR,
    ],
    SwarmTopology.RING: [
        AgentRole.CODER,
        AgentRole.REVIEWER,
        AgentRole.TESTER,
    ],
    SwarmTopology.STAR: [
        AgentRole.ARCHITECT,
        AgentRole.CODER,
        AgentRole.SECURITY_AUDITOR,
    ],
}


def _build_synthesis_prompt(task: str, agent_outputs: dict[str, str]) -> list[dict]:
    agent_section = "\n\n".join(
        f"## {role.upper()} AGENT\n{output}" for role, output in agent_outputs.items()
    )
    return [
        {
            "role": "system",
            "content": (
                "You are the queen coordinator of a multi-agent swarm. Your job is to "
                "synthesize insights from all specialized agents into a single coherent, "
                "actionable response. Preserve the most valuable contributions from each "
                "agent while resolving any conflicts. Be comprehensive but concise."
            ),
        },
        {
            "role": "user",
            "content": (
                f"ORIGINAL TASK:\n{task}\n\n"
                f"AGENT OUTPUTS:\n{agent_section}\n\n"
                "Synthesize these into a unified, high-quality response:"
            ),
        },
    ]


class RufloSwarmBlock(AIBlockBase):
    """
    Coordinates a swarm of specialized AI agents to collaboratively solve a task.

    Inspired by Ruflo v3.5 multi-agent orchestration:
    - Spawns agents with specialized roles (architect, coder, reviewer, tester, security-auditor)
    - Supports hierarchical, mesh, ring, and star topologies
    - Aggregates outputs via synthesis, majority-vote, or first-valid consensus
    """

    class Input(BlockSchemaInput):
        task: str = SchemaField(
            description="The task or question for the agent swarm to solve.",
            placeholder="Describe what you want the swarm to accomplish...",
        )
        topology: SwarmTopology = SchemaField(
            default=SwarmTopology.HIERARCHICAL,
            description=(
                "Swarm topology: hierarchical (architect-led, 4 agents), "
                "mesh (all 5 roles), ring (3 roles, sequential), "
                "star (3 roles, architect-centered)."
            ),
        )
        consensus: ConsensusStrategy = SchemaField(
            default=ConsensusStrategy.SYNTHESIS,
            description=(
                "How agents reach a final answer: synthesis (queen aggregates all), "
                "majority_vote (most common answer wins), "
                "first_valid (first non-empty response used)."
            ),
        )
        custom_roles: list[AgentRole] = SchemaField(
            default=[],
            description=(
                "Override the topology's default roles with a custom set. "
                "Leave empty to use the topology defaults."
            ),
        )
        model: LlmModel = SchemaField(
            title="LLM Model",
            default=DEFAULT_LLM_MODEL,
            description="The language model each agent in the swarm will use.",
            advanced=False,
        )
        credentials: AICredentials = AICredentialsField()

    class Output(BlockSchemaOutput):
        result: str = SchemaField(
            description="Final synthesized output from the swarm."
        )
        agent_outputs: dict = SchemaField(
            description="Individual outputs keyed by agent role name."
        )
        swarm_id: str = SchemaField(
            description="Unique identifier for this swarm execution."
        )
        roles_used: list = SchemaField(
            description="List of agent roles that participated in this swarm."
        )
        error: str = SchemaField(description="Error message if the swarm fails.")

    def __init__(self):
        super().__init__(
            id="a3f9c2e1-7b4d-4e8a-9c6f-1d2e3f4a5b6c",
            input_schema=RufloSwarmBlock.Input,
            output_schema=RufloSwarmBlock.Output,
            description=(
                "Ruflo multi-agent swarm orchestration. Spawns specialized agents "
                "(architect, coder, reviewer, tester, security-auditor) that collaborate "
                "on a task using configurable topologies and consensus strategies."
            ),
            categories={BlockCategory.AI, BlockCategory.AGENT},
            test_input={
                "task": "Write a Python function to validate email addresses",
                "topology": SwarmTopology.RING,
                "consensus": ConsensusStrategy.SYNTHESIS,
                "custom_roles": [],
                "model": DEFAULT_LLM_MODEL,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("swarm_id", lambda v: isinstance(v, str) and len(v) > 0),
                ("roles_used", lambda v: isinstance(v, list) and len(v) > 0),
                ("agent_outputs", lambda v: isinstance(v, dict)),
                ("result", lambda v: isinstance(v, str) and len(v) > 0),
            ],
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        swarm_id = str(uuid.uuid4())
        roles = (
            input_data.custom_roles
            if input_data.custom_roles
            else TOPOLOGY_ROLE_SETS[input_data.topology]
        )

        yield "swarm_id", swarm_id
        yield "roles_used", [r.value for r in roles]

        agent_outputs: dict[str, str] = {}
        for role in roles:
            prompt = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPTS[role]},
                {"role": "user", "content": input_data.task},
            ]
            try:
                response = await llm_call(
                    credentials=credentials,
                    llm_model=input_data.model,
                    prompt=prompt,
                    max_tokens=None,
                )
                agent_outputs[role.value] = response.response
                self.merge_stats(
                    NodeExecutionStats(
                        input_token_count=response.prompt_tokens,
                        output_token_count=response.completion_tokens,
                    )
                )
            except Exception as e:
                agent_outputs[role.value] = f"[ERROR: {e}]"

        yield "agent_outputs", agent_outputs

        if not agent_outputs:
            yield "result", ""
            yield "error", "All agents failed to produce output"
            return

        result = await self._apply_consensus(
            input_data.task,
            agent_outputs,
            input_data.consensus,
            credentials,
            input_data.model,
        )
        yield "result", result

    async def _apply_consensus(
        self,
        task: str,
        agent_outputs: dict[str, str],
        strategy: ConsensusStrategy,
        credentials: APIKeyCredentials,
        model: LlmModel,
    ) -> str:
        valid_outputs = {
            k: v for k, v in agent_outputs.items() if not v.startswith("[ERROR")
        }

        if not valid_outputs:
            return next(iter(agent_outputs.values()), "")

        if strategy == ConsensusStrategy.FIRST_VALID:
            return next(iter(valid_outputs.values()))

        if strategy == ConsensusStrategy.MAJORITY_VOTE:
            counts = Counter(valid_outputs.values())
            return counts.most_common(1)[0][0]

        # SYNTHESIS: queen coordinator aggregates all agent outputs
        synthesis_prompt = _build_synthesis_prompt(task, valid_outputs)
        try:
            response = await llm_call(
                credentials=credentials,
                llm_model=model,
                prompt=synthesis_prompt,
                max_tokens=None,
            )
            self.merge_stats(
                NodeExecutionStats(
                    input_token_count=response.prompt_tokens,
                    output_token_count=response.completion_tokens,
                )
            )
            return response.response
        except Exception:
            return "\n\n---\n\n".join(
                f"**{role}**: {output}" for role, output in valid_outputs.items()
            )


class RufloTaskRouterBlock(AIBlockBase):
    """
    Routes a task to the appropriate complexity tier, following Ruflo's 3-tier model:
    - Simple: direct answer (no swarm needed)
    - Medium: single specialist agent
    - Complex: full swarm orchestration recommendation
    """

    class Input(BlockSchemaInput):
        task: str = SchemaField(
            description="The task to classify and route.",
            placeholder="Describe your task...",
        )
        model: LlmModel = SchemaField(
            title="LLM Model",
            default=DEFAULT_LLM_MODEL,
            description="The language model used for routing classification.",
            advanced=True,
        )
        credentials: AICredentials = AICredentialsField()

    class Output(BlockSchemaOutput):
        complexity: str = SchemaField(
            description="Detected complexity tier: simple, medium, or complex."
        )
        recommended_topology: str = SchemaField(
            description="Recommended Ruflo swarm topology for the task."
        )
        recommended_roles: list = SchemaField(
            description="Recommended agent roles for the detected complexity."
        )
        routing_reason: str = SchemaField(
            description="Explanation of why this complexity tier was chosen."
        )
        error: str = SchemaField(description="Error message if routing fails.")

    def __init__(self):
        super().__init__(
            id="b7e4d1f2-8c5a-4f9b-0e7d-2a3b4c5d6e7f",
            input_schema=RufloTaskRouterBlock.Input,
            output_schema=RufloTaskRouterBlock.Output,
            description=(
                "Ruflo intelligent task router. Classifies task complexity "
                "(simple/medium/complex) and recommends the optimal swarm topology "
                "and agent roles to use with the RufloSwarmBlock."
            ),
            categories={BlockCategory.AI, BlockCategory.LOGIC},
            test_input={
                "task": "What is 2 + 2?",
                "model": DEFAULT_LLM_MODEL,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("complexity", lambda v: v in ("simple", "medium", "complex")),
                ("recommended_topology", lambda v: isinstance(v, str)),
                ("recommended_roles", lambda v: isinstance(v, list)),
                ("routing_reason", lambda v: isinstance(v, str)),
            ],
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        routing_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a task routing agent for a multi-agent AI system. "
                    "Classify the given task into one of three complexity tiers:\n"
                    "- simple: factual lookups, arithmetic, single-step transformations\n"
                    "- medium: coding tasks, explanations, single-domain analysis\n"
                    "- complex: multi-step architecture, security audits, cross-domain problems\n\n"
                    "Respond with valid JSON only:\n"
                    '{"complexity": "simple|medium|complex", '
                    '"recommended_topology": "hierarchical|mesh|ring|star", '
                    '"recommended_roles": ["role1", "role2"], '
                    '"routing_reason": "brief explanation"}'
                ),
            },
            {
                "role": "user",
                "content": f"Classify this task:\n\n{input_data.task}",
            },
        ]

        try:
            response = await llm_call(
                credentials=credentials,
                llm_model=input_data.model,
                prompt=routing_prompt,
                max_tokens=None,
                force_json_output=True,
            )
            self.merge_stats(
                NodeExecutionStats(
                    input_token_count=response.prompt_tokens,
                    output_token_count=response.completion_tokens,
                )
            )
            routing = json.loads(response.response)
            yield "complexity", routing.get("complexity", "medium")
            yield "recommended_topology", routing.get(
                "recommended_topology", "hierarchical"
            )
            yield "recommended_roles", routing.get(
                "recommended_roles", ["coder", "reviewer"]
            )
            yield "routing_reason", routing.get("routing_reason", "")
        except Exception as e:
            yield "complexity", "medium"
            yield "recommended_topology", "hierarchical"
            yield "recommended_roles", ["architect", "coder", "reviewer"]
            yield "routing_reason", "Defaulted to medium/hierarchical due to routing error"
            yield "error", str(e)
