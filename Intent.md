Mind_Shift_AI — Base Project Intent
Core Purpose

Mind_Shift_AI is a longitudinal AI reflection and decision-support system.

The project should not merely summarize a user's journal or show emotional statistics. Its deeper purpose is to use the user's own history, memories, emotional patterns, behavioral patterns, goals, and current situation to help them:

understand what is happening,
identify the real decision/problem,
compare realistic alternatives,
recognize when temporary emotions or recurring behavioral patterns may be influencing the decision,
receive a reasoned recommendation when enough evidence exists,
and convert that recommendation into concrete next actions.

The intended progression is:

Journal / conversation
      ↓
Understand the user
      ↓
Retrieve relevant personal history
      ↓
Identify patterns and evidence
      ↓
Understand the current decision/problem
      ↓
Consider multiple possible actions
      ↓
Evaluate trade-offs
      ↓
Give a reasoned recommendation
      ↓
Produce an actionable next step
Existing Project Philosophy

Mind_Shift already contains useful infrastructure such as:

journal processing,
safety/crisis checks,
emotion detection,
NLP extraction,
FAISS semantic retrieval,
SQLite structured memory,
historical memory retrieval,
behavioral/emotional analytics,
user profile information,
deterministic analysis,
LLM-based response generation.

Do not rebuild these systems unnecessarily.

New functionality should reuse and extend the existing architecture wherever possible.

The current project is primarily:

personal data
→ analysis
→ retrieval
→ insights
→ reflection

The target extension is:

personal data
→ analysis
→ retrieval
→ decision understanding
→ reasoning over alternatives
→ guidance
→ action
Primary Development Goal

Add a Guidance / Decision-Support Layer on top of the existing system.

The system should be capable of handling prompts such as:

"I got criticised at work and now I want to quit."

"I have two job opportunities and cannot decide."

"I keep avoiding interviews because I feel underprepared."

"Should I confront this person or leave the situation alone?"

Instead of replying only with:

"You have experienced similar anxiety several times."

the system should eventually be able to respond more like:

"Based on your previous behavior and the available evidence,
I don't think resigning immediately is the best option.

Your confidence tends to fall sharply after criticism,
but previous entries show that this reaction usually stabilizes.

A better next step is to collect specific feedback,
separate the criticism from the larger career decision,
and reconsider after the emotional spike settles."

The recommendation should be:

evidence-aware,
explainable,
grounded in available user history,
uncertainty-aware,
and followed by concrete actions.
Important Architectural Principle

Do not make the LLM the entire reasoning system.

Prefer a hybrid architecture:

LLM
    → interpretation
    → intent understanding
    → option generation
    → planning
    → natural-language explanation

Deterministic/application logic
    → safety
    → retrieval
    → memory
    → scoring constraints
    → validation
    → permissions
    → evaluation

The model may reason, but important system guarantees should not rely entirely on an unconstrained LLM response.

Desired Guidance Flow

A useful target flow is:

User input
    ↓
Existing safety system
    ↓
Determine whether this is:
    - normal journaling/reflection
    - information request
    - decision/guidance request
    ↓
For guidance requests:
    ↓
Extract structured decision state
    ↓
Retrieve relevant personal context
    ↓
Generate realistic options
    ↓
Evaluate trade-offs
    ↓
Check uncertainty / missing information
    ↓
Produce recommendation
    ↓
Generate concrete action plan

Normal journaling should continue using the existing system.

Do not force every request through the guidance engine.

Structured Decision State

Internally, decision-oriented requests should ideally be represented explicitly rather than remaining only as free-form text.

Example conceptual state:

problem
desired_outcome
available_options
fears
constraints
relevant_goals
current_emotional_state
missing_information
uncertainty

Exact schemas should be determined after inspecting the current repository.

Personal Context Should Be Used as Evidence

Relevant historical data may include:

similar previous situations,
previous decisions,
consequences of previous decisions,
recurring emotional patterns,
recurring avoidance behavior,
goals,
preferences,
known relationships/entities,
previous successful coping/actions.

However:

Do not dump the user's entire history into the prompt.

The system should retrieve and construct only the context relevant to the current problem.

Agentic AI Intent

An agent may later be introduced, but the purpose of the agent is not to make the project look "agentic."

The agent should exist only if dynamic tool selection improves the problem.

Useful tools may wrap existing project capabilities such as:

search_similar_memories()
get_emotional_patterns()
get_user_profile()
get_goal_history()
get_previous_decision_outcomes()

The important behavior is:

understand problem
    ↓
decide which information is needed
    ↓
call appropriate tool
    ↓
observe result
    ↓
gather additional evidence if necessary
    ↓
make recommendation

The agent should have bounded execution and should not run uncontrolled loops.

Safety Boundary

Mind_Shift should not become an autonomous medical or psychological authority.

It may provide strong guidance for ordinary life decisions such as:

career decisions,
studying,
procrastination,
habits,
difficult conversations,
personal planning,
ordinary relationship conflicts,
choosing between multiple reasonable alternatives.

For areas involving:

self-harm,
suicide,
abuse,
immediate danger,
medical treatment,
medication,
severe mental-health symptoms,
other high-risk irreversible decisions,

the normal recommendation pipeline must be restricted or redirected through the existing safety path.

Existing deterministic safety rules should have priority over LLM judgment.

Product Positioning

Do not design or describe the product as:

"AI therapist that knows what is right for you."

The intended positioning is closer to:

"A personal AI reflection and decision-support system with longitudinal memory."

It can make recommendations when appropriate, but should expose uncertainty and the evidence behind its recommendation.

Two-Day Implementation Constraint

The immediate work has a strict time constraint.

Therefore:

Prefer a small complete vertical slice over a broad incomplete architecture.

Current implementation priorities are roughly:

1. Understand existing integration points
2. Structured decision representation
3. Decision-focused context retrieval
4. Simple guidance engine
5. Small tool-using agent
6. Minimal evaluation
7. Minimal tracing
8. LangGraph only if genuinely useful and time permits
Explicit Non-Goals for This Iteration

Do not spend the two-day implementation window on:

MCP,
fine-tuning,
RL/RLHF,
replacing FAISS,
replacing SQLite,
new vector databases,
full Memory V2 redesign,
multi-agent systems,
complicated LangGraph architecture,
OpenTelemetry infrastructure,
voice/multimodal features,
large frontend redesign,
broad refactoring unrelated to the guidance feature.

These can be future improvements.

Engineering Rules

When planning implementation:

Inspect the actual repository before designing new abstractions.
Reuse current services whenever possible.
Preserve existing working behavior.
Keep existing tests passing.
Do not introduce unnecessary dependencies.
Avoid speculative abstractions or over-engineering.
Implement one vertical feature at a time.
Add tests for new behavior.
Prefer explicit, understandable code over framework-heavy solutions.
Do not silently change the fundamental intent described in this document.
Definition of Success

A successful first implementation should allow at least one realistic end-to-end case such as:

User:
"I received negative feedback today and I'm thinking of resigning."

System:
1. identifies that this is a decision request,
2. extracts the actual decision,
3. retrieves relevant historical evidence,
4. identifies possible alternatives,
5. weighs relevant trade-offs,
6. produces a clear recommendation,
7. explains why,
8. states uncertainty or missing information,
9. provides concrete next actions,
10. respects safety boundaries.

If this single flow works cleanly using the current Mind_Shift architecture, the iteration is successful.

Instruction to Coding Agent

Treat everything above as the product intent, not as a mandatory implementation design.

Inspect the repository first.

Then create your own technically sound implementation plan based on the real codebase.

You may change file structure, abstractions, schemas, or implementation details if the repository suggests a better approach.

Do not change the core product intent, safety boundary, two-day scope, or non-goals without explicitly identifying the reason.