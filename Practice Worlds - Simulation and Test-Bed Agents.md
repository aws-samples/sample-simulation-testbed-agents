# Practice Worlds: Simulation and Test-Bed Agents

*Let agents reason, act, and learn in a safe virtual environment before they ever touch production*

---

This is the ninth post in our series on [AWS Prescriptive Guidance for Agentic AI Patterns](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/). Each post focuses on the concepts and patterns behind a single agent type, paired with a [hands-on sample on GitHub](README.md).

## Introduction

A function is tested with inputs and assertions. An agent that takes actions can't be tested that way, its behavior emerges from a loop between reasoning, action, and feedback, and the loop is what you actually want to evaluate. Whether an agent that restarts services, edits code, or moves money has a strategy that holds together is invisible from any single input/output pair.

Simulation and sandboxing, just like with human developers, are where we can run that loop safely. The real question we want to evaluate is "does the agent's strategy hold up when the environment pushes back?" i.e. Does it recover when its first move fails? Notice when it's stuck? Adapt when the situation changes mid-task? Those are the failure modes that bite in production, and they only surface inside a loop.

By the end of this post, you'll understand:
- Why a simulated environment is the right place to develop and test acting agents
- The closed loop that distinguishes simulation agents from one-shot agents
- The common patterns for agentic simulations such as sandboxes, scenario simulators, and learning test-beds
- When to simulate, and what simulation can't tell you

---

## The Road to Simulation Agents: A Brief History

Learning by simulation predates LLMs by decades; agents are the newest inhabitants of an old idea.

### Reinforcement learning in simulated worlds

Simulation became the engine of modern learning agents. Reinforcement learning needs enormous numbers of trials, far more than reality could supply safely. Game and robotics environments provided those trials cheaply, and famously produced agents that mastered games and control tasks through millions of simulated episodes. The closed loop of *act → observe → reward → adjust* is the heart of this era.

### LLM agents in test-beds

Now the agent in the loop is often an LLM-based planner rather than a pure RL policy, and the "environment" might be a CLI sandbox, a synthetic data stream, or a simulated business process. The same logic applies: run the agent in a safe replica, score its behavior, and iterate before production. AWS has written explicitly about this — see [LLMs in generative agent-based simulation](https://aws.amazon.com/blogs/hpc/llms-the-new-frontier-in-generative-agent-based-simulation/) and [evaluation tooling for testing agents systematically](https://aws.amazon.com/blogs/machine-learning/evaluating-ai-agents-real-world-lessons-from-building-agentic-systems-at-amazon/).

---

## The Closed Loop

<img src="images/simulation-testbed-agents.png" width="600" alt="Diagram of a simulation agent: the agent perceives a simulated environment's state, retrieves its goal and memory, reasons and plans with an LLM, executes simulated actions, and learns from the outcomes in a closed loop." />

What sets a simulation agent apart from the earlier patterns is the **closed loop**. A basic reasoning agent answers and stops. A simulation agent acts, observes how the environment responded, and feeds that observation back into its next decision.

That loop only works because the environment *pushes back*. A practice world isn't just a place to run commands; it's a model that produces realistic feedback. The quality of that feedback determines how much the agent can learn. A rich environment teaches; a hollow one just records.

---

## Common Patterns

Simulation agents take a few recognizable forms.

| Pattern | The environment | What you learn |
|-------|-----------------|----------------|
| **Sandbox** | A confined space to act in (e.g., a temp file system) | How the agent behaves when it can act freely but safely |
| **Scenario simulator** | A modeled situation with feedback (e.g., a broken server) | How the agent diagnoses and solves in a closed loop |
| **Learning test-bed** | A repeatable game with a reward signal | How a strategy improves over repeated trials |

A **sandbox** is a bounded environment where an agent can experiment without escaping into anything real. A **scenario simulator** models a situation that pushes back: drop the agent into a simulated incident with randomized faults and watch it perceive, reason, act, and observe across the full loop. A **learning test-bed** is the purest form: a repeatable task with a clear reward, where you can watch a strategy emerge from self-reflection across runs rather than from explicit instruction. The first tests *behavior*, the second tests *problem-solving*, the third tests *learning*.

---

## When to Use Simulation Agents

Simulate when actions have real consequences, trials are needed, or safety must be proven first.

| Use Case | Example |
|----------|---------|
| **Reinforcement learning** | Training policies for robotics, drones, or games |
| **Autonomous training** | Vehicles learning on virtual roads |
| **DevOps test-beds** | Simulated CLIs and systems for safe automation testing |
| **Emergent behavior** | Social simulations and multi-agent experiments |
| **Safety validation** | Proving decision logic before it reaches production |

### What Simulation Can't Tell You

A simulation is only as good as its model of reality and no model is complete. An agent that performs flawlessly in the practice world can still fail on the messy edge cases the simulation didn't capture (this is the "sim-to-real gap" in robotics). Treat strong simulated results as necessary, not sufficient. Simulation simply tells you an agent is *ready to try* for real.

---

## What's Next

You now understand why a practice world is the right place to develop acting agents, the closed loop that defines them, and the common patterns simulation takes. The natural next step is to see them run. The **[companion sample](README.md)** builds a CLI sandbox, a closed-loop scenario simulator, and a self-improving strategy learner.

Simulation lets agents act safely; the next question is how we *watch* agents once they're live. In the [next post](https://github.com/aws-samples/sample-observer-monitoring-agents/blob/main/Watching%20the%20Watched%20-%20Observer%20and%20Monitoring%20Agents.md), we'll explore observer and monitoring agents and the patterns for keeping an eye on agents in production.

---

## Resources

- [Companion sample: Simulation and Test-Bed Agents](README.md)
- [AWS Prescriptive Guidance - Simulation and test-bed agents](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/simulation-and-test-bed-agents.html)
- [LLMs: the new frontier in generative agent-based simulation](https://aws.amazon.com/blogs/hpc/llms-the-new-frontier-in-generative-agent-based-simulation/)
- [Strands Agents Documentation](https://strandsagents.com/)
- [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)

---

**Tim Sitze** is a Solutions Architect at Amazon Web Services, where he works with cybersecurity ISVs to design and scale their products on AWS. He specializes in security, AI/ML, IoT and data platform architectures, and has partnered on workloads spanning identity threat intelligence, agentic AI, and cloud-native security operations. Tim is based in the Washington, D.C. area.  
