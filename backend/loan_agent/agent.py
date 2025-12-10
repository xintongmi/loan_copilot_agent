import json
import asyncio
from google.adk.agents.llm_agent import Agent
from google.adk.runners import InMemoryRunner
from loan_agent.initialization_agent import (
    create_loan_initialization_agent,
    LoanInitializationOutput,
)
from loan_agent.shared import session_service
from google.adk.tools import FunctionTool


def create_dynamic_main_agent(resource_id: str):
    """
    Creates the 'Main Agent' graph injected with the specific Cache ID.
    This runs ONLY after the ID is found.
    """
    pre_qual_agent = Agent(
        model="gemini-2.5-flash",
        name="pre_qualification_sub_agent",
        instruction=(
            f"Operating on Guideline Cache: {resource_id}. "
            "Calculate DTI and Reserves using the guideline formulas. "
            "Validate document dates and extract numbers."
        ),
    )

    underwriting_agent = Agent(
        model="gemini-2.5-flash",
        name="underwriting_sub_agent",
        instruction=(
            f"Operating on Guideline Cache: {resource_id}. "
            "Verify bank statements and flag abnormal cashflow."
        )
    )

    return Agent(
        model="gemini-2.5-flash",
        name="main_business_agent",
        description="Manager for the loan process.",
        instruction=(
            f"You are the Loan Manager using cache {resource_id}. "
            "Delegate tasks to pre_qualification_sub_agent or underwriting_sub_agent."
        ),
        sub_agents=[pre_qual_agent, underwriting_agent]
    )


@FunctionTool
def run_underwriting_workflow(resource_id: str, user_query: str) -> str:
    """
    Spins up the dynamic underwriting agents for a specific resource ID.
    """
    print(f"DEBUG: Starting dynamic workflow for Cache ID: {resource_id}")

    # 1. Build the agent graph dynamically
    dynamic_agent = create_dynamic_main_agent(resource_id)

    # 2. Create a nested runner
    runner = InMemoryRunner(agent=dynamic_agent, session_service=session_service)

    # 3. Execute the inner agent
    try:
        # Using await since we are in an async context
        response = asyncio.run(
            runner.run(user_query, session_id=f"nested_{resource_id}")
        )
        return response.text
    except Exception as e:
        return f"An error occurred during underwriting: {e}"


async def get_agent_response(query: str, session_id: str):
    """
    Invokes the root agent with a user query.
    """
    loan_initialization_agent = create_loan_initialization_agent()
    root_agent = Agent(
        model="gemini-2.5-flash",
        name="root_agent",
        description="Orchestrator for the Underwriting Copilot.",
        instruction=(
            "You are the System Orchestrator. Follow this strict sequence:\n"
            "1. Delegate to `loan_initialization_agent` to identify the program.\n"
            "2. READ the output. If it contains a `resource_id`:\n"
            "3. CALL the tool `run_underwriting_workflow`.\n"
            "   - Argument `resource_id`: The ID returned by the initialization agent.\n"
            "   - Argument `user_query`: The original user question.\n"
            "4. Output the result from the tool as your final answer."
        ),
        sub_agents=[loan_initialization_agent],
        tools=[run_underwriting_workflow]
    )

    runner = InMemoryRunner(agent=root_agent, session_service=session_service)
    response = await runner.run(query, session_id=session_id)
    # The final output of the root agent is the response
    return response.text
