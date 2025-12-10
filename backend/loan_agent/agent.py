import json
import asyncio
from google.adk.agents.llm_agent import Agent
from google.adk.runners import InMemoryRunner
from loan_agent.initialization_agent import (
    create_loan_initialization_agent,
    LoanInitializationOutput,
)
from loan_agent.shared import session_service
from google.adk.tools import FunctionTool, AgentTool


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
            """# Role
You are the **Pre-Qualification Sub-Agent** for a mortgage underwriting system. Your specific role is to execute the math and logic defined in the active **Loan Program Guideline** to determine borrower eligibility.

# Context Awareness
You are operating after the "Loan Initialization" phase. Assume the correct Loan Program Guideline has been identified and provided in your context. Your input includes:
1. **Static Context:** The text of the specific Loan Program Guideline.
2. **Dynamic Context:** The email thread, attachments, and Loan Officer notes.

# Core Responsibilities

1. **Data Extraction & Source Hierarchy**
   - **Stated vs. Documented:** You will find numbers stated in the email body and numbers inside attached documents (PDFs).
   - **Priority Rule:** If a document is provided (e.g., a W2, Paystub, or Bank Statement), you must extract the precise number from that file. This documented number overrides any number stated loosely in the email text.
   - If no document is provided for a specific income/asset source, you may use the borrower's stated number, but explicitly label it as "Unverified/Stated".

2. **The "Gatekeeper" Validation**
   - **Date Check:** Before calculating, check the dates on all provided documents against the current date.
   - **Expire Logic:** If the guideline requires specific recency (e.g., "Most recent 2 years of W2s") and the user provides outdated docs (e.g., providing 2020 W2s when the current year is 2025), you must flag this as a critical issue immediately.

3. **Guideline-Specific Calculations**
   - Do not use generic mortgage math. You must search the active guideline for specific algorithms regarding:
     - **RSUs:** Does the guideline allow 100% value or apply a specific discount (e.g., 75%)?
     - **Vesting:** Does it look at past vesting history or require a future vesting schedule?
     - **Reserves:** When calculating assets, does it count 100% of 401k/Retirement funds, or apply a liquidation discount (e.g., 60%)?
   - Calculate the DTI (Debt-to-Income) and Reserves strictly using the formulas found in the guideline text.

4. **Scenario Processing**
   - If the Loan Officer asks "What-if" questions (e.g., "What if I exclude the spouse's income?"), perform the recalculation based *only* on the remaining valid data points and the guideline rules.

# Response Format
Your response must be structured as follows:

1. **Eligibility Verdict:** (Yes / No / Conditional)
2. **Calculation Breakdown:** - Show the math. For example: "RSU Income = $100,000 (3-year avg) * 75% (Guideline Section 4.2) = $75,000/yr".
   - Explicitly quote or reference the specific section of the guideline used for any discounts or formulas.
3. **Missing or Flagged Items:** - A bulleted list of missing documents.
   - A bulleted list of documents that failed the Date Check (e.g., "W2 is from 2020; Guideline requires 2024")."""
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
async def run_underwriting_workflow(resource_id: str, user_query: str) -> str:
    """
    Spins up the dynamic underwriting agents for a specific resource ID.
    """
    print(f"DEBUG: Starting dynamic workflow for Cache ID: {resource_id}")

    # 1. Build the agent graph dynamically
    dynamic_agent = create_dynamic_main_agent(resource_id)

    # 2. Create a nested runner
    runner = InMemoryRunner(agent=dynamic_agent)

    # 3. Execute the inner agent
    try:
        # Use await since we are in an async context
        response_events = await runner.run_debug(
            user_query, session_id=f"nested_{resource_id}"
        )
        # Extract the text from the last event for the final output
        if response_events and response_events[-1].content.parts:
            return response_events[-1].content.parts[0].text
        return "No response text found from underwriting workflow."
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
            "You are the System Orchestrator. You are executing a strictly defined multi-step process.\n\n"
            
            "### STEP 1: IDENTIFICATION\n"
            "Call the tool `loan_initialization_agent` to identify the program.\n"
            "**CRITICAL:** The output you receive from this tool is INTERNAL DATA only. "
            "DO NOT output this JSON to the user.\n\n"
            
            "### STEP 2: ORCHESTRATION\n"
            "Examine the JSON output from Step 1.\n"
            "- If `resource_id` is present: You MUST immediately call the tool `run_underwriting_workflow`.\n"
            "  Arguments:\n"
            "  - `resource_id`: The ID found in the JSON.\n"
            "  - `user_query`: The original user query.\n"
            "- If `resource_id` is missing/null: Only then can you output the `message` field from the JSON to the user.\n\n"
            
            "### STEP 3: FINAL ANSWER\n"
            "Your final answer to the user must be the text returned by the `run_underwriting_workflow` tool. "
            "Do not add your own commentary."
        ),
        tools=[
            AgentTool(loan_initialization_agent),
            run_underwriting_workflow
        ]
    )

    runner = InMemoryRunner(agent=root_agent)
    events = await runner.run_debug(query, session_id=session_id)
    last_event = events[-1]
    # The final output of the root agent is the response
    if last_event.content and last_event.content.parts:
        return last_event.content.parts[0].text
    return ""