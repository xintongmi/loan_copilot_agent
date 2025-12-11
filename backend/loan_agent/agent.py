import json
from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from loan_agent.initialization_agent import (
    create_loan_initialization_agent,
    LoanInitializationOutput,
)
from loan_agent.pre_qualification_agent import create_pre_qualification_agent
from loan_agent.underwriting_agent import create_underwriting_agent
from loan_agent.shared import session_service


def create_main_agent(guideline_content: str):
    """
    Creates the 'Main Agent' that routes to the appropriate sub-agent.
    """
    pre_qualification_agent = create_pre_qualification_agent(guideline_content)
    underwriting_agent = create_underwriting_agent(guideline_content)

    return Agent(
        model="gemini-2.5-flash",
        name="main_agent",
        description="Routes user requests to the appropriate sub-agent based on the query.",
        instruction="""You are the **Main Agent**. Your job is to determine which sub-agent is best suited to handle the user's query and route the request to them.

- If the user is asking for calculations, eligibility, or has a "what-if" scenario, route to the `pre_qualification_agent`.
- If the user is asking a general question about the guidelines, route to the `underwriting_agent`.
""",
        sub_agents=[pre_qualification_agent, underwriting_agent],
    )

async def get_agent_response(query: str, session_id: str):
    print(f"--- get_agent_response called with query: '{query}' ---")
    """
    Invokes the agent graph with a user query, following a two-step process:
    1. Run initialization agent to extract guideline logic and determine if we should proceed.
    2. If successful and should_proceed is True, run the main agent to route to the appropriate sub-agent.
    """

    # --- Step 1: Run the loan initialization agent ---
    loan_initialization_agent = create_loan_initialization_agent()
    init_runner = Runner(
        agent=loan_initialization_agent, 
        session_service=session_service,
        app_name="loan_copilot_app"
    )
    print("--- Running Initialization Agent ---")
    init_events = await init_runner.run_debug(query, session_id=session_id)
    
    last_event = init_events[-1]
    try:
        init_output_json = json.loads(last_event.content.parts[0].text)
        init_output = LoanInitializationOutput(**init_output_json)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"ERROR: Could not parse output from initialization agent: {e}")
        return "Sorry, an internal error occurred during the initialization phase."

    # --- Step 2: Conditionally run the main agent ---
    if init_output.should_proceed and init_output.guideline_content:
        print(f"--- Initialization complete. Proceeding to Main Agent. ---")
        
        main_agent = create_main_agent(init_output.guideline_content)
        main_runner = Runner(
            agent=main_agent, 
            session_service=session_service,
            app_name="loan_copilot_app"
        )
        
        main_events = await main_runner.run_debug(query, session_id=session_id)
        
        final_event = main_events[-1]
        if final_event.content and final_event.content.parts:
            return final_event.content.parts[0].text
        return "The main agent executed but produced no specific output."
    else:
        print("--- Initialization failed or should not proceed. ---")
        return init_output.message or "The loan program could not be determined or the process was stopped."