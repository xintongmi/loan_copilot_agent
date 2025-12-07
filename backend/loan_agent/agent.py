import json
from google.adk.agents.llm_agent import Agent
from google.adk.runners import InMemoryRunner
from loan_agent.initialization_agent import (
    create_loan_initialization_agent,
    LoanInitializationOutput,
)
from loan_agent.shared import session_service
from google.adk.tools import FunctionTool




def create_dynamic_main_agent_graph(resource_id: str):
    """Creates the main agent graph dynamically based on the resource ID."""
    guideline_file_name = resource_id.split(":", 1)[1]

    @FunctionTool
    def run_dynamic_underwriting(user_query: str) -> str:
        """
        A placeholder tool for running dynamic underwriting.
        In a real scenario, this would use the resource_id to perform tasks.
        """
        return f"Running underwriting for resource {resource_id} with query: {user_query} using guideline file: {guideline_file_name}"

    return Agent(
        model="gemini-2.5-flash",
        name="dynamic_main_agent",
        description="A dynamic agent that uses a resource ID.",
        instruction="You are a dynamic agent. Use your tools to answer the user's question.",
        tools=[run_dynamic_underwriting],
    )


async def get_agent_response(query: str, session_id: str):
    """
    Invokes the root agent with a user query.
    """
    loan_initialization_agent = create_loan_initialization_agent()
    init_runner = InMemoryRunner(
        agent=loan_initialization_agent
    )
    events = await init_runner.run_debug(
        query, session_id=session_id
    )
    last_event = events[-1]
    # The output of the agent is a JSON string in the last event.
    init_output_json = json.loads(last_event.content.parts[0].text)
    init_output = LoanInitializationOutput(**init_output_json)

    if init_output.resource_id:
        main_agent = create_dynamic_main_agent_graph(init_output.resource_id)
        runner = InMemoryRunner(
            agent=main_agent
        )
        response = await runner.run_debug(query, session_id=session_id)
        return response
    else:
        return init_output.message or "The loan program could not be determined."