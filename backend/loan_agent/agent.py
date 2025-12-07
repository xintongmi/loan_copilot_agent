from google.adk.agents.llm_agent import Agent
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import tool
from google.cloud import storage
from pydantic import BaseModel, Field
from typing import Optional

GCS_PATH = "gs://copilot_loan_guidelines/guidelines/"
session_service = InMemorySessionService()

class LoanInitializationOutput(BaseModel):
    resource_id: Optional[str] = Field(
        default=None, description="The resource ID of the guideline cache."
    )
    message: Optional[str] = Field(
        default=None,
        description=(
            "A message indicating why the loan program couldn't be determined "
            "or other information."
        ),
    )

@tool
def get_guideline_cache_resource_id(
    guideline_file_name: str, session: Session
) -> str:
    """
    Gets the resource ID for the guideline cache.
    If the resource ID has expired, it recreates the context caching resource.
    """
    if "resource_id" in session.state:
        print(f"Using cached resource ID for session {session.id}")
        return session.state["resource_id"]

    print(
        "Creating new resource ID for "
        f"{guideline_file_name} in session {session.id}"
    )
    resource_id = f"guideline_cache_resource_id_for_session_{session.id}"
    session.state["resource_id"] = resource_id
    return resource_id


def _list_gcs_files(gcs_path: str) -> list[str]:
    """Lists all files under a given GCS path."""
    # gs://bucket-name/prefix
    path_parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    return [blob.name for blob in blobs]


def _create_loan_initialization_agent():
    file_names = _list_gcs_files(GCS_PATH)
    file_names_str = "\n".join(file_names)

    return Agent(
        model="gemini-2.5-flash",
        name="loan_initialization_agent",
        description=(
            "An agent responsible for checking if there is a consensus on the "
            "loan program from the given context, finding the loan guideline, "
            "and ensuring the guideline cache is valid."
        ),
        instruction=f"""You are the **Loan Initialization Agent** for an underwriting copilot.

### Your Goal
Your sole responsibility is to identify the correct Loan Program Guideline based on the current email context and Loan Officer notes, and then load that guideline into the system cache.

### Available Guideline Files
You must strictly match the loan program to one of the following files. Do not invent filenames.
{file_names_str}

### Instructions
1. **Analyze Context:** Read the email thread and any notes provided by the Loan Officer. Look for specific loan program names (e.g., "Jumbo", "Conforming", "FHA", "VA", "5/1 ARM").
2. **Check Consensus:**
   - If the Loan Officer explicitly names a program in the notes, prioritize that.
   - If the email implies one program but the notes imply another, or if the request is vague (e.g., "new loan"), STOP. You cannot proceed.
3. **Action:**
   - **IF** a single, clear loan program is identified and matches a file in the list above:
     - Call the tool `get_guideline_cache_resource_id(file_name)`.
     - Output the resulting resource ID.
   - **IF** the program is ambiguous, missing, or does not match a file:
     - Output a clear message explaining why you need human help.

### Output Format
You must output a structured JSON object matching this schema exactly:
{{
  "resource_id": "string or null", // The ID returned by the tool, or null if failed
  "message": "string or null"      // Explanation if you cannot determine the program
}}

### Examples
User: "I have a new Jumbo loan application here." (File 'jumbo_v1.pdf' exists)
Output: {{ "resource_id": "projects/123/.../cachedContents/xyz", "message": null }}

User: "Customer asking for rates." (No program specified)
Output: {{ "resource_id": null, "message": "I cannot determine the loan program from the email. Please specify if this is Jumbo, Conforming, or FHA." }}
""",
        tools=[get_guideline_cache_resource_id],
        output_model=LoanInitializationOutput,
    )


def _create_root_agent():
    return Agent(
        model="gemini-2.5-flash",
        name="root_agent",
        description=(
            "You are a loan agent, the loan process has several steps. "
            "Try to figure out what stage you are in."
        ),
        instruction="Answer user questions to the best of your knowledge",
    )


async def get_agent_response(query: str, session_id: str):
    """
    Invokes the root agent with a user query.
    """
    loan_initialization_agent = _create_loan_initialization_agent()
    init_runner = InMemoryRunner(
        agent=loan_initialization_agent, session_service=session_service
    )
    init_output: LoanInitializationOutput = await init_runner.run_debug(
        query, session_id=session_id
    )

    if init_output.resource_id:
        root_agent = _create_root_agent()
        # TODO: Pass the resource_id to the root_agent's context.
        runner = InMemoryRunner(
            agent=root_agent, session_service=session_service
        )
        response = await runner.run_debug(query, session_id=session_id)
        return response
    else:
        return init_output.message or "The loan program could not be determined."