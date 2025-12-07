from google.adk.agents.llm_agent import Agent
from loan_agent.shared import session_service
from google.adk.tools import FunctionTool
from google.api_core import exceptions
from google.cloud.storage import Client as StorageClient
from google.genai.types import CachedContent
from pydantic import BaseModel, Field
from typing import Optional
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext # New import


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


@FunctionTool
async def get_guideline_cache_resource_id(
    guideline_file_name: str, session_id: str, tool_context: ToolContext
) -> str:
    """
    Gets the resource ID for the guideline cache.
    If the resource ID has expired, it recreates the context caching resource.
    """
    app_name = tool_context._invocation_context.session.app_name
    user_id = tool_context._invocation_context.session.user_id

    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    if not session:
        session = await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state={},
        )

    if "resource_id" in session.state:
        stored_id = session.state["resource_id"]
        print(f"Checking validity of cached resource: {stored_id}")

        try:
            # 2. VALIDATION: Try to fetch the real resource from Vertex AI
            # If the cache has expired, this raises a 404 NotFound error.
            CachedContent(name=stored_id)

            # Optional: You could also check cached_content.expire_time here
            # if you want to refresh it preemptively before it dies.

            print(f"Resource {stored_id} is valid.")
            return stored_id

        except (exceptions.NotFound, ValueError):
            # 3. Handle Expiration
            print(
                f"Resource {stored_id} has expired or was not found. "
                "Recreating..."
            )
            # Clean up the stale ID from the session
            del session.state["resource_id"]
        except Exception as e:
            print(f"Unexpected error checking cache: {e}. Recreating...")
            del session.state["resource_id"]

    print(
        "Creating new resource ID for "
        f"{guideline_file_name} in session {session.id}"
    )
    resource_id = f"guideline_cache_resource_id_for_session_{session.id}:{guideline_file_name}"
    session.state["resource_id"] = resource_id
    return resource_id


def _list_gcs_files(gcs_path_param: str) -> list[str]:
    """Lists all files under a given GCS path."""
    # gs://bucket-name/prefix
    path_parts = gcs_path_param.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""

    storage_client = StorageClient()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    return [blob.name for blob in blobs]



def create_loan_initialization_agent():
    GCS_PATH = "gs://copilot_loan_guidelines/guidelines/" # Moved definition inside
    file_names = _list_gcs_files(GCS_PATH)
    file_names_str = "\n".join(file_names)

    async def _get_loan_initialization_instruction(context: ReadonlyContext) -> str: # New function
        # Access session_id from the context
        current_session_id = context.session.id
        return f"""You are the **Loan Initialization Agent** for an underwriting copilot.

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
     - Call the tool `get_guideline_cache_resource_id(guideline_file_name=file_name, session_id="{current_session_id}")`.
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
"""

    return Agent(
        model="gemini-2.5-flash",
        name="loan_initialization_agent",
        description=(
            "An agent responsible for checking if there is a consensus on the "
            "loan program from the given context, finding the loan guideline, "
            "and ensuring the guideline cache is valid."
        ),
        instruction=_get_loan_initialization_instruction, # Pass the function here
        tools=[get_guideline_cache_resource_id],
        output_schema=LoanInitializationOutput,
    )
