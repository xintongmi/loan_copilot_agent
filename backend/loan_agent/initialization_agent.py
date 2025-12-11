from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool
from google.api_core import exceptions
from google.cloud.storage import Client as StorageClient
from pydantic import BaseModel, Field
from typing import Optional, Dict
from google.adk.tools.tool_context import ToolContext
import os
import json
import asyncio
import base64

# --- Vertex AI SDK Imports & Initialization ---
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

# The .env file is loaded by main.py, so we can access the variables here.
try:
    vertexai.init(
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION")
    )
    print("Vertex AI SDK Initialized.")
except Exception as e:
    print(f"ERROR: Failed to initialize Vertex AI SDK: {e}")


# --- Configuration ---
GEMINI_MODEL_ID = "gemini-2.5-flash"
GCS_GUIDELINES_PATH = "gs://copilot_loan_guidelines/guidelines/"


# --- Pydantic Schemas ---
class LoanInitializationOutput(BaseModel):
    """The output schema for the loan initialization agent."""
    guideline_content: Optional[str] = Field(
        default=None, description="The extracted text content of the underwriting guideline."
    )
    message: Optional[str] = Field(
        default=None,
        description="A message indicating why the loan program couldn't be determined or other information."
    )
    should_proceed: bool = Field(
        default=False, description="Whether the main agent should proceed."
    )

# --- Helper Functions & Tools ---
def _list_gcs_files(gcs_path_param: str) -> list[str]:
    """Lists all files under a given GCS path, returning only the base filenames."""
    path_parts = gcs_path_param.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""

    try:
        storage_client = StorageClient()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        return [blob.name[len(prefix):] for blob in blobs if blob.name.startswith(prefix) and blob.name != prefix]
    except Exception as e:
        print(f"ERROR: Failed to list GCS files at {gcs_path_param}. Assuming no files. Error: {e}")
        return []


@FunctionTool
async def extract_guideline_logic(
    guideline_file_name: str, tool_context: ToolContext
) -> str:
    """
    Downloads a guideline PDF from GCS, extracts its text content using a Gemini model,
    and returns the extracted text as a string.
    """
    gcs_uri = f"{GCS_GUIDELINES_PATH}{guideline_file_name}"
    print(f"DEBUG: Attempting to download {gcs_uri}")

    # 1. Download PDF from GCS
    storage_client = StorageClient()
    path_parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1]
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    print(f"DEBUG: Downloading bytes from GCS for {gcs_uri}...")
    try:
        pdf_bytes = await asyncio.to_thread(blob.download_as_bytes)
    except exceptions.NotFound:
        print(f"ERROR: File not found in GCS: {gcs_uri}")
        raise ValueError(f"Guideline file '{guideline_file_name}' not found.")

    print(f"DEBUG: PDF downloaded successfully. Size: {len(pdf_bytes)} bytes.")
    # 2. Prepare request for Gemini
    prompt = "Extract all text from the provided PDF document. Return only the raw text content."

    pdf_part = Part.from_data(
        data=pdf_bytes,
        mime_type="application/pdf",
    )

    # 3. Call Gemini to extract structured data
    model = GenerativeModel(GEMINI_MODEL_ID)
    
    print("DEBUG: Calling Gemini to extract text from PDF...")
    response = await model.generate_content_async(
        contents=[prompt, pdf_part],
    )
    print("DEBUG: Gemini call complete.")

    extracted_text = response.text
    print(f"DEBUG: Gemini response received. Text length: {len(extracted_text)}")

    return extracted_text


# --- Agent Factory ---
def create_loan_initialization_agent():
    file_names = _list_gcs_files(GCS_GUIDELINES_PATH)
    file_names_str = "\n".join(file_names) or "No files found."

    instruction = f"""You are the **Loan Initialization Agent** for an underwriting copilot. Your sole responsibility is to identify the correct Loan Program Guideline and determine if the process should continue.

### Available Guideline Files
You must strictly match the loan program to one of the following files. Do not invent filenames.
{file_names_str}

### Instructions
1.  **Analyze Context:** Read the user's query to identify a loan program.
2.  **Action:**
    *   **IF** a single, clear loan program is identified and matches a file in the list above:
        - Call the tool `extract_guideline_logic` with the matched `guideline_file_name`.
        - Set `should_proceed` to `True`.
        - Populate `guideline_content` with the result from the tool.
    *   **IF** the program is ambiguous, missing, or does not match a file:
        - Set `should_proceed` to `False`.
        - Populate `message` with a clear explanation of why you are stopping.

### Output Format
You must output a structured JSON object matching this schema exactly:
{LoanInitializationOutput.model_json_schema()}
"""

    return Agent(
        model="gemini-2.5-flash",
        name="loan_initialization_agent",
        description="Identifies a loan guideline and determines if the process should proceed.",
        instruction=instruction,
        tools=[extract_guideline_logic],
        output_schema=LoanInitializationOutput,
    )