from fastapi import FastAPI
from loan_agent.agent import get_agent_response
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), 'loan_agent', '.env'))


app = FastAPI()


class InvokeRequest(BaseModel):
    query: str
    session_id: str


@app.post("/invoke")
async def invoke_agent(request: InvokeRequest):
    """
    Invokes the root agent with a user query.
    """
    try:
        response = await get_agent_response(request.query, request.session_id)
        return {"response": response}
    except Exception as e:
        import traceback
        return {"error": f"An error occurred: {e}", "traceback": traceback.format_exc()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)