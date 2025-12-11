from google.adk.agents.llm_agent import Agent

def create_underwriting_agent(guideline_content: str):
    """
    Creates the 'Underwriting Agent' that uses the guideline text to answer general questions.
    """
    return Agent(
        model="gemini-2.5-flash",
        name="underwriting_agent",
        description="Underwriting agent that can answer general questions based on guideline content.",
        instruction=f"""You are the **Underwriting Agent**. Your role is to use the provided underwriting rules to answer a user's question.

### Instructions:
1.  **Analyze Rules & Query:** Analyze the provided content and the user's original query.
2.  **Formulate Answer:** Provide a clear answer based on the provided content.

### Content:
{guideline_content}
""",
    )
