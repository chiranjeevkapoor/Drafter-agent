from typing import Annotated, TypedDict, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os
load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


document_content = []

@tool
def update(content:str):
    """Updates the document with the provided content"""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"


@tool
def save(filename:str):
    """Saves the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """
    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"
    
    try:
        with open(filename,'w') as file:
            file.write(document_content)
        print(f"\n Document has been saved successfully to {filename}")
        return f"\n Document has been saved successfully to {filename}"
    except Exception as e:
        return f"Error saving document : {str(e)}"

tools = [update, save]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY_K")).bind_tools(tools)

def our_agent(state: AgentState)->AgentState:
    system_prompt = SystemMessage(content="""
    You are Drafter, A helpful writing assistant. You are going to help the user update and modify documents.
                                  
    -If the user wants to update the document, use the 'update' tool with the complete updated content.
    -If the user wants to save and finish, you need to use the 'save' tool.
    -Make sure to always show the current document state after modifications.
                                  
    The current document content is: {document_content}""")

    if not state['messages']:
        user_input = "I'am ready to help you update a document, What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = "What would you like to do with the document?"
        print(f"\n USER:{user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + user_message

    response = model.invoke(all_messages)

    print(f"\n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}]")

    return {"messages":list(state['messages'] + [user_message, response])}

