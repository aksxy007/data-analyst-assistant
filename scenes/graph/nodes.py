# Over the whole file single LLM is used to lets set it up
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from langchain_core.messages import AIMessage,ToolMessage,HumanMessage
from langgraph.graph import END
import re
from langsmith import traceable

from dotenv import load_dotenv
import os
from scenes.graph.tool_registry import TOOL_REGISTRY

from scenes.graph.states import AgentState
from scenes.DataModels.ToolInvocation import ToolInvocation
from scenes.graph.tools import python_code_executor


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
# openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatGroq(model_name="llama-3.3-70b-versatile",temperature=0.9,max_tokens=4000,api_key=api_key)
# chat_llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.9,max_tokens=4000,api_key=openai_api_key)
# coder_llm = ChatGroq(model_name="qwen-2.5-coder-32b",temperature=0,max_tokens=8000,api_key=api_key)


tools = [python_code_executor]
chat_llm=llm.bind_tools(tools)

with open("scenes/prompts/analyst_prompt.md") as f:
    analyst_prompt = f.read()

chat_template = ChatPromptTemplate.from_messages([
    ("system",analyst_prompt),
    ("placeholder","{messages}"),
    ]
)

model = chat_template | chat_llm

def remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def create_data_summary(state:AgentState) -> str:
    summary = ""
    variables = []
    
    for d in state['input_data']:
        variables.append(d.variable_name)
        summary+=f"Variable: {d.variable_name}\n"
        summary+=f"Description: {d.data_description}\n"
    
    if "current_variables" in state:
        remaining_variables = [v for v in state["current_variables"] if v not in variables]
        summary+="\n".join([f"Variable: {v}" for v in remaining_variables])
        
    return summary

@traceable
def call_model(state:AgentState):
    
    current_data_template = """The following data is available for analysis:\n{data_summary}"""
    # Add the current data summary into the prompt for reference
    current_data_messages = HumanMessage(content=current_data_template.format(data_summary=create_data_summary(state)))
    state["messages"] = [current_data_messages] + state["messages"]
    
    # print("state in call_model",state)
    llm_outputs = model.invoke(state)
    # print("State",state)
    # if isinstance(llm_outputs, AIMessage) and hasattr(llm_outputs, "tool_calls"):
    #     print("Tool Calls Generated:", llm_outputs.tool_calls)  # 
    
    return {"messages":[llm_outputs] ,"intermediate_outputs":[current_data_messages.content]}

def route_to_tools(state:AgentState) -> Literal["tools","__end__"]:
    """
        Use in the conditional_edge to route to the ToolNode if the last message
        has tool calls. Otherwise, route back to the agent.
    """

    if messages := state.get("messages"):
        # Get the lastest AI message
        ai_message = messages[-1]
    else:
        raise Exception(f"No AI messages in state: {state}")


    if hasattr(ai_message,"tool_calls") and len(ai_message.tool_calls)>0:
        return "tools"
    
    return "__end__"

# Call tools separately to update the graph states if need be in the tool calling step
def call_tools(state:AgentState):
    last_message = state["messages"][-1]
    tool_messages=[]
    state_updates = {}

    if isinstance(last_message,AIMessage) and hasattr(last_message,"tool_calls"):
        
        
        for tc in last_message.tool_calls:
            # print(tc)
            tool_name  = tc['name']
            tool_args = tc['args']
            
            if "thought" not in tool_args or "python_code" not in tool_args:
                raise Exception(f"Malformed tool arguments: {tool_args}")
            
            if tool_name not in TOOL_REGISTRY:
                raise Exception(f"Tool {tool_name} not found in tool registry")
            
            tool= TOOL_REGISTRY[tool_name]
            # print("tool",tool)
            input_data = {
                'graph_state': state,
                'python_code': tool_args.get('python_code', ''),
                'thought': tool_args.get('thought', '')
            }
            
            response = tool.invoke(input=input_data)
            # if response is an expception raise it
            
            if isinstance(response,Exception):
                raise Exception(f"Error in tool {tc['name']}: {response}")
            else:
                message, updates = response
                print("updates",updates)
                print("message",message)
                
                tool_messages.append(
                    ToolMessage(
                        tool=tc['name'],
                        content=str(message),
                        tool_call_id=tc['id']
                    )
                )
                state_updates.update(updates)
            
            if 'messages' not in state_updates:
                state_updates['messages'] = []
            
            state_updates['messages']=tool_messages
            
        return state_updates
                
