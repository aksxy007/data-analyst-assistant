from langchain_core.messages import HumanMessage
from typing import List
from scenes.DataModels.InputData import InputData
from langgraph.graph import StateGraph,END,START
from scenes.graph.nodes import call_model, call_tools, route_to_tools
from scenes.graph.states import AgentState


class Chatbot:
    def __init__(self):
        super().__init__()
        self.reset_chat()
        self.graph=self.build_graph()
        
    
    def user_message(self,user_query:str,input_data:List[InputData]):
        starting_image_paths_set = set(sum(self.output_image_paths.values(), []))
        input_state = {
            "messages": self.chat_history + [HumanMessage(content=user_query)],
            "output_image_paths": list(starting_image_paths_set),
            "input_data": input_data,
        }

        result = self.graph.invoke(input_state, {"recursion_limit": 15})
        self.chat_history = result["messages"]
        # print("intermediate_ouputs",self.intermediate_outputs)
        new_image_paths = set(result["output_image_paths"]) - starting_image_paths_set
        self.output_image_paths[len(self.chat_history) - 1] = list(new_image_paths)
        
        if "intermediate_outputs" in result:
            self.intermediate_outputs.extend(result["intermediate_outputs"])
    
    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node('agent',call_model)
        workflow.add_node('tools',call_tools)
        
        workflow.add_conditional_edges('agent',route_to_tools)
        workflow.add_edge('tools','agent')
        workflow.set_entry_point('agent')
        return workflow.compile()
        
    def reset_chat(self):
        self.chat_history = [] 
        self.intermediate_outputs=[]
        self.output_image_paths = {}
        # self.input_data = []