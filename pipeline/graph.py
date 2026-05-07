from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import PipelineNodes


def create_footagent_graph(nodes: PipelineNodes):
    """
    Creates the LangGraph execution flow for FootAgent.
    """
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("tracking", nodes.tracking_node)
    workflow.add_node("trigger", nodes.trigger_node)
    workflow.add_node("off_ball", nodes.off_ball_node)
    workflow.add_node("var", nodes.var_node)
    workflow.add_node("memory", nodes.memory_node)

    # Define Edges
    workflow.set_entry_point("tracking")
    workflow.add_edge("tracking", "trigger")
    
    # Conditional Routing from Trigger
    def route_incident(state: AgentState):
        if state["is_incident"]:
            return "var"
        return "off_ball"

    workflow.add_conditional_edges(
        "trigger",
        route_incident,
        {
            "var": "var",
            "off_ball": "off_ball"
        }
    )

    workflow.add_edge("var", "off_ball")
    workflow.add_edge("off_ball", "memory")
    workflow.add_edge("memory", END)

    # Compile
    return workflow.compile()


if __name__ == "__main__":
    from .model_loaders import ModelLoader
    from .vram_scheduler import VRAMScheduler
    
    loader = ModelLoader()
    scheduler = VRAMScheduler(loader)
    nodes = PipelineNodes(loader, scheduler)
    app = create_footagent_graph(nodes)
    print("LangGraph Pipeline compiled successfully.")
