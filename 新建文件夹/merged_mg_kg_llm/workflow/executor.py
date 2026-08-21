"""工作流执行器"""
from typing import Iterator, Dict, Any
from workflow.graph_builder import WorkflowBuilder
from workflow.state_schema import MGWorkflowState


class WorkflowExecutor:
    """工作流执行器"""

    def __init__(self):
        """初始化执行器"""
        self.builder = WorkflowBuilder()
        self.workflow = self.builder.create_workflow()

    def execute(self, user_input: str) -> MGWorkflowState:
        """
        执行完整工作流

        Args:
            user_input: 用户输入的自然语言场景描述

        Returns:
            最终状态
        """
        initial_state = self.builder.create_initial_state(user_input)
        final_state = self.workflow.invoke(initial_state)
        return final_state

    def stream_execute(self, user_input: str) -> Iterator[MGWorkflowState]:
        """
        流式执行工作流

        Args:
            user_input: 用户输入的自然语言场景描述

        Yields:
            每个步骤的状态
        """
        initial_state = self.builder.create_initial_state(user_input)

        for output in self.workflow.stream(initial_state):
            for node_name, state in output.items():
                yield state
