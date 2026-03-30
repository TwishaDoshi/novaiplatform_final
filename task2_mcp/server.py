from typing import Any, Dict

from tools import (
    get_order_status,
    create_return_request,
    get_customer_profile,
    search_product_catalog,
    recommend_products,
)


class NovaMCPServer:
    """
    Lightweight MCP-style tool server for the assessment.
    Exposes tool registry + unified execute() method.
    """

    def __init__(self):
        self.tool_registry = {
            "get_order_status": get_order_status,
            "create_return_request": create_return_request,
            "get_customer_profile": get_customer_profile,
            "search_product_catalog": search_product_catalog,
            "recommend_products": recommend_products,
        }

    def list_tools(self):
        return list(self.tool_registry.keys())

    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        if tool_name not in self.tool_registry:
            return {
                "success": False,
                "message": f"Unknown tool: {tool_name}",
                "tool_name": tool_name,
                "data": None,
            }

        try:
            tool_fn = self.tool_registry[tool_name]
            result = tool_fn(**kwargs)
            return {
                "success": True,
                "message": "Tool executed successfully.",
                "tool_name": tool_name,
                "data": result,
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Tool execution failed: {str(e)}",
                "tool_name": tool_name,
                "data": None,
            }


if __name__ == "__main__":
    server = NovaMCPServer()
    print("Available tools:")
    for tool in server.list_tools():
        print("-", tool)