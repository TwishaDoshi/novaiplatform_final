from server import NovaMCPServer


class NovaMCPClient:
    """
    Simple client wrapper for calling tools from the MCP server.
    """

    def __init__(self):
        self.server = NovaMCPServer()

    def call_tool(self, tool_name: str, **kwargs):
        print(f"\n[CLIENT] Calling tool: {tool_name}")
        print(f"[CLIENT] Input: {kwargs}")

        result = self.server.execute(tool_name, **kwargs)

        print(f"[CLIENT] Output: {result}")
        return result


if __name__ == "__main__":
    client = NovaMCPClient()

    client.call_tool("get_order_status", order_id="O1001")
    client.call_tool("get_customer_profile", customer_id="C001")
    client.call_tool("search_product_catalog", query="serum")
    client.call_tool("recommend_products", customer_id="C001")