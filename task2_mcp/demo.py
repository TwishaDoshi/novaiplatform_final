import json
from client import NovaMCPClient


def pretty_print(title: str, obj):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def run_compound_demo():
    """
    Compound demo scenario for Task 2:
    1. Get order status
    2. Get customer profile
    3. Create return request
    4. Search product catalog
    5. Recommend products
    """
    client = NovaMCPClient()

    # Scenario assumptions:
    # Customer C001 asks:
    # "Where is my order O1001? I may want to return it.
    #  Also recommend something for my skin type."

    order_result = client.call_tool("get_order_status", order_id="O1001")
    pretty_print("STEP 1 - ORDER STATUS RESULT", order_result)

    customer_result = client.call_tool("get_customer_profile", customer_id="C001")
    pretty_print("STEP 2 - CUSTOMER PROFILE RESULT", customer_result)

    return_result = client.call_tool(
        "create_return_request",
        order_id="O1001",
        reason="Changed mind after purchase"
    )
    pretty_print("STEP 3 - RETURN REQUEST RESULT", return_result)

    product_search_result = client.call_tool(
        "search_product_catalog",
        query="serum"
    )
    pretty_print("STEP 4 - PRODUCT SEARCH RESULT", product_search_result)

    recommendation_result = client.call_tool(
        "recommend_products",
        customer_id="C001"
    )
    pretty_print("STEP 5 - PRODUCT RECOMMENDATION RESULT", recommendation_result)

    print("\nDemo completed successfully.")
    print("Check task2_mcp/audit_log.jsonl for the full audit trail.")


if __name__ == "__main__":
    run_compound_demo()