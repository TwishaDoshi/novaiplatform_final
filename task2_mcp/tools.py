import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "audit_log.jsonl")


def _safe_value(value):
    """Convert pandas/NaN values to JSON-safe Python values."""
    if pd.isna(value):
        return None
    return value


def _row_to_dict(row: pd.Series) -> Dict[str, Any]:
    return {k: _safe_value(v) for k, v in row.to_dict().items()}


def log_audit_event(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Dict[str, Any],
    success: bool = True,
    error: Optional[str] = None
) -> None:
    """Append a tool call log entry to audit_log.jsonl."""
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "success": success,
        "error": error,
    }

    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class NovaMockDB:
    def __init__(self):
        self.customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
        self.orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
        self.products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
        self.returns_df = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"))
        self.support_tickets = pd.read_csv(os.path.join(DATA_DIR, "support_tickets.csv"))

        self._normalize()

    def _normalize(self):
        for df, cols in [
            (self.customers, ["customer_id", "name", "country", "skin_type", "size_preference"]),
            (self.orders, ["order_id", "customer_id", "items", "status"]),
            (self.products, ["product_id", "name", "category", "ingredients", "target"]),
            (self.returns_df, ["return_id", "order_id", "customer_id", "reason", "status"]),
            (self.support_tickets, ["ticket_id", "customer_id", "query", "intent", "status"]),
        ]:
            for col in cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)

    def save_returns(self):
        self.returns_df.to_csv(os.path.join(DATA_DIR, "returns.csv"), index=False)


db = NovaMockDB()


def get_order_status(order_id: str) -> Dict[str, Any]:
    tool_name = "get_order_status"
    tool_input = {"order_id": order_id}

    try:
        order_id = str(order_id).strip().upper()
        result_df = db.orders[db.orders["order_id"].str.upper() == order_id]

        if result_df.empty:
            output = {
                "found": False,
                "message": f"No order found for order_id={order_id}"
            }
            log_audit_event(tool_name, tool_input, output, success=True)
            return output

        order = _row_to_dict(result_df.iloc[0])

        output = {
            "found": True,
            "message": "Order found successfully.",
            "order": order
        }
        log_audit_event(tool_name, tool_input, output, success=True)
        return output

    except Exception as e:
        output = {"found": False, "message": "Order lookup failed."}
        log_audit_event(tool_name, tool_input, output, success=False, error=str(e))
        return output


def create_return_request(order_id: str, reason: str) -> Dict[str, Any]:
    tool_name = "create_return_request"
    tool_input = {"order_id": order_id, "reason": reason}

    try:
        order_id = str(order_id).strip().upper()
        reason = str(reason).strip()

        order_df = db.orders[db.orders["order_id"].str.upper() == order_id]
        if order_df.empty:
            output = {
                "created": False,
                "message": f"Cannot create return. No order found for order_id={order_id}"
            }
            log_audit_event(tool_name, tool_input, output, success=True)
            return output

        existing_return = db.returns_df[db.returns_df["order_id"].str.upper() == order_id]
        if not existing_return.empty:
            existing = _row_to_dict(existing_return.iloc[0])
            output = {
                "created": False,
                "message": f"Return already exists for order_id={order_id}",
                "return_request": existing
            }
            log_audit_event(tool_name, tool_input, output, success=True)
            return output

        order_row = order_df.iloc[0]
        new_return = {
            "return_id": f"R{uuid.uuid4().hex[:6].upper()}",
            "order_id": order_id,
            "customer_id": order_row["customer_id"],
            "reason": reason,
            "status": "pending"
        }

        db.returns_df = pd.concat([db.returns_df, pd.DataFrame([new_return])], ignore_index=True)
        db.save_returns()

        output = {
            "created": True,
            "message": "Return request created successfully.",
            "return_request": new_return
        }
        log_audit_event(tool_name, tool_input, output, success=True)
        return output

    except Exception as e:
        output = {"created": False, "message": "Return creation failed."}
        log_audit_event(tool_name, tool_input, output, success=False, error=str(e))
        return output


def get_customer_profile(customer_id: str) -> Dict[str, Any]:
    tool_name = "get_customer_profile"
    tool_input = {"customer_id": customer_id}

    try:
        customer_id = str(customer_id).strip().upper()
        result_df = db.customers[db.customers["customer_id"].str.upper() == customer_id]

        if result_df.empty:
            output = {
                "found": False,
                "message": f"No customer found for customer_id={customer_id}"
            }
            log_audit_event(tool_name, tool_input, output, success=True)
            return output

        customer = _row_to_dict(result_df.iloc[0])

        output = {
            "found": True,
            "message": "Customer profile found successfully.",
            "customer": customer
        }
        log_audit_event(tool_name, tool_input, output, success=True)
        return output

    except Exception as e:
        output = {"found": False, "message": "Customer lookup failed."}
        log_audit_event(tool_name, tool_input, output, success=False, error=str(e))
        return output


def search_product_catalog(query: str) -> Dict[str, Any]:
    tool_name = "search_product_catalog"
    tool_input = {"query": query}

    try:
        q = str(query).strip().lower()
        if not q:
            output = {"found": False, "message": "Empty search query.", "products": []}
            log_audit_event(tool_name, tool_input, output, success=True)
            return output

        matches: List[Dict[str, Any]] = []

        for _, row in db.products.iterrows():
            row_dict = _row_to_dict(row)
            searchable_text = " ".join([
                str(row_dict.get("product_id", "")).lower(),
                str(row_dict.get("name", "")).lower(),
                str(row_dict.get("category", "")).lower(),
                str(row_dict.get("ingredients", "")).lower(),
                str(row_dict.get("target", "")).lower(),
            ])

            if q in searchable_text:
                matches.append(row_dict)

        output = {
            "found": len(matches) > 0,
            "message": f"{len(matches)} product(s) matched.",
            "products": matches[:10]
        }
        log_audit_event(tool_name, tool_input, output, success=True)
        return output

    except Exception as e:
        output = {"found": False, "message": "Product search failed.", "products": []}
        log_audit_event(tool_name, tool_input, output, success=False, error=str(e))
        return output


def recommend_products(customer_id: str, skin_type: Optional[str] = None) -> Dict[str, Any]:
    tool_name = "recommend_products"
    tool_input = {"customer_id": customer_id, "skin_type": skin_type}

    try:
        customer_id = str(customer_id).strip().upper()

        customer_df = db.customers[db.customers["customer_id"].str.upper() == customer_id]
        if customer_df.empty:
            output = {
                "found": False,
                "message": f"No customer found for customer_id={customer_id}",
                "recommendations": []
            }
            log_audit_event(tool_name, tool_input, output, success=True)
            return output

        customer = _row_to_dict(customer_df.iloc[0])
        effective_skin_type = (skin_type or customer.get("skin_type") or "").lower()

        recommendations = []
        for _, row in db.products.iterrows():
            row_dict = _row_to_dict(row)
            target = str(row_dict.get("target", "")).lower()
            category = str(row_dict.get("category", "")).lower()

            # Simple recommendation logic:
            # prioritize skincare/makeup items targeted to customer skin type
            if effective_skin_type and effective_skin_type in target:
                recommendations.append(row_dict)
            elif effective_skin_type == "" and category in {"skincare", "makeup"}:
                recommendations.append(row_dict)

        # fallback: if nothing matched, return top few products
        if not recommendations:
            recommendations = [_row_to_dict(r) for _, r in db.products.head(3).iterrows()]

        output = {
            "found": True,
            "message": f"Generated {len(recommendations[:5])} recommendation(s).",
            "customer_id": customer_id,
            "skin_type_used": effective_skin_type,
            "recommendations": recommendations[:5]
        }
        log_audit_event(tool_name, tool_input, output, success=True)
        return output

    except Exception as e:
        output = {"found": False, "message": "Recommendation failed.", "recommendations": []}
        log_audit_event(tool_name, tool_input, output, success=False, error=str(e))
        return output