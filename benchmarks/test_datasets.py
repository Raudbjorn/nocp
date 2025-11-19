"""
Synthetic test datasets for benchmarking.

Generates realistic datasets for different scenarios:
- RAG retrieval with large document sets
- API calls with verbose responses
- Database queries with structured results
"""

from typing import Dict, Any, List, Literal
import random
import string


def generate_random_text(word_count: int) -> str:
    """
    Generate random lorem ipsum style text.

    Args:
        word_count: Number of words to generate

    Returns:
        Random text string
    """
    words = [
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
        "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
        "et", "dolore", "magna", "aliqua", "enim", "ad", "minim", "veniam",
        "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi",
        "aliquip", "ex", "ea", "commodo", "consequat", "duis", "aute", "irure",
        "in", "reprehenderit", "voluptate", "velit", "esse", "cillum", "fugiat",
        "nulla", "pariatur", "excepteur", "sint", "occaecat", "cupidatat",
        "non", "proident", "sunt", "culpa", "qui", "officia", "deserunt",
        "mollit", "anim", "id", "est", "laborum", "system", "data", "query",
        "result", "process", "function", "method", "class", "interface",
        "implementation", "architecture", "design", "pattern", "service",
    ]

    return " ".join(random.choice(words) for _ in range(word_count))


def generate_rag_dataset(
    size: Literal["small", "medium", "large"]
) -> Dict[str, Any]:
    """
    Generate RAG retrieval dataset.

    Args:
        size: Dataset size (small/medium/large)

    Returns:
        Dictionary with input and output data
    """
    # Size configurations
    configs = {
        "small": {"num_docs": 10, "words_per_doc": 200, "num_chunks": 5},
        "medium": {"num_docs": 50, "words_per_doc": 300, "num_chunks": 10},
        "large": {"num_docs": 200, "words_per_doc": 500, "num_chunks": 20},
    }

    config = configs[size]

    # Generate documents
    documents = []
    for i in range(config["num_docs"]):
        doc = {
            "id": f"doc_{i:04d}",
            "title": f"Document {i}: {generate_random_text(5)}",
            "content": generate_random_text(config["words_per_doc"]),
            "metadata": {
                "source": f"source_{random.randint(1, 10)}",
                "timestamp": f"2024-01-{random.randint(1, 28):02d}T12:00:00Z",
                "relevance_score": random.uniform(0.5, 0.99),
            },
        }
        documents.append(doc)

    # Create input context (all documents as string)
    input_text = "Retrieved documents from vector database:\n\n"
    for doc in documents:
        input_text += f"[Document {doc['id']}]\n"
        input_text += f"Title: {doc['title']}\n"
        input_text += f"Content: {doc['content']}\n"
        input_text += f"Source: {doc['metadata']['source']}\n"
        input_text += f"Relevance: {doc['metadata']['relevance_score']:.3f}\n\n"

    # Create output (structured response)
    output = {
        "summary": generate_random_text(100),
        "top_chunks": [
            {
                "document_id": doc["id"],
                "title": doc["title"],
                "snippet": doc["content"][:100],
                "relevance_score": doc["metadata"]["relevance_score"],
            }
            for doc in sorted(documents, key=lambda d: d["metadata"]["relevance_score"], reverse=True)[:config["num_chunks"]]
        ],
        "total_documents": len(documents),
        "sources": list(set(doc["metadata"]["source"] for doc in documents)),
    }

    return {
        "input": input_text,
        "output": output,
        "metadata": {
            "scenario": "rag_retrieval",
            "size": size,
            "num_documents": len(documents),
        },
    }


def generate_api_dataset(
    size: Literal["small", "medium", "large"]
) -> Dict[str, Any]:
    """
    Generate API call dataset with verbose responses.

    Args:
        size: Dataset size (small/medium/large)

    Returns:
        Dictionary with input and output data
    """
    # Size configurations
    configs = {
        "small": {"num_users": 5, "num_transactions": 10, "description_words": 50},
        "medium": {"num_users": 20, "num_transactions": 50, "description_words": 100},
        "large": {"num_users": 100, "num_transactions": 200, "description_words": 200},
    }

    config = configs[size]

    # Generate users with verbose details
    users = []
    for i in range(config["num_users"]):
        user = {
            "user_id": f"user_{i:06d}",
            "username": f"user{i}",
            "email": f"user{i}@example.com",
            "full_name": f"User {i} Full Name",
            "profile": {
                "bio": generate_random_text(config["description_words"]),
                "interests": [generate_random_text(3) for _ in range(5)],
                "location": f"City {random.randint(1, 100)}",
                "join_date": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            },
            "settings": {
                "notifications": random.choice([True, False]),
                "theme": random.choice(["light", "dark"]),
                "language": random.choice(["en", "es", "fr"]),
            },
            "metadata": {
                "last_login": f"2024-01-{random.randint(1, 28):02d}T{random.randint(0, 23):02d}:00:00Z",
                "total_logins": random.randint(1, 1000),
                "account_status": "active",
            },
        }
        users.append(user)

    # Generate transactions
    transactions = []
    for i in range(config["num_transactions"]):
        transaction = {
            "transaction_id": f"tx_{i:08d}",
            "user_id": random.choice(users)["user_id"],
            "type": random.choice(["purchase", "refund", "transfer"]),
            "amount": round(random.uniform(10, 1000), 2),
            "currency": "USD",
            "description": generate_random_text(config["description_words"] // 2),
            "timestamp": f"2024-01-{random.randint(1, 28):02d}T{random.randint(0, 23):02d}:00:00Z",
            "status": random.choice(["completed", "pending", "failed"]),
        }
        transactions.append(transaction)

    # Create input context
    input_text = "API Response:\n\n"
    input_text += f"Total Users: {len(users)}\n"
    input_text += f"Total Transactions: {len(transactions)}\n\n"
    input_text += "Users:\n"
    for user in users:
        input_text += f"  - {user['user_id']}: {user['full_name']}\n"
        input_text += f"    Email: {user['email']}\n"
        input_text += f"    Bio: {user['profile']['bio']}\n"
        input_text += f"    Interests: {', '.join(user['profile']['interests'])}\n\n"

    input_text += "Transactions:\n"
    for tx in transactions:
        input_text += f"  - {tx['transaction_id']}: {tx['type']} ${tx['amount']} {tx['currency']}\n"
        input_text += f"    User: {tx['user_id']}\n"
        input_text += f"    Description: {tx['description']}\n"
        input_text += f"    Status: {tx['status']}\n\n"

    # Create output (structured)
    output = {
        "users": users,
        "transactions": transactions,
        "summary": {
            "total_users": len(users),
            "total_transactions": len(transactions),
            "total_amount": sum(tx["amount"] for tx in transactions),
            "by_status": {
                "completed": sum(1 for tx in transactions if tx["status"] == "completed"),
                "pending": sum(1 for tx in transactions if tx["status"] == "pending"),
                "failed": sum(1 for tx in transactions if tx["status"] == "failed"),
            },
        },
    }

    return {
        "input": input_text,
        "output": output,
        "metadata": {
            "scenario": "api_call",
            "size": size,
            "num_users": len(users),
            "num_transactions": len(transactions),
        },
    }


def generate_database_dataset(
    size: Literal["small", "medium", "large"]
) -> Dict[str, Any]:
    """
    Generate database query dataset with structured results.

    Args:
        size: Dataset size (small/medium/large)

    Returns:
        Dictionary with input and output data
    """
    # Size configurations
    configs = {
        "small": {"num_products": 20, "num_orders": 30, "num_customers": 15},
        "medium": {"num_products": 100, "num_orders": 200, "num_customers": 80},
        "large": {"num_products": 500, "num_orders": 1000, "num_customers": 400},
    }

    config = configs[size]

    # Generate products
    products = []
    for i in range(config["num_products"]):
        product = {
            "product_id": f"prod_{i:06d}",
            "name": f"Product {i}: {generate_random_text(3)}",
            "description": generate_random_text(50),
            "category": random.choice(["Electronics", "Clothing", "Books", "Home", "Sports"]),
            "price": round(random.uniform(10, 500), 2),
            "stock": random.randint(0, 1000),
            "rating": round(random.uniform(1, 5), 1),
            "reviews_count": random.randint(0, 500),
        }
        products.append(product)

    # Generate customers
    customers = []
    for i in range(config["num_customers"]):
        customer = {
            "customer_id": f"cust_{i:06d}",
            "name": f"Customer {i}",
            "email": f"customer{i}@example.com",
            "phone": f"+1-555-{random.randint(1000000, 9999999)}",
            "address": f"{random.randint(1, 9999)} Main St, City {random.randint(1, 100)}",
            "join_date": f"202{random.randint(1, 4)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            "loyalty_tier": random.choice(["Bronze", "Silver", "Gold", "Platinum"]),
        }
        customers.append(customer)

    # Generate orders
    orders = []
    for i in range(config["num_orders"]):
        order_products = random.sample(products, k=random.randint(1, min(5, len(products))))
        order = {
            "order_id": f"order_{i:08d}",
            "customer_id": random.choice(customers)["customer_id"],
            "products": [
                {
                    "product_id": p["product_id"],
                    "name": p["name"],
                    "quantity": random.randint(1, 5),
                    "unit_price": p["price"],
                }
                for p in order_products
            ],
            "total": sum(p["price"] * random.randint(1, 5) for p in order_products),
            "status": random.choice(["pending", "shipped", "delivered", "cancelled"]),
            "order_date": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        }
        orders.append(order)

    # Create input context
    input_text = "Database Query Results:\n\n"
    input_text += "PRODUCTS:\n"
    for product in products:
        input_text += f"  {product['product_id']} | {product['name']} | ${product['price']:.2f} | {product['stock']} in stock\n"
        input_text += f"    Category: {product['category']} | Rating: {product['rating']}/5 ({product['reviews_count']} reviews)\n"

    input_text += "\nCUSTOMERS:\n"
    for customer in customers:
        input_text += f"  {customer['customer_id']} | {customer['name']} | {customer['email']}\n"
        input_text += f"    {customer['address']} | {customer['loyalty_tier']} member since {customer['join_date']}\n"

    input_text += "\nORDERS:\n"
    for order in orders:
        input_text += f"  {order['order_id']} | Customer: {order['customer_id']} | Total: ${order['total']:.2f}\n"
        input_text += f"    Status: {order['status']} | Date: {order['order_date']}\n"
        input_text += f"    Products: {len(order['products'])} items\n"

    # Create output (structured)
    output = {
        "products": products,
        "customers": customers,
        "orders": orders,
        "analytics": {
            "total_products": len(products),
            "total_customers": len(customers),
            "total_orders": len(orders),
            "total_revenue": sum(order["total"] for order in orders),
            "avg_order_value": sum(order["total"] for order in orders) / len(orders) if orders else 0,
            "products_by_category": {
                cat: len([p for p in products if p["category"] == cat])
                for cat in set(p["category"] for p in products)
            },
            "orders_by_status": {
                status: len([o for o in orders if o["status"] == status])
                for status in set(o["status"] for o in orders)
            },
        },
    }

    return {
        "input": input_text,
        "output": output,
        "metadata": {
            "scenario": "database_query",
            "size": size,
            "num_products": len(products),
            "num_customers": len(customers),
            "num_orders": len(orders),
        },
    }


if __name__ == "__main__":
    """Test dataset generation."""
    print("Generating test datasets...\n")

    for size in ["small", "medium", "large"]:
        print(f"\n{'='*60}")
        print(f"Size: {size.upper()}")
        print('='*60)

        # RAG dataset
        rag = generate_rag_dataset(size)
        print(f"\nRAG Dataset:")
        print(f"  Input length: {len(rag['input'])} chars")
        print(f"  Documents: {rag['metadata']['num_documents']}")
        print(f"  Output keys: {list(rag['output'].keys())}")

        # API dataset
        api = generate_api_dataset(size)
        print(f"\nAPI Dataset:")
        print(f"  Input length: {len(api['input'])} chars")
        print(f"  Users: {api['metadata']['num_users']}")
        print(f"  Transactions: {api['metadata']['num_transactions']}")

        # Database dataset
        db = generate_database_dataset(size)
        print(f"\nDatabase Dataset:")
        print(f"  Input length: {len(db['input'])} chars")
        print(f"  Products: {db['metadata']['num_products']}")
        print(f"  Customers: {db['metadata']['num_customers']}")
        print(f"  Orders: {db['metadata']['num_orders']}")
