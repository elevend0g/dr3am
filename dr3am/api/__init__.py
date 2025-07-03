"""API integrations for dr3am"""

from .search_providers import WebSearchModule
from .shopping_apis import ShoppingModule
from .news_apis import NewsModule

__all__ = [
    "WebSearchModule",
    "ShoppingModule",
    "NewsModule",
]