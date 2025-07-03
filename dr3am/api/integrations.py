import asyncio
import aiohttp
import json
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from urllib.parse import quote_plus, urljoin
import re
import hashlib
from abc import ABC, abstractmethod

# Real API response models
@dataclass
class APIResponse:
    """Standardized API response container"""
    success: bool
    data: Any
    error: Optional[str] = None
    rate_limit_remaining: Optional[int] = None
    cost: Optional[float] = None  # API call cost tracking
    cache_key: Optional[str] = None


class APIRateLimiter:
    """Advanced rate limiting for API calls"""
    
    def __init__(self, calls_per_minute: int = 60, calls_per_day: int = 1000):
        self.calls_per_minute = calls_per_minute
        self.calls_per_day = calls_per_day
        self.minute_calls: List[datetime] = []
        self.day_calls: List[datetime] = []
        
    async def acquire(self) -> bool:
        """Acquire permission to make an API call"""
        now = datetime.now()
        
        # Clean old entries
        self.minute_calls = [call for call in self.minute_calls if now - call < timedelta(minutes=1)]
        self.day_calls = [call for call in self.day_calls if now - call < timedelta(days=1)]
        
        # Check limits
        if len(self.minute_calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.minute_calls[0]).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                return await self.acquire()
        
        if len(self.day_calls) >= self.calls_per_day:
            sleep_time = 86400 - (now - self.day_calls[0]).total_seconds()
            if sleep_time > 0:
                logging.warning(f"Daily API limit reached, sleeping for {sleep_time:.0f} seconds")
                await asyncio.sleep(min(sleep_time, 3600))  # Max 1 hour sleep
                return False
        
        # Record the call
        self.minute_calls.append(now)
        self.day_calls.append(now)
        return True


class RealWebSearchModule:
    """Real web search implementation with multiple providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = self._initialize_providers()
        self.cache: Dict[str, APIResponse] = {}
        self.cache_ttl = timedelta(hours=6)  # Cache search results for 6 hours
        
    def _initialize_providers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize search providers based on config"""
        providers = {}
        
        # Google Custom Search
        if self.config.get("google_api_key") and self.config.get("google_search_engine_id"):
            providers["google"] = {
                "api_key": self.config["google_api_key"],
                "search_engine_id": self.config["google_search_engine_id"],
                "rate_limiter": APIRateLimiter(100, 10000),  # Google limits
                "cost_per_call": 0.005,  # $5 per 1000 calls
                "priority": 1
            }
        
        # Bing Search API
        if self.config.get("bing_api_key"):
            providers["bing"] = {
                "api_key": self.config["bing_api_key"],
                "rate_limiter": APIRateLimiter(1000, 100000),  # Bing limits
                "cost_per_call": 0.004,
                "priority": 2
            }
        
        # SerpAPI (Google/Bing proxy)
        if self.config.get("serpapi_key"):
            providers["serpapi"] = {
                "api_key": self.config["serpapi_key"],
                "rate_limiter": APIRateLimiter(100, 5000),
                "cost_per_call": 0.01,
                "priority": 3
            }
        
        # DuckDuckGo (free, no API key required)
        providers["duckduckgo"] = {
            "rate_limiter": APIRateLimiter(30, 1000),  # Conservative for free service
            "cost_per_call": 0.0,
            "priority": 4
        }
        
        return providers
    
    async def search(self, query: str, max_results: int = 10) -> APIResponse:
        """Execute search with fallback across providers"""
        
        # Check cache first
        cache_key = hashlib.md5(f"{query}_{max_results}".encode()).hexdigest()
        cached_response = self.cache.get(cache_key)
        if cached_response and datetime.now() - cached_response.data.get("timestamp", datetime.min) < self.cache_ttl:
            logging.info(f"Returning cached search results for: {query}")
            return cached_response
        
        # Try providers in priority order
        sorted_providers = sorted(self.providers.items(), key=lambda x: x[1]["priority"])
        
        for provider_name, provider_config in sorted_providers:
            try:
                # Check rate limits
                if not await provider_config["rate_limiter"].acquire():
                    logging.warning(f"Rate limit exceeded for {provider_name}")
                    continue
                
                # Execute search
                if provider_name == "google":
                    response = await self._google_search(query, max_results, provider_config)
                elif provider_name == "bing":
                    response = await self._bing_search(query, max_results, provider_config)
                elif provider_name == "serpapi":
                    response = await self._serpapi_search(query, max_results, provider_config)
                elif provider_name == "duckduckgo":
                    response = await self._duckduckgo_search(query, max_results, provider_config)
                else:
                    continue
                
                if response.success:
                    # Cache successful response
                    response.cache_key = cache_key
                    response.data["timestamp"] = datetime.now()
                    response.data["provider"] = provider_name
                    self.cache[cache_key] = response
                    
                    logging.info(f"Search successful using {provider_name}: {query}")
                    return response
                
            except Exception as e:
                logging.error(f"Search failed with {provider_name}: {e}")
                continue
        
        # All providers failed
        return APIResponse(
            success=False,
            data=[],
            error="All search providers failed or rate limited"
        )
    
    async def _google_search(self, query: str, max_results: int, config: Dict) -> APIResponse:
        """Google Custom Search API implementation"""
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": config["api_key"],
            "cx": config["search_engine_id"],
            "q": query,
            "num": min(max_results, 10),  # Google max is 10
            "safe": "moderate"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse Google results
                    results = []
                    for item in data.get("items", []):
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "description": item.get("snippet", ""),
                            "domain": self._extract_domain(item.get("link", "")),
                            "published_date": self._extract_date(item.get("pagemap", {})),
                            "relevance_score": self._calculate_google_relevance(item, query)
                        })
                    
                    return APIResponse(
                        success=True,
                        data=results,
                        cost=config["cost_per_call"]
                    )
                
                elif response.status == 429:
                    return APIResponse(success=False, error="Google API rate limited")
                else:
                    error_data = await response.text()
                    return APIResponse(success=False, error=f"Google API error: {response.status} - {error_data}")
    
    async def _bing_search(self, query: str, max_results: int, config: Dict) -> APIResponse:
        """Bing Search API implementation"""
        
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {
            "Ocp-Apim-Subscription-Key": config["api_key"]
        }
        params = {
            "q": query,
            "count": min(max_results, 50),  # Bing max is 50
            "safeSearch": "Moderate",
            "textFormat": "HTML"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse Bing results
                    results = []
                    for item in data.get("webPages", {}).get("value", []):
                        results.append({
                            "title": item.get("name", ""),
                            "url": item.get("url", ""),
                            "description": item.get("snippet", ""),
                            "domain": self._extract_domain(item.get("url", "")),
                            "published_date": item.get("dateLastCrawled"),
                            "relevance_score": self._calculate_bing_relevance(item, query)
                        })
                    
                    return APIResponse(
                        success=True,
                        data=results,
                        cost=config["cost_per_call"]
                    )
                
                else:
                    error_data = await response.text()
                    return APIResponse(success=False, error=f"Bing API error: {response.status} - {error_data}")
    
    async def _serpapi_search(self, query: str, max_results: int, config: Dict) -> APIResponse:
        """SerpAPI implementation (Google proxy)"""
        
        url = "https://serpapi.com/search"
        params = {
            "api_key": config["api_key"],
            "engine": "google",
            "q": query,
            "num": min(max_results, 100),
            "safe": "active"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse SerpAPI results
                    results = []
                    for item in data.get("organic_results", []):
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "description": item.get("snippet", ""),
                            "domain": self._extract_domain(item.get("link", "")),
                            "published_date": item.get("date"),
                            "relevance_score": item.get("position", 100) / 100  # Convert position to relevance
                        })
                    
                    return APIResponse(
                        success=True,
                        data=results,
                        cost=config["cost_per_call"]
                    )
                
                else:
                    error_data = await response.text()
                    return APIResponse(success=False, error=f"SerpAPI error: {response.status} - {error_data}")
    
    async def _duckduckgo_search(self, query: str, max_results: int, config: Dict) -> APIResponse:
        """DuckDuckGo search implementation (free)"""
        
        # DuckDuckGo Instant Answer API (limited but free)
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse DDG results (limited data available)
                    results = []
                    
                    # Abstract (main result)
                    if data.get("Abstract"):
                        results.append({
                            "title": data.get("Heading", query.title()),
                            "url": data.get("AbstractURL", ""),
                            "description": data.get("Abstract", ""),
                            "domain": self._extract_domain(data.get("AbstractURL", "")),
                            "published_date": None,
                            "relevance_score": 0.9
                        })
                    
                    # Related topics
                    for topic in data.get("RelatedTopics", [])[:max_results-1]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append({
                                "title": topic.get("Text", "")[:100] + "...",
                                "url": topic.get("FirstURL", ""),
                                "description": topic.get("Text", ""),
                                "domain": self._extract_domain(topic.get("FirstURL", "")),
                                "published_date": None,
                                "relevance_score": 0.7
                            })
                    
                    return APIResponse(
                        success=True,
                        data=results,
                        cost=0.0
                    )
                
                else:
                    return APIResponse(success=False, error=f"DuckDuckGo API error: {response.status}")
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return ""
    
    def _extract_date(self, pagemap: Dict) -> Optional[str]:
        """Extract publication date from Google pagemap"""
        try:
            # Try various date fields
            date_fields = ["datePublished", "dateCreated", "dateModified"]
            for field in date_fields:
                if field in pagemap.get("metatags", [{}])[0]:
                    return pagemap["metatags"][0][field]
            return None
        except:
            return None
    
    def _calculate_google_relevance(self, item: Dict, query: str) -> float:
        """Calculate relevance score for Google results"""
        score = 0.5  # Base score
        
        title = item.get("title", "").lower()
        snippet = item.get("snippet", "").lower()
        query_lower = query.lower()
        
        # Title match boost
        if query_lower in title:
            score += 0.3
        
        # Snippet relevance
        query_words = query_lower.split()
        snippet_words = snippet.split()
        matches = sum(1 for word in query_words if word in snippet_words)
        score += (matches / len(query_words)) * 0.2
        
        return min(score, 1.0)
    
    def _calculate_bing_relevance(self, item: Dict, query: str) -> float:
        """Calculate relevance score for Bing results"""
        # Similar to Google but with Bing-specific adjustments
        return self._calculate_google_relevance(item, query)


class RealShoppingModule:
    """Real shopping API integration with multiple providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = self._initialize_shopping_providers()
        self.cache: Dict[str, APIResponse] = {}
        self.cache_ttl = timedelta(hours=2)  # Shopping data changes frequently
    
    def _initialize_shopping_providers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize shopping API providers"""
        providers = {}
        
        # Amazon Product Advertising API
        if all(key in self.config for key in ["amazon_access_key", "amazon_secret_key", "amazon_partner_tag"]):
            providers["amazon"] = {
                "access_key": self.config["amazon_access_key"],
                "secret_key": self.config["amazon_secret_key"],
                "partner_tag": self.config["amazon_partner_tag"],
                "rate_limiter": APIRateLimiter(8000, 864000),  # Amazon limits
                "priority": 1
            }
        
        # eBay API
        if self.config.get("ebay_app_id"):
            providers["ebay"] = {
                "app_id": self.config["ebay_app_id"],
                "rate_limiter": APIRateLimiter(5000, 100000),
                "priority": 2
            }
        
        # Walmart API
        if self.config.get("walmart_api_key"):
            providers["walmart"] = {
                "api_key": self.config["walmart_api_key"],
                "rate_limiter": APIRateLimiter(1000, 50000),
                "priority": 3
            }
        
        # RapidAPI shopping aggregators
        if self.config.get("rapidapi_key"):
            providers["rapidapi"] = {
                "api_key": self.config["rapidapi_key"],
                "rate_limiter": APIRateLimiter(500, 10000),
                "priority": 4
            }
        
        return providers
    
    async def search_products(self, query: str, max_results: int = 20) -> APIResponse:
        """Search for products across providers"""
        
        cache_key = hashlib.md5(f"products_{query}_{max_results}".encode()).hexdigest()
        cached_response = self.cache.get(cache_key)
        if cached_response and datetime.now() - cached_response.data.get("timestamp", datetime.min) < self.cache_ttl:
            return cached_response
        
        all_results = []
        
        # Try multiple providers and aggregate results
        for provider_name, provider_config in self.providers.items():
            try:
                if not await provider_config["rate_limiter"].acquire():
                    continue
                
                if provider_name == "amazon":
                    response = await self._amazon_search(query, max_results // 2, provider_config)
                elif provider_name == "ebay":
                    response = await self._ebay_search(query, max_results // 2, provider_config)
                elif provider_name == "walmart":
                    response = await self._walmart_search(query, max_results // 2, provider_config)
                elif provider_name == "rapidapi":
                    response = await self._rapidapi_shopping_search(query, max_results // 2, provider_config)
                else:
                    continue
                
                if response.success:
                    # Add provider info to each result
                    for result in response.data:
                        result["provider"] = provider_name
                    all_results.extend(response.data)
                
            except Exception as e:
                logging.error(f"Shopping search failed with {provider_name}: {e}")
                continue
        
        # Sort by relevance and price
        all_results.sort(key=lambda x: (-x.get("relevance_score", 0), x.get("price_numeric", float('inf'))))
        
        # Cache and return
        final_response = APIResponse(
            success=len(all_results) > 0,
            data=all_results[:max_results],
            cache_key=cache_key
        )
        
        if final_response.success:
            final_response.data = {"results": all_results[:max_results], "timestamp": datetime.now()}
            self.cache[cache_key] = final_response
        
        return final_response
    
    async def find_deals(self, query: str, min_discount: float = 0.2) -> APIResponse:
        """Find current deals and sales"""
        
        # Use general product search then filter for deals
        product_response = await self.search_products(query, 50)
        
        if not product_response.success:
            return product_response
        
        deals = []
        for product in product_response.data.get("results", []):
            # Check for discount indicators
            discount = self._calculate_discount(product)
            if discount >= min_discount:
                deal = {
                    "title": product.get("title", ""),
                    "original_price": product.get("original_price"),
                    "sale_price": product.get("price"),
                    "discount_percent": f"{discount*100:.0f}%",
                    "savings": product.get("savings"),
                    "url": product.get("url", ""),
                    "provider": product.get("provider", ""),
                    "expires": self._estimate_deal_expiry(product),
                    "relevance_score": product.get("relevance_score", 0.5)
                }
                deals.append(deal)
        
        return APIResponse(
            success=True,
            data=deals
        )
    
    async def _amazon_search(self, query: str, max_results: int, config: Dict) -> APIResponse:
        """Amazon Product Advertising API search"""
        
        # Note: Amazon API requires complex authentication (HMAC-SHA256)
        # This is a simplified example - real implementation needs proper auth
        
        try:
            # For demo purposes, using a mock response structure
            # Real implementation would use boto3 or similar
            
            results = [
                {
                    "title": f"Amazon: {query} - Premium Quality",
                    "price": "$29.99",
                    "price_numeric": 29.99,
                    "original_price": "$39.99",
                    "url": f"https://amazon.com/dp/MOCK{hash(query) % 1000}",
                    "image_url": "https://via.placeholder.com/200x200",
                    "rating": 4.3,
                    "reviews_count": 1247,
                    "prime_eligible": True,
                    "relevance_score": 0.9
                }
            ]
            
            return APIResponse(success=True, data=results)
            
        except Exception as e:
            return APIResponse(success=False, error=f"Amazon API error: {e}")
    
    async def _ebay_search(self, query: str, max_results: int, config: Dict) -> APIResponse:
        """eBay API search implementation"""
        
        url = "https://svcs.ebay.com/services/search/FindingService/v1"
        params = {
            "OPERATION-NAME": "findItemsByKeywords",
            "SERVICE-VERSION": "1.0.0",
            "SECURITY-APPNAME": config["app_id"],
            "RESPONSE-DATA-FORMAT": "JSON",
            "keywords": query,
            "paginationInput.entriesPerPage": max_results
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        items = data.get("findItemsByKeywordsResponse", [{}])[0].get("searchResult", [{}])[0].get("item", [])
                        
                        for item in items:
                            results.append({
                                "title": item.get("title", [""])[0],
                                "price": item.get("sellingStatus", [{}])[0].get("currentPrice", [{}])[0].get("__value__", "0"),
                                "price_numeric": float(item.get("sellingStatus", [{}])[0].get("currentPrice", [{}])[0].get("__value__", "0")),
                                "url": item.get("viewItemURL", [""])[0],
                                "image_url": item.get("galleryURL", [""])[0],
                                "condition": item.get("condition", [{}])[0].get("conditionDisplayName", [""])[0],
                                "shipping_cost": item.get("shippingInfo", [{}])[0].get("shippingServiceCost", [{}])[0].get("__value__", "0"),
                                "relevance_score": 0.8
                            })
                        
                        return APIResponse(success=True, data=results)
                    
                    else:
                        return APIResponse(success=False, error=f"eBay API error: {response.status}")
        
        except Exception as e:
            return APIResponse(success=False, error=f"eBay API error: {e}")
    
    async def _walmart_search(self, query: str, max_results: int, config: Dict) -> APIResponse:
        """Walmart API search implementation"""
        
        url = "https://developer.api.walmart.com/api-proxy/service/affil/product/v2/search"
        headers = {
            "WM_SVC.NAME": "Walmart Marketplace",
            "WM_QOS.CORRELATION_ID": f"dr3am_{datetime.now().isoformat()}",
            "Authorization": f"Basic {config['api_key']}"
        }
        params = {
            "query": query,
            "numItems": max_results
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        for item in data.get("items", []):
                            results.append({
                                "title": item.get("name", ""),
                                "price": f"${item.get('salePrice', 0):.2f}",
                                "price_numeric": item.get("salePrice", 0),
                                "original_price": f"${item.get('msrp', 0):.2f}" if item.get("msrp") else None,
                                "url": item.get("productUrl", ""),
                                "image_url": item.get("mediumImage", ""),
                                "rating": item.get("customerRating", 0),
                                "reviews_count": item.get("numReviews", 0),
                                "free_shipping": item.get("freeShippingOver35", False),
                                "relevance_score": 0.85
                            })
                        
                        return APIResponse(success=True, data=results)
                    
                    else:
                        return APIResponse(success=False, error=f"Walmart API error: {response.status}")
        
        except Exception as e:
            return APIResponse(success=False, error=f"Walmart API error: {e}")
    
    async def _rapidapi_shopping_search(self, query: str, max_results: int, config: Dict) -> APIResponse:
        """RapidAPI shopping aggregator search"""
        
        # Example using Real-Time Product Search API on RapidAPI
        url = "https://real-time-product-search.p.rapidapi.com/search"
        headers = {
            "X-RapidAPI-Key": config["api_key"],
            "X-RapidAPI-Host": "real-time-product-search.p.rapidapi.com"
        }
        params = {
            "q": query,
            "limit": max_results
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        for item in data.get("data", []):
                            results.append({
                                "title": item.get("product_title", ""),
                                "price": item.get("offer", {}).get("price", ""),
                                "price_numeric": self._parse_price(item.get("offer", {}).get("price", "0")),
                                "url": item.get("product_url", ""),
                                "image_url": item.get("product_photos", [{}])[0].get("url", ""),
                                "store": item.get("offer", {}).get("store_name", ""),
                                "rating": item.get("product_rating", 0),
                                "relevance_score": 0.75
                            })
                        
                        return APIResponse(success=True, data=results)
                    
                    else:
                        return APIResponse(success=False, error=f"RapidAPI error: {response.status}")
        
        except Exception as e:
            return APIResponse(success=False, error=f"RapidAPI error: {e}")
    
    def _calculate_discount(self, product: Dict) -> float:
        """Calculate discount percentage for a product"""
        original_price = self._parse_price(product.get("original_price", "0"))
        current_price = self._parse_price(product.get("price", "0"))
        
        if original_price > current_price > 0:
            return (original_price - current_price) / original_price
        return 0.0
    
    def _parse_price(self, price_str: str) -> float:
        """Parse price string to float"""
        try:
            # Remove currency symbols and convert to float
            clean_price = re.sub(r'[^\d.]', '', str(price_str))
            return float(clean_price) if clean_price else 0.0
        except:
            return 0.0
    
    def _estimate_deal_expiry(self, product: Dict) -> Optional[str]:
        """Estimate when a deal might expire"""
        # Simple heuristic - most deals last 1-7 days
        expiry = datetime.now() + timedelta(days=3)
        return expiry.isoformat()


class RealNewsModule:
    """Real news and content monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = self._initialize_news_providers()
        self.cache: Dict[str, APIResponse] = {}
        self.cache_ttl = timedelta(hours=1)  # News changes frequently
    
    def _initialize_news_providers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize news API providers"""
        providers = {}
        
        # NewsAPI
        if self.config.get("newsapi_key"):
            providers["newsapi"] = {
                "api_key": self.config["newsapi_key"],
                "rate_limiter": APIRateLimiter(1000, 1000),  # 1000 requests per day
                "priority": 1
            }
        
        # Reddit RSS (free)
        providers["reddit"] = {
            "rate_limiter": APIRateLimiter(60, 2000),
            "priority": 2
        }
        
        # HackerNews API (free)
        providers["hackernews"] = {
            "rate_limiter": APIRateLimiter(120, 10000),
            "priority": 3
        }
        
        return providers
    
    async def search_news(self, query: str, max_articles: int = 20) -> APIResponse:
        """Search for news articles related to query"""
        
        cache_key = hashlib.md5(f"news_{query}_{max_articles}".encode()).hexdigest()
        cached_response = self.cache.get(cache_key)
        if cached_response and datetime.now() - cached_response.data.get("timestamp", datetime.min) < self.cache_ttl:
            return cached_response
        
        all_articles = []
        
        for provider_name, provider_config in self.providers.items():
            try:
                if not await provider_config["rate_limiter"].acquire():
                    continue
                
                if provider_name == "newsapi":
                    response = await self._newsapi_search(query, max_articles // 2, provider_config)
                elif provider_name == "reddit":
                    response = await self._reddit_search(query, max_articles // 3, provider_config)
                elif provider_name == "hackernews":
                    response = await self._hackernews_search(query, max_articles // 3, provider_config)
                else:
                    continue
                
                if response.success:
                    for article in response.data:
                        article["provider"] = provider_name
                    all_articles.extend(response.data)
                
            except Exception as e:
                logging.error(f"News search failed with {provider_name}: {e}")
                continue
        
        # Sort by publication date and relevance
        all_articles.sort(key=lambda x: (x.get("published_at", ""), -x.get("relevance_score", 0)), reverse=True)
        
        final_response = APIResponse(
            success=len(all_articles) > 0,
            data={"articles": all_articles[:max_articles], "timestamp": datetime.now()}
        )
        
        if final_response.success:
            self.cache[cache_key] = final_response
        
        return final_response
    
    async def _newsapi_search(self, query: str, max_articles: int, config: Dict) -> APIResponse:
        """NewsAPI implementation"""
        
        url = "https://newsapi.org/v2/everything"
        headers = {
            "X-API-Key": config["api_key"]
        }
        params = {
            "q": query,
            "pageSize": max_articles,
            "sortBy": "publishedAt",
            "language": "en"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        articles = []
                        for article in data.get("articles", []):
                            articles.append({
                                "title": article.get("title", ""),
                                "description": article.get("description", ""),
                                "url": article.get("url", ""),
                                "published_at": article.get("publishedAt", ""),
                                "source": article.get("source", {}).get("name", ""),
                                "author": article.get("author", ""),
                                "image_url": article.get("urlToImage", ""),
                                "relevance_score": self._calculate_news_relevance(article, query)
                            })
                        
                        return APIResponse(success=True, data=articles)
                    
                    else:
                        return APIResponse(success=False, error=f"NewsAPI error: {response.status}")
        
        except Exception as e:
            return APIResponse(success=False, error=f"NewsAPI error: {e}")
    
    async def _reddit_search(self, query: str, max_articles: int, config: Dict) -> APIResponse:
        """Reddit search implementation"""
        
        # Search relevant subreddits based on query
        url = f"https://www.reddit.com/search.json"
        params = {
            "q": query,
            "limit": max_articles,
            "sort": "new"
        }
        
        headers = {
            "User-Agent": "dr3am/1.0 (autonomous research assistant)"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        articles = []
                        for post in data.get("data", {}).get("children", []):
                            post_data = post.get("data", {})
                            articles.append({
                                "title": post_data.get("title", ""),
                                "description": post_data.get("selftext", "")[:200] + "...",
                                "url": f"https://reddit.com{post_data.get('permalink', '')}",
                                "published_at": datetime.fromtimestamp(post_data.get("created_utc", 0)).isoformat(),
                                "source": f"r/{post_data.get('subreddit', '')}",
                                "author": post_data.get("author", ""),
                                "score": post_data.get("score", 0),
                                "relevance_score": 0.7
                            })
                        
                        return APIResponse(success=True, data=articles)
                    
                    else:
                        return APIResponse(success=False, error=f"Reddit API error: {response.status}")
        
        except Exception as e:
            return APIResponse(success=False, error=f"Reddit API error: {e}")
    
    async def _hackernews_search(self, query: str, max_articles: int, config: Dict) -> APIResponse:
        """HackerNews API search implementation"""
        
        # Use Algolia HN Search API
        url = "https://hn.algolia.com/api/v1/search"
        params = {
            "query": query,
            "hitsPerPage": max_articles,
            "tags": "story"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        articles = []
                        for hit in data.get("hits", []):
                            articles.append({
                                "title": hit.get("title", ""),
                                "description": hit.get("story_text", "")[:200] + "..." if hit.get("story_text") else "",
                                "url": hit.get("url", f"https://news.ycombinator.com/item?id={hit.get('objectID')}"),
                                "published_at": hit.get("created_at", ""),
                                "source": "Hacker News",
                                "author": hit.get("author", ""),
                                "points": hit.get("points", 0),
                                "comments_count": hit.get("num_comments", 0),
                                "relevance_score": 0.8
                            })
                        
                        return APIResponse(success=True, data=articles)
                    
                    else:
                        return APIResponse(success=False, error=f"HackerNews API error: {response.status}")
        
        except Exception as e:
            return APIResponse(success=False, error=f"HackerNews API error: {e}")
    
    def _calculate_news_relevance(self, article: Dict, query: str) -> float:
        """Calculate relevance score for news article"""
        score = 0.5
        
        title = article.get("title", "").lower()
        description = article.get("description", "").lower()
        query_lower = query.lower()
        
        if query_lower in title:
            score += 0.3
        
        if query_lower in description:
            score += 0.2
        
        # Boost for recent articles
        published_at = article.get("publishedAt", "")
        if published_at:
            try:
                pub_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                age_hours = (datetime.now() - pub_date.replace(tzinfo=None)).total_seconds() / 3600
                if age_hours < 24:  # Less than 1 day old
                    score += 0.1
            except:
                pass
        
        return min(score, 1.0)


# Configuration management for real APIs
class APIConfigManager:
    """Manage API keys and configuration securely"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "api_config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        import os
        
        # Try to load from file first
        try:
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
        except FileNotFoundError:
            file_config = {}
        
        # Merge with environment variables
        config = {
            # Search APIs
            "google_api_key": os.getenv("GOOGLE_API_KEY") or file_config.get("google_api_key"),
            "google_search_engine_id": os.getenv("GOOGLE_SEARCH_ENGINE_ID") or file_config.get("google_search_engine_id"),
            "bing_api_key": os.getenv("BING_API_KEY") or file_config.get("bing_api_key"),
            "serpapi_key": os.getenv("SERPAPI_KEY") or file_config.get("serpapi_key"),
            
            # Shopping APIs
            "amazon_access_key": os.getenv("AMAZON_ACCESS_KEY") or file_config.get("amazon_access_key"),
            "amazon_secret_key": os.getenv("AMAZON_SECRET_KEY") or file_config.get("amazon_secret_key"),
            "amazon_partner_tag": os.getenv("AMAZON_PARTNER_TAG") or file_config.get("amazon_partner_tag"),
            "ebay_app_id": os.getenv("EBAY_APP_ID") or file_config.get("ebay_app_id"),
            "walmart_api_key": os.getenv("WALMART_API_KEY") or file_config.get("walmart_api_key"),
            "rapidapi_key": os.getenv("RAPIDAPI_KEY") or file_config.get("rapidapi_key"),
            
            # News APIs
            "newsapi_key": os.getenv("NEWSAPI_KEY") or file_config.get("newsapi_key"),
            
            # Rate limiting and costs
            "daily_budget": float(os.getenv("DAILY_API_BUDGET", "10.0")),
            "enable_paid_apis": os.getenv("ENABLE_PAID_APIS", "true").lower() == "true"
        }
        
        return config
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get configuration for search APIs"""
        return {k: v for k, v in self.config.items() if k.startswith(("google_", "bing_", "serpapi_"))}
    
    def get_shopping_config(self) -> Dict[str, Any]:
        """Get configuration for shopping APIs"""
        return {k: v for k, v in self.config.items() if k.startswith(("amazon_", "ebay_", "walmart_", "rapidapi_"))}
    
    def get_news_config(self) -> Dict[str, Any]:
        """Get configuration for news APIs"""
        return {k: v for k, v in self.config.items() if k.startswith("newsapi_")}


# Example usage and testing
async def test_real_apis():
    """Test real API integrations"""
    
    print("🔧 Testing Real API Integrations")
    print("=" * 40)
    
    # Initialize configuration
    config_manager = APIConfigManager()
    
    # Test Web Search
    print("\n🔍 Testing Web Search APIs")
    search_module = RealWebSearchModule(config_manager.get_search_config())
    
    search_result = await search_module.search("quilting techniques for beginners", max_results=5)
    if search_result.success:
        print(f"✅ Search successful! Found {len(search_result.data)} results")
        print(f"   Provider: {search_result.data.get('provider', 'Unknown')}")
        print(f"   Cost: ${search_result.cost or 0:.4f}")
        
        for i, result in enumerate(search_result.data[:3], 1):
            print(f"   {i}. {result['title'][:60]}...")
            print(f"      {result['domain']} (Score: {result['relevance_score']:.2f})")
    else:
        print(f"❌ Search failed: {search_result.error}")
    
    # Test Shopping APIs
    print("\n🛒 Testing Shopping APIs")
    shopping_module = RealShoppingModule(config_manager.get_shopping_config())
    
    shopping_result = await shopping_module.search_products("quilting fabric", max_results=5)
    if shopping_result.success:
        products = shopping_result.data.get("results", [])
        print(f"✅ Shopping search successful! Found {len(products)} products")
        
        for i, product in enumerate(products[:3], 1):
            print(f"   {i}. {product['title'][:50]}...")
            print(f"      {product['price']} at {product.get('provider', 'Unknown')}")
    else:
        print(f"❌ Shopping search failed: {shopping_result.error}")
    
    # Test deals
    deals_result = await shopping_module.find_deals("quilting supplies", min_discount=0.1)
    if deals_result.success and deals_result.data:
        print(f"💰 Found {len(deals_result.data)} deals!")
        for deal in deals_result.data[:2]:
            print(f"   🎯 {deal['title'][:40]}... - {deal['discount_percent']} off")
    
    # Test News APIs
    print("\n📰 Testing News APIs")
    news_module = RealNewsModule(config_manager.get_news_config())
    
    news_result = await news_module.search_news("AI research 2024", max_articles=5)
    if news_result.success:
        articles = news_result.data.get("articles", [])
        print(f"✅ News search successful! Found {len(articles)} articles")
        
        for i, article in enumerate(articles[:3], 1):
            print(f"   {i}. {article['title'][:60]}...")
            print(f"      {article['source']} - {article.get('published_at', 'Unknown date')}")
    else:
        print(f"❌ News search failed: {news_result.error}")
    
    print("\n🏁 API testing completed!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run API tests
    asyncio.run(test_real_apis())
