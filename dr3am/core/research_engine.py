import asyncio
import aiohttp
import json
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from urllib.parse import quote_plus
import re


# Research result models
@dataclass
class ResearchResult:
    """Container for research findings"""
    research_id: str
    topic: str
    research_type: str
    findings: List[Dict[str, Any]]
    summary: str
    insights: List[str]
    sources: List[str]
    confidence_score: float
    research_timestamp: datetime
    expiry_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionableInsight:
    """Specific action user can take based on research"""
    action_type: str  # "purchase", "read", "visit", "learn", "monitor"
    title: str
    description: str
    url: Optional[str] = None
    urgency: str = "medium"  # "low", "medium", "high", "urgent"
    estimated_value: Optional[str] = None
    deadline: Optional[datetime] = None


@dataclass
class ResearchTask:
    """Individual research task to be executed"""
    task_id: str
    opportunity: 'ResearchOpportunity'  # From semantic analyzer
    assigned_modules: List[str]
    priority: float
    created_at: datetime
    scheduled_for: datetime
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[ResearchResult] = None


class ResearchModule(ABC):
    """Base class for pluggable research modules"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self.rate_limit_delay = self.config.get("rate_limit_delay", 1.0)
        self.last_request_time = datetime.min
    
    @abstractmethod
    async def can_handle(self, opportunity: 'ResearchOpportunity') -> bool:
        """Check if this module can handle the research opportunity"""
        pass
    
    @abstractmethod
    async def execute_research(self, opportunity: 'ResearchOpportunity') -> ResearchResult:
        """Execute the research and return findings"""
        pass
    
    async def _rate_limit(self):
        """Simple rate limiting"""
        elapsed = datetime.now() - self.last_request_time
        if elapsed.total_seconds() < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed.total_seconds())
        self.last_request_time = datetime.now()


class WebSearchModule(ResearchModule):
    """Module for web search research"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.search_api_key = self.config.get("search_api_key")
        self.search_engine_id = self.config.get("search_engine_id")
        self.max_results = self.config.get("max_results", 10)
    
    async def can_handle(self, opportunity) -> bool:
        """Can handle most general research topics"""
        return opportunity.research_type in ["find_resources", "deep_dive", "monitor"]
    
    async def execute_research(self, opportunity) -> ResearchResult:
        """Execute web search research"""
        
        # Generate search queries based on opportunity
        queries = self._generate_search_queries(opportunity)
        
        all_results = []
        sources = []
        
        async with aiohttp.ClientSession() as session:
            for query in queries:
                await self._rate_limit()
                
                search_results = await self._perform_search(session, query)
                all_results.extend(search_results)
                
                # Extract sources
                for result in search_results:
                    if result.get("link"):
                        sources.append(result["link"])
        
        # Synthesize findings
        findings = self._process_search_results(all_results)
        summary = self._generate_summary(findings, opportunity)
        insights = self._extract_insights(findings, opportunity)
        
        return ResearchResult(
            research_id=self._generate_research_id(opportunity),
            topic=opportunity.interest_topic,
            research_type=opportunity.research_type,
            findings=findings,
            summary=summary,
            insights=insights,
            sources=sources[:10],  # Limit sources
            confidence_score=self._calculate_confidence(findings),
            research_timestamp=datetime.now(),
            expiry_time=datetime.now() + timedelta(days=7),  # Web results expire in a week
            metadata={"queries_used": queries, "total_results": len(all_results)}
        )
    
    def _generate_search_queries(self, opportunity) -> List[str]:
        """Generate search queries based on research opportunity"""
        
        base_topic = opportunity.interest_topic
        
        query_templates = {
            "find_resources": [
                f"{base_topic} tutorials beginner guide",
                f"best {base_topic} resources 2024",
                f"{base_topic} community forum reddit",
                f"learn {base_topic} step by step"
            ],
            "deep_dive": [
                f"{base_topic} complete guide comprehensive",
                f"{base_topic} expert advice tips",
                f"{base_topic} latest research studies",
                f"{base_topic} best practices 2024"
            ],
            "monitor": [
                f"{base_topic} news updates 2024",
                f"{base_topic} latest developments",
                f"{base_topic} trending discussions",
                f"recent {base_topic} breakthroughs"
            ]
        }
        
        queries = query_templates.get(opportunity.research_type, [f"{base_topic} information guide"])
        return queries[:3]  # Limit to 3 queries to avoid rate limits
    
    async def _perform_search(self, session: aiohttp.ClientSession, query: str) -> List[Dict]:
        """Perform actual web search"""
        
        if self.search_api_key and self.search_engine_id:
            # Use Google Custom Search API
            return await self._google_custom_search(session, query)
        else:
            # Fallback to mock search for development
            return self._mock_search_results(query)
    
    async def _google_custom_search(self, session: aiohttp.ClientSession, query: str) -> List[Dict]:
        """Use Google Custom Search API"""
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.search_api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(self.max_results, 10)
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("items", [])
                else:
                    logging.warning(f"Search API error: {response.status}")
                    return self._mock_search_results(query)
        except Exception as e:
            logging.error(f"Search error: {e}")
            return self._mock_search_results(query)
    
    def _mock_search_results(self, query: str) -> List[Dict]:
        """Mock search results for development/testing"""
        
        return [
            {
                "title": f"Complete Guide to {query.split()[0]}",
                "link": f"https://example.com/guide-{hash(query) % 1000}",
                "snippet": f"Comprehensive guide covering everything about {query.split()[0]}...",
                "displayLink": "example.com"
            },
            {
                "title": f"Best {query.split()[0]} Resources for Beginners",
                "link": f"https://tutorial-site.com/{query.split()[0].lower()}-basics",
                "snippet": f"Top resources and tutorials for learning {query.split()[0]}...",
                "displayLink": "tutorial-site.com"
            },
            {
                "title": f"{query.split()[0]} Community Forum",
                "link": f"https://reddit.com/r/{query.split()[0].lower()}",
                "snippet": f"Active community discussing {query.split()[0]} topics...",
                "displayLink": "reddit.com"
            }
        ]
    
    def _process_search_results(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """Process and structure search results"""
        
        processed = []
        for result in results:
            processed_result = {
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "description": result.get("snippet", ""),
                "domain": result.get("displayLink", ""),
                "relevance_score": self._calculate_relevance(result),
                "content_type": self._classify_content_type(result)
            }
            processed.append(processed_result)
        
        # Sort by relevance
        processed.sort(key=lambda x: x["relevance_score"], reverse=True)
        return processed[:10]  # Top 10 results
    
    def _calculate_relevance(self, result: Dict) -> float:
        """Calculate relevance score for a search result"""
        
        score = 0.5  # Base score
        
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        domain = result.get("displayLink", "").lower()
        
        # Boost for educational/authoritative domains
        authoritative_domains = ["edu", "gov", "wikipedia", "tutorial", "guide"]
        if any(domain_type in domain for domain_type in authoritative_domains):
            score += 0.3
        
        # Boost for guide/tutorial content
        guide_keywords = ["guide", "tutorial", "how to", "beginner", "learn", "course"]
        if any(keyword in title or keyword in snippet for keyword in guide_keywords):
            score += 0.2
        
        # Boost for recent content
        if "2024" in title or "2024" in snippet:
            score += 0.1
        
        return min(score, 1.0)
    
    def _classify_content_type(self, result: Dict) -> str:
        """Classify the type of content"""
        
        title = result.get("title", "").lower()
        url = result.get("link", "").lower()
        
        if "youtube.com" in url or "video" in title:
            return "video"
        elif "reddit.com" in url or "forum" in url:
            return "community"
        elif "tutorial" in title or "guide" in title or "how" in title:
            return "tutorial"
        elif "course" in title or "edu" in url:
            return "course"
        elif "blog" in url or "article" in title:
            return "article"
        else:
            return "resource"
    
    def _generate_summary(self, findings: List[Dict], opportunity) -> str:
        """Generate summary of research findings"""
        
        if not findings:
            return f"No significant resources found for {opportunity.interest_topic}"
        
        content_types = {}
        for finding in findings:
            content_type = finding["content_type"]
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        summary_parts = [
            f"Found {len(findings)} relevant resources for {opportunity.interest_topic}."
        ]
        
        if content_types:
            type_descriptions = []
            for content_type, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
                type_descriptions.append(f"{count} {content_type}{'s' if count > 1 else ''}")
            
            summary_parts.append(f"Including {', '.join(type_descriptions)}.")
        
        # Highlight top resource
        if findings:
            top_resource = findings[0]
            summary_parts.append(f"Top resource: '{top_resource['title']}' on {top_resource['domain']}")
        
        return " ".join(summary_parts)
    
    def _extract_insights(self, findings: List[Dict], opportunity) -> List[str]:
        """Extract actionable insights from findings"""
        
        insights = []
        
        if not findings:
            insights.append(f"Consider searching for more specific {opportunity.interest_topic} terms")
            return insights
        
        # Analyze content types
        content_types = {}
        high_quality_sources = []
        
        for finding in findings:
            content_type = finding["content_type"]
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            if finding["relevance_score"] > 0.8:
                high_quality_sources.append(finding)
        
        # Generate insights based on findings
        if "tutorial" in content_types and content_types["tutorial"] >= 2:
            insights.append(f"Multiple tutorial resources available - good for hands-on learning")
        
        if "community" in content_types:
            insights.append(f"Active community discussions found - good for getting help and tips")
        
        if "video" in content_types:
            insights.append(f"Video content available - helpful for visual learning")
        
        if high_quality_sources:
            insights.append(f"Found {len(high_quality_sources)} high-quality authoritative sources")
        
        # Check for learning progression
        beginner_content = sum(1 for f in findings if "beginner" in f["title"].lower() or "basic" in f["title"].lower())
        if beginner_content >= 2:
            insights.append("Good beginner-friendly resources available")
        
        return insights
    
    def _calculate_confidence(self, findings: List[Dict]) -> float:
        """Calculate confidence in research results"""
        
        if not findings:
            return 0.2
        
        # Base confidence from number of results
        base_confidence = min(len(findings) / 10, 0.7)
        
        # Boost for high-relevance results
        high_relevance_count = sum(1 for f in findings if f["relevance_score"] > 0.7)
        relevance_boost = min(high_relevance_count / 5, 0.2)
        
        # Boost for diverse content types
        content_types = set(f["content_type"] for f in findings)
        diversity_boost = min(len(content_types) / 6, 0.1)
        
        return min(base_confidence + relevance_boost + diversity_boost, 0.95)
    
    def _generate_research_id(self, opportunity) -> str:
        """Generate unique research ID"""
        unique_string = f"{opportunity.interest_topic}_{opportunity.research_type}_{datetime.now().date()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]


class ShoppingResearchModule(ResearchModule):
    """Module for shopping and deal research"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.deal_apis = self.config.get("deal_apis", [])
        self.price_tracking = self.config.get("enable_price_tracking", True)
    
    async def can_handle(self, opportunity) -> bool:
        """Handle shopping-related research"""
        shopping_keywords = ["buy", "purchase", "price", "deal", "sale", "store", "product"]
        topic_lower = opportunity.interest_topic.lower()
        return any(keyword in topic_lower for keyword in shopping_keywords)
    
    async def execute_research(self, opportunity) -> ResearchResult:
        """Execute shopping research"""
        
        findings = []
        sources = []
        
        # Search for deals and products
        async with aiohttp.ClientSession() as session:
            # Mock shopping research - would integrate with actual APIs
            deals = await self._find_deals(session, opportunity.interest_topic)
            price_comparisons = await self._compare_prices(session, opportunity.interest_topic)
            
            findings.extend(deals)
            findings.extend(price_comparisons)
        
        # Generate insights specific to shopping
        insights = self._generate_shopping_insights(findings, opportunity)
        summary = self._generate_shopping_summary(findings, opportunity)
        
        return ResearchResult(
            research_id=self._generate_research_id(opportunity),
            topic=opportunity.interest_topic,
            research_type="shopping_research",
            findings=findings,
            summary=summary,
            insights=insights,
            sources=sources,
            confidence_score=self._calculate_confidence(findings),
            research_timestamp=datetime.now(),
            expiry_time=datetime.now() + timedelta(days=1),  # Shopping results expire quickly
            metadata={"price_tracking_enabled": self.price_tracking}
        )
    
    async def _find_deals(self, session: aiohttp.ClientSession, topic: str) -> List[Dict]:
        """Find current deals and sales"""
        
        # Mock implementation - would use real deal APIs
        await asyncio.sleep(0.5)  # Simulate API call
        
        return [
            {
                "type": "deal",
                "title": f"{topic.title()} Sale - 40% Off",
                "store": "Example Store",
                "discount": "40%",
                "original_price": "$100.00",
                "sale_price": "$60.00",
                "url": f"https://example-store.com/{topic.lower()}-sale",
                "expires": (datetime.now() + timedelta(days=3)).isoformat(),
                "relevance_score": 0.9
            },
            {
                "type": "deal",
                "title": f"Premium {topic.title()} Bundle",
                "store": "Quality Shop",
                "discount": "25%",
                "original_price": "$200.00", 
                "sale_price": "$150.00",
                "url": f"https://quality-shop.com/{topic.lower()}-bundle",
                "expires": (datetime.now() + timedelta(days=7)).isoformat(),
                "relevance_score": 0.8
            }
        ]
    
    async def _compare_prices(self, session: aiohttp.ClientSession, topic: str) -> List[Dict]:
        """Compare prices across retailers"""
        
        await asyncio.sleep(0.3)
        
        return [
            {
                "type": "price_comparison",
                "product": f"Basic {topic.title()} Kit",
                "prices": [
                    {"store": "Store A", "price": "$45.99", "rating": 4.5},
                    {"store": "Store B", "price": "$42.99", "rating": 4.2},
                    {"store": "Store C", "price": "$48.99", "rating": 4.8}
                ],
                "best_deal": {"store": "Store B", "price": "$42.99", "savings": "$3.00"},
                "relevance_score": 0.85
            }
        ]
    
    def _generate_shopping_insights(self, findings: List[Dict], opportunity) -> List[str]:
        """Generate shopping-specific insights"""
        
        insights = []
        
        # Analyze deals
        deals = [f for f in findings if f.get("type") == "deal"]
        if deals:
            best_deal = max(deals, key=lambda x: float(x.get("discount", "0%").rstrip("%")))
            insights.append(f"Best current deal: {best_deal['discount']} off at {best_deal['store']}")
            
            # Check for expiring deals
            urgent_deals = []
            for deal in deals:
                if deal.get("expires"):
                    expires = datetime.fromisoformat(deal["expires"])
                    if expires <= datetime.now() + timedelta(days=2):
                        urgent_deals.append(deal)
            
            if urgent_deals:
                insights.append(f"{len(urgent_deals)} deals expire within 2 days - act quickly")
        
        # Analyze price comparisons
        price_comparisons = [f for f in findings if f.get("type") == "price_comparison"]
        if price_comparisons:
            for comparison in price_comparisons:
                best_deal = comparison.get("best_deal", {})
                if best_deal:
                    insights.append(f"Best price found: {best_deal['price']} at {best_deal['store']}")
        
        return insights
    
    def _generate_shopping_summary(self, findings: List[Dict], opportunity) -> str:
        """Generate shopping research summary"""
        
        deals_count = len([f for f in findings if f.get("type") == "deal"])
        comparisons_count = len([f for f in findings if f.get("type") == "price_comparison"])
        
        summary_parts = [f"Shopping research for {opportunity.interest_topic}:"]
        
        if deals_count:
            summary_parts.append(f"Found {deals_count} active deals")
        
        if comparisons_count:
            summary_parts.append(f"Compared prices across {comparisons_count} product categories")
        
        # Highlight best opportunity
        deals = [f for f in findings if f.get("type") == "deal"]
        if deals:
            best_deal = max(deals, key=lambda x: float(x.get("discount", "0%").rstrip("%")))
            summary_parts.append(f"Top opportunity: {best_deal['discount']} off at {best_deal['store']}")
        
        return ". ".join(summary_parts) + "."


class HealthResearchModule(ResearchModule):
    """Module for health and wellness research"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.medical_apis = self.config.get("medical_apis", [])
        self.include_experimental = self.config.get("include_experimental", False)
    
    async def can_handle(self, opportunity) -> bool:
        """Handle health-related research"""
        health_keywords = ["pain", "health", "wellness", "medical", "symptom", "treatment", "therapy", "exercise", "diet"]
        topic_lower = opportunity.interest_topic.lower()
        return any(keyword in topic_lower for keyword in health_keywords)
    
    async def execute_research(self, opportunity) -> ResearchResult:
        """Execute health research with appropriate disclaimers"""
        
        findings = []
        
        # Health research with disclaimers
        general_info = await self._find_general_health_info(opportunity.interest_topic)
        professional_resources = await self._find_professional_resources(opportunity.interest_topic)
        
        findings.extend(general_info)
        findings.extend(professional_resources)
        
        insights = self._generate_health_insights(findings, opportunity)
        summary = self._generate_health_summary(findings, opportunity)
        
        return ResearchResult(
            research_id=self._generate_research_id(opportunity),
            topic=opportunity.interest_topic,
            research_type="health_research",
            findings=findings,
            summary=summary,
            insights=insights,
            sources=[],
            confidence_score=self._calculate_confidence(findings),
            research_timestamp=datetime.now(),
            expiry_time=datetime.now() + timedelta(days=30),  # Health info stays relevant longer
            metadata={
                "disclaimer": "This information is for educational purposes only and should not replace professional medical advice",
                "recommend_professional_consultation": True
            }
        )
    
    async def _find_general_health_info(self, topic: str) -> List[Dict]:
        """Find general health information"""
        
        await asyncio.sleep(0.5)
        
        return [
            {
                "type": "general_info",
                "title": f"Understanding {topic.title()}",
                "content": f"General information about {topic} including common causes and management approaches.",
                "source": "Health Education Resource",
                "credibility": "high",
                "last_updated": "2024",
                "relevance_score": 0.9
            }
        ]
    
    async def _find_professional_resources(self, topic: str) -> List[Dict]:
        """Find professional healthcare resources"""
        
        await asyncio.sleep(0.3)
        
        return [
            {
                "type": "professional_resource",
                "title": f"Find {topic.title()} Specialists",
                "description": f"Directory of healthcare professionals specializing in {topic}",
                "resource_type": "provider_directory",
                "url": f"https://healthcare-directory.com/{topic.lower()}-specialists",
                "relevance_score": 0.95
            }
        ]
    
    def _generate_health_insights(self, findings: List[Dict], opportunity) -> List[str]:
        """Generate health-specific insights with appropriate disclaimers"""
        
        insights = [
            "⚠️ Always consult with healthcare professionals for medical concerns",
            "Educational resources found for general understanding"
        ]
        
        professional_resources = [f for f in findings if f.get("type") == "professional_resource"]
        if professional_resources:
            insights.append("Professional healthcare providers available in your area")
        
        return insights
    
    def _generate_health_summary(self, findings: List[Dict], opportunity) -> str:
        """Generate health research summary with disclaimers"""
        
        return f"Health research for {opportunity.interest_topic}: Found educational resources and professional provider information. Remember: this is for informational purposes only - consult healthcare professionals for medical advice."


class ResearchOrchestrator:
    """Orchestrates research execution across multiple modules"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.modules: List[ResearchModule] = []
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, ResearchTask] = {}
        self.completed_research: Dict[str, ResearchResult] = {}
        self.running = False
        
        # Performance tracking
        self.research_stats = {
            "total_completed": 0,
            "total_failed": 0,
            "average_execution_time": 0.0,
            "module_usage": {}
        }
        
        # Initialize default modules
        self._initialize_default_modules()
    
    def _initialize_default_modules(self):
        """Initialize default research modules"""
        
        web_config = self.config.get("web_search", {})
        shopping_config = self.config.get("shopping", {})
        health_config = self.config.get("health", {})
        
        self.register_module(WebSearchModule(web_config))
        self.register_module(ShoppingResearchModule(shopping_config))
        self.register_module(HealthResearchModule(health_config))
    
    def register_module(self, module: ResearchModule):
        """Register a new research module"""
        self.modules.append(module)
        self.research_stats["module_usage"][module.name] = 0
        logging.info(f"Registered research module: {module.name}")
    
    async def start_orchestrator(self):
        """Start the research orchestrator background loop"""
        self.running = True
        
        # Start background task processor
        asyncio.create_task(self._process_task_queue())
        logging.info("Research orchestrator started")
    
    async def stop_orchestrator(self):
        """Stop the research orchestrator"""
        self.running = False
        logging.info("Research orchestrator stopped")
    
    async def schedule_research(self, opportunity: 'ResearchOpportunity', priority: float = 0.5, delay: timedelta = None) -> str:
        """Schedule research for execution"""
        
        task_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(opportunity.interest_topic) % 1000}"
        
        # Find appropriate modules
        assigned_modules = []
        for module in self.modules:
            if await module.can_handle(opportunity):
                assigned_modules.append(module.name)
        
        if not assigned_modules:
            assigned_modules = ["WebSearchModule"]  # Fallback to web search
        
        # Create research task
        scheduled_time = datetime.now()
        if delay:
            scheduled_time += delay
        
        task = ResearchTask(
            task_id=task_id,
            opportunity=opportunity,
            assigned_modules=assigned_modules,
            priority=priority,
            created_at=datetime.now(),
            scheduled_for=scheduled_time
        )
        
        # Add to queue
        await self.task_queue.put(task)
        self.active_tasks[task_id] = task
        
        logging.info(f"Scheduled research task {task_id} for {opportunity.interest_topic}")
        return task_id
    
    async def execute_immediate_research(self, opportunity: 'ResearchOpportunity') -> ResearchResult:
        """Execute research immediately and return result"""
        
        # Find best module for this opportunity
        best_module = None
        for module in self.modules:
            if await module.can_handle(opportunity):
                best_module = module
                break
        
        if not best_module:
            best_module = self.modules[0]  # Fallback to first module
        
        # Execute research
        start_time = datetime.now()
        try:
            result = await best_module.execute_research(opportunity)
            
            # Update stats
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(best_module.name, execution_time, success=True)
            
            # Cache result
            self.completed_research[result.research_id] = result
            
            logging.info(f"Completed immediate research for {opportunity.interest_topic} using {best_module.name}")
            return result
            
        except Exception as e:
            logging.error(f"Research failed for {opportunity.interest_topic}: {e}")
            self._update_stats(best_module.name, 0, success=False)
            raise
    
    async def get_research_result(self, research_id: str) -> Optional[ResearchResult]:
        """Get completed research result"""
        return self.completed_research.get(research_id)
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of research task"""
        task = self.active_tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "status": task.status,
            "topic": task.opportunity.interest_topic,
            "assigned_modules": task.assigned_modules,
            "created_at": task.created_at.isoformat(),
            "scheduled_for": task.scheduled_for.isoformat(),
            "retry_count": task.retry_count,
            "result_available": task.result is not None
        }
    
    async def _process_task_queue(self):
        """Background task queue processor"""
        
        while self.running:
            try:
                # Get next task (with timeout to allow periodic checks)
                task = await asyncio.wait_for(self.task_queue.get(), timeout=5.0)
                
                # Check if it's time to execute
                if datetime.now() >= task.scheduled_for:
                    await self._execute_task(task)
                else:
                    # Put back in queue for later
                    await self.task_queue.put(task)
                    await asyncio.sleep(1)
                
            except asyncio.TimeoutError:
                # Periodic check for shutdown
                continue
            except Exception as e:
                logging.error(f"Task queue processing error: {e}")
                await asyncio.sleep(5)
    
    async def _execute_task(self, task: ResearchTask):
        """Execute a research task"""
        
        task.status = "running"
        start_time = datetime.now()
        
        try:
            # Find module to execute
            module = None
            for mod in self.modules:
                if mod.name in task.assigned_modules:
                    module = mod
                    break
            
            if not module:
                raise ValueError(f"No module found for task {task.task_id}")
            
            # Execute research
            result = await module.execute_research(task.opportunity)
            
            # Store result
            task.result = result
            task.status = "completed"
            self.completed_research[result.research_id] = result
            
            # Update stats
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(module.name, execution_time, success=True)
            
            logging.info(f"Completed task {task.task_id} using {module.name}")
            
        except Exception as e:
            task.status = "failed"
            task.retry_count += 1
            
            logging.error(f"Task {task.task_id} failed: {e}")
            
            # Retry if under limit
            if task.retry_count <= task.max_retries:
                task.status = "pending"
                task.scheduled_for = datetime.now() + timedelta(minutes=5 * task.retry_count)
                await self.task_queue.put(task)
                logging.info(f"Retrying task {task.task_id} in {5 * task.retry_count} minutes")
            else:
                self._update_stats(task.assigned_modules[0] if task.assigned_modules else "unknown", 0, success=False)
    
    def _update_stats(self, module_name: str, execution_time: float, success: bool):
        """Update research execution statistics"""
        
        if success:
            self.research_stats["total_completed"] += 1
            
            # Update average execution time
            total = self.research_stats["total_completed"]
            current_avg = self.research_stats["average_execution_time"]
            self.research_stats["average_execution_time"] = ((current_avg * (total - 1)) + execution_time) / total
            
        else:
            self.research_stats["total_failed"] += 1
        
        if module_name in self.research_stats["module_usage"]:
            self.research_stats["module_usage"][module_name] += 1
    
    def get_research_stats(self) -> Dict[str, Any]:
        """Get research execution statistics"""
        return self.research_stats.copy()
    
    async def generate_actionable_insights(self, research_results: List[ResearchResult]) -> List[ActionableInsight]:
        """Generate actionable insights from multiple research results"""
        
        insights = []
        
        for result in research_results:
            # Convert research findings to actionable insights
            topic_insights = await self._extract_topic_insights(result)
            insights.extend(topic_insights)
        
        # Deduplicate and prioritize
        unique_insights = self._deduplicate_insights(insights)
        prioritized_insights = sorted(unique_insights, key=lambda x: self._calculate_insight_priority(x), reverse=True)
        
        return prioritized_insights[:10]  # Top 10 insights
    
    async def _extract_topic_insights(self, result: ResearchResult) -> List[ActionableInsight]:
        """Extract actionable insights from a research result"""
        
        insights = []
        
        # Shopping insights
        if result.research_type == "shopping_research":
            for finding in result.findings:
                if finding.get("type") == "deal" and finding.get("discount"):
                    discount_pct = float(finding["discount"].rstrip("%"))
                    if discount_pct >= 30:  # Significant discount
                        insights.append(ActionableInsight(
                            action_type="purchase",
                            title=f"Great Deal: {finding['title']}",
                            description=f"{finding['discount']} off at {finding['store']} - save {finding.get('original_price', '')}",
                            url=finding.get("url"),
                            urgency="high" if discount_pct >= 50 else "medium",
                            estimated_value=finding.get("sale_price"),
                            deadline=datetime.fromisoformat(finding["expires"]) if finding.get("expires") else None
                        ))
        
        # Learning insights
        elif result.research_type in ["find_resources", "deep_dive"]:
            high_quality_resources = [f for f in result.findings if f.get("relevance_score", 0) > 0.8]
            if high_quality_resources:
                for resource in high_quality_resources[:3]:  # Top 3
                    insights.append(ActionableInsight(
                        action_type="learn",
                        title=f"Learn: {resource['title']}",
                        description=resource.get("description", ""),
                        url=resource.get("url"),
                        urgency="medium"
                    ))
        
        # Health insights (with disclaimers)
        elif result.research_type == "health_research":
            for finding in result.findings:
                if finding.get("type") == "professional_resource":
                    insights.append(ActionableInsight(
                        action_type="visit",
                        title="Consult Healthcare Professional",
                        description=f"Consider consulting a specialist about {result.topic}",
                        url=finding.get("url"),
                        urgency="medium"
                    ))
        
        return insights
    
    def _deduplicate_insights(self, insights: List[ActionableInsight]) -> List[ActionableInsight]:
        """Remove duplicate insights"""
        
        seen_titles = set()
        unique_insights = []
        
        for insight in insights:
            if insight.title not in seen_titles:
                unique_insights.append(insight)
                seen_titles.add(insight.title)
        
        return unique_insights
    
    def _calculate_insight_priority(self, insight: ActionableInsight) -> float:
        """Calculate priority score for an insight"""
        
        priority = 0.5  # Base priority
        
        # Urgency boost
        urgency_scores = {"low": 0.1, "medium": 0.3, "high": 0.5, "urgent": 0.8}
        priority += urgency_scores.get(insight.urgency, 0.3)
        
        # Deadline boost
        if insight.deadline:
            days_until_deadline = (insight.deadline - datetime.now()).days
            if days_until_deadline <= 1:
                priority += 0.3
            elif days_until_deadline <= 7:
                priority += 0.2
        
        # Action type boost
        action_priorities = {"purchase": 0.4, "visit": 0.3, "learn": 0.2, "read": 0.1, "monitor": 0.1}
        priority += action_priorities.get(insight.action_type, 0.2)
        
        return min(priority, 1.0)


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.append(".")  # Add current directory to path
    from semantic_analyzer import ResearchOpportunity
    
    async def test_research_engine():
        """Test the research execution engine"""
        
        # Initialize orchestrator
        config = {
            "web_search": {
                "max_results": 10,
                "rate_limit_delay": 1.0
            },
            "shopping": {
                "enable_price_tracking": True
            },
            "health": {
                "include_experimental": False
            }
        }
        
        orchestrator = ResearchOrchestrator(config)
        await orchestrator.start_orchestrator()
        
        # Create test research opportunities
        test_opportunities = [
            ResearchOpportunity(
                interest_topic="quilting techniques",
                research_question="Find new resources and tutorials for quilting techniques",
                priority_score=0.8,
                estimated_effort="medium",
                suggested_tools=["web_search"],
                research_type="find_resources",
                reasoning="User has shown active engagement in quilting"
            ),
            ResearchOpportunity(
                interest_topic="fabric shopping",
                research_question="Find current fabric sales and deals",
                priority_score=0.9,
                estimated_effort="low",
                suggested_tools=["shopping_search"],
                research_type="monitor",
                reasoning="User frequently mentions fabric costs"
            )
        ]
        
        # Execute immediate research
        print("=== Executing Immediate Research ===")
        for opportunity in test_opportunities:
            print(f"\n🔍 Researching: {opportunity.interest_topic}")
            
            try:
                result = await orchestrator.execute_immediate_research(opportunity)
                print(f"✅ Research completed!")
                print(f"📊 Confidence: {result.confidence_score:.2f}")
                print(f"📝 Summary: {result.summary}")
                print(f"💡 Insights: {len(result.insights)} found")
                
                for insight in result.insights[:3]:  # Show first 3 insights
                    print(f"   - {insight}")
                
            except Exception as e:
                print(f"❌ Research failed: {e}")
        
        # Generate actionable insights
        print("\n=== Generating Actionable Insights ===")
        all_results = list(orchestrator.completed_research.values())
        
        if all_results:
            actionable_insights = await orchestrator.generate_actionable_insights(all_results)
            
            print(f"🎯 Generated {len(actionable_insights)} actionable insights:")
            for insight in actionable_insights[:5]:  # Show top 5
                print(f"   {insight.action_type.upper()}: {insight.title}")
                print(f"      {insight.description}")
                if insight.urgency != "medium":
                    print(f"      Urgency: {insight.urgency}")
                print()
        
        # Show statistics
        print("=== Research Statistics ===")
        stats = orchestrator.get_research_stats()
        print(f"Total completed: {stats['total_completed']}")
        print(f"Total failed: {stats['total_failed']}")
        print(f"Average execution time: {stats['average_execution_time']:.2f}s")
        print(f"Module usage: {stats['module_usage']}")
        
        await orchestrator.stop_orchestrator()
    
    # Run the test
    asyncio.run(test_research_engine())
