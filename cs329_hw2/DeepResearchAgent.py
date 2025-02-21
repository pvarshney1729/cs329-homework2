
import os
import requests
from datetime import datetime
import openai
import json
from googleapiclient.discovery import build
from textblob import TextBlob
from openai import OpenAI
from pydantic import BaseModel
from typing import Dict, Any, List

from cs329_hw2.api_manager import APIManager
from cs329_hw2.utils import generate_openai, generate_anthropic, generate_together
from cs329_hw2.multi_lm_agent_v6_49 import MultiLMAgent
from concurrent.futures import ThreadPoolExecutor

import openai
import json
import re

class DeepResearchAgent(MultiLMAgent):
    def __init__(self, api_manager):
       super().__init__(
            api_manager=api_manager,
            decomposition_model="gpt-4o-mini",
            iterative_refinement_model="gpt-4o-mini", 
            fusion_model="gpt-4o-mini",
            generation_temp=0.7,
            fusion_temp=0.5
        )
        
        
    def research(self, query: str) -> Dict[str, Any]:
        """
        Conducts deep research on a given query.
        
        Args:
            query: Complex research question to investigate
            
        Returns:
            Dictionary containing:
            - report: str, synthesized findings with citations
            - sources: List[str], list of source URLs or references
        """

        research_plan = self._create_enhanced_research_plan(query)
        
        section_findings = []
        sources = set()
        
        def process_aspect(aspect):
            results = self.decompose_query(aspect["sub_query"], optimize_query_flag=True)
            processed = self._process_research_results(results, aspect["focus"])
            
            return {
                "section": aspect["focus"],
                "findings": [{
                    "aspect": aspect["focus"],
                    "findings": processed["findings"],
                    "key_metrics": self._extract_key_metrics(processed["findings"]),
                    "temporal_data": self._extract_temporal_data(processed["findings"])
                }],
                "sources": processed["sources"]
            }
    
            
        with ThreadPoolExecutor() as executor:
            future_results = executor.map(process_aspect, research_plan["aspects"])
            for result in future_results:
                section_findings.append(result)
                sources.update(result["sources"])
    
        report = self._generate_structured_report(query, section_findings, research_plan)

        sources_section = report.split("## References")[-1].strip()
        sources_list = [line.strip() for line in sources_section.split("\n") if line.strip().startswith("[")]
        
        return {
            "report": report,
            "sources": sources_list
        }



    def _create_enhanced_research_plan(self, query: str) -> dict:
        """Creates a detailed research plan using OpenAI function calling to ensure JSON output."""

        client = OpenAI()

        system_prompt = """You are an expert research planner. Your task is to break down complex queries into 
        clear research aspects. Always respond in the exact JSON format specified."""
        
        
        function_schema = {
            "name": "create_research_plan",
            "description": "Creates a detailed research plan for a given research query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "aspects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "focus": {"type": "string", "description": "Specific aspect to research"},
                                "sub_query": {"type": "string", "description": "A targeted search query for that aspect"},
                                "key_points": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Key points to investigate for this aspect"
                                }
                            },
                            "required": ["focus", "sub_query", "key_points"]
                        },
                        "description": "List of research aspects."
                    }
                },
                "required": ["aspects"]
            }
        }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a detailed research plan for this query: {query}\n\nPlease ensure that the output follows the provided JSON schema exactly."}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            functions=[function_schema],
            function_call={"name": "create_research_plan"},
            temperature=0.3
        )
        
        message = response.choices[0].message

        if "function_call" in message:
            try:
                plan = json.loads(json.loads(message.function_call.arguments))
                return plan
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to decode function call arguments: {e}")
        return {
            "aspects": [
                {
                    "focus": "Recent Developments",
                    "sub_query": f"latest developments in {query} 2024",
                    "key_points": ["technical breakthroughs", "company announcements"]
                },
                {
                    "focus": "Challenges",
                    "sub_query": f"main challenges and obstacles in {query}",
                    "key_points": ["technical challenges", "implementation issues"]
                },
                {
                    "focus": "Future Outlook",
                    "sub_query": f"future predictions and roadmap for {query}",
                    "key_points": ["expected developments", "industry trends"]
                }
            ]
        }

    def _extract_key_metrics(self, findings: List[Dict]) -> Dict[str, Any]:
        """Extracts and structures key numerical metrics from findings."""
        metrics = {}
        
        for finding in findings:
            matches = re.finditer(
                r'(\d+(?:\.\d+)?(?:%|percent|million|billion|trillion)?)\s*(?:in|for|of)?\s*([^,.;]+)',
                finding['content']
            )
            
            for match in matches:
                value, context = match.groups()
                metrics[context.strip()] = value
                
        return metrics

    def _extract_temporal_data(self, findings: List[Dict]) -> Dict[str, List[Dict]]:
        """Extracts and organizes temporal data from findings."""
        temporal_data = {}
        
        for finding in findings:
            dates = re.finditer(
                r'(?:in|by|during|since)?\s*'
                r'(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)|'
                r'(?:Q[1-4])|'
                r'(?:\d{4}))',
                finding['content']
            )
            
            for date in dates:
                date_str = date.group()
                context = finding['content'][max(0, date.start()-50):min(len(finding['content']), date.end()+100)]
                
                if date_str not in temporal_data:
                    temporal_data[date_str] = []
                    
                temporal_data[date_str].append({
                    'context': context,
                    'source': finding['source']
                })
                
        return temporal_data

    def _generate_structured_report(self, query: str, section_findings: List[Dict], research_plan: Dict) -> str:
        """Generates a well-structured report with proper markdown formatting."""
        system_prompt = """You are an expert research report writer who creates comprehensive, 
        well-structured reports with proper markdown formatting and academic-style citations."""
        
        synthesis_prompt = f"""
        Generate a comprehensive research report answering:
        {query}

        MARKDOWN FORMATTING REQUIREMENTS:
        1. Title: Use a single # for the main title
        2. Section Headers:
        - Use ## for main sections (Executive Summary, Introduction, etc.)
        - Use ### for subsections
        - Use #### for sub-subsections if needed
        3. Emphasis:
        - Use **bold** for important terms or metrics
        - Use *italics* for emphasis where appropriate
        4. Lists:
        - Use proper markdown bullet points (*)
        - Use numbered lists (1., 2., etc.) for sequential items
        5. Citations:
        - Use [1], [2], etc. for inline citations
        - Include a References section at the end
        
        STRUCTURE:
        ## Title
        
        ### Executive Summary
        (2-3 paragraphs)
        
        ### Key Developments
        (with subsections and paragraphs)
        
        ### Challenges and Limitations
        (with subsections and paragraphs)
        
        ### Future Outlook
        (with subsections and paragraphs)
        
        ### Conclusion
        (1-2 paragraphs)
        
        ### References
        [1] Source 1
        [2] Source 2
        etc.

        Make sure the report reads like a professional research document with academic-style citations.
        """
        
        report = self.generate_with_system_prompt(
            system_prompt,
            synthesis_prompt,
            model="gpt-4o",
            temperature=0.3
        )
        
        return report.strip()

    def _format_section_findings(self, section_findings: List[Dict]) -> str:
        """Formats section findings for report generation."""
        formatted = ""
        for section in section_findings:
            formatted += f"\nSECTION: {section['section']}\n"
            for finding in section['findings']:
                formatted += f"\nAspect: {finding['aspect']}\n"
                formatted += "Key Metrics:\n"
                for metric, value in finding['key_metrics'].items():
                    formatted += f"- {metric}: {value}\n"
                    
                formatted += "\nTemporal Data:\n"
                for date, items in finding['temporal_data'].items():
                    formatted += f"- {date}:\n"
                    for item in items:
                        formatted += f"  * {item['context']}\n"
                        formatted += f"    Source: {item['source']}\n"
                        
            formatted += "\nSources:\n"
            for source in section['sources']:
                formatted += f"- {source}\n"
                
        return formatted
    

    def _process_research_results(self, results: List[Dict], focus: str) -> Dict:
        """
        Process raw research results into structured findings.
        
        Args:
            results: List of raw research results
            focus: Research aspect focus area
            
        Returns:
            Dict containing:
            - findings: List of processed findings
            - sources: List of unique sources
        """
        processed_findings = []
        sources = set()
        
        for result in results:
            content = result.get('content', '')
            source = result.get('source', '')
            
            if content and source:
                # Add to findings
                processed_findings.append({
                    'content': content,
                    'source': source
                })
                
                # Add to sources
                sources.add(source)
        
        return {
            "findings": processed_findings,
            "sources": list(sources)
        }