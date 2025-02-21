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
import concurrent

import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cs329_hw2.api_manager import APIManager
from cs329_hw2.utils import generate_openai, generate_anthropic, generate_together

from . import prompts

from concurrent.futures import ThreadPoolExecutor, as_completed

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

import re

class SubQuery(BaseModel):
    api: str
    params: Dict
    order: int

class DecompositionResponse(BaseModel):
    sub_queries: List[SubQuery]

def extract_between_tags(text: str, tag: str) -> str:
    """Extract content between XML-style tags"""
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    try:
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag)
        
        if start_idx == -1 or end_idx == -1:
            return text
            
        start = start_idx + len(start_tag)
        return text[start:end_idx].strip()
    except Exception as e:
        print(f"Error extracting content between {tag} tags: {str(e)}")
        return text
    
    
class MultiLMAgent:
    """A class to manage multiple language models for generation, iterative refinement, and fusion"""
    
    def __init__(
        self,
        api_manager: APIManager,
        decomposition_model: str = "gpt-4o-mini",
        iterative_refinement_model: str = "gpt-4o-mini",
        fusion_model: str = "gpt-4o-mini",
        generation_temp: float = 0.7,
        fusion_temp: float = 0.5
    ):
        """
        INPUT:
            api_manager: APIManager - Instance of APIManager for API interactions
            decomposition_model: str - Model to use for query decomposition
            iterative_refinement_model: str - Model to use for iterative refinement
            fusion_model: str - Model to use for fusing responses
            generation_temp: float - Temperature for generation (0.0-1.0)
            fusion_temp: float - Temperature for fusion (0.0-1.0)
        """
        self.generation_temp=generation_temp
        self.api_manager=api_manager
        ################ CODE STARTS HERE ###############
        
        self.decomposition_model = decomposition_model
        self.iterative_refinement_model = iterative_refinement_model
        self.fusion_model = fusion_model
        self.fusion_temp = fusion_temp

        ################ CODE ENDS HERE ###############

    def generate(self, prompt: str, model: str = "gpt-4o-mini") -> List[Dict]:
        """
        INPUT:
            prompt: str - Input prompt for generation
            model: str - The model to use for generation
        OUTPUT:
            response: str - Generated response
        """

        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates responses to user queries."},
            {"role": "user", "content": prompt}
        ]
        
        if "gpt" in model:
            response = generate_openai(messages=messages, model=model, temperature=self.generation_temp)
        elif "claude" in model:
            response = generate_anthropic(messages=messages, model=model, temperature=self.generation_temp)
        else:
            response = generate_together(messages=messages, model=model, temperature=self.generation_temp)
        return response
    

    def generate_with_system_prompt(self, system_prompt: str, prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> List[Dict]:
        """
        INPUT:
            prompt: str - Input prompt for generation
            model: str - The model to use for generation
        OUTPUT:
            response: str - Generated response
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        if "gpt" in model:
            response = generate_openai(messages=messages, model=model, temperature=temperature)
        elif "claude" in model:
            response = generate_anthropic(messages=messages, model=model, temperature=temperature)
        else:
            response = generate_together(messages=messages, model=model, temperature=temperature)
        return response
    
    def single_LM_with_single_API_call(self, query: str, model: str) -> str:
        """
        INPUT:
            query: str - The user's query to be decomposed
            model: str - The model to use for generation

        OUTPUT:
            str - The response from the model
        """
        ################ CODE STARTS HERE ###############

        api_response = self.api_manager.route_query(query)

        enhanced_prompt = self.generate_prompt(query, [api_response])

        response = self.generate(enhanced_prompt, model)

        return response if response else ""

        ################ CODE ENDS HERE ###############

    def clean_ui_elements(self, text: str) -> str:
        """Clean scraped text while preserving structure"""
        if not text:
            return ""
            
        # Remove common UI elements
        ui_patterns = [
            r'Menu|Navigation|Search|Login|Sign in|Subscribe',
            r'Follow us|Share|Tweet|Like|Comment',
            r'Download|Upload|Print|View',
            r'Privacy Policy|Terms of Use',
            r'©.*?reserved'
        ]
        for pattern in ui_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Normalize multiple newlines
        
        return text.strip()


    def decompose_query(self, query: str, optimize_query_flag: bool = False) -> List[Dict]:
        """
        INPUT:
            query: str - The user's original query to be decomposed
            
        OUTPUT:
            List[Dict] - List of dictionaries containing:
                - api: str - Name of the API to call
                - params: Dict - Parameters for the API call
                - results: Dict - Gathered results from the API call
                - status: str - Success/error status
                - order: int - Order of execution
        """
        ################ CODE STARTS HERE ###############

        prompt = self.get_decomposition_prompt(query)
        system_prompt = "You are an expert at breaking down complex queries into simpler sub-queries that when answered together will help answer the original query."
        response = self.generate_with_system_prompt(system_prompt, prompt, self.decomposition_model)

        cleaned_response = response.strip()
        cleaned_response = self.clean_ui_elements(cleaned_response)

        if not response:
            return []

        try:

            decomposed_queries = json.loads(cleaned_response)
            sub_queries = []

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_query = {
                    executor.submit(
                        self.api_manager.route_query, 
                        sub_query["query"],
                        optimize_query_flag
                    ): sub_query 
                    for sub_query in decomposed_queries.get("sub_queries", [])
                }

                for future in concurrent.futures.as_completed(future_to_query):
                    sub_query = future_to_query[future]

                    try:
                        api_response = future.result()
                        sub_queries.append({
                            "api": api_response.get("api_used", "unknown"),
                            "params": api_response.get("params", {}),
                            "results": api_response.get("results", {}),
                            "status": "success" if api_response.get("results") else "error",
                            "order": sub_query.get("order", 0),
                            "purpose": sub_query.get("purpose", "")
                        })

                    except Exception as e:
                        print(f"Error executing sub-query: {str(e)}")
                        return [{
                            "api": "error",
                            "params": {},
                            "results": {},
                            "status": "error",
                            "order": 0,
                            "purpose": str(e)
                        }]

            return sorted(sub_queries, key=lambda x: x["order"])
        
        except Exception as e:
            print(f"Error in decompose_query: {str(e)}")
            return []


        ################ CODE ENDS HERE ###############

    def get_decomposition_prompt(self, query: str) -> str:
        """
        INPUT:
            query: str - The user's query to be decomposed
            
        OUTPUT:
            str - Formatted prompt for decomposition model
        """
        ################ CODE STARTS HERE ###############

        return prompts.decompose_query_prompt.format(query=query)

    
        ################ CODE ENDS HERE ###############
        
    def generate_prompt(self, query: str, decomposed_queries: List[Dict] = None, summarization=False) -> str:
        """
        INPUT:
            query: str - Original user query
            decomposed_queries: List[Dict] - List of decomposed query results
            
        OUTPUT:
            str - Enhanced prompt including original query and API results
        """
        ################ CODE STARTS HERE ###############

        system_prompt = """You are an expert information synthesizer. Your task is to analyze multiple pieces of information and provide a comprehensive, well-reasoned response to a user's query."""

        prompt = f"""
        <ORIGINAL_QUERY>
        {query}
        </ORIGINAL_QUERY>

        <GATHERED_INFORMATION>
        """

        if decomposed_queries and len(decomposed_queries) > 0:
            for i, result_dict in enumerate(decomposed_queries, 1):
                prompt += f"\nSource {i}:"
                if result_dict.get("status") == "success":
                    
                    prompt += f"\n- Parameters Used: {json.dumps(result_dict.get('params', {}), indent=2)}"
                    
                    results = result_dict.get('results', {})
                    if result_dict.get('api') == 'google_search':
                        prompt += "\n- Search Results:"
                        if isinstance(results, list):
                            for j, result in enumerate(results[:3], 1):
                                prompt += f"\n  Result {j}:"
                                prompt += f"\n    URL: {result.get('link', '')}"
                                prompt += f"\n    Snippet: {result.get('snippet', '')}"
                                
                                webpage_content = result.get('webpage_content', {})
                                if webpage_content:
                                    prompt += f"\n    Page Title: {webpage_content.get('title', '')}"
                                    raw_text = webpage_content.get('text_content', '')
                                    extracted_content = self._clean_and_extract_relevant(raw_text, query, summarization)
                                    # MAX_LENGTH = 5000
                                    # if len(extracted_content) > MAX_LENGTH:
                                        # extracted_content = extracted_content[:MAX_LENGTH] + "..."
                                    prompt += f"\n    Extracted Content: {extracted_content}"
                                    
                                    # Add image information
                                    images = webpage_content.get('images', [])
                                    if images:
                                        prompt += "\n    Page Images Content:"
                                        for k, img in enumerate(images[:5], 1):  # Limit to first 3 images
                                            if img.get('alt'):
                                                prompt += f"\n      {k}. {img.get('alt', '')}"
                                prompt += "\n    ---"
                    else:
                        prompt += f"\n- Results: {json.dumps(results, indent=2)}"
                    
                    prompt += "\n---"

        prompt += """
        </GATHERED_INFORMATION>

        <OUTPUT_FORMAT>
        Your response MUST follow this exact structure:

        <DIRECT_ANSWER>
        Provide a clear, concise answer to the query in a few words or 1 sentence maximum.
        </DIRECT_ANSWER>

        <KEY_FINDINGS>
        List 1-2 facts/key findings with source attribution that support your answer.
        </KEY_FINDINGS>

        <CONFIDENCE_ASSESSMENT>
        Rate claims 1-5 based on evidence quality, information completeness and agreement
        </CONFIDENCE_ASSESSMENT>
        </OUTPUT_FORMAT>

        <INSTRUCTIONS>
        1. Follow the output format EXACTLY, including all XML tags
        2. In DIRECT_ANSWER:
           - Use precise numbers and facts
           - No hedging or qualifiers unless uncertainty is significant
        
        3. In KEY_FINDINGS:
           - Each finding must cite specific source(s)
           - Include only verifiable facts, not interpretations
        
        4. In CONFIDENCE_ASSESSMENT:
           - Base ratings on source reliability and evidence consistency

        5. Do not add any qualifiers or anything else outside of the tags.

        6 If the original query mentions precise information such as awards names, locations, etc, then make sure to search for them in the gathered information.

        CRITICAL RULES:
        - Never speculate beyond the evidence
        - Always maintain XML tag structure
        - Be concise but precise
        </INSTRUCTIONS>"""

        return prompt

        ################ CODE ENDS HERE ###############

    def iterative_refine(self, prompt: str, max_iterations: int = 3, models_to_query: List[str] = None) -> str:
        """
        INPUT:
            prompt: str - Input prompt for generation
            max_iterations: int - Maximum number of refinement iterations
            models_to_query: List[str] - List of models to query

        OUTPUT:
            str - Final fused response
        """
        ################ CODE STARTS HERE ###############

        current_prompt = prompt

        if models_to_query is None:
            models_to_query = [
                "gpt-4o-mini",
                "claude-3-5-haiku-latest",
                "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
            ]
        

        fused_responses = []

        for iteration in range(max_iterations):
            
            decomposed_queries = self.decompose_query(current_prompt)

            generated_prompts = self.generate_prompt(current_prompt, decomposed_queries, summarization=True)

            fused_response = self.fuse(generated_prompts, models_to_query)

            fused_responses.append(fused_response)
            
            system_prompt = "You are a rigorous evaluator focused on ensuring responses are accurate, complete, and well-reasoned. Your goal is to identify any gaps or areas for improvement in the response."            

            formatted_responses = "\n".join([f"Iteration {i+1}:\n{response}" for i, response in enumerate(fused_responses)])

            verification_prompt = prompts.verification_prompt.format(
                current_prompt=current_prompt,
                fused_responses=formatted_responses
            )                                                                   

            evaluation = self.generate_with_system_prompt(system_prompt, verification_prompt, self.fusion_model, temperature=0.3)

            is_satisfactory = extract_between_tags(evaluation, "SATISFACTORY").upper() in  ["YES" or "TRUE"]
            missing_information = extract_between_tags(evaluation, "MISSING_INFORMATION")

            print(f"Iteration {iteration + 1} Is Response Satisfactory: {is_satisfactory}")
            if iteration == max_iterations - 1:
                print("Terminating refinement loop: max_iterations reached")
                return fused_response
            
            if is_satisfactory or "NONE" in missing_information.upper():
                print("Terminating refinement loop: with satisfactory response obtained.")
                return fused_response
                

            current_prompt = current_prompt + f"""
            The following information is still missing to answer the original query:
            <MISSING_INFORMATION>
            {missing_information}
            </MISSING_INFORMATION>

            The following information relevant to the query has been retrieved: 
            <QUERY_RESPONSES_SO_FAR>
            {fused_responses}
            </QUERY_RESPONSES_SO_FAR>
            """



        ################ CODE ENDS HERE ###############


    def _truncate_json(self, obj: Any, max_chars: int = 1000) -> Any:
        """Helper method to truncate JSON objects while preserving structure"""
        if isinstance(obj, str):
            return obj[:max_chars] + "..." if len(obj) > max_chars else obj
        elif isinstance(obj, dict):
            return {k: self._truncate_json(v, max_chars // len(obj)) for k, v in obj.items()}
        elif isinstance(obj, list):
            chars_per_item = max_chars // len(obj) if obj else max_chars
            return [self._truncate_json(item, chars_per_item) for item in obj[:3]]  # Limit array items
        return obj

    
    def fuse(self, prompt: str, models_to_query: List[str] = [
            "gpt-4o-mini",
            "claude-3-5-haiku-latest",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        ]) -> str:
        """
        Queries multiple models with the same prompt and fuses the responses 
        by combining the best elements from each response.

        INPUT:
            prompt: str - Original prompt
            
        OUTPUT:
            str - Single fused response combining the best elements
        """
        ################ CODE STARTS HERE ###############

        model_responses = []

        truncated_prompt = prompt

        for model in models_to_query:
            try:
                response = self.generate(truncated_prompt, model)
                if response:
                    model_responses.append({
                        "model": model,
                        "response": response,
                    })
            except Exception as e:
                print(f"Error in generating response from {model}: {str(e)}")
                continue

        if not model_responses:
            return "Unable to generate responses from any model."
        
        system_prompt = "You are an expert answer synthesizer. Analyze and combine the following model responses into a single comprehensive answer."

        fusion_prompt = prompts.fusion_prompt.format(
            source_responses=self._format_model_responses(model_responses)
        )

        try:
            fused_response = self.generate_with_system_prompt(system_prompt, fusion_prompt, self.fusion_model, temperature=0.3)
        except Exception as e:
            # Fallback to haiku's response if fusion fails
            fused_response = model_responses[1]['response']
            print(f"Fusion failed, using haiku's response: {fused_response}")

        return extract_between_tags(fused_response, "DIRECT_ANSWER")

        ################ CODE ENDS HERE ###############

    def _format_model_responses(self, responses: List[Dict]) -> str:
        """Helper to format model responses for fusion"""
        formatted = ""
        for i, response in enumerate(responses, 1):
            formatted += f"Model: {response['model']}\n"
            # formatted += f"Confidence: {response['confidence']:.2f}\n"
            formatted += f"Response: {response['response']}\n"
            formatted += "-" * 50 + "\n"
        return formatted
      
    def run_pipeline(self, query: str, iterations: int = 4) -> str:
        """
        INPUT:
            query: str - User's natural language query
            iterations: int - Number of iterations of iterative refinement
        OUTPUT:
            str - Final fused response or error message
        """
        ################ CODE STARTS HERE ###############
        simplified_query = self.preprocess_query(query, self.decomposition_model)

        simplified_original_query = simplified_query.get('Rewritten Query', query)

        knowledge_base = []
        response_list = []

        missing_information = None

        current_query = simplified_original_query

        for iteration in range(iterations):

            decomposed_results = self.decompose_query(current_query, optimize_query_flag=False)

            knowledge_base = self.update_knowledge_base(decomposed_results, current_query, knowledge_base)

            enhanced_prompt = self.generate_prompt_from_processed_kb(current_query, knowledge_base)

            system_prompt = "You are an expert answer generator. Analyze and combine the information so far into a single concise and precise reponse that accuractely answers the user query."

            response = self.generate_with_system_prompt(system_prompt, enhanced_prompt, model="gpt-4o")

            response = extract_between_tags(response, "DIRECT_ANSWER")

            response_list.append(response)

            if iteration == iterations - 1:
                print("Terminating refinement loop: max_iterations reached")
                return response

            system_prompt = "You are a rigorous evaluator focused on ensuring responses are accurate, complete, and well-reasoned. Your goal is to identify any gaps or areas for improvement in the response."

            formatted_responses = "\n".join([f"Iteration {i+1}:\n{response}" for i, response in enumerate(response_list)])

            verification_prompt = prompts.verification_prompt.format(
                current_prompt=query,
                fused_responses=formatted_responses
            )

            evaluation = self.generate_with_system_prompt(system_prompt, verification_prompt, self.fusion_model, temperature=0.3)

            is_satisfactory = extract_between_tags(evaluation, "SATISFACTORY").upper() in  ["YES" or "TRUE"]
            missing_information = extract_between_tags(evaluation, "MISSING_INFORMATION")
            
            if (is_satisfactory or "NONE" in missing_information.upper()) and iteration > 1:
                print("Terminating refinement loop: with satisfactory response obtained.")
                return response
            
            current_query = current_query + f"""
            The following information is still missing to answer the original query:
            <MISSING_INFORMATION>
            {missing_information}
            </MISSING_INFORMATION>

            The following information relevant to the query has been retrieved: 
            <QUERY_RESPONSES_SO_FAR>
            {formatted_responses}
            </QUERY_RESPONSES_SO_FAR>
            """

        return response
            
            
        ################ CODE ENDS HERE ###############


    def preprocess_query(self, query: str, model: str = "gpt-4o-mini") -> str:
        """
        Use LLM to extract essential query elements and identify relevant APIs.
        Returns simplified query optimized for API calls.
        """
        prompt = f"""Given this query: "{query}"

        1. Extract the core question, removing unnecessary context and narrative elements
        2. Identify key entities, dates, and facts that need to be verified
        3. Rewrite as a simple, direct question
        
        Format your response strictly as a JSON object:
        {{
            "Core Question": "simplified question",
            "Key Elements": "comma-separated list of key entities/facts to verify",
            "Rewritten Query": "rewritten query"
        }}.
        
        IMPORTANT: Failure to return a valid JSON object will lead to punishment.
        """

        response = self.generate(
            prompt,
            model=model,
        )

        try:
            return json.loads(response.strip(), strict=False)
        except json.JSONDecodeError:
            try:
                # Extract JSON-like content
                match = re.search(r'\{[^}]+\}', response)
                if match:
                    json_str = match.group(0)
                    return json.loads(json_str, strict=False)
                
                # Clean and restructure the response
                cleaned_response = response.strip()
                # Remove any markdown code block indicators
                cleaned_response = re.sub(r'```json\s*|\s*```', '', cleaned_response)
                # Remove any leading/trailing whitespace or newlines
                cleaned_response = cleaned_response.strip()
                return json.loads(cleaned_response, strict=False)
            except Exception as e:
                print(f"Error parsing LLM response: {str(e)}")
                print(f"Raw response: {response}")

                return {
                    "Core Question": query,
                    "Key Elements": query,
                    "Rewritten Query": query
                }


    def clean_scraped_content(self, text: str) -> str:
        """
        Clean scraped content for insertion into the LLM context window.
        
        Steps:
          1. Ensure the text is in English.
          2. Remove control characters (except newlines/tabs to preserve structure).
          3. Remove URLs, email addresses, and common UI navigation elements.
          4. Normalize whitespace while preserving line breaks.
        
        Args:
            text: The scraped raw input text.
            
        Returns:
            A cleaned string that retains key context and structure.
        """
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)

        ui_elements = (
            r'\b(Navigation|Menu|Search|Subscribe|Log ?in|Sign ?in|Create Account|'
            r'Donate|Home|Main page|Contents|About|Contact|Help|Special pages|'
            r'Follow us|Share|Tweet|Like|Comment|SECTION|NEWSLETTER|TOPICS|'
            r'e-Paper|LOGIN|Account|FREE TRIAL|GIFT|Live Now|Skip to content|'
            r'Advertisement|Privacy Policy|Terms of Service|Sign Up|Sign Out|'
            r'Facebook|Twitter|Instagram|LinkedIn|YouTube|Pinterest|Reddit|'
            r'Cookie Policy|Site Map|Customer Service|Feedback|FAQ|'
            r'All rights reserved|© \d{4})\b'
        )

        text = re.sub(ui_elements, ' ', text, flags=re.IGNORECASE)
        
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', '\n', text)

        redundant_patterns = (
            r'^\s*$',  # Empty lines
            r'^\s*[-–—]+\s*$',  # Lines with only dashes
            r'^\s*\|\s*$',  # Lines with only pipes
        )
        for pattern in redundant_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)

        return text.strip()
        
    def _clean_and_extract_relevant(self, response: str, query: str, summarization: bool = False) -> str:
        """Enhanced cleaning with relevance-based extraction."""
        # Split into sentences
        cleaned_text = self.clean_scraped_content(response)

        if summarization:
        
            try:
                system_prompt = """
You are an expert information extraction and synthesis system designed to:
- Identify and preserve key information from source text
- Maintain factual accuracy and original context 
"""

                llm_prompt = f"""
Given a query and some scraped webpage content, extract and structure all important information that might be relevant to answer the query.
{query}
</QUERY>

<TEXT>
{' '.join(cleaned_text)}
</TEXT>


<EXTRACTION_RULES>
1. PRESERVE EXACTLY:
   - Numbers, measurements, and quantities
   - Dates and timestamps
   - Names of people, organizations, pin codes, and places
   - Direct quotes and key phrases
   - Product codes, model numbers, and IDs
   - Technical specifications

2. MAINTAIN INTEGRITY:
   - Keep original sequence of key facts
   - Retain important qualifiers and conditions
   - Preserve relationships between connected facts. E.g. X said this quote, Y won that prize, Z was mentioned in this article, etc. 

3. HANDLING RELEVANCE:
   - Prioritize query-relevant information
   - Flag critical information even if not directly related to query

5. PROHIBITED:
   - No external knowledge, explanations, commentary or interpretations
   - No summarization that loses precision
</EXTRACTION_RULES>

EXTRACTED INFORMATION:"""
                
                # Use a faster/smaller model for this extraction
                relevant_content = self.generate_with_system_prompt(system_prompt, llm_prompt, model="gpt-4o-mini", temperature=0.3)
                
                if relevant_content:
                    return relevant_content
                
                return cleaned_text
                
            except Exception as e:
                print(f"Error in relevance extraction: {str(e)}")
                return cleaned_text
            
        return cleaned_text

    def process_decomposed_result(self, result_dict: Dict, query: str, summarization: bool = False) -> Dict:
        """Helper function to process a decomposed query result into a cleaner format"""
        processed = {
            "source_type": result_dict.get("api", "unknown"),
            "status": result_dict.get("status", "error"),
            "order": result_dict.get("order", 0),
            "purpose": result_dict.get("purpose", ""),
            "processed_results": []
        }
        
        if result_dict.get("status") != "success":
            return processed
            
        results = result_dict.get("results", {})
        if result_dict.get("api") == "google_search" and isinstance(results, list):
            for result in results:  # Limit to top 3 results
                processed_result = {
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "title": "",
                    "extracted_content": "",
                    "relevant_images": []
                }
                
                webpage_content = result.get("webpage_content", {})
                if webpage_content:
                    processed_result["title"] = webpage_content.get("title", "")
                    raw_text = webpage_content.get("text_content", "")
                    processed_result["extracted_content"] = self._clean_and_extract_relevant(
                        raw_text, query, summarization
                    )
                    
                    # Process relevant images
                    images = webpage_content.get("images", [])
                    processed_result["relevant_images"] = [
                        img.get("alt", "") for img in images[:5] if img.get("alt")
                    ]
                
                processed["processed_results"].append(processed_result)
        else:
            # For non-Google Search APIs, store results directly
            processed["processed_results"] = results
            
        return processed

    def update_knowledge_base(self, decomposed_results: List[Dict], query: str, knowledge_base: List[Dict]) -> List[Dict]:
        """Helper function to process and update knowledge base with new results"""
        new_entries = []
        
        for result in decomposed_results:
            processed_result = self.process_decomposed_result(result, query)
            
            if processed_result == None:
                pass

            is_duplicate = False
            for existing in knowledge_base:
                if (existing["source_type"] == processed_result["source_type"] and 
                    existing["processed_results"] == processed_result["processed_results"]):
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                new_entries.append(processed_result)
        
        return knowledge_base + new_entries

    def generate_prompt_from_processed_kb(self, query: str, knowledge_base: List[Dict]) -> str:
        """Generate prompt using pre-processed knowledge base"""
        system_prompt = """You are an expert information synthesizer. Your task is to analyze multiple pieces of information and provide a comprehensive, well-reasoned response to a user's query."""

        prompt = f"""
        <ORIGINAL_QUERY>
        {query}
        </ORIGINAL_QUERY>

        <GATHERED_INFORMATION>
        """

        if knowledge_base:
            for i, entry in enumerate(knowledge_base, 1):
                    prompt += f"\nSource {i} ({entry['source_type']}):"
                    prompt += f"\nPurpose: {entry['purpose']}"
                    
                    if entry["source_type"] == "google_search":
                        for j, result in enumerate(entry["processed_results"], 1):
                            prompt += f"\n  Result {j}:"
                            if result["title"]:
                                prompt += f"\n    Title: {result['title']}"
                            prompt += f"\n    URL: {result['url']}"
                            prompt += f"\n    Snippet: {result['snippet']}"
                            
                            if result["extracted_content"]:
                                prompt += f"\n    Content: {result['extracted_content']}"
                                
                            if result["relevant_images"]:
                                prompt += "\n    Relevant Images:"
                                for img_alt in result["relevant_images"]:
                                    prompt += f"\n      - {img_alt}"
                            prompt += "\n    ---"
                    else:
                        prompt += f"\n  Results: {json.dumps(entry['processed_results'], indent=2)}"
                    
                    prompt += "\n---"

        prompt += """
        </GATHERED_INFORMATION>

        <OUTPUT_FORMAT>
        Your response MUST follow this exact structure:

        <DIRECT_ANSWER>
        Provide a clear and concise answer to the query.
        </DIRECT_ANSWER>

        <KEY_FINDINGS>
        List 2-3 facts/key findings with source attribution that support your answer.
        </KEY_FINDINGS>

        <CONFIDENCE_ASSESSMENT>
        Rate claims 1-5 based on evidence quality, information completeness and agreement
        </CONFIDENCE_ASSESSMENT>
        </OUTPUT_FORMAT>

        <INSTRUCTIONS>
        1. Follow the output format EXACTLY, including all XML tags
        2. In DIRECT_ANSWER:
           - Use precise numbers and facts
           - No hedging or qualifiers unless uncertainty is significant
           - Follow the output format if mentioned in the query.
           Examples:
            Query: On what day, month, and year was Algerian artist Mohammed Racim born?
            Bad Response: Mohammed Racim was born on June 24, 1896.
            Good Response: June 24th, 1896.

            Query: How much money, in euros, was the surgeon held responsible for Stella Obasanjo's death ordered to pay her son?
            Bad Response: The surgeon was ordered to pay €120,000 to Stella Obasanjo's son.
            Good Response: 120,000

            Query: The object in the British Museum's collection with a museum number of 2012,5015.17 is the shell of a particular mollusk species. According to the abstract of a research article published in Science Advances in 2021, beads made from the shells of this species were found that are at least how many thousands of years old?
            Bad Response: 142,000
            Good Response: 142

        
        3. In KEY_FINDINGS:
           - Each finding must cite specific source(s)
           - Include only verifiable facts, not interpretations
        
        4. In CONFIDENCE_ASSESSMENT:
           - Base ratings on source reliability and evidence consistency

        5. Do not add any qualifiers or anything else outside of the tags.

        6 If the original query mentions precise information such as awards names, locations, etc, then make sure to search for them in the gathered information.

        CRITICAL RULES:
        - Never speculate beyond the evidence
        - Always maintain XML tag structure
        - Be concise but precise
        </INSTRUCTIONS>"""

        return prompt