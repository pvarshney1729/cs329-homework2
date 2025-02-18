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

class SubQuery(BaseModel):
    api: str
    params: Dict
    order: int

class DecompositionResponse(BaseModel):
    sub_queries: List[SubQuery]

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
        
        pass 

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

    def single_LM_with_single_API_call(self, query: str, model: str) -> str:
        """
        INPUT:
            query: str - The user's query to be decomposed
            model: str - The model to use for generation

        OUTPUT:
            str - The response from the model
        """
        ################ CODE STARTS HERE ###############

        pass

        ################ CODE ENDS HERE ###############

    def decompose_query(self, query: str) -> List[Dict]:
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

        pass

        ################ CODE ENDS HERE ###############

    def get_decomposition_prompt(self, query: str) -> str:
        """
        INPUT:
            query: str - The user's query to be decomposed
            
        OUTPUT:
            str - Formatted prompt for decomposition model
        """
        ################ CODE STARTS HERE ###############

        pass

        ################ CODE ENDS HERE ###############
        
    def generate_prompt(self, query: str, decomposed_queries: List[Dict] = None) -> str:
        """
        INPUT:
            query: str - Original user query
            decomposed_queries: List[Dict] - List of decomposed query results
            
        OUTPUT:
            str - Enhanced prompt including original query and API results
        """
        ################ CODE STARTS HERE ###############

        pass

        ################ CODE ENDS HERE ###############

    def iterative_refine(self, prompt: str, max_iterations: int = 3) -> str:
        """
        INPUT:
            prompt: str - Input prompt for generation
            max_iterations: int - Maximum number of refinement iterations
            
        OUTPUT:
            str - Final refined response
        """
        ################ CODE STARTS HERE ###############

        pass

        ################ CODE ENDS HERE ###############
    
    def fuse(self, prompt: str) -> str:
        """
        Queries multiple models with the same prompt and fuses the responses 
        by combining the best elements from each response.

        INPUT:
            prompt: str - Original prompt
            
        OUTPUT:
            str - Single fused response combining the best elements
        """
        ################ CODE STARTS HERE ###############

        pass

        ################ CODE ENDS HERE ###############
        
    def run_pipeline(self, query: str, iterations: int) -> str:
        """
        INPUT:
            query: str - User's natural language query
            iterations: int - Number of iterations of iterative refinement
        OUTPUT:
            str - Final fused response or error message
        """
        ################ CODE STARTS HERE ###############

        pass

        ################ CODE ENDS HERE ###############
