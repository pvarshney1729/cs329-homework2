parse_query_params_prompt = """You are a precise parameter extraction system. Your task is to extract parameters for a specific API function from a natural language query.
            
            <function_name>
            {function_name}
            </function_name>

            <query>
            {query}
            </query>

            Instructions:
            1. Extract explicit parameter values mentioned in the query
            2. Infer implicit parameters based on context
            3. Apply default values where appropriate
            4. Validate parameter formats match requirements
            5. Only include parameters that are defined in the function schema
            6. Maintain correct data types for each parameter
            7. Follow format specifications (e.g., dates as YYYY-MM-DD)

            Example Response:
            {{
                "parameters": {{
                    "location": "New York",
                    "date": "2024-02-17"
                }},
            }}"""

route_query_prompt = """You are an API selection master who picks the appropriate API endpoint to gather information for a given user query from the provided catalog.

            Input Query:
            <query>
            {query}
            </query>

            Available API Catalog:
            <api_catalog>
            {api_catalog}
            </api_catalog>

            Instructions:
            1. Analyze the query for key intent, entities, and required information
            2. Compare query requirements against each API's capabilities
            3. Consider both explicit and implicit parameters in the query
            4. Select the most appropriate API based on functionality match
            5. Validate parameter formats against API specifications
            6. If parameters are missing but required, use reasonable defaults
            7. If parameters are ambiguous, choose the most likely interpretation
            8. If no API clearly matches, select the closest match and explain limitations
            9. If multiple APIs could work, select the most specific/appropriate one
            10. Maintain consistent parameter formats (dates as YYYY-MM-DD, etc.)

            Example:
            For query "What's the weather like in Seattle tomorrow?", response would be:
            {{
                "api_name": "get_weather",
                "parameters": {{
                    "location": "Seattle, WA",
                    "date": "2024-02-18"
                }},
                "reason": "Query explicitly asks for weather information with clear location and time indicators"
            }}
"""



decompose_query_prompt = """
        <QUERY>
        {query}
        </QUERY>

        <INSTRUCTIONS>
        1. Analyze what information is needed to fully answer the query
        2. Create sub-queries that:
        - Are specific and focused on one piece of information
        - Can be answered independently
        - Are in natural language (don't worry about API specifics)
        - Build towards the final answer
        3. Order the sub-queries logically:
        - Consider dependencies between information
        - Put prerequisite information first
        - Structure for building a complete answer
        </INSTRUCTIONS>

        IMPORTANT: Return ONLY a valid JSON object with this EXACT structure:
        {{
            "sub_queries": [
                {{
                    "query": "natural language sub-query",
                    "order": "execution order (1, 2, 3, etc.)",
                    "purpose": "why this information is needed"
                }}
            ],
            "reasoning": "explanation of how these sub-queries will help answer the original query"
        }}

        EXAMPLES:

        Complex Query: "Should I invest in AAPL given their new product announcement and current market conditions?"

        {{
            "sub_queries": [
                {{
                    "query": "What new product did Apple recently announce?",
                    "order": 1,
                    "purpose": "Understand the product announcement's potential impact"
                }},
                {{
                    "query": "What is the current market sentiment towards tech stocks?",
                    "order": 2,
                    "purpose": "Understand broader market context"
                }},
                {{
                    "query": "How has Apple's stock been performing recently?",
                    "order": 3,
                    "purpose": "Understand current stock performance"
                }}
            ],
            "reasoning": "To make an investment decision, we need to understand the product announcement's significance, evaluate it in the context of current market conditions, and consider recent stock performance"
        }}

        IMPORTANT:
        - Focus on WHAT information is needed, not HOW to get it
        - Write sub-queries in natural language
        - Don't specify APIs or parameters (this will be handled separately)
        - Ensure each sub-query has a clear, specific purpose
        - Order queries logically

        NOW DECOMPOSE THE GIVEN QUERY."""




verification_prompt = """The following query has been asked by a user: {current_prompt}
                                      
                                      To answer this query, the following answers were generated incrementally using information retrieved using APIs:
                                      {fused_responses}

                                    Analyze this response for:
                                    1. Completeness: Are all aspects of the query addressed?
                                    2. Accuracy: Is the information provided accurate and well-supported?
                                    3. Clarity: Is the response clear and well-organized?

                                    If improvements are needed, specify exactly what information is missing or clarification is needed.
                                    If the response is satisfactory, state that "no further refinement needed".

                                    Provide your analysis STRICTLY in this structured format: 

                                    <EVALUATION>
                                    Accuracy: [Score: 1-5] - [Brief justification]
                                    Completeness: [Score: 1-5] - [Brief justification]
                                    Reasoning: [Score: 1-5] - [Brief justification]
                                    Depth: [Score: 1-5] - [Brief justification]
                                    </EVALUATION>
                                        
                                    <MISSING_INFORMATION>
                                    [Clear, specific information/clarification that is still missing or could improve the answer the original query well]
                                    </MISSING_INFORMATION>

                                    <SATISFACTORY>
                                    [ONE WORD: YES if all the response answers the query well, NO if it does not]
                                    </SATISFACTORY>
                                    
                                    EXAMPLES:
                                    Query: "What is the capital of France?"
                                    Response: "The capital of France is Paris."
                                    <MISSING_INFORMATION>
                                    None
                                    </MISSING_INFORMATION>
                                    <SATISFACTORY>
                                    YES
                                    </SATISFACTORY>

                                    Query: "List the top 3 highest mountains in the world with their heights."
                                    Response: "Mount Everest, K2, Kangchenjunga."
                                    <MISSING_INFORMATION>
                                    Specify the heights of each mountain.
                                    </MISSING_INFORMATION>
                                    <SATISFACTORY>
                                    NO
                                    </SATISFACTORY>

                                    Query: "Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited? Just give me the city name without abbreviations.
                                    Response': St. Petersburg,
                                    <MISSING_INFORMATION>
                                    The query expects city name without ANY abbreviations.
                                    The query expects city name without ANY abbreviations.
                                    </MISSING_INFORMATION>
                                    <SATISFACTORY>
                                    NO
                                    </SATISFACTORY>
                                    """




fusion_prompt = """
SOURCE RESPONSES:
{source_responses}

<OUTPUT_FORMAT>
Your response MUST follow these guideliens:
- A clear, direct answer synthesized from all sources (A few words or 1 sentence maximum).
- 1-2 key findings from model responsesthat support your final conclusion.
- A confidence score from 1-5, where 5 is the highest confidence.
- List the primary model source (most reliable/detailed model) and supporting model sources (that provided corroborating evidence)
</OUTPUT_FORMAT>

<INSTRUCTIONS>
1. Maintain XML tags exactly as shown
2. For each section:
   - DIRECT_ANSWER: Focus on consensus, note uncertainty when significant
   - KEY_FINDINGS: List only facts with explicit agreement from the model responses
   - CONFIDENCE_ASSESSMENT: Rate claims based on source quality and agreement
3. Do not add any qualifiers or anything else outside of the tags.
4. If there can be multiple answers to the same query, e.g., pincodes for a particular species, then list all the pincodes in the DIRECT_ANSWER tag.

CRITICAL RULES:
- Never invent information not present in source responses
- Always cite specific models for claims
- Use consistent terminology across sections
- Be explicit about uncertainty and confidence levels
- Prioritize precision over completeness

Generate your response following this exact structure."""


consensus_prompt = """Given multiple model responses to this question, determine the single most reliable answer.

ORIGINAL QUESTION: {original_query}

MODEL RESPONSES:
{model_responses}

INSTRUCTIONS:
1. Analyze the responses and select the most reliable answer based on:
   - Agreement across multiple responses
   - Specificity and precision of answers
   - Consistency with question format
   
2. Response Requirements:
   - Provide ONLY the answer with no explanation
   - Match the exact format of the original question
   - If no clear consensus, return the claude model's response
   - Do not add qualifiers or additional context
   - Do not explain your reasoning
   
3. Format Examples:
   Question: "What year did X happen?"
   Valid Answer: "1999"
   Invalid Answer: "Based on the responses, it occurred in 1999"
   
   Question: "Who won the award?"
   Valid Answer: "John Smith"
   Invalid Answer: "Multiple sources indicate John Smith won"

Your answer (ONLY the answer, no explanation):"""



search_query_optimization_prompt = """Given the user question, optimize the search query while preserving crucial context and intent.

<User Question>
{query}
</User Question>

Original Search Query: {search_query}

CORE PRINCIPLES:
1. Maintain all specific names (ESPECIALLY PERSON NAMES), dates, and identifying details from the original query
2. Keep unique identifiers (episode numbers, titles, specific events)
3. Don't generalize specific questions into broad topics. Keep the original search goal clear and focused. Think: Does the optimized query still answer the original question?
4. IMPORTANT: Preserve the type of information being requested (who, what, when, where)
5. Add quotation marks around exact phrases, dates, etc. that must appear together
6. Use site-specific searches only when highly relevant (e.g., site:github.com for code issues). Include factual constraints (after:YYYY, before:YYYY) when time-relevance is crucial.

EXAMPLES:

Original: "Who was the recipient of the Best VTuber Streamer award at the 2022 Streamer Awards?"
BAD: "Ironmouse Best VTuber Streamer Award 2022 Streamer Awards winner confirmed"
GOOD: "2022 Streamer Awards Best VTuber winner"

Original: "What is the season, episode number, and title of the Archer episode featuring Sterling Archer's rampage?"
BAD: "Archer TV show Sterling Archer violent rampage episode season number title"
GOOD: "Archer TV series Sterling Archer rampage episode title season"

HANDLING AMBIGUOUS QUERIES:
- If the query lacks context, optimize based only on available information
- Don't add assumptions or specifics not present in the original query

PLEASE RETURN ONLY THE OPTIMIZED QUERY NOTHING ELSE. NO EXPLANATION OR REASONING IN YOUR RESPONSE."""