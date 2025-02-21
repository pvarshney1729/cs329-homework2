import os
import requests
from datetime import datetime
from openai import OpenAI
import json
from googleapiclient.discovery import build
from textblob import TextBlob
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote
import pytz
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

import ssl
import certifi
import httplib2
from cs329_hw2.utils import generate_openai, generate_anthropic

import re
from . import prompts

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

    
function_schemas = {
    "google_search": {
        "description": "Performs a Google search and returns relevant search results",
        "parameters": {
            "search_term": "string",
            "num_results": "integer (optional, default: 3)"
        },
        "parameter_descriptions": {
            "search_term": "The query string to search for. Can include Boolean operators (AND, OR, NOT) and special operators like site:, filetype:",
            "num_results": "Number of search results to return. Must be between 1 and 100. Defaults to 3 if not specified."
        },
        "example": {
            "input": {
                "search_term": "machine learning tutorials site:github.com",
                "num_results": 3
            }
        }
    },
    "get_stock_data": {
        "description": "Retrieves historical or real-time stock market data for a given symbol",
        "parameters": {
            "symbol": "string (e.g., AAPL, TSLA)",
            "date": "string (optional, YYYY-MM-DD format)"
        },
        "parameter_descriptions": {
            "symbol": "Stock ticker symbol. Must be a valid symbol listed on major exchanges (NYSE, NASDAQ, etc.)",
            "date": "Historical date for stock data. If omitted, returns latest available data. Must not be a future date or weekend."
        },
        "example": {
            "input": {
                "symbol": "AAPL",
                "date": "2024-02-15"
            }
        }
    },
    "get_weather": {
        "description": "Retrieves weather forecast or historical weather data for a specific location",
        "parameters": {
            "location": "string (e.g., 'Palo Alto, CA')",
            "date": "string (YYYY-MM-DD format)",
            "hour": "string (24-hour format)"
        },
        "parameter_descriptions": {
            "location": "City name with optional state/country. Can also accept latitude/longitude coordinates",
            "date": "Date for weather data. Cannot be more than 7 days in future or 30 days in past",
            "hour": "Specific hour for weather data in 24-hour format (00-23). If omitted, returns full day forecast"
        },
        "example": {
            "input": {
                "location": "Palo Alto, CA",
                "date": "2024-02-17",
                "hour": "14"
            }
        }
    },
    "analyze_sentiment": {
        "description": "Performs sentiment analysis on provided text using natural language processing",
        "parameters": {
            "text": "string (text to analyze)"
        },
        "parameter_descriptions": {
            "text": "Input text for sentiment analysis. Must be between 1 and 10000 characters. Supports multiple languages."
        },
        "example": {
            "input": {
                "text": "I absolutely love this new feature! It's incredibly useful and well-designed."
            }
        }
    }
}

class APIManager:
    """A unified class to manage various API interactions"""
    
    def __init__(self, google_api_key: str, google_cx_id: str, alpha_vantage_key: str):
        """
        INPUT:
            google_api_key: str - Google API key for Custom Search
            google_cx_id: str - Google Custom Search Engine ID
            alpha_vantage_key: str - Alpha Vantage API key
            
        Initializes API keys and configurations
        """
        ################ CODE STARTS HERE ###############

        self.google_api_key = google_api_key
        self.google_cx_id = google_cx_id
        self.alpha_vantage_key = alpha_vantage_key
        self.google_search_client = build("customsearch", "v1", developerKey=self.google_api_key)

        ################ CODE ENDS HERE ###############

    def parse_query_params(self, query: str, function_name: str) -> Optional[Dict]:
        """
        INPUT:
            query: str - Natural language query from user
            function_name: str - Name of the function to parse parameters for
            
        OUTPUT:
            Optional[Dict] - Parameters needed for the specified function or None if parsing fails
        """
        ################ CODE STARTS HERE ###############

        try:

            client = OpenAI()

            schema = function_schemas[function_name]

            function_dict = {
                "name": function_name,
                "parameters": {
                    "type": "object",
                    "properties": {
                        # Only include the main parameters without complex nesting
                        param: {
                            "type": "string" if "string" in param_type.lower() else "integer",
                            "description": schema["parameter_descriptions"][param]
                        }
                        for param, param_type in schema["parameters"].items()
                    },
                    "required": [param for param in schema["parameters"].keys() 
                            if "optional" not in schema["parameters"][param].lower()]
                }
            }

            prompt = prompts.parse_query_params_prompt.format(
                function_name=function_name,
                query=query
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                functions=[function_dict],
                function_call={"name": function_name}
            )

            message = response.choices[0].message

            return json.loads(message.function_call.arguments)
                        
        except Exception as e:
            print(f"Error in parse_query_params: {str(e)}")
            return None            

        ################ CODE ENDS HERE ###############

    def route_query(self, query: str, optimize_query_flag: bool = False) -> Dict:
        """
        INPUT:
            query: str - Natural language query to route
            
        OUTPUT:
            Dict containing:
                - results: Any - Results from the API call
                - api_used: str - Name of the API that was used
                - error: str (optional) - Error message if something went wrong
        """
        ################ CODE STARTS HERE ###############

        try: 

            client = OpenAI()

            function_definition = {
                "name": "route_query",
                "description": "Select the best matching API for the query and extract necessary parameters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "api_name": {
                            "type": "string",
                            "enum": list(function_schemas.keys()),
                            "description": "The name of the API to use."
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Dictionary of key-value pairs of parameters for the selected API."
                        },
                        "reason": {
                            "type": "string",
                            "description": "Clear explanation of the selection rationale."
                        }
                    },
                    "required": ["api_name", "parameters", "reason"]
                }
            }

            prompt = prompts.route_query_prompt.format(
                query=query,
                api_catalog=json.dumps(function_schemas, indent=2)
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                functions=[function_definition],
                function_call="auto",
            )

            message = response.choices[0].message

            if message.function_call:
                routing_info = json.loads(message.function_call.arguments)
                # print("\nRouting Info:")
                # print(routing_info)
                api_name = routing_info["api_name"]
            else:
                print("No function call returned in route_query.")
                return None

            api_name = routing_info["api_name"]

            params = self.parse_query_params(query, api_name)

            if not params:
                return {
                    "results": None,
                    "api_used": api_name,
                    "params": params,
                    "error": "Failed to parse parameters"
                }
            
            if api_name == "google_search":
                results = self.google_search(**params)
                return {
                    "results": results,
                    "api_used": api_name,
                    "params": params,
                    "error": None
                }
            elif api_name == "get_stock_data":
                results = self.get_stock_data(**params)
            elif api_name == "get_weather":
                results = self.get_weather(**params)
            elif api_name == "analyze_sentiment":
                results = self.analyze_sentiment(**params)
            else:
                results = self.google_search(**params)
                return {
                    "results": results,
                    "api_used": api_name,
                    "params": params,
                    "error": None
                }
            
            return {
                    "results": results,
                    "api_used": api_name,
                    "params": params,
                    "error": None
                }

        except Exception as e:
            print(f"Error in route_query: {str(e)}")
            return {
                "results": None,
                "api_used": None,
                "params": None,
                "error": f"Routing error: {str(e)}"
            }

        ################ CODE ENDS HERE ###############
        
    def google_search(self, search_term: str, num_results: int = 3) -> List[Dict]:
        """
        INPUT:
            search_term: str - The search query
            num_results: int - Number of results to return (default: 3)
            
        OUTPUT:
            List[Dict] - List of search results, each containing:
                - title: str
                - link: str
                - snippet: str
                - webpage_content: Dict (optional)
        """
        ################ CODE STARTS HERE ###############

        try:

            search_term = f"{search_term} -filetype:pdf"
        
        
            http = httplib2.Http(
                timeout=5,
                ca_certs=certifi.where()
            )
            service = build(
                "customsearch", 
                "v1", 
                developerKey=self.google_api_key,
                http=http
            )

            search_results = service.cse().list(
                q=search_term, 
                cx=self.google_cx_id, 
                num=num_results
            ).execute()

            results = []
            if 'items' in search_results:
                for item in search_results['items']:

                    if item.get('link', '').lower().endswith('.pdf'):
                        continue

                    parsed_url = urlparse(item.get('link', ''))
                    clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{quote(parsed_url.path)}"
                
                    results.append({
                        'title': item.get('title', ''),
                        'link': clean_url,
                        'snippet': item.get('snippet', '')
                    })

                    try:
                        # Skip content fetching for PDF files
                        if not item['link'].lower().endswith('.pdf'):
                            response = requests.get(
                                clean_url,
                                timeout=10,
                                verify=certifi.where(),
                                headers={
                                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                                }
                            )

                            if response.status_code == 200:
                                soup = BeautifulSoup(response.text, 'html.parser')

                                page_title = ''
                                meta_title = soup.find('meta', {'property': 'og:title'})
                                if meta_title:
                                    page_title = meta_title.get('content', '')
                                if not page_title:
                                    title_tag = soup.find('title')
                                    if title_tag:
                                        page_title = title_tag.strip()

                                images = []
                                for img in soup.find_all('img'):
                                    img_info = {
                                        'alt': img.get('alt', ''),
                                        'src': img.get('src', ''),
                                    }
                                    if img_info['alt'] or img_info['src']:
                                        images.append(img_info)

                                content = soup.get_text()[:10000]
                                results[-1]['webpage_content'] = {
                                    'title': page_title,
                                    'clean_url': clean_url,
                                    'text_content': content,
                                    'images': images
                                }
                    except Exception as e:
                        # print(f"Error fetching content from {clean_url}: {e}")
                        None
                return results
        
        except Exception as google_error:
            print(f"Google API error: {str(google_error)}")
            return []

        return []
                        
        ################ CODE ENDS HERE ###############
    
    def get_stock_data(self, symbol: str, date: Optional[str] = None) -> Dict:
        """
        INPUT:
            symbol: str - Stock symbol (e.g., 'AAPL')
            date: Optional[str] - Date in format 'YYYY-MM-DD'
            
        OUTPUT:
            Dict containing either:
                Current data:
                    - symbol: str
                    - price: float
                    - change: float
                    - change_percent: str
                Historical data:
                    - date: str
                    - open: float
                    - high: float
                    - low: float
                    - close: float
                    - volume: int
        """
        ################ CODE STARTS HERE ###############

        try:
                base_url = "https://www.alphavantage.co/query"
                
                if date:
                    # Get historical data
                    params = {
                        "function": "TIME_SERIES_DAILY",
                        "symbol": symbol,
                        "apikey": self.alpha_vantage_key,
                        "outputsize": "full"  # Get full data to ensure we have historical dates
                    }
                    
                    response = requests.get(base_url, params=params)
                    data = response.json()
                    
                    # Check for error messages
                    if "Error Message" in data:
                        print(f"API Error: {data['Error Message']}")
                        return {}
                        
                    # Get historical data for specific date
                    if "Time Series (Daily)" in data and date in data["Time Series (Daily)"]:
                        daily_data = data["Time Series (Daily)"][date]
                        return {
                            "date": date,
                            "open": float(daily_data["1. open"]),
                            "high": float(daily_data["2. high"]),
                            "low": float(daily_data["3. low"]),
                            "close": float(daily_data["4. close"]),
                            "volume": int(daily_data["5. volume"])
                        }
                    else:
                        print(f"No data available for {symbol} on {date}")
                        return {}
                        
                else:
                    # Get current data using GLOBAL_QUOTE endpoint
                    params = {
                        "function": "GLOBAL_QUOTE",
                        "symbol": symbol,
                        "apikey": self.alpha_vantage_key
                    }
                    
                    response = requests.get(base_url, params=params)
                    data = response.json()
                    
                    # Check for error messages
                    if "Error Message" in data:
                        print(f"API Error: {data['Error Message']}")
                        return {}
                        
                    if "Global Quote" in data:
                        quote = data["Global Quote"]
                        return {
                            "symbol": symbol,
                            "price": float(quote["05. price"]),
                            "change": float(quote["09. change"]),
                            "change_percent": quote["10. change percent"].rstrip('%')  # Remove % symbol
                        }
                    else:
                        print(f"No current data available for {symbol}")
                        return {}
                        
        except Exception as e:
                print(f"Error fetching stock data: {str(e)}")
                return {}
        ################ CODE ENDS HERE ###############
    
    @staticmethod
    def analyze_sentiment(text: str) -> Dict:
        """
        INPUT:
            text: str - Text to analyze
            
        OUTPUT:
            Dict containing:
                - sentiment: str - "positive", "negative", or "neutral"
                - polarity: float - Sentiment polarity score
                - subjectivity: float - Subjectivity score
        """
        ################ CODE STARTS HERE ###############

        try:

            blob = TextBlob(text)

            polarity = blob.sentiment.polarity

            if polarity > 0:
                sentiment = 'positive'
            elif polarity < 0:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            return {
                'sentiment': sentiment,
                'polarity': polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
            
        except Exception as e:
            print(f"Error in analyze_sentiment: {str(e)}")
            return {
                'sentiment': 'neutral',
                'polarity': 0,
                'subjectivity': 0
            }
            
        ################ CODE ENDS HERE ###############
    
    @staticmethod
    def get_weather(location: str, date: str, hour: str = "12") -> Dict:
        """
        INPUT:
            location: str - Location string (e.g., "Palo Alto, CA")
            date: str - Date in YYYY-MM-DD format
            hour: str - Hour in 24-hour format (default: "12")
            
        OUTPUT:
            Dict containing:
                - temperature: str
                - weather_description: str
                - humidity: str
                - wind_speed: str, any wind speed value is acceptable
        """
        ################ CODE STARTS HERE ###############

        try:

            coordinates = APIManager._get_coordinates(location)

            if not coordinates:
                print(f"Could not find coordinates for {location}")
                return {}   
            
            latitude, longitude = coordinates
            
            base_url = "https://api.open-meteo.com/v1/forecast"

            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": ["temperature_2m", "relativehumidity_2m", "windspeed_10m", "weathercode",
                           "precipitation"],
                "timezone": "auto",
                "start_date": date,
                "end_date": date
            }
            
            response = requests.get(base_url, params=params)

            if response.status_code != 200:
                print(f"Error fetching weather data: {response.status_code}")
                return {}

            data = response.json()

            try:
                hour_int = int(hour)
                if hour_int < 0 or hour_int > 23:
                    print(f"Invalid hour: {hour}")
                    hour_int = 12  # Default to noon if invalid
            except (TypeError, ValueError):
                print(f"Invalid hour format: {hour}")
                hour_int = 12  # Default to noon if invalid
            
            hourly_data = data["hourly"]

            temperature = str(hourly_data["temperature_2m"][hour_int])
            precipitation = str(hourly_data["precipitation"][hour_int])
            humidity = str(hourly_data["relativehumidity_2m"][hour_int])
            wind_speed = str(hourly_data["windspeed_10m"][hour_int])
            weather_description = APIManager._get_weather_description(hourly_data["weathercode"][hour_int])


            weather_data = {
                "temperature": temperature,
                "precipitation": precipitation,
                "wind_speed": wind_speed,
                "humidity": humidity,
                "weather_description": weather_description
            }

            return weather_data
                
            
        except Exception as e:
            print(f"Error in get_weather: {str(e)}")
            return {}
            
            
            

        ################ CODE ENDS HERE ###############
    
    @staticmethod
    def _get_coordinates(location: str) -> Optional[tuple]:
        """
        INPUT:
            location: str - Location name to geocode
            
        OUTPUT:
            Optional[tuple] - (latitude: float, longitude: float) or None if not found
        """
        ################ CODE STARTS HERE ###############

        try:

            geolocator = Nominatim(user_agent="my_weather_app")

            location = geolocator.geocode(location)

            if location:
                return (location.latitude, location.longitude)
            else:
                return None
            
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"Error in Geocoding _get_coordinates: {str(e)}")
            return None
        
        except Exception as e:
            print(f"Error in _get_coordinates: {str(e)}")
            return None
        
        ################ CODE ENDS HERE ###############
    
    @staticmethod
    def _get_weather_description(code: int) -> str:
        """
        INPUT:
            code: int - Weather condition code
            
        OUTPUT:
            str - Human-readable weather description
        """
        ################ CODE STARTS HERE ###############

        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }
        return weather_codes.get(code, "Unknown weather condition")

        ################ CODE ENDS HERE ###############