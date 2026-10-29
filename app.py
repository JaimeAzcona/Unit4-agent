import os
import gradio as gr
import requests
import inspect
import pandas as pd
import random
import datasets
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from huggingface_hub import HfApi
from smolagents import Tool, CodeAgent, OpenAIModel
from youtube_transcript_api import YouTubeTranscriptApi
from tavily import TavilyClient
import base64
from dotenv import load_dotenv

# --- Constants ---
load_dotenv()
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
API_KEY= os.getenv("OPENAI_API_KEY")

# 2. Web Search Tool / Web Page Viewer
class TavilySearchTool(Tool):
    name = "web_search"
    description = "Searches the web for up-to-date information. Use this to find facts, news, and external data."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query."
        }
    }
    output_type = "string"

    def forward(self, query: str):
        try:
            client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            # include_raw_content pulls the actual page text so the agent doesn't have to visit the URL
            response = client.search(query=query, include_raw_content=True) 
            
            formatted_results = []
            for res in response['results']:
                formatted_results.append(f"Title: {res['title']}\nContent: {res['content']}")
            return "\n\n".join(formatted_results)
        except Exception as e:
            error_msg = f"Search error: {e}"
            print(error_msg)
            return error_msg



class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = "Visits a webpage and extracts its readable text content."
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL of the webpage to visit."
        }
    }
    output_type = "string"

    def forward(self, url: str):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator=' ', strip=True)
            if len(text) > 5000:
                return text[:5000] + "... [Content truncated for length]"
            return text
        except Exception as e:
            return f"Error visiting {url}: {str(e)}"

class TranscribeAudioTool(Tool):
    name = "transcribe_audio"
    description = "Transcribes an audio file (like MP3) into text. Use this when the task mentions an audio file."
    inputs = {
        "file_path": {
            "type": "string",
            "description": "The local path to the audio file."
        }
    }
    output_type = "string"

    def forward(self, file_path: str):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=API_KEY)
            with open(file_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return transcript.text
        except Exception as e:
            return f"Error transcribing audio: {e}"

class AnalyzeImageTool(Tool):
    name = "analyze_image"
    description = "Analyzes an image file and answers a question about it. Use this when the task provides an image file."
    inputs = {
        "file_path": {
            "type": "string",
            "description": "The local path to the image file."
        },
        "question": {
            "type": "string",
            "description": "The specific question to ask about the image."
        }
    }
    output_type = "string"

    def forward(self, file_path: str, question: str):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=API_KEY)
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"Error transcribing audio: {e}"
            print(error_msg) # This prints to your terminal so you can debug it
            return error_msg
class YouTubeTranscriptTool(Tool):
    name = "get_youtube_transcript"
    description = "Fetches the text transcript of a YouTube video. Use this to find out what is spoken in a YouTube video."
    inputs = {
        "video_url": {
            "type": "string",
            "description": "The full YouTube URL (e.g., https://www.youtube.com/watch?v=...)"
        }
    }
    output_type = "string"

    def forward(self, video_url: str):
        try:
            # Extract the video ID from the URL
            if "v=" in video_url:
                video_id = video_url.split("v=")[-1].split("&")[0]
            elif "youtu.be/" in video_url:
                video_id = video_url.split("youtu.be/")[-1].split("?")[0]
            else:
                return "Error: Could not extract video ID from URL."
            
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            # Combine all the text blocks into one readable string
            full_text = " ".join([entry['text'] for entry in transcript])
            return full_text
        except Exception as e:
            return f"Error fetching YouTube transcript: {e}"

# Initialize tools
search_tool = TavilySearchTool()
visit_tool = VisitWebpageTool()
transcribe_tool = TranscribeAudioTool()
analyze_image_tool = AnalyzeImageTool()
youtube_tool = YouTubeTranscriptTool()

# --- Agent Definition ---


class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
        self.model = OpenAIModel(model_id="gpt-4o", api_key=API_KEY)
        self.agent = CodeAgent(
            tools=[visit_tool, search_tool, transcribe_tool, analyze_image_tool, youtube_tool],
            model=self.model,
            additional_authorized_imports=['pandas', 'openpyxl'],
            max_steps=6,
            verbosity_level=1
        )

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        
        gag_order_rules = """
        
        CRITICAL RULES FOR EXECUTION:
        1. NO CONVERSATION: You must output ONLY the final raw answer. No preambles, no explanations, no polite filler. If the answer is 5, output "5".
        2. NO SURRENDER: If code or a tool fails, write a new script and try again. Never tell the user to do it manually.
        3. LOCAL FILES: If this task involves a downloaded file, it is located in your current working directory. Pass the exact filename to your tools, do not invent file paths.
        """
        
        enforced_question = question + gag_order_rules
        
        try:
            result = self.agent.run(enforced_question)
            return str(result).strip()
        except Exception as e:
            print(f"Error running agent: {e}")
            return f"Agent Error: {e}"

def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")

    username= "jazcona"
    print(f"User logged in: {username}")

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             return "Fetched questions list is empty or invalid format.", None
    except Exception as e:
        return f"Error fetching questions: {e}", None

    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name") # Extract the file name
        
        if not task_id or question_text is None:
            continue
        if file_name:
            print(f"Downloading attachment: {file_name}...")
            file_url = f"https://agents-course-unit4-scoring.hf.space/files/{task_id}"
            try:
                file_response = requests.get(file_url)
                if file_response.status_code == 200:
                    with open(file_name, "wb") as f:
                        f.write(file_response.content)
                    print(f"Successfully downloaded {file_name}.")
                    # Inform the agent that the file is available locally
                    question_text += f"\n\n[Note: A file named '{file_name}' has been downloaded to your local directory for this task. CRITICAL INSTRUCTION: You are an autonomous coding agent. Do NOT provide manual, conversational tutorials. You must write and execute Python code (using pandas) to read this file and output the exact final calculated answer.]"
                else:
                     print(f"Failed to download {file_name} (HTTP {file_response.status_code}).")
            except Exception as e:
                print(f"Error downloading file {file_name}: {e}")

        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})
             answers_payload.append({"task_id": task_id, "submitted_answer": f"Error: {e}"})

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        return final_status, pd.DataFrame(results_log)
    except Exception as e:
        return f"Submission Failed: {e}", pd.DataFrame(results_log)

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown("Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.")
    
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
    
    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
