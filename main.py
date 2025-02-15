from fastapi import FastAPI, HTTPException, UploadFile, File
import os
import json
import aiohttp
import re
from datetime import datetime
from typing import Dict, Any, List
import sqlite3
import glob
from dotenv import load_dotenv
import base64
import subprocess
import difflib

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

class AIProxy:
    def __init__(self, token: str):
        self.token = token
        self.api_url = "https://api.aiproxy.cloud/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    async def get_completion(self, prompt: str) -> str:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                if response.status != 200:
                    raise HTTPException(status_code=500, detail="AI Proxy request failed")
                data = await response.json()
                return data['choices'][0]['message']['content']

class TaskHandler:
    def __init__(self, ai_proxy: AIProxy):
        self.ai_proxy = ai_proxy

    # Dispatcher: examines the task description and calls the corresponding handler.
    async def handle_task(self, task_description: str) -> Dict[str, Any]:
        lower_task = task_description.lower()
        if "datagen" in lower_task or "generate data" in lower_task:
            return await self.handle_datagen(task_description)
        elif "format" in lower_task and "prettier" in lower_task:
            return await self.handle_format_file(task_description)
        elif "wednesday" in lower_task and "date" in lower_task:
            return await self.handle_count_wednesdays(task_description)
        elif "sort contacts" in lower_task:
            return await self.handle_sort_contacts(task_description)
        elif "recent logs" in lower_task:
            return await self.handle_recent_logs(task_description)
        elif "extract headers" in lower_task:
            return await self.handle_extract_headers(task_description)
        elif "extract email" in lower_task:
            return await self.handle_extract_email(task_description)
        elif "extract card" in lower_task:
            return await self.handle_extract_card(task_description)
        elif "similar comments" in lower_task or "most similar" in lower_task:
            return await self.handle_similar_comments(task_description)
        elif "ticket sales" in lower_task:
            return await self.handle_ticket_sales(task_description)
        else:
            raise ValueError(f"Unknown task: {task_description}")

    # A1: Data Generation using external script
    async def handle_datagen(self, task_description: str) -> Dict[str, Any]:
        # Get the user email from environment variable
        user_email = os.environ.get("USER_EMAIL")
        if not user_email:
            raise HTTPException(status_code=400, detail="USER_EMAIL environment variable not set")
        # Download the datagen.py script
        datagen_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
        download_cmd = ["curl", "-fsSL", datagen_url, "-o", "/tmp/datagen.py"]
        result_download = subprocess.run(download_cmd, capture_output=True, text=True)
        if result_download.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Failed to download datagen.py: {result_download.stderr}")
        # Run the datagen.py script with the user's email
        run_cmd = ["python", "/tmp/datagen.py", user_email]
        result_run = subprocess.run(run_cmd, capture_output=True, text=True)
        if result_run.returncode != 0:
            raise HTTPException(status_code=500, detail=f"datagen.py execution failed: {result_run.stderr}")
        return {"status": "success", "output": result_run.stdout}

    # A2: Format /data/format.md using prettier@3.4.2
    async def handle_format_file(self, task_description: str) -> Dict[str, Any]:
        cmd = ["npx", "prettier@3.4.2", "--write", "/data/format.md"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Prettier formatting failed: {result.stderr}")
        return {"status": "success", "output": result.stdout}

    # A3: Count Wednesdays in /data/dates.txt
    async def handle_count_wednesdays(self, task_description: str) -> Dict[str, Any]:
        input_file = "/data/dates.txt"
        output_file = "/data/dates-wednesdays.txt"
        if not os.path.exists(input_file):
            raise HTTPException(status_code=404, detail=f"File not found: {input_file}")
        count = 0
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    # Try parsing the date; adjust the format if needed.
                    dt = datetime.strptime(line, "%Y-%m-%d")
                    if dt.weekday() == 2:  # Monday=0, Tuesday=1, Wednesday=2
                        count += 1
                except Exception as e:
                    # If date parsing fails, skip the line
                    continue
        with open(output_file, 'w') as f:
            f.write(str(count))
        return {"status": "success", "wednesdays_count": count}

    # A4: Sort Contacts (already implemented)
    async def handle_sort_contacts(self, task_description: str) -> Dict[str, Any]:
        input_file = "/data/contacts.json"
        output_file = "/data/contacts-sorted.json"
        if not os.path.exists(input_file):
            raise HTTPException(status_code=404, detail=f"Input file not found: {input_file}")
        with open(input_file, 'r') as f:
            contacts = json.load(f)
        sorted_contacts = sorted(contacts, key=lambda x: (x.get('last_name', ''), x.get('first_name', '')))
        with open(output_file, 'w') as f:
            json.dump(sorted_contacts, f, indent=2)
        return {"status": "success", "contacts_sorted": len(sorted_contacts)}

    # A5: Recent Logs (already implemented)
    async def handle_recent_logs(self, task_description: str) -> Dict[str, Any]:
        log_dir = "/data/logs/"
        output_file = "/data/logs-recent.txt"
        log_files = glob.glob(f"{log_dir}*.log")
        recent_logs = sorted(log_files, key=lambda x: os.path.getmtime(x), reverse=True)[:10]
        first_lines = []
        for log_file in recent_logs:
            with open(log_file, 'r') as f:
                first_lines.append(f.readline().strip())
        with open(output_file, 'w') as f:
            f.write("\n".join(first_lines))
        return {"status": "success", "logs_processed": len(first_lines)}

    # A6: Extract Headers from Markdown files (already implemented)
    async def handle_extract_headers(self, task_description: str) -> Dict[str, Any]:
        docs_dir = "/data/docs/"
        output_file = "/data/docs/index.json"
        md_files = glob.glob(f"{docs_dir}**/*.md", recursive=True)
        headers = {}
        for md_file in md_files:
            with open(md_file, 'r') as f:
                content = f.read()
                match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                if match:
                    relative_path = os.path.relpath(md_file, docs_dir)
                    headers[relative_path] = match.group(1)
        with open(output_file, 'w') as f:
            json.dump(headers, f, indent=2)
        return {"status": "success", "files_processed": len(headers)}

    # A7: Extract Email (already implemented)
    async def handle_extract_email(self, task_description: str) -> Dict[str, Any]:
        input_file = "/data/email.txt"
        output_file = "/data/email-sender.txt"
        if not os.path.exists(input_file):
            raise HTTPException(status_code=404, detail=f"Input file not found: {input_file}")
        with open(input_file, 'r') as f:
            email_content = f.read()
        prompt = f"Extract just the sender's email address from this email:\n{email_content}\nReturn only the email address."
        email = await self.ai_proxy.get_completion(prompt)
        with open(output_file, 'w') as f:
            f.write(email.strip())
        return {"status": "success", "email": email.strip()}

    # A8: Extract Card (already implemented)
    async def handle_extract_card(self, task_description: str) -> Dict[str, Any]:
        input_file = "/data/credit-card.png"
        output_file = "/data/credit-card.txt"
        if not os.path.exists(input_file):
            raise HTTPException(status_code=404, detail=f"Input file not found: {input_file}")
        with open(input_file, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode()
        prompt = "Extract the credit card number from this image. Return only the numbers without spaces."
        card_number = await self.ai_proxy.get_completion(prompt)
        card_number = re.sub(r'\D', '', card_number)
        with open(output_file, 'w') as f:
            f.write(card_number)
        return {"status": "success", "card_number": card_number}

    # A9: Find Most Similar Pair of Comments using a simple similarity measure
    async def handle_similar_comments(self, task_description: str) -> Dict[str, Any]:
        input_file = "/data/comments.txt"
        output_file = "/data/comments-similar.txt"
        if not os.path.exists(input_file):
            raise HTTPException(status_code=404, detail=f"File not found: {input_file}")
        with open(input_file, 'r') as f:
            comments = [line.strip() for line in f if line.strip()]
        if len(comments) < 2:
            raise HTTPException(status_code=400, detail="Not enough comments to compare")
        best_ratio = 0
        best_pair = ("", "")
        for i in range(len(comments)):
            for j in range(i+1, len(comments)):
                ratio = difflib.SequenceMatcher(None, comments[i], comments[j]).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pair = (comments[i], comments[j])
        with open(output_file, 'w') as f:
            f.write(best_pair[0] + "\n" + best_pair[1])
        return {"status": "success", "similarity_ratio": best_ratio}

    # A10: Calculate Ticket Sales (already implemented)
    async def handle_ticket_sales(self, task_description: str) -> Dict[str, Any]:
        db_file = "/data/ticket-sales.db"
        output_file = "/data/ticket-sales-gold.txt"
        if not os.path.exists(db_file):
            raise HTTPException(status_code=404, detail=f"Database file not found: {db_file}")
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
        total_sales = cursor.fetchone()[0]
        conn.close()
        with open(output_file, 'w') as f:
            f.write(str(total_sales))
        return {"status": "success", "total_sales": total_sales}

# Initialize AIProxy and TaskHandler
ai_proxy = AIProxy(os.environ.get("AIPROXY_TOKEN", ""))
task_handler = TaskHandler(ai_proxy)

@app.post("/run")
async def run_task(task: str):
    try:
        result = await task_handler.handle_task(task)
        return {"status": "success", "result": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read")
async def read_file(path: str):
    try:
        # Restrict file access to within /data for security
        if not path.startswith("/data/"):
            raise HTTPException(status_code=400, detail="Access denied: Path must be within /data directory")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        with open(path, 'r') as file:
            content = file.read()
        return content
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the LLM-based Automation Agent API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
