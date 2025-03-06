from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv
import asyncio
import logging
import colorama
from colorama import Fore, Style
from openai import OpenAI
import uvicorn
import json
from datetime import datetime

# Initialize colorama for cross-platform color support
colorama.init()

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Custom formatter with colors
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: Fore.CYAN + "%(asctime)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + "%(asctime)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + "%(asctime)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
        logging.ERROR: Fore.RED + "%(asctime)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + "%(asctime)s - %(levelname)s - %(message)s" + Style.RESET_ALL
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

# Configure root logger
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="Chat API", 
              description="API for basic chat application using DeepSeek Reasoner")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(8192, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Controls randomness, lower is more deterministic")
    frequency_penalty: Optional[float] = Field(0.0, description="Avoids repetitive dialogue")
    presence_penalty: Optional[float] = Field(0.0, description="Allows revisiting established themes")
    conversation_id: str = Field(..., description="Required unique ID for each conversation")
    stream: Optional[bool] = Field(True, description="Whether to stream the response")
    language: Optional[str] = Field("english", description="Language for the response")

class ChatResponse(BaseModel):
    content: str                      # The final content ready for display
    reasoning: Optional[str] = None   # The reasoning/thought process (if available)
    total_tokens: int
    usage: dict
    conversation_id: str

# Global store for conversation histories
conversation_histories = {}

# Helper function to log AI outputs
def log_ai_output(content_type, content, truncate=200):
    """Log AI model output with appropriate formatting"""
    if content_type == "reasoning":
        color = Fore.CYAN
        prefix = "[AI REASONING]"
    elif content_type == "content":
        color = Fore.GREEN
        prefix = "[AI CONTENT]"
    else:
        color = Fore.WHITE
        prefix = "[AI OUTPUT]"
        
    # Truncate long content for console display
    display_content = content
    if len(content) > truncate:
        display_content = content[:truncate] + "..."
        
    # Log to both logger and print to console
    logger.info(f"{prefix} {display_content}")
    print(f"{color}{prefix} {display_content}{Style.RESET_ALL}")
    
    # For very important outputs, you might want to save to a dedicated file
    with open("logs/ai_outputs.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} - {content_type}: {content}\n\n")

# DeepSeek Reasoner Client
class DeepSeekReasonerClient:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            logger.error("DEEPSEEK_API_KEY not found in environment variables")
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")

        logger.info("Initializing DeepSeek Reasoner client")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )

    async def generate_content_stream(self, request: ChatRequest):
        """Generate content and stream the response with improved console output"""
        # Get or create conversation history
        conversation_id = request.conversation_id
        
        # If conversation_id is not provided, raise an error - we require a fixed ID
        if not conversation_id:
            logger.error("No conversation_id provided")
            raise HTTPException(status_code=400, detail="conversation_id is required for all requests")
        
        # Check if this is a new conversation
        is_new_conversation = conversation_id not in conversation_histories
        
        if is_new_conversation:
            logger.info(f"Starting new chat with ID: {conversation_id}")
            # Initialize with system message
            system_messages = [msg for msg in request.messages if msg.role.lower() == "system"]
            if system_messages:
                # Append language instruction to system message
                system_content = system_messages[0].content
                if hasattr(request, 'language') and request.language and request.language.lower() != "english":
                    system_content += f" Please respond in {request.language}."
                conversation_histories[conversation_id] = [
                    {"role": "system", "content": system_content}
                ]
            else:
                system_content = (
                    "You are a helpful, friendly assistant. Provide accurate, concise and useful responses."
                )
                # Add language instruction if needed
                if hasattr(request, 'language') and request.language and request.language.lower() != "english":
                    system_content += f" Please respond in {request.language}."
                conversation_histories[conversation_id] = [
                    {"role": "system", "content": system_content},
                ]
            print(f"{Fore.YELLOW}[NEW CHAT] Starting a new chat with conversation ID: {conversation_id}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}[CONTINUING] Continuing existing chat with conversation ID: {conversation_id}{Style.RESET_ALL}")

        # Get the current conversation history
        conversation_history = conversation_histories[conversation_id]
        
        # Process the new messages
        new_messages = []
        for msg in request.messages:
            # Skip system messages if already in history
            if msg.role.lower() == "system" and len(conversation_history) > 0 and conversation_history[0]["role"] == "system":
                continue
                
            message_dict = {"role": msg.role, "content": msg.content}
            if msg.name:
                message_dict["name"] = msg.name
            new_messages.append(message_dict)
        
        # Ensure we have at least one user message
        has_user_message = any(msg["role"] == "user" for msg in new_messages)
        if not has_user_message:
            logger.warning("No user message found in request, adding a default user message")
            new_messages.append({
                "role": "user",
                "content": "Hello, can you help me?"
            })
        
        # For continuing conversations, add new messages to history
        if not is_new_conversation:
            for msg in new_messages:
                # Only add if not already in history to avoid duplicates
                if msg not in conversation_history:
                    conversation_history.append(msg)
        else:
            # For new conversations, just use the system message + new messages
            if len(conversation_history) > 0 and conversation_history[0]["role"] == "system":
                conversation_history = [conversation_history[0]] + new_messages
            else:
                conversation_history = new_messages
        
        # Ensure the last message is from the user
        if conversation_history[-1]["role"] != "user":
            logger.warning("Last message is not from user, rearranging messages")
            # Find the last user message
            user_messages = [i for i, msg in enumerate(conversation_history) if msg["role"] == "user"]
            if user_messages:
                # Move the last user message to the end
                last_user_idx = user_messages[-1]
                last_user_msg = conversation_history.pop(last_user_idx)
                conversation_history.append(last_user_msg)
            else:
                # If no user message exists, add a default one
                conversation_history.append({
                    "role": "user",
                    "content": "Hello, can you help me?"
                })

        # Print the conversation history for debugging
        logger.info(f"Using conversation history with {len(conversation_history)} messages")
        logger.info(f"Last message content: {conversation_history[-1]['content'][:200]}...")
        logger.info(f"Last message role: {conversation_history[-1]['role']}")
        
        # Print complete conversation history for debugging
        print(f"{Fore.YELLOW}[DEBUG] Complete conversation history:{Style.RESET_ALL}")
        for i, msg in enumerate(conversation_history):
            content_preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
            print(f"{Fore.YELLOW}[{i}] {msg['role']}: {content_preview}{Style.RESET_ALL}")

        # Update the conversation history in the global store
        conversation_histories[conversation_id] = conversation_history

        try:
            logger.info("Sending streaming request to DeepSeek Reasoner API...")
            print(f"{Fore.BLUE}[API] Sending streaming request to DeepSeek Reasoner...{Style.RESET_ALL}")

            stream = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="deepseek-reasoner",
                messages=conversation_history,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stream=True
            )

            # Variables to collect the full response
            full_content = ""
            reasoning_content = ""
            current_reasoning_chunk = ""
            current_content_chunk = ""

            # Stream the response with improved console formatting
            async def response_stream():
                nonlocal full_content, reasoning_content, current_reasoning_chunk, current_content_chunk
                
                # Add colorized console output
                content_color = Fore.GREEN
                reasoning_color = Fore.CYAN
                
                # Buffered output for console
                reasoning_buffer = ""
                content_buffer = ""
                
                for chunk in stream:
                    response_dict = {}
                    
                    # Extract reasoning content
                    if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                        reasoning = chunk.choices[0].delta.reasoning_content
                        reasoning_content += reasoning
                        reasoning_buffer += reasoning
                        
                        # Only print when we have a complete sentence or substantial chunk
                        if ('.' in reasoning_buffer or '!' in reasoning_buffer or '?' in reasoning_buffer or 
                            len(reasoning_buffer) > 80 or '\n' in reasoning_buffer):
                            print(f"{reasoning_color}[REASONING] {reasoning_buffer}{Style.RESET_ALL}")
                            reasoning_buffer = ""
                        
                        response_dict["type"] = "reasoning"
                        response_dict["content"] = reasoning
                    
                    # Extract regular content
                    elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_content += content
                        content_buffer += content
                        
                        # Only print when we have a complete sentence or substantial chunk
                        if ('.' in content_buffer or '!' in content_buffer or '?' in content_buffer or 
                            len(content_buffer) > 80 or '\n' in content_buffer):
                            print(f"{content_color}[CONTENT] {content_buffer}{Style.RESET_ALL}")
                            content_buffer = ""
                        
                        response_dict["type"] = "content"
                        response_dict["content"] = content
                    
                    # Only yield if we have content to send
                    if response_dict:
                        yield f"data: {json.dumps(response_dict)}\n\n"
                
                # Print any remaining buffered content
                if reasoning_buffer:
                    print(f"{reasoning_color}[REASONING] {reasoning_buffer}{Style.RESET_ALL}")
                if content_buffer:
                    print(f"{content_color}[CONTENT] {content_buffer}{Style.RESET_ALL}")
                
                # After streaming is complete, save to conversation history
                if full_content:  # Only add if we actually got content
                    conversation_history.append({"role": "assistant", "content": full_content})
                    conversation_histories[conversation_id] = conversation_history
                    
                    # Print completion message to console
                    print(f"\n{content_color}[COMPLETE]{Style.RESET_ALL} Generation finished. Total content length: {len(full_content)} chars")
                    print(f"{reasoning_color}[COMPLETE]{Style.RESET_ALL} Total reasoning length: {len(reasoning_content)} chars")
                    
                    # Log full output to file
                    with open(f"logs/chat_{conversation_id}.log", "a", encoding="utf-8") as f:
                        f.write(f"\n\n--- {datetime.now().isoformat()} ---\n")
                        f.write(f"REASONING:\n{reasoning_content}\n\n")
                        f.write(f"CONTENT:\n{full_content}\n\n")
                
                # Send completion message
                yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
            
            return response_stream()

        except Exception as e:
            logger.error(f"Error calling DeepSeek Reasoner API: {str(e)}")
            # Print error with red color
            print(f"{Fore.RED}[ERROR] DeepSeek Reasoner API error: {str(e)}{Style.RESET_ALL}")
            raise HTTPException(status_code=500, detail=f"DeepSeek Reasoner API error: {str(e)}")

    async def generate_content(self, request: ChatRequest):
        """Generate content and return the full response"""
        # Get or create conversation history
        conversation_id = request.conversation_id
        
        # If conversation_id is not provided, raise an error
        if not conversation_id:
            logger.error("No conversation_id provided")
            raise HTTPException(status_code=400, detail="conversation_id is required for all requests")
        
        # Check if this is a new conversation
        is_new_conversation = conversation_id not in conversation_histories
        
        if is_new_conversation:
            logger.info(f"Starting new conversation with ID: {conversation_id}")
            # Initialize with system message
            system_messages = [msg for msg in request.messages if msg.role.lower() == "system"]
            if system_messages:
                # Append language instruction to system message
                system_content = system_messages[0].content
                if hasattr(request, 'language') and request.language and request.language.lower() != "english":
                    system_content += f" Please respond in {request.language}."
                conversation_histories[conversation_id] = [
                    {"role": "system", "content": system_content}
                ]
            else:
                system_content = (
                    "You are a helpful, friendly assistant. Provide accurate, concise and useful responses."
                )
                # Add language instruction if needed
                if hasattr(request, 'language') and request.language and request.language.lower() != "english":
                    system_content += f" Please respond in {request.language}."
                conversation_histories[conversation_id] = [
                    {"role": "system", "content": system_content},
                ]
            print(f"{Fore.YELLOW}[NEW CHAT] Starting a new chat with conversation ID: {conversation_id}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}[CONTINUING] Continuing existing chat with conversation ID: {conversation_id}{Style.RESET_ALL}")

        # Get the current conversation history
        conversation_history = conversation_histories[conversation_id]
        
        # Print the conversation history before modification for debugging
        logger.info(f"BEFORE - Conversation ID {conversation_id} history:")
        for i, msg in enumerate(conversation_history):
            content_preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
            logger.info(f"[{i}] {msg['role']}: {content_preview}")
        
        # Process the new messages
        new_messages = []
        for msg in request.messages:
            # Skip system messages if already in history
            if msg.role.lower() == "system" and len(conversation_history) > 0 and conversation_history[0]["role"] == "system":
                continue
                
            message_dict = {"role": msg.role, "content": msg.content}
            if msg.name:
                message_dict["name"] = msg.name
            new_messages.append(message_dict)
        
        # Ensure we have at least one user message
        has_user_message = any(msg["role"] == "user" for msg in new_messages)
        if not has_user_message:
            logger.warning("No user message found in request, adding a default user message")
            new_messages.append({
                "role": "user",
                "content": "Hello, can you help me?"
            })
        
        # For continuing conversations, add new messages to history
        if not is_new_conversation:
            for msg in new_messages:
                # Only add if not already in history to avoid duplicates
                if msg not in conversation_history:
                    conversation_history.append(msg)
        else:
            # For new conversations, just use the system message + new messages
            if len(conversation_history) > 0 and conversation_history[0]["role"] == "system":
                conversation_history = [conversation_history[0]] + new_messages
            else:
                conversation_history = new_messages
        
        # Ensure the last message is from the user
        if conversation_history[-1]["role"] != "user":
            logger.warning("Last message is not from user, rearranging messages")
            # Find the last user message
            user_messages = [i for i, msg in enumerate(conversation_history) if msg["role"] == "user"]
            if user_messages:
                # Move the last user message to the end
                last_user_idx = user_messages[-1]
                last_user_msg = conversation_history.pop(last_user_idx)
                conversation_history.append(last_user_msg)
            else:
                # If no user message exists, add a default one
                conversation_history.append({
                    "role": "user",
                    "content": "Hello, can you help me?"
                })

        # Update the conversation history in the global store
        conversation_histories[conversation_id] = conversation_history
        
        # Print the conversation history after modification for debugging
        logger.info(f"AFTER - Conversation ID {conversation_id} history:")
        for i, msg in enumerate(conversation_history):
            content_preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
            logger.info(f"[{i}] {msg['role']}: {content_preview}")

        try:
            logger.info("Sending request to DeepSeek Reasoner API...")
            print(f"{Fore.BLUE}[API] Sending request with {len(conversation_history)} messages{Style.RESET_ALL}")
            for i, msg in enumerate(conversation_history):
                content_preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
                print(f"{Fore.BLUE}[API] [{i}] {msg['role']}: {content_preview}{Style.RESET_ALL}")

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="deepseek-reasoner",
                messages=conversation_history,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stream=False
            )

            # Extract content and reasoning
            full_content = response.choices[0].message.content
            reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)

            # Print the response to console
            print(f"{Fore.GREEN}[RESPONSE] Content (first 200 chars): {full_content[:200]}...{Style.RESET_ALL}")
            if reasoning_content:
                print(f"{Fore.CYAN}[RESPONSE] Reasoning (first 200 chars): {reasoning_content[:200]}...{Style.RESET_ALL}")
                
            logger.info(f"Response from DeepSeek Reasoner (first 200 chars): {full_content[:200]}...")
            logger.info(f"Reasoning from DeepSeek Reasoner (first 200 chars): {reasoning_content[:200] if reasoning_content else 'No reasoning provided'}...")
            
            # Add the response to the conversation history
            conversation_history.append({"role": "assistant", "content": full_content})
            conversation_histories[conversation_id] = conversation_history

            # Log full output to file
            with open(f"logs/chat_{conversation_id}.log", "a", encoding="utf-8") as f:
                f.write(f"\n\n--- {datetime.now().isoformat()} ---\n")
                f.write(f"REASONING:\n{reasoning_content if reasoning_content else 'None'}\n\n")
                f.write(f"CONTENT:\n{full_content}\n\n")

            # Estimate token usage (roughly ~4 chars per token)
            estimated_prompt_tokens = sum(len(msg.get("content", "")) // 4 for msg in conversation_history[:-1])
            estimated_completion_tokens = len(full_content) // 4
            usage = {
                "prompt_tokens": estimated_prompt_tokens,
                "completion_tokens": estimated_completion_tokens,
                "total_tokens": estimated_prompt_tokens + estimated_completion_tokens
            }

            # Construct response with separate reasoning and content
            response = ChatResponse(
                content=full_content,
                reasoning=reasoning_content if reasoning_content else None,
                total_tokens=sum(usage.values()),
                usage=usage,
                conversation_id=conversation_id
            )

            return response

        except Exception as e:
            logger.error(f"Error calling DeepSeek Reasoner API: {str(e)}")
            print(f"{Fore.RED}[ERROR] DeepSeek Reasoner API error: {str(e)}{Style.RESET_ALL}")
            raise HTTPException(status_code=500, detail=f"DeepSeek Reasoner API error: {str(e)}")

# Dependency
def get_deepseek_client():
    return DeepSeekReasonerClient()


# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Add a route to serve the index.html file
@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse("static/index.html")

# Endpoints
@app.post("/chat")
async def generate_chat_response(
    request: ChatRequest,
    client: DeepSeekReasonerClient = Depends(get_deepseek_client)
):
    """Generate chat response using DeepSeek Reasoner."""
    try:
        # Validate conversation_id is provided
        if not request.conversation_id:
            raise HTTPException(status_code=400, detail="conversation_id is required for all requests")
            
        # Convert string "true"/"false" to boolean if needed
        if isinstance(request.stream, str):
            request.stream = request.stream.lower() == "true"
            
        if request.stream:
            # Return a streaming response
            stream_response = await client.generate_content_stream(request)
            return StreamingResponse(
                stream_response,
                media_type="text/event-stream"
            )
        else:
            # Return a regular response
            return await client.generate_content(request)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Unexpected error in generate_chat_response")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear the conversation history for a specific ID."""
    if conversation_id in conversation_histories:
        del conversation_histories[conversation_id]
        return {"status": "success", "message": f"Conversation {conversation_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

@app.get("/conversations/{conversation_id}/raw")
async def get_raw_conversation(conversation_id: str):
    """Get the raw conversation history for debugging."""
    if conversation_id in conversation_histories:
        return {"conversation_id": conversation_id, "history": conversation_histories[conversation_id]}
    else:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")

@app.get("/conversations")
async def list_conversations():
    """List all active conversation IDs."""
    return {
        "conversation_ids": list(conversation_histories.keys()),
        "count": len(conversation_histories)
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "active_conversations": len(conversation_histories)}

if __name__ == "__main__":
    logger.info("Starting chat API server with DeepSeek Reasoner...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
