# DeepSeek-R1-API-Usage-with-Live-Streaming-Frontend

## Overview

The DeepSeek Reasoner Chat API is a FastAPI-based application that provides a robust interface for interacting with the DeepSeek Reasoner language model. This API enables both streaming and non-streaming chat completions, conversation management, and integration with a built-in frontend.

## Key Features

- **Streaming Responses**: Real-time streaming of AI responses with separate reasoning and content streams
- **Conversation Management**: Persistent conversation history with conversation-specific IDs
- **Multi-language Support**: Ability to respond in languages other than English
- **Built-in Frontend**: HTML/JavaScript frontend for immediate use
- **Colored Console Logging**: Rich, color-coded logging for easier debugging
- **CORS Support**: Cross-Origin Resource Sharing enabled for frontend integration
- **Docker Support**: Ready-to-use Docker configuration for easy deployment
- **Health Checks**: API health monitoring endpoint

## Setup and Installation

### Prerequisites

- Python 3.8+ (for local installation)
- DeepSeek API key
- Docker and Docker Compose (for containerized deployment)

### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SakibAhmedShuva/DeepSeek-R1-API-Usage-with-Live-Streaming-Frontend.git
   cd DeepSeek-R1-API-Usage-with-Live-Streaming-Frontend
   ```

2. Create a `.env` file in the root directory with:
   ```
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```

3. Install dependencies:
   ```bash
   pip install fastapi uvicorn pydantic python-dotenv colorama openai
   ```

4. Run the application:
   ```bash
   python app.py
   ```

The server will start on `http://0.0.0.0:8000`.

### Docker Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SakibAhmedShuva/DeepSeek-R1-API-Usage-with-Live-Streaming-Frontend.git
   cd DeepSeek-R1-API-Usage-with-Live-Streaming-Frontend
   ```

2. Create a `.env` file in the root directory with:
   ```
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```

3. Build and start the container:
   ```bash
   docker-compose up -d
   ```

The server will be accessible at `http://localhost:8000`.

## Frontend Interface

The application includes a built-in HTML/JavaScript frontend for interacting with the DeepSeek Reasoner model. The frontend is served from the root URL (`/`) and offers the following features:

- **Chat Interface**: Text-based chat interface for sending messages to the model
- **Stream Display**: Real-time display of both reasoning and content as they are generated
- **Conversation Management**: Options to start new conversations and view conversation history
- **Parameter Controls**: Adjustments for temperature, max tokens, and other model parameters
- **Multi-language Support**: Option to specify response language

To access the frontend, navigate to `http://localhost:8000` after starting the application.

## Docker Configuration

### Dockerfile

The application includes a Dockerfile that:

1. Uses Python 3.9 as the base image
2. Sets up the working directory
3. Installs all required dependencies
4. Exposes port 8000
5. Launches the API on container start

### Docker Compose

The `docker-compose.yml` file configures:

1. The build process for the application
2. Port mapping (8000:8000)
3. Environment variable passing from the host
4. Volume mounting for logs persistence

Example `docker-compose.yml`:
```yaml
version: '3'
services:
  deepseek-reasoner-api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

## API Reference

### Chat Completion

Generate a chat completion using DeepSeek Reasoner.

**Endpoint**: `POST /chat`

**Request Body**:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "max_tokens": 8192,
  "temperature": 0.7,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "conversation_id": "unique-conversation-id",
  "stream": true,
  "language": "english"
}
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| messages | array | Required | Array of message objects with role and content |
| max_tokens | integer | 8192 | Maximum tokens to generate |
| temperature | float | 0.7 | Controls randomness (lower is more deterministic) |
| frequency_penalty | float | 0.0 | Avoids repetitive dialogue |
| presence_penalty | float | 0.0 | Allows revisiting established themes |
| conversation_id | string | Required | Unique ID for conversation tracking |
| stream | boolean | true | Whether to stream the response |
| language | string | "english" | Language for the response |

**Response**:

For streaming responses (`stream=true`):
- Event stream with "data" events containing JSON objects
- Each chunk includes either content or reasoning with an identified type

For non-streaming responses (`stream=false`):
```json
{
  "content": "Hello! I'm doing well. How can I assist you today?",
  "reasoning": "The user is greeting me and asking how I am. I should respond politely and ask how I can help them.",
  "total_tokens": 42,
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 32,
    "total_tokens": 42
  },
  "conversation_id": "unique-conversation-id"
}
```

### Clear Conversation

Clear conversation history for a specific conversation ID.

**Endpoint**: `DELETE /conversations/{conversation_id}`

**Response**:
```json
{
  "status": "success",
  "message": "Conversation unique-conversation-id cleared"
}
```

### Get Raw Conversation

Retrieve raw conversation history (for debugging).

**Endpoint**: `GET /conversations/{conversation_id}/raw`

**Response**:
```json
{
  "conversation_id": "unique-conversation-id",
  "history": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "Hello! I'm doing well. How can I assist you today?"}
  ]
}
```

### List Conversations

List all active conversation IDs.

**Endpoint**: `GET /conversations`

**Response**:
```json
{
  "conversation_ids": ["conversation-id-1", "conversation-id-2"],
  "count": 2
}
```

### Health Check

Check API health status.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "ok",
  "active_conversations": 2
}
```

## Message Handling Logic

### Conversation Management

- Each conversation is identified by a unique `conversation_id`
- New conversations are initialized with a system message
- Conversation histories are stored in memory during the application lifecycle
- Language preferences can be specified for each conversation

### Message Processing Rules

1. System messages are only used once at the beginning of a conversation
2. Each request must contain at least one user message
3. The API ensures the last message in a conversation is always from the user
4. Messages are deduplicated to avoid redundancy

## Streaming Response Format

Streaming responses are sent as Server-Sent Events (SSE) with the following format:

```
data: {"type": "reasoning", "content": "I should greet the user..."}

data: {"type": "content", "content": "Hello! How can I help..."}

data: {"type": "done", "conversation_id": "unique-conversation-id"}
```

- `reasoning`: Internal thought process of the model (when available)
- `content`: The actual response to be shown to the user
- `done`: Sent when the response is complete

## Frontend Integration

### Built-in Frontend

The application serves a built-in frontend at the root URL (`/`). This frontend:

1. Connects to the API endpoints
2. Handles both streaming and non-streaming responses
3. Displays reasoning and content in separate sections
4. Manages conversation history
5. Provides controls for model parameters

### Custom Frontend Integration

To integrate with a custom frontend:

1. Use the `/chat` endpoint for generating responses
2. Implement SSE handling for streaming responses
3. Use the conversation management endpoints to handle chat history
4. Set the appropriate CORS headers if needed

Example JavaScript for SSE handling:

```javascript
const eventSource = new EventSource('/chat?stream=true&conversation_id=123');

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  if (data.type === 'reasoning') {
    // Update reasoning display
  } else if (data.type === 'content') {
    // Update content display
  } else if (data.type === 'done') {
    // Complete the operation
    eventSource.close();
  }
};
```

## Logging and Debugging

The application uses a colorized logging system to facilitate debugging:

- **Green**: Regular content from the model
- **Cyan**: Reasoning content from the model
- **Yellow**: System information and debugging
- **Red**: Errors and exceptions

Logs are written to the `logs` directory:
- `logs/ai_outputs.log`: Contains all AI-generated outputs
- `logs/chat_{conversation_id}.log`: Conversation-specific logs

When using Docker, these logs are persisted via a volume mount.

## Error Handling

The API returns appropriate HTTP status codes for different error scenarios:

- `400 Bad Request`: Missing required parameters or invalid input
- `404 Not Found`: Conversation ID not found
- `500 Internal Server Error`: API errors or unexpected exceptions

## Performance Considerations

- The application uses `asyncio` to handle async operations
- Token estimation is approximate (uses 4 characters per token rule of thumb)
- In-memory storage of conversations can lead to memory issues with many active conversations
- Docker container memory should be monitored with high traffic

## Security Notes

- API keys should be stored securely in environment variables
- No authentication mechanism is built into this API - add authentication middleware for production use
- Consider implementing rate limiting for production deployment
- For Docker deployments, use Docker secrets for sensitive information

## License

This project is available under the terms specified in the repository.
