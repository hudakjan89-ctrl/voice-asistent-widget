# Force rebuild v3 - fix static file routing order
"""
Ultra-Low Latency Voice Assistant Backend
Main FastAPI server with WebSocket audio streaming pipeline.
"""
import asyncio
import json
import logging
import sys
import traceback
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import httpx
from openai import AsyncOpenAI

from config import (
    DEEPGRAM_API_KEY,
    OPENROUTER_API_KEY,
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    ELEVENLABS_MODEL,
    ELEVENLABS_WS_URL,
    LLM_MODEL,
    OPENROUTER_BASE_URL,
    DEEPGRAM_MODEL,
    DEEPGRAM_LANGUAGE,
    SESSION_INACTIVITY_TIMEOUT,
    MAX_CONVERSATION_HISTORY,
    get_system_prompt,
    get_greeting_prompt,
    validate_api_keys,
    get_config_summary,
)
from errors import (
    ConfigurationError,
    ServiceConnectionError,
    STTError,
    LLMError,
    TTSError,
    VoiceAssistantError,
)
from text_normalizer import normalize_text

# Configure logging - DEBUG level for maximum visibility
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    stream=sys.stdout,
    force=True
)

# Set all loggers to DEBUG
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Log startup
logger.info("=" * 60)
logger.info("VOICE ASSISTANT SERVER STARTING")
logger.info("=" * 60)
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {__file__}")

# OpenRouter client (OpenAI compatible)
openai_client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Voice Assistant Server...")
    
    # Validate API keys at startup
    try:
        validate_api_keys()
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Server will start but voice features will not work without valid API keys.")
    
    # Log configuration summary
    config = get_config_summary()
    logger.info(f"LLM Model: {config['llm_model']}")
    logger.info(f"Deepgram Model: {config['deepgram_model']}")
    logger.info(f"Deepgram Language: {config['deepgram_language']}")
    logger.info(f"ElevenLabs Voice: {config['elevenlabs_voice_id']}")
    
    yield
    logger.info("Shutting down Voice Assistant Server...")


app = FastAPI(
    title="Voice Assistant API",
    description="Ultra-low latency voice assistant with streaming STT, LLM, and TTS",
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response logging middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming requests and outgoing responses."""
    
    async def dispatch(self, request: Request, call_next):
        # Log incoming request
        start_time = time.time()
        logger.info(f">>> REQUEST: {request.method} {request.url.path}")
        logger.debug(f"    Headers: {dict(request.headers)}")
        logger.debug(f"    Query params: {dict(request.query_params)}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(f"<<< RESPONSE: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(f"!!! ERROR: {request.method} {request.url.path} - {type(e).__name__}: {e} - Time: {process_time:.3f}s")
            logger.error(f"    Traceback:\n{traceback.format_exc()}")
            raise


# Add middleware
app.add_middleware(RequestLoggingMiddleware)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    config = get_config_summary()
    return JSONResponse({
        "status": "healthy",
        "service": "voice-assistant",
        "config": {
            "llm_model": config["llm_model"],
            "deepgram_model": config["deepgram_model"],
            "deepgram_language": config["deepgram_language"],
            "api_keys_configured": config["api_keys_configured"],
        }
    })


@app.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with external service connectivity tests.
    Use this for debugging configuration issues.
    """
    results = {
        "status": "healthy",
        "service": "voice-assistant",
        "checks": {}
    }
    
    # Check Deepgram API connectivity
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "https://api.deepgram.com/v1/projects",
                headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
            )
            results["checks"]["deepgram"] = {
                "status": "ok" if response.status_code in [200, 401] else "error",
                "reachable": True,
                "authenticated": response.status_code == 200
            }
    except Exception as e:
        results["checks"]["deepgram"] = {
            "status": "error",
            "reachable": False,
            "error": str(e)
        }
    
    # Check OpenRouter API connectivity
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{OPENROUTER_BASE_URL}/models",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
            )
            results["checks"]["openrouter"] = {
                "status": "ok" if response.status_code in [200, 401] else "error",
                "reachable": True,
                "authenticated": response.status_code == 200
            }
    except Exception as e:
        results["checks"]["openrouter"] = {
            "status": "error",
            "reachable": False,
            "error": str(e)
        }
    
    # Check ElevenLabs API connectivity
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "https://api.elevenlabs.io/v1/user",
                headers={"xi-api-key": ELEVENLABS_API_KEY}
            )
            results["checks"]["elevenlabs"] = {
                "status": "ok" if response.status_code in [200, 401] else "error",
                "reachable": True,
                "authenticated": response.status_code == 200
            }
    except Exception as e:
        results["checks"]["elevenlabs"] = {
            "status": "error",
            "reachable": False,
            "error": str(e)
        }
    
    # Overall status
    all_ok = all(
        check.get("status") == "ok" 
        for check in results["checks"].values()
    )
    results["status"] = "healthy" if all_ok else "degraded"
    
    return JSONResponse(results)


class VoiceSession:
    """
    Manages a single voice conversation session.
    Handles the full pipeline: Deepgram (STT) -> LLM -> ElevenLabs (TTS)
    """
    
    # Reconnection settings
    MAX_RECONNECT_ATTEMPTS = 3
    RECONNECT_DELAY_BASE = 1.0  # Base delay in seconds
    
    def __init__(self, client_ws: WebSocket):
        self.client_ws = client_ws
        self.conversation_history = []
        self.system_prompt = get_system_prompt()
        
        # State management
        self.is_speaking = False  # Bot is currently outputting audio
        self.is_listening = True  # Accepting user audio
        self.should_interrupt = False  # Barge-in detected
        self.session_active = True  # Session is still active
        
        # Async tasks and connections
        self.deepgram_ws: Optional[WebSocket] = None
        self.elevenlabs_ws: Optional[WebSocket] = None
        self.current_response_task: Optional[asyncio.Task] = None
        self.deepgram_receiver_task: Optional[asyncio.Task] = None
        
        # Reconnection tracking
        self.deepgram_reconnect_attempts = 0
        
        # Text accumulator for LLM response
        self.pending_text = ""
        self.text_send_task: Optional[asyncio.Task] = None
        
        # Session statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "audio_chunks_sent": 0,
            "audio_chunks_received": 0,
            "reconnections": 0,
        }
        
        # Inactivity tracking
        self.last_activity_time = asyncio.get_event_loop().time()
        self.inactivity_check_task: Optional[asyncio.Task] = None
        
        logger.info("New voice session created")
    
    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity_time = asyncio.get_event_loop().time()
    
    async def check_inactivity(self):
        """Periodically check for session inactivity and cleanup if needed."""
        check_interval = 30  # Check every 30 seconds
        
        while self.session_active:
            await asyncio.sleep(check_interval)
            
            if not self.session_active:
                break
            
            current_time = asyncio.get_event_loop().time()
            inactive_duration = current_time - self.last_activity_time
            
            if inactive_duration > SESSION_INACTIVITY_TIMEOUT:
                logger.info(f"Session inactive for {inactive_duration:.0f}s, initiating cleanup")
                await self.send_to_client({
                    "type": "session_timeout",
                    "message": "Session ended due to inactivity"
                })
                self.session_active = False
                break
            elif inactive_duration > SESSION_INACTIVITY_TIMEOUT - 60:
                # Warn client 60 seconds before timeout
                remaining = SESSION_INACTIVITY_TIMEOUT - inactive_duration
                logger.debug(f"Session will timeout in {remaining:.0f}s")
    
    async def send_to_client(self, message: dict):
        """Send JSON message to client."""
        try:
            await self.client_ws.send_json(message)
            self.stats["messages_sent"] += 1
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
    
    async def send_audio_to_client(self, audio_data: bytes):
        """Send binary audio data to client."""
        try:
            await self.client_ws.send_bytes(audio_data)
        except Exception as e:
            logger.error(f"Error sending audio to client: {e}")
    
    async def handle_barge_in(self):
        """Handle user interruption (barge-in)."""
        logger.info("Barge-in detected! Stopping current response...")
        
        self.should_interrupt = True
        self.is_speaking = False
        
        # Cancel current response generation
        if self.current_response_task and not self.current_response_task.done():
            self.current_response_task.cancel()
            try:
                await self.current_response_task
            except asyncio.CancelledError:
                pass
        
        # Send clear audio command to client
        await self.send_to_client({"type": "clear_audio"})
        
        # Close ElevenLabs connection to stop TTS
        if self.elevenlabs_ws:
            try:
                await self.elevenlabs_ws.close()
            except:
                pass
            self.elevenlabs_ws = None
        
        # Reset state
        self.should_interrupt = False
        self.pending_text = ""
        
        logger.info("Barge-in handled, ready for new input")
    
    async def generate_llm_response(self, user_text: str, is_greeting: bool = False):
        """Generate streaming response from LLM."""
        try:
            if is_greeting:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_text}
                ]
            else:
                # Add user message to history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_text
                })
                
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    *self.conversation_history
                ]
            
            logger.info(f"Generating LLM response for: {user_text[:50]}...")
            
            stream = await openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                stream=True,
                max_tokens=500,
                temperature=0.7,
            )
            
            full_response = ""
            
            async for chunk in stream:
                if self.should_interrupt:
                    logger.info("LLM generation interrupted by barge-in")
                    break
                
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    
                    # Send token to TTS pipeline
                    await self.send_text_to_tts(token)
                    
                    # Also send transcript to client for display
                    await self.send_to_client({
                        "type": "assistant_text",
                        "text": token,
                        "is_final": False
                    })
            
            # Flush any remaining text to TTS
            await self.flush_tts_buffer()
            
            # Send final indicator
            await self.send_to_client({
                "type": "assistant_text",
                "text": "",
                "is_final": True
            })
            
            # Add to conversation history if not interrupted
            if not self.should_interrupt and full_response and not is_greeting:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                
                # Trim conversation history to prevent memory issues
                if len(self.conversation_history) > MAX_CONVERSATION_HISTORY:
                    # Keep the most recent messages, preserving pairs if possible
                    self.conversation_history = self.conversation_history[-MAX_CONVERSATION_HISTORY:]
                    logger.debug(f"Trimmed conversation history to {MAX_CONVERSATION_HISTORY} messages")
            
            logger.info(f"LLM response complete: {len(full_response)} chars")
            
        except asyncio.CancelledError:
            logger.info("LLM generation cancelled")
            raise
        except Exception as e:
            error = LLMError(str(e))
            logger.error(f"Error generating LLM response: {e}")
            await self.send_to_client(error.to_dict())
    
    async def send_text_to_tts(self, text: str):
        """
        Accumulate text and send to TTS when we have enough for natural speech.
        Implements optimistic pipelining - sends chunks as soon as possible.
        """
        self.pending_text += text
        
        # Send on sentence boundaries or punctuation for natural speech
        # Also send if we have accumulated enough text
        send_markers = [".", "!", "?", ",", ";", ":", "\n"]
        
        should_send = False
        for marker in send_markers:
            if marker in self.pending_text:
                should_send = True
                break
        
        # Also send if we have enough text (for long sentences)
        if len(self.pending_text) > 100:
            should_send = True
        
        if should_send and self.pending_text.strip():
            text_to_send = self.pending_text
            self.pending_text = ""
            
            # Normalize text for TTS
            normalized_text = normalize_text(text_to_send)
            
            # Send to ElevenLabs
            await self.stream_to_elevenlabs(normalized_text)
    
    async def flush_tts_buffer(self):
        """Flush any remaining text in the TTS buffer."""
        if self.pending_text.strip():
            normalized_text = normalize_text(self.pending_text)
            self.pending_text = ""
            await self.stream_to_elevenlabs(normalized_text)
        
        # Send end of stream to ElevenLabs
        if self.elevenlabs_ws:
            try:
                await self.elevenlabs_ws.send(json.dumps({"text": ""}))
            except:
                pass
    
    async def connect_elevenlabs(self):
        """Establish WebSocket connection to ElevenLabs."""
        import websockets
        
        if self.elevenlabs_ws:
            return
        
        url = ELEVENLABS_WS_URL.format(
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL
        )
        
        try:
            self.elevenlabs_ws = await websockets.connect(
                url,
                extra_headers={"xi-api-key": ELEVENLABS_API_KEY}
            )
            
            # Send initial configuration
            init_message = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                },
                "generation_config": {
                    "chunk_length_schedule": [50, 80, 120, 150]
                },
                "xi_api_key": ELEVENLABS_API_KEY
            }
            await self.elevenlabs_ws.send(json.dumps(init_message))
            
            # Start receiving audio in background
            asyncio.create_task(self.receive_elevenlabs_audio())
            
            logger.info("Connected to ElevenLabs")
            
        except Exception as e:
            logger.error(f"Error connecting to ElevenLabs: {e}")
            self.elevenlabs_ws = None
    
    async def stream_to_elevenlabs(self, text: str):
        """Stream text to ElevenLabs for TTS."""
        if not text.strip():
            return
        
        if not self.elevenlabs_ws:
            await self.connect_elevenlabs()
        
        if self.elevenlabs_ws:
            try:
                message = {
                    "text": text,
                    "try_trigger_generation": True
                }
                await self.elevenlabs_ws.send(json.dumps(message))
                self.is_speaking = True
            except Exception as e:
                logger.error(f"Error sending to ElevenLabs: {e}")
                self.elevenlabs_ws = None
    
    async def receive_elevenlabs_audio(self):
        """Receive audio chunks from ElevenLabs and forward to client."""
        import base64
        
        try:
            async for message in self.elevenlabs_ws:
                if self.should_interrupt:
                    break
                
                data = json.loads(message)
                
                if "audio" in data and data["audio"]:
                    # Decode base64 audio and send to client
                    audio_bytes = base64.b64decode(data["audio"])
                    await self.send_audio_to_client(audio_bytes)
                
                if data.get("isFinal"):
                    self.is_speaking = False
                    await self.send_to_client({"type": "audio_end"})
                    # Tell client we're ready to listen again
                    await self.send_to_client({"type": "listening"})
                    logger.info("TTS audio finished, ready for next user input")
                    break
                    
        except Exception as e:
            if not self.should_interrupt:
                logger.error(f"Error receiving from ElevenLabs: {e}")
        finally:
            self.is_speaking = False
            if self.elevenlabs_ws:
                try:
                    await self.elevenlabs_ws.close()
                except:
                    pass
                self.elevenlabs_ws = None
    
    async def connect_deepgram(self):
        """Establish WebSocket connection to Deepgram for STT."""
        import websockets
        
        # Build URL with keywords for better recognition
        base_url = f"wss://api.deepgram.com/v1/listen?model={DEEPGRAM_MODEL}&language={DEEPGRAM_LANGUAGE}&punctuate=true&smart_format=true&endpointing=300&interim_results=true&utterance_end_ms=1000&vad_events=true&encoding=linear16&sample_rate=16000"
        
        # Add keywords with very high boost for company-specific terms
        keywords_params = "&keywords=EniQ:5&keywords=automatizace:3&keywords=digitální:2.5&keywords=procesů:2.5&keywords=firma:2&keywords=společnost:2&keywords=služby:2&keywords=Alex:2"
        url = base_url + keywords_params
        
        logger.info(f"Connecting to Deepgram: model={DEEPGRAM_MODEL}, language={DEEPGRAM_LANGUAGE}")
        
        try:
            self.deepgram_ws = await websockets.connect(
                url,
                extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                ping_interval=20,
                ping_timeout=10,
            )
            
            logger.info("Connected to Deepgram successfully with keywords boost")
            
        except Exception as e:
            logger.error(f"Error connecting to Deepgram: {e}")
            self.deepgram_ws = None
            raise
    
    async def receive_deepgram_transcripts(self):
        """Receive and process transcripts from Deepgram with reconnection support."""
        logger.info("Starting Deepgram transcript receiver...")
        message_count = 0
        
        while self.session_active:
            try:
                if not self.deepgram_ws:
                    logger.warning("Deepgram WebSocket not connected, attempting to reconnect...")
                    await self.reconnect_deepgram()
                    if not self.deepgram_ws:
                        logger.error("Failed to reconnect to Deepgram")
                        break
                
                logger.info("Listening for Deepgram messages...")
                async for message in self.deepgram_ws:
                    message_count += 1
                    if message_count % 10 == 1:
                        logger.debug(f"Deepgram message #{message_count}")
                    if not self.session_active:
                        break
                    
                    self.stats["messages_received"] += 1
                    data = json.loads(message)
                    
                    # Handle speech started event (for barge-in)
                    if data.get("type") == "SpeechStarted":
                        logger.info("Speech started detected")
                        if self.is_speaking:
                            await self.handle_barge_in()
                        continue
                    
                    # Handle transcript results
                    if "channel" in data:
                        alternatives = data.get("channel", {}).get("alternatives", [])
                        if alternatives:
                            transcript = alternatives[0].get("transcript", "")
                            is_final = data.get("is_final", False)
                            speech_final = data.get("speech_final", False)
                            
                            if transcript:
                                # Send interim results to client
                                await self.send_to_client({
                                    "type": "user_text",
                                    "text": transcript,
                                    "is_final": is_final or speech_final
                                })
                                
                                # Process final transcripts
                                if is_final or speech_final:
                                    logger.info(f"Final transcript: {transcript}")
                                    
                                    # Generate response (don't await - let it run in background)
                                    self.current_response_task = asyncio.create_task(
                                        self.generate_llm_response(transcript)
                                    )
                    
                    # Handle utterance end
                    if data.get("type") == "UtteranceEnd":
                        logger.info("Utterance end detected")
                
                # Connection closed normally, try to reconnect if session still active
                if self.session_active:
                    logger.warning("Deepgram connection closed, attempting reconnection...")
                    self.deepgram_ws = None
                    await self.reconnect_deepgram()
                        
            except Exception as e:
                logger.error(f"Error receiving from Deepgram: {e}")
                self.deepgram_ws = None
                
                if self.session_active:
                    await self.reconnect_deepgram()
                    
        # Cleanup
        if self.deepgram_ws:
            try:
                await self.deepgram_ws.close()
            except:
                pass
            self.deepgram_ws = None
    
    async def reconnect_deepgram(self):
        """Attempt to reconnect to Deepgram with exponential backoff."""
        if not self.session_active:
            return
        
        self.deepgram_reconnect_attempts += 1
        
        if self.deepgram_reconnect_attempts > self.MAX_RECONNECT_ATTEMPTS:
            error = STTError(f"Max reconnection attempts ({self.MAX_RECONNECT_ATTEMPTS}) reached")
            logger.error(error.message)
            await self.send_to_client(error.to_dict())
            return
        
        delay = self.RECONNECT_DELAY_BASE * (2 ** (self.deepgram_reconnect_attempts - 1))
        logger.info(f"Reconnecting to Deepgram (attempt {self.deepgram_reconnect_attempts}) in {delay}s...")
        
        await asyncio.sleep(delay)
        
        try:
            await self.connect_deepgram()
            self.deepgram_reconnect_attempts = 0  # Reset on successful connection
            self.stats["reconnections"] += 1
            logger.info("Successfully reconnected to Deepgram")
        except Exception as e:
            logger.error(f"Reconnection to Deepgram failed: {e}")
    
    async def send_audio_to_deepgram(self, audio_data: bytes):
        """Forward audio data to Deepgram."""
        # Update activity on receiving audio from client
        self.update_activity()
        
        if self.deepgram_ws and self.is_listening:
            try:
                await self.deepgram_ws.send(audio_data)
                self.stats["audio_chunks_sent"] += 1
                # Log every 50 chunks
                if self.stats["audio_chunks_sent"] % 50 == 0:
                    logger.debug(f"Audio chunks sent to Deepgram: {self.stats['audio_chunks_sent']}")
            except Exception as e:
                logger.error(f"Error sending to Deepgram: {e}")
                # Mark connection as closed to trigger reconnection
                self.deepgram_ws = None
        elif not self.deepgram_ws:
            logger.warning("Cannot send audio - Deepgram not connected")
        elif not self.is_listening:
            logger.debug("Not listening - audio ignored")
    
    async def generate_greeting(self):
        """Generate and speak the initial greeting."""
        logger.info("Generating greeting...")
        
        # Connect to ElevenLabs and wait for connection
        await self.connect_elevenlabs()
        
        # Wait a moment for WebSocket to be fully ready
        await asyncio.sleep(0.5)
        
        # Get greeting text based on time of day
        from datetime import datetime
        now = datetime.now()
        hour = now.hour
        
        if 6 <= hour < 12:
            greeting_text = "Dobré ráno, tady Alex z EniQ. Jak vám mohu dnes pomoci s automatizací vašich procesů?"
        elif 12 <= hour < 18:
            greeting_text = "Dobré odpoledne, tady Alex z EniQ. Jak vám mohu dnes pomoci s automatizací vašich procesů?"
        elif 18 <= hour < 22:
            greeting_text = "Dobrý večer, tady Alex z EniQ. Jak vám mohu dnes pomoci s automatizací vašich procesů?"
        else:
            greeting_text = "Dobrou noc, tady Alex z EniQ. Jak vám mohu dnes pomoci s automatizací vašich procesů?"
        
        # Send greeting text directly to client (for display)
        await self.send_to_client({
            "type": "assistant_text",
            "text": greeting_text,
            "is_final": True
        })
        
        # Send greeting to TTS
        try:
            await self.stream_to_elevenlabs(greeting_text)
            logger.info("Greeting sent to TTS successfully")
        except Exception as e:
            logger.error(f"Error sending greeting to TTS: {e}")
        
        # Send listening status after greeting
        await self.send_to_client({"type": "listening"})
    
    async def start(self):
        """Start the voice session."""
        logger.info("Starting voice session...")
        
        # Connect to Deepgram
        try:
            await self.connect_deepgram()
        except Exception as e:
            error = ServiceConnectionError("Deepgram", str(e))
            logger.error(f"Failed to connect to Deepgram: {e}")
            await self.send_to_client(error.to_dict())
            return
        
        # Start receiving transcripts in background
        self.deepgram_receiver_task = asyncio.create_task(
            self.receive_deepgram_transcripts()
        )
        
        # Start inactivity checker
        self.inactivity_check_task = asyncio.create_task(
            self.check_inactivity()
        )
        
        # Update activity timestamp
        self.update_activity()
        
        await self.send_to_client({
            "type": "session_started",
            "message": "Voice session started"
        })
        
        # Generate automatic greeting
        logger.info("Generating automatic greeting...")
        await self.generate_greeting()
        
        logger.info("Session ready - listening for user input")
    
    async def cleanup(self):
        """Clean up all connections."""
        logger.info("Cleaning up voice session...")
        
        # Mark session as inactive to stop reconnection attempts
        self.session_active = False
        self.is_listening = False
        
        # Cancel any running tasks
        tasks_to_cancel = [
            self.current_response_task,
            self.deepgram_receiver_task,
            self.inactivity_check_task,
        ]
        
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close WebSocket connections
        if self.deepgram_ws:
            try:
                await self.deepgram_ws.close()
            except:
                pass
            self.deepgram_ws = None
        
        if self.elevenlabs_ws:
            try:
                await self.elevenlabs_ws.close()
            except:
                pass
            self.elevenlabs_ws = None
        
        # Log session statistics
        logger.info(f"Session stats: {self.stats}")
        logger.info("Voice session cleaned up")


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for audio streaming.
    
    Client Protocol:
    - Send binary audio data (PCM 16-bit, 16kHz, mono)
    - Receive binary audio data (MP3 from ElevenLabs)
    - Receive JSON messages for control/text
    
    JSON Message Types:
    - {"type": "session_started"} - Session is ready
    - {"type": "user_text", "text": "...", "is_final": bool} - User transcript
    - {"type": "assistant_text", "text": "...", "is_final": bool} - Assistant response
    - {"type": "clear_audio"} - Clear audio buffer (barge-in)
    - {"type": "audio_end"} - Audio stream complete
    - {"type": "session_timeout"} - Session ended due to inactivity
    - {"type": "error", "message": "..."} - Error occurred
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    session = VoiceSession(websocket)
    
    try:
        # Start the session (connects to services, generates greeting)
        await session.start()
        
        # Main receive loop - forward audio to Deepgram
        while session.session_active:
            try:
                # Use a timeout to check session status periodically
                data = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=5.0
                )
                
                # Check for disconnect
                if data.get("type") == "websocket.disconnect":
                    logger.info("WebSocket disconnect message received")
                    break
                
                if "bytes" in data:
                    # Audio data from client
                    await session.send_audio_to_deepgram(data["bytes"])
                    
                elif "text" in data:
                    # JSON control message from client
                    try:
                        message = json.loads(data["text"])
                        msg_type = message.get("type")
                        
                        if msg_type == "ping":
                            await websocket.send_json({"type": "pong"})
                            session.update_activity()
                        
                        elif msg_type == "stop":
                            logger.info("Stop requested by client")
                            break
                            
                    except json.JSONDecodeError:
                        pass
                        
            except asyncio.TimeoutError:
                # Just a timeout, continue checking session status
                logger.debug("Receive timeout, continuing...")
                continue
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected by client")
                break
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                # Don't break on error, try to continue
                continue
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        
    finally:
        await session.cleanup()
        logger.info("WebSocket connection closed")


# Error handlers
@app.exception_handler(VoiceAssistantError)
async def voice_assistant_error_handler(request: Request, exc: VoiceAssistantError):
    """Handle custom voice assistant errors."""
    logger.error(f"Voice assistant error: {exc.code} - {exc.message}")
    logger.error(f"Request: {request.method} {request.url}")
    return JSONResponse(
        status_code=400,
        content=exc.to_dict()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors with full traceback."""
    # Get full traceback
    tb_str = traceback.format_exc()
    
    # Log detailed error information
    logger.error("=" * 60)
    logger.error("UNHANDLED EXCEPTION - INTERNAL SERVER ERROR")
    logger.error("=" * 60)
    logger.error(f"Request: {request.method} {request.url}")
    logger.error(f"Path: {request.url.path}")
    logger.error(f"Headers: {dict(request.headers)}")
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Exception message: {exc}")
    logger.error(f"Full traceback:\n{tb_str}")
    logger.error("=" * 60)
    
    return JSONResponse(
        status_code=500,
        content={
            "type": "error",
            "code": "INTERNAL_ERROR",
            "message": f"Internal server error: {type(exc).__name__}: {str(exc)}",
            "traceback": tb_str
        }
    )


# STATIC FILE ROUTES - MUST BE ABSOLUTE LAST!
@app.get("/")
async def serve_index():
    """Serve index.html"""
    logger.info("Serving index.html")
    return FileResponse("/app/client/index.html")

logger.info("Static file routes configured for /app/client")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server via __main__")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="debug"
    )
