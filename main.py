# Force rebuild v4 - Google Speech V2 + Ultra-Fast Pipeline
"""
Ultra-Low Latency Voice Assistant Backend with Google Cloud Speech V2
Main FastAPI server with WebSocket audio streaming pipeline.
"""
import asyncio
import json
import logging
import sys
import traceback
import time
import os
import base64
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import httpx
from openai import AsyncOpenAI

# Google Cloud Speech V2
from google.cloud.speech_v2 import SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

from config import (
    GOOGLE_APPLICATION_CREDENTIALS,
    GOOGLE_CLOUD_PROJECT_ID,
    GOOGLE_SPEECH_MODEL,
    GOOGLE_SPEECH_LANGUAGES,
    GOOGLE_PHRASE_SETS,
    GOOGLE_PHRASE_BOOST,
    VAD_SILENCE_TIMEOUT_MS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    ELEVENLABS_MODEL,
    ELEVENLABS_OUTPUT_FORMAT,
    ELEVENLABS_WS_URL,
    ELEVENLABS_OPTIMIZE_LATENCY,
    LLM_MODEL,
    SESSION_INACTIVITY_TIMEOUT,
    MAX_CONVERSATION_HISTORY,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_ENCODING,
    get_system_prompt,
    get_greeting_text,
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
logger.info("ULTRA-FAST VOICE ASSISTANT SERVER STARTING")
logger.info("=" * 60)
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {__file__}")

# Set Google Cloud credentials from environment
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
logger.info(f"📁 Google Cloud credentials file: {GOOGLE_APPLICATION_CREDENTIALS}")
logger.info(f"🆔 Google Cloud Project ID: {GOOGLE_CLOUD_PROJECT_ID if GOOGLE_CLOUD_PROJECT_ID else '(will auto-detect)'}")

# OpenRouter client (OpenAI-compatible API for Llama 3.1 70B)
openai_client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Ultra-Fast Voice Assistant Server...")
    
    # Validate API keys at startup
    try:
        validate_api_keys()
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Server will start but voice features will not work without valid API keys.")
    
    # Log configuration summary
    config = get_config_summary()
    logger.info(f"LLM Model: {config['llm_model']}")
    logger.info(f"STT Service: {config['stt_service']}")
    logger.info(f"STT Languages: {config['stt_languages']}")
    logger.info(f"ElevenLabs Voice: {config['elevenlabs_voice_id']} ({ELEVENLABS_MODEL})")
    logger.info(f"Target Latency: <1.5s")
    
    yield
    logger.info("Shutting down Voice Assistant Server...")


app = FastAPI(
    title="Ultra-Fast Voice Assistant API",
    description="Ultra-low latency voice assistant with Google Speech V2, Llama 3.3 70B, and ElevenLabs Flash v2.5",
    version="4.0.0",
    lifespan=lifespan
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests."""
    
    async def dispatch(self, request: Request, call_next):
        logger.info(f">>> REQUEST: {request.method} {request.url.path}")
        logger.debug(f"    Headers: {dict(request.headers)}")
        logger.debug(f"    Query params: {dict(request.query_params)}")
        
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        logger.info(f"<<< RESPONSE: {request.method} {request.url.path} - Status: {response.status_code} - Time: {duration:.3f}s")
        return response

app.add_middleware(LoggingMiddleware)


class VoiceSession:
    """
    Manages a single ultra-fast voice conversation session.
    Pipeline: Google Speech V2 (long model) -> Llama 3.1 70B -> ElevenLabs Flash v2.5
    Target latency: <1.5s end-to-end
    """
    
    def __init__(self, client_ws: WebSocket):
        self.client_ws = client_ws
        self.conversation_history = []
        self.system_prompt = get_system_prompt()
        
        # State management
        self.is_speaking = False  # Bot is currently outputting audio
        self.is_listening = True  # Accepting user audio
        self.should_interrupt = False  # Barge-in detected
        self.session_active = True  # Session is still active
        
        # Google Speech V2 client and streaming
        self.speech_client: Optional[SpeechAsyncClient] = None
        self.speech_stream: Optional[AsyncGenerator] = None
        self.speech_request_queue: Optional[asyncio.Queue] = None
        self.speech_receiver_task: Optional[asyncio.Task] = None
        
        # ElevenLabs connection
        self.elevenlabs_ws: Optional[any] = None
        self.current_response_task: Optional[asyncio.Task] = None
        
        # Text accumulator for TTS pipeline
        self.pending_text = ""
        
        # VAD (Voice Activity Detection) for silence timeout
        self.last_speech_time = time.time()
        self.vad_task: Optional[asyncio.Task] = None
        self.pending_transcript = ""  # Accumulate interim transcripts
        
        # Session statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "audio_chunks_sent": 0,
            "audio_chunks_received": 0,
        }
        
        # Inactivity tracking
        self.last_activity_time = asyncio.get_event_loop().time()
        self.inactivity_check_task: Optional[asyncio.Task] = None
        
        # WebSocket keep-alive
        self.keepalive_task: Optional[asyncio.Task] = None
        
        # ElevenLabs audio receiver task
        self.elevenlabs_receive_task: Optional[asyncio.Task] = None
        
        logger.info("New ultra-fast voice session created")
    
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
    
    async def send_keepalive(self):
        """Send periodic ping to keep WebSocket connection alive."""
        ping_interval = 20  # Send ping every 20 seconds
        
        while self.session_active:
            await asyncio.sleep(ping_interval)
            
            if not self.session_active:
                break
            
            try:
                await self.send_to_client({"type": "ping"})
                logger.debug("📡 Sent keepalive ping to client")
            except Exception as e:
                logger.warning(f"Failed to send keepalive: {e}")
                break
    
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
            chunk_size = len(audio_data)
            await self.client_ws.send_bytes(audio_data)
            logger.debug(f"📤 Sent audio chunk to client: {chunk_size} bytes")
        except Exception as e:
            logger.error(f"❌ Error sending audio to client: {e}")
    
    async def handle_barge_in(self):
        """Handle user interruption (barge-in) - ULTRA-FAST response."""
        logger.info("⚡ BARGE-IN! Stopping current response immediately...")
        
        self.should_interrupt = True
        self.is_speaking = False
        
        # Cancel current response generation (LLM)
        if self.current_response_task and not self.current_response_task.done():
            self.current_response_task.cancel()
            try:
                await self.current_response_task
            except asyncio.CancelledError:
                pass
        
        # Send clear audio command to client (stop playback)
        await self.send_to_client({"type": "clear_audio"})
        
        # Close ElevenLabs connection to stop TTS immediately
        if self.elevenlabs_ws:
            try:
                await self.elevenlabs_ws.close()
            except:
                pass
            self.elevenlabs_ws = None
        
        # Reset state
        self.should_interrupt = False
        self.pending_text = ""
        
        logger.info("✅ Barge-in handled, ready for new input")
    
    async def generate_llm_response(self, user_text: str, is_greeting: bool = False):
        """Generate streaming response from Llama 3.1 70B (via OpenRouter - ultra-fast)."""
        try:
            if is_greeting:
                # For greeting, just send hardcoded Czech text
                return
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
            
            logger.info(f"🧠 LLM (Llama 3.1 70B) generating response for: {user_text[:50]}...")
            
            stream = await openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                stream=True,
                max_tokens=300,  # Keep responses short for fast TTS
                temperature=0.7,
            )
            
            full_response = ""
            token_count = 0
            
            async for chunk in stream:
                if self.should_interrupt:
                    logger.info("LLM generation interrupted by barge-in")
                    break
                
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    token_count += 1
                    
                    # Debug logging every 5 tokens
                    if token_count % 5 == 0:
                        logger.debug(f"💬 LLM token #{token_count}: '{token}' (total: {len(full_response)} chars)")
                    
                    # Send token to TTS pipeline immediately
                    await self.send_text_to_tts(token)
                    
                    # Also send transcript to client for display
                    await self.send_to_client({
                        "type": "assistant_text",
                        "text": token,
                        "is_final": False
                    })
            
            logger.info(f"✅ LLM generated {token_count} tokens, {len(full_response)} chars total")
            
            # Flush any remaining text to TTS
            await self.flush_tts_buffer()
            
            # Send final indicator
            await self.send_to_client({
                "type": "assistant_text",
                "text": "",
                "is_final": True
            })
            
            # Add to conversation history if not interrupted
            if not self.should_interrupt and full_response:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                
                # Trim conversation history to prevent memory issues
                if len(self.conversation_history) > MAX_CONVERSATION_HISTORY:
                    self.conversation_history = self.conversation_history[-MAX_CONVERSATION_HISTORY:]
            
            logger.info(f"✅ LLM response complete: {len(full_response)} chars")
            
        except asyncio.CancelledError:
            logger.info("LLM generation cancelled")
            raise
        except Exception as e:
            error = LLMError(str(e))
            logger.error(f"❌ Error generating LLM response: {e}")
            await self.send_to_client(error.to_dict())
    
    async def send_text_to_tts(self, text: str):
        """
        Accumulate text and send to TTS when we have enough for natural speech.
        Ultra-fast chunking for minimal latency.
        """
        self.pending_text += text
        
        # Send on sentence boundaries for natural speech
        send_markers = [".", "!", "?", ",", ";"]
        
        should_send = False
        for marker in send_markers:
            if marker in self.pending_text:
                should_send = True
                break
        
        # Also send if we have enough text
        if len(self.pending_text) > 80:
            should_send = True
        
        if should_send and self.pending_text.strip():
            text_to_send = self.pending_text
            self.pending_text = ""
            
            logger.info(f"📝 Sending to TTS: '{text_to_send[:50]}...' ({len(text_to_send)} chars)")
            
            # Normalize text for TTS
            normalized_text = normalize_text(text_to_send)
            
            # Send to ElevenLabs Flash v2.5
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
        """Establish WebSocket connection to ElevenLabs Flash v2.5 (ultra-low latency)."""
        import websockets
        
        if self.elevenlabs_ws:
            return True
        
        url = ELEVENLABS_WS_URL.format(
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL,
            latency=ELEVENLABS_OPTIMIZE_LATENCY,
            output_format=ELEVENLABS_OUTPUT_FORMAT
        )
        
        try:
            logger.info(f"🔌 Connecting to ElevenLabs WebSocket (output_format={ELEVENLABS_OUTPUT_FORMAT})...")
            logger.debug(f"   URL: {url}")
            self.elevenlabs_ws = await websockets.connect(
                url,
                extra_headers={"xi-api-key": ELEVENLABS_API_KEY}
            )
            
            # Send initial configuration for Flash v2.5
            init_message = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                    "style": 0.0,
                    "use_speaker_boost": True
                },
                "generation_config": {
                    "chunk_length_schedule": [50]  # Single value for Flash model (ultra-fast)
                },
                "xi_api_key": ELEVENLABS_API_KEY
            }
            await self.elevenlabs_ws.send(json.dumps(init_message))
            
            # Wait for WebSocket to be fully ready (handshake complete)
            await asyncio.sleep(0.3)
            
            # Start receiving audio in background - CRITICAL: Store task reference!
            self.elevenlabs_receive_task = asyncio.create_task(self.receive_elevenlabs_audio())
            logger.info("🔄 Started ElevenLabs audio receiver task")
            
            logger.info("✅ ElevenLabs WebSocket connected and ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error connecting to ElevenLabs: {e}")
            self.elevenlabs_ws = None
            return False
    
    async def stream_to_elevenlabs(self, text: str):
        """Stream text to ElevenLabs for TTS generation."""
        if not self.elevenlabs_ws:
            logger.warning("⚠️ ElevenLabs WebSocket not connected, connecting now...")
            await self.connect_elevenlabs()
        
        if self.elevenlabs_ws and text.strip():
            try:
                message = {
                    "text": text,
                    "try_trigger_generation": True
                }
                logger.debug(f"🎙️ Streaming to ElevenLabs: '{text[:30]}...' ({len(text)} chars)")
                await self.elevenlabs_ws.send(json.dumps(message))
                logger.debug(f"✅ Sent to ElevenLabs successfully")
            except Exception as e:
                logger.error(f"❌ Error streaming to ElevenLabs: {e}")
                # Reconnect on error
                self.elevenlabs_ws = None
                await self.connect_elevenlabs()
        elif not text.strip():
            logger.debug("⚠️ Empty text, skipping ElevenLabs stream")
        else:
            logger.error("❌ ElevenLabs WebSocket is None after connection attempt!")
    
    async def receive_elevenlabs_audio(self):
        """Receive audio chunks from ElevenLabs and forward to client."""
        logger.info("🎧 ElevenLabs audio receiver task STARTED")
        message_count = 0
        
        try:
            async for message in self.elevenlabs_ws:
                try:
                    if self.should_interrupt:
                        logger.info("🛑 ElevenLabs receiver interrupted by should_interrupt flag")
                        break
                    
                    message_count += 1
                    message_type = type(message).__name__
                    
                    # DEBUG: Log message type
                    logger.debug(f"📩 Message #{message_count} from ElevenLabs: type={message_type}, size={len(message) if hasattr(message, '__len__') else 'N/A'}")
                    
                    # ElevenLabs sends both JSON (metadata) and binary (audio) messages
                    # Check for bytes or bytearray
                    if isinstance(message, (bytes, bytearray)):
                        # Binary audio data (PCM)
                        chunk_size = len(message)
                        logger.info(f"🔊 Received PCM audio chunk from ElevenLabs: {chunk_size} bytes")
                        await self.send_audio_to_client(bytes(message))
                        self.stats["audio_chunks_received"] += 1
                        
                        # Log every 10 chunks at INFO level for visibility
                        if self.stats["audio_chunks_received"] % 10 == 0:
                            logger.info(f"📦 Received {self.stats['audio_chunks_received']} audio chunks from ElevenLabs")
                            
                    elif isinstance(message, str):
                        # JSON metadata or base64-encoded audio
                        try:
                            data = json.loads(message)
                            
                            # Check for error
                            if "error" in data and data["error"]:
                                logger.error(f"❌ ElevenLabs error: {data['error']}")
                                continue
                            
                            # CRITICAL: ElevenLabs sends PCM audio as base64 in JSON!
                            if "audio" in data and data["audio"]:
                                audio_base64 = data["audio"]
                                
                                # Decode base64 to raw PCM bytes
                                try:
                                    pcm_bytes = base64.b64decode(audio_base64)
                                    chunk_size = len(pcm_bytes)
                                    logger.info(f"🔊 Decoded base64 audio from ElevenLabs: {chunk_size} bytes PCM")
                                    
                                    # Send to client
                                    await self.send_audio_to_client(pcm_bytes)
                                    self.stats["audio_chunks_received"] += 1
                                    
                                    if self.stats["audio_chunks_received"] % 10 == 0:
                                        logger.info(f"📦 Received {self.stats['audio_chunks_received']} audio chunks from ElevenLabs")
                                except Exception as decode_err:
                                    logger.error(f"❌ Failed to decode base64 audio: {decode_err}")
                            else:
                                # Regular metadata (isFinal, alignment, etc.)
                                logger.debug(f"📨 ElevenLabs metadata: {data}")
                        
                        except Exception as e:
                            logger.warning(f"⚠️ Non-JSON text message from ElevenLabs: {message[:100]}")
                    else:
                        logger.warning(f"⚠️ Unknown message type from ElevenLabs: {message_type}")
                
                except Exception as inner_e:
                    logger.error(f"❌ Error processing ElevenLabs message #{message_count}: {inner_e}")
                    logger.error(traceback.format_exc())
                    continue  # Keep loop alive
        
        except asyncio.CancelledError:
            logger.info(f"🛑 ElevenLabs receiver task CANCELLED (processed {message_count} messages)")
            raise  # Re-raise to properly cancel
        
        except Exception as e:
            logger.error(f"❌ FATAL error in ElevenLabs receiver (after {message_count} messages): {e}")
            logger.error(traceback.format_exc())
        
        finally:
            logger.info(f"🎧 ElevenLabs audio receiver task ENDED (total messages: {message_count})")
    
    async def init_google_speech(self):
        """Initialize Google Cloud Speech V2 client with 'long' model configuration."""
        try:
            # Create Speech client (using default global endpoint for multi-language support)
            # CRITICAL: Multi-language (sk-SK, cs-CZ) only supported in global location
            self.speech_client = SpeechAsyncClient()
            logger.info(f"🌍 Google Speech client endpoint: speech.googleapis.com (global)")
            
            # Define recognizer path (required for V2 API)
            # CRITICAL: Using global location to support multiple languages (sk-SK, cs-CZ)
            self.recognizer_path = f"projects/{GOOGLE_CLOUD_PROJECT_ID}/locations/global/recognizers/_"
            logger.info(f"📍 Google Speech recognizer: {self.recognizer_path}")
            
            # Configure streaming recognition with 'long' model
            # CRITICAL: Minimal config - 'long' model does NOT support:
            #   - speech_adaptation_boost (phrase sets)
            #   - automatic_punctuation
            logger.info(f"⚙️ Using minimal config for 'long' model (no phrase adaptation, no auto-punctuation)")
            
            recognition_config = cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=AUDIO_SAMPLE_RATE,
                    audio_channel_count=AUDIO_CHANNELS,
                ),
                language_codes=GOOGLE_SPEECH_LANGUAGES,  # sk-SK, cs-CZ
                model=GOOGLE_SPEECH_MODEL,  # long
                # NO features block - not supported by this model
                # NO adaptation block - not supported by this model
            )
            
            streaming_config = cloud_speech.StreamingRecognitionConfig(
                config=recognition_config,
                streaming_features=cloud_speech.StreamingRecognitionFeatures(
                    interim_results=True,
                ),
            )
            
            # Store config for request generator
            self.streaming_config = streaming_config
            
            # Create request queue for audio streaming
            self.speech_request_queue = asyncio.Queue()
            
            # CRITICAL FIX: streaming_recognize() may return a coroutine in some Speech V2 versions
            # We don't await it here - the async iterator is used directly in receive_google_transcripts
            # The method returns an AsyncIterable that can be used with 'async for'
            self.speech_stream = self.speech_client.streaming_recognize(
                requests=self._audio_request_generator()
            )
            
            # Start receiving transcripts in background
            # The task will iterate over the speech_stream async iterator
            self.speech_receiver_task = asyncio.create_task(self.receive_google_transcripts())
            
            # Start VAD monitoring
            self.vad_task = asyncio.create_task(self.monitor_vad())
            
            logger.info(f"✅ Google Speech V2 (long model) initialized for project: {GOOGLE_CLOUD_PROJECT_ID}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Google Speech: {e}")
            logger.error(traceback.format_exc())
            raise STTError(f"Failed to initialize Google Speech V2: {e}")
    
    async def _audio_request_generator(self):
        """
        Generate streaming audio requests for Google Speech V2.
        First request must contain recognizer and streaming_config.
        Subsequent requests contain only audio data.
        """
        # First request: recognizer + config
        yield cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer_path,
            streaming_config=self.streaming_config
        )
        
        # Subsequent requests: audio only
        while self.session_active and self.is_listening:
            audio_chunk = await self.speech_request_queue.get()
            if audio_chunk is None:  # Sentinel to stop
                break
            yield cloud_speech.StreamingRecognizeRequest(audio=audio_chunk)
    
    async def send_audio_to_google(self, audio_data: bytes):
        """Send raw PCM audio to Google Speech V2."""
        self.update_activity()
        if self.speech_request_queue and self.is_listening:
            try:
                await self.speech_request_queue.put(audio_data)
                self.stats["audio_chunks_sent"] += 1
            except Exception as e:
                logger.error(f"Error sending audio to Google Speech: {e}")
    
    async def receive_google_transcripts(self):
        """Receive and process transcripts from Google Speech V2."""
        try:
            # CRITICAL FIX: In Speech V2 API, streaming_recognize may return a coroutine
            # Check if it's a coroutine and await it first to get the actual async iterator
            import inspect
            if inspect.iscoroutine(self.speech_stream):
                logger.debug("⚠️ speech_stream is coroutine, awaiting to get async iterator...")
                self.speech_stream = await self.speech_stream
            
            # Now iterate over the async iterator
            async for response in self.speech_stream:
                if not self.session_active:
                    break
                
                for result in response.results:
                    if not result.alternatives:
                        continue
                    
                    transcript = result.alternatives[0].transcript
                    is_final = result.is_final
                    
                    # Detect language from result
                    language_code = result.language_code if result.language_code else "cs"
                    language = language_code.split("-")[0]  # sk-SK -> sk
                    
                    # LOG: Every received transcript (both interim and final)
                    logger.info(f"🎤 Google STT {'[FINAL]' if is_final else '[interim]'} ({language}): {transcript}")
                    logger.debug(f"📝 Transcript ({'final' if is_final else 'interim'}): {transcript}")
                    
                    if is_final:
                        # Reset VAD timer on final transcript
                        self.last_speech_time = time.time()
                        self.pending_transcript = ""
                        
                        # Send final transcript to client
                        await self.send_to_client({
                            "type": "user_text",
                            "text": transcript,
                            "is_final": True,
                            "language": language
                        })
                        
                        # Check if bot is speaking (barge-in detection)
                        if self.is_speaking:
                            await self.handle_barge_in()
                        
                        # CRITICAL: Wait 700ms before responding to avoid jumping in too early
                        # Google STT sends is_final=True quickly, but user might continue speaking
                        logger.info(f"🎯 Final transcript ({language}): {transcript}")
                        logger.debug("⏳ Waiting 700ms to ensure user finished speaking...")
                        await asyncio.sleep(0.7)
                        
                        # Check if user started speaking again during the wait
                        time_since_last_speech = time.time() - self.last_speech_time
                        if time_since_last_speech < 0.5:
                            logger.debug("🔄 User continued speaking, skipping response")
                            continue
                        
                        # Send to LLM for response
                        self.current_response_task = asyncio.create_task(
                            self.generate_llm_response(transcript.strip())
                        )
                    else:
                        # Interim transcript - accumulate for VAD
                        self.pending_transcript = transcript
                        self.last_speech_time = time.time()
                        
                        # Send interim to client for real-time display
                        await self.send_to_client({
                            "type": "user_text",
                            "text": transcript,
                            "is_final": False,
                            "language": language
                        })
        
        except Exception as e:
            logger.error(f"❌ Error receiving Google transcripts: {e}")
            logger.error(traceback.format_exc())
    
    async def monitor_vad(self):
        """
        Monitor Voice Activity Detection.
        Send accumulated transcript to LLM after VAD_SILENCE_TIMEOUT_MS of silence.
        """
        while self.session_active:
            await asyncio.sleep(0.1)  # Check every 100ms
            
            if self.pending_transcript and not self.is_speaking:
                silence_duration = (time.time() - self.last_speech_time) * 1000  # ms
                
                if silence_duration > VAD_SILENCE_TIMEOUT_MS:
                    # Silence timeout reached - process accumulated transcript
                    transcript = self.pending_transcript.strip()
                    if transcript:
                        logger.info(f"⏱️ VAD timeout ({silence_duration:.0f}ms) - processing: {transcript}")
                        
                        # Send to client as final
                        await self.send_to_client({
                            "type": "user_text",
                            "text": transcript,
                            "is_final": True,
                            "language": "cs"  # Default to Czech
                        })
                        
                        # Send to LLM
                        self.current_response_task = asyncio.create_task(
                            self.generate_llm_response(transcript)
                        )
                        
                        self.pending_transcript = ""
    
    async def generate_greeting(self):
        """Generate initial greeting in Czech (hardcoded)."""
        try:
            logger.info("🎉 Generating greeting...")
            
            # Get time-appropriate Czech greeting
            greeting_text = get_greeting_text()
            
            logger.info(f"📢 Greeting: {greeting_text}")
            
            # Send greeting text to client
            await self.send_to_client({
                "type": "assistant_text",
                "text": greeting_text,
                "is_final": False
            })
            
            # Connect to ElevenLabs and wait for stable connection
            logger.info("⏳ Waiting for ElevenLabs WebSocket to be ready...")
            connected = await self.connect_elevenlabs()
            
            if not connected:
                logger.error("❌ Failed to connect to ElevenLabs, greeting aborted")
                return
            
            # Additional wait to ensure WebSocket is fully stable
            await asyncio.sleep(0.5)
            logger.info("✅ ElevenLabs connection stable, sending greeting...")
            
            # Send greeting to TTS
            normalized_greeting = normalize_text(greeting_text)
            await self.stream_to_elevenlabs(normalized_greeting)
            
            # End of greeting
            await asyncio.sleep(0.2)
            await self.flush_tts_buffer()
            
            await self.send_to_client({
                "type": "assistant_text",
                "text": "",
                "is_final": True
            })
            
            logger.info("✅ Greeting complete")
            
        except Exception as e:
            logger.error(f"❌ Error generating greeting: {e}")
    
    async def start(self):
        """Start the voice session pipeline."""
        try:
            logger.info("🚀 Starting ultra-fast voice session...")
            
            # Initialize Google Speech V2
            await self.init_google_speech()
            
            # Generate greeting
            await self.generate_greeting()
            
            # Start inactivity monitoring
            self.inactivity_check_task = asyncio.create_task(self.check_inactivity())
            
            # Start WebSocket keep-alive (ping every 20s)
            self.keepalive_task = asyncio.create_task(self.send_keepalive())
            logger.info("📡 WebSocket keep-alive started (ping every 20s)")
            
            logger.info("✅ Voice session started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start voice session: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def cleanup(self):
        """Clean up all connections and tasks."""
        logger.info("🧹 Cleaning up voice session...")
        
        self.session_active = False
        self.is_listening = False
        
        # Cancel VAD task
        if self.vad_task:
            self.vad_task.cancel()
            try:
                await self.vad_task
            except asyncio.CancelledError:
                pass
        
        # Cancel inactivity check
        if self.inactivity_check_task:
            self.inactivity_check_task.cancel()
            try:
                await self.inactivity_check_task
            except asyncio.CancelledError:
                pass
        
        # Cancel keep-alive task
        if self.keepalive_task:
            self.keepalive_task.cancel()
            try:
                await self.keepalive_task
            except asyncio.CancelledError:
                pass
        
        # Cancel current response
        if self.current_response_task and not self.current_response_task.done():
            self.current_response_task.cancel()
            try:
                await self.current_response_task
            except asyncio.CancelledError:
                pass
        
        # Stop Google Speech streaming
        if self.speech_request_queue:
            await self.speech_request_queue.put(None)  # Sentinel to stop
        
        if self.speech_receiver_task:
            self.speech_receiver_task.cancel()
            try:
                await self.speech_receiver_task
            except asyncio.CancelledError:
                pass
        
        # Close Google Speech client
        if self.speech_client:
            try:
                await self.speech_client.close()
            except:
                pass
        
        # Cancel ElevenLabs audio receiver task
        if hasattr(self, 'elevenlabs_receive_task') and self.elevenlabs_receive_task:
            self.elevenlabs_receive_task.cancel()
            try:
                await self.elevenlabs_receive_task
            except asyncio.CancelledError:
                pass
            logger.info("🛑 ElevenLabs receiver task cancelled")
        
        # Close ElevenLabs connection
        if self.elevenlabs_ws:
            try:
                await self.elevenlabs_ws.close()
            except:
                pass
        
        logger.info(f"📊 Session stats: {self.stats}")
        logger.info("✅ Voice session cleaned up")


# FastAPI WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for voice assistant."""
    await websocket.accept()
    logger.info("🔌 WebSocket connection accepted")
    
    session = VoiceSession(websocket)
    
    try:
        # Start the session
        await session.start()
        
        # Main message loop
        while session.session_active:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=1.0
                )
                
                if "bytes" in message:
                    # Binary audio data from client
                    audio_data = message["bytes"]
                    await session.send_audio_to_google(audio_data)
                
                elif "text" in message:
                    # JSON control message from client
                    data = json.loads(message["text"])
                    message_type = data.get("type")
                    
                    if message_type == "start":
                        logger.info("▶️ Client requested start")
                        session.is_listening = True
                    
                    elif message_type == "stop":
                        logger.info("⏹️ Stop requested by client")
                        session.is_listening = False
                        break
                    
                    elif message_type == "ping":
                        await session.send_to_client({"type": "pong"})
            
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                logger.info("🔌 Client disconnected")
                break
    
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        await session.cleanup()
        logger.info("🔌 WebSocket connection closed")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint for Coolify."""
    config = get_config_summary()
    return JSONResponse({
        "status": "healthy",
        "service": "ultra-fast-voice-assistant",
        "config": {
            "llm_model": config["llm_model"],
            "stt_service": config["stt_service"],
            "stt_languages": config["stt_languages"],
            "api_keys_configured": config["api_keys_configured"],
        }
    })


# Detailed health check endpoint
@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with service connectivity tests."""
    results = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    # Check Google Cloud credentials
    try:
        if os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
            results["checks"]["google_cloud"] = {
                "status": "ok",
                "credentials_file": "exists",
                "project_id": GOOGLE_CLOUD_PROJECT_ID
            }
        else:
            results["checks"]["google_cloud"] = {
                "status": "error",
                "credentials_file": "missing"
            }
            results["status"] = "degraded"
    except Exception as e:
        results["checks"]["google_cloud"] = {"status": "error", "error": str(e)}
        results["status"] = "degraded"
    
    # Check OpenRouter connectivity
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                timeout=5.0
            )
            results["checks"]["openrouter"] = {
                "status": "ok" if response.status_code == 200 else "error",
                "reachable": True
            }
    except Exception as e:
        results["checks"]["openrouter"] = {"status": "error", "error": str(e)}
        results["status"] = "degraded"
    
    # Check ElevenLabs API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                timeout=5.0
            )
            results["checks"]["elevenlabs"] = {
                "status": "ok" if response.status_code == 200 else "error",
                "reachable": True
            }
    except Exception as e:
        results["checks"]["elevenlabs"] = {"status": "error", "error": str(e)}
        results["status"] = "degraded"
    
    return JSONResponse(results)


# Static file serving
import os
from pathlib import Path

# Determine client directory path
client_dir = Path(__file__).parent / "client"
logger.info(f"Static file routes configured for {client_dir}")


@app.get("/")
async def serve_index():
    """Serve the main index.html page."""
    logger.info("Serving index.html")
    index_path = client_dir / "index.html"
    if not index_path.exists():
        logger.error(f"index.html not found at {index_path}")
        return JSONResponse({"error": "Frontend not found"}, status_code=404)
    return FileResponse(index_path)


@app.get("/{file_path:path}")
async def serve_static_files(file_path: str):
    """Serve static files from the client directory."""
    file_full_path = client_dir / file_path
    
    if file_full_path.exists() and file_full_path.is_file():
        return FileResponse(file_full_path)
    
    # If file doesn't exist and it's not a known API route, serve index.html (SPA fallback)
    if not file_path.startswith(("ws", "health", "api")):
        index_path = client_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
    
    return JSONResponse({"error": "Not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        log_level="debug",
        reload=False
    )
