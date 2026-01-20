"""
Ultra-Low Latency Voice Assistant Backend
Main FastAPI server with WebSocket audio streaming pipeline.
"""
import asyncio
import json
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import httpx
from openai import AsyncOpenAI

from server.config import (
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
    get_system_prompt,
    get_greeting_prompt,
)
from server.text_normalizer import normalize_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# OpenRouter client (OpenAI compatible)
openai_client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Voice Assistant Server...")
    logger.info(f"LLM Model: {LLM_MODEL}")
    logger.info(f"Deepgram Model: {DEEPGRAM_MODEL}")
    logger.info(f"ElevenLabs Voice: {ELEVENLABS_VOICE_ID}")
    yield
    logger.info("Shutting down Voice Assistant Server...")


app = FastAPI(
    title="Voice Assistant API",
    description="Ultra-low latency voice assistant with streaming STT, LLM, and TTS",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files for client
app.mount("/static", StaticFiles(directory="client"), name="static")


@app.get("/")
async def root():
    """Serve the test client."""
    return FileResponse("client/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "voice-assistant",
        "llm_model": LLM_MODEL,
    })


class VoiceSession:
    """
    Manages a single voice conversation session.
    Handles the full pipeline: Deepgram (STT) -> LLM -> ElevenLabs (TTS)
    """
    
    def __init__(self, client_ws: WebSocket):
        self.client_ws = client_ws
        self.conversation_history = []
        self.system_prompt = get_system_prompt()
        
        # State management
        self.is_speaking = False  # Bot is currently outputting audio
        self.is_listening = True  # Accepting user audio
        self.should_interrupt = False  # Barge-in detected
        
        # Async tasks and connections
        self.deepgram_ws: Optional[WebSocket] = None
        self.elevenlabs_ws: Optional[WebSocket] = None
        self.current_response_task: Optional[asyncio.Task] = None
        
        # Text accumulator for LLM response
        self.pending_text = ""
        self.text_send_task: Optional[asyncio.Task] = None
        
        logger.info("New voice session created")
    
    async def send_to_client(self, message: dict):
        """Send JSON message to client."""
        try:
            await self.client_ws.send_json(message)
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
            
            logger.info(f"LLM response complete: {len(full_response)} chars")
            
        except asyncio.CancelledError:
            logger.info("LLM generation cancelled")
            raise
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            await self.send_to_client({
                "type": "error",
                "message": f"LLM error: {str(e)}"
            })
    
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
                    "chunk_length_schedule": [120, 160, 250, 290]
                }
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
        
        url = f"wss://api.deepgram.com/v1/listen?model={DEEPGRAM_MODEL}&language={DEEPGRAM_LANGUAGE}&punctuate=true&endpointing=300&interim_results=true&utterance_end_ms=1000&vad_events=true"
        
        try:
            self.deepgram_ws = await websockets.connect(
                url,
                extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
            )
            
            # Start receiving transcripts in background
            asyncio.create_task(self.receive_deepgram_transcripts())
            
            logger.info("Connected to Deepgram")
            
        except Exception as e:
            logger.error(f"Error connecting to Deepgram: {e}")
            self.deepgram_ws = None
    
    async def receive_deepgram_transcripts(self):
        """Receive and process transcripts from Deepgram."""
        try:
            async for message in self.deepgram_ws:
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
                                
                                # Generate response
                                self.current_response_task = asyncio.create_task(
                                    self.generate_llm_response(transcript)
                                )
                
                # Handle utterance end
                if data.get("type") == "UtteranceEnd":
                    logger.info("Utterance end detected")
                    
        except Exception as e:
            logger.error(f"Error receiving from Deepgram: {e}")
        finally:
            if self.deepgram_ws:
                try:
                    await self.deepgram_ws.close()
                except:
                    pass
                self.deepgram_ws = None
    
    async def send_audio_to_deepgram(self, audio_data: bytes):
        """Forward audio data to Deepgram."""
        if self.deepgram_ws and self.is_listening:
            try:
                await self.deepgram_ws.send(audio_data)
            except Exception as e:
                logger.error(f"Error sending to Deepgram: {e}")
    
    async def generate_greeting(self):
        """Generate and speak the initial greeting."""
        logger.info("Generating greeting...")
        
        greeting_prompt = get_greeting_prompt()
        
        # Connect to ElevenLabs first
        await self.connect_elevenlabs()
        
        # Generate greeting
        self.current_response_task = asyncio.create_task(
            self.generate_llm_response(greeting_prompt, is_greeting=True)
        )
    
    async def start(self):
        """Start the voice session."""
        logger.info("Starting voice session...")
        
        # Connect to Deepgram
        await self.connect_deepgram()
        
        # Generate immediate greeting
        await self.generate_greeting()
        
        await self.send_to_client({
            "type": "session_started",
            "message": "Voice session started"
        })
    
    async def cleanup(self):
        """Clean up all connections."""
        logger.info("Cleaning up voice session...")
        
        # Cancel any running tasks
        if self.current_response_task and not self.current_response_task.done():
            self.current_response_task.cancel()
        
        # Close WebSocket connections
        if self.deepgram_ws:
            try:
                await self.deepgram_ws.close()
            except:
                pass
        
        if self.elevenlabs_ws:
            try:
                await self.elevenlabs_ws.close()
            except:
                pass
        
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
    - {"type": "error", "message": "..."} - Error occurred
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    session = VoiceSession(websocket)
    
    try:
        # Start the session (connects to services, generates greeting)
        await session.start()
        
        # Main receive loop - forward audio to Deepgram
        while True:
            try:
                data = await websocket.receive()
                
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
                        
                        elif msg_type == "stop":
                            logger.info("Stop requested by client")
                            break
                            
                    except json.JSONDecodeError:
                        pass
                        
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected by client")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        
    finally:
        await session.cleanup()
        logger.info("WebSocket connection closed")


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
