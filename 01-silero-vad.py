#
# 01-silero-vad.py
#
# Turn detection approach: Silero VAD (local ONNX, acoustic silence detection)
# STT: Cartesia Ink-Whisper (cloud)
# TTS: Cartesia (cloud)
# LLM: AWS Bedrock (Claude Haiku 4.5)
#
# The simplest approach — Silero listens for silence to decide when the user
# has finished speaking. No turn model involved.
#
# Run: python 01-silero-vad.py
# Then open http://localhost:7860
#

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport):
    stt = CartesiaSTTService(api_key=os.environ["CARTESIA_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = AWSBedrockLLMService(
        aws_region=os.environ.get("AWS_REGION", "us-east-1"),
        settings=AWSBedrockLLMService.Settings(
            model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
            system_instruction="You are a friendly travel assistant. Help users plan trips — flights, hotels, destinations, itineraries. Keep responses short and conversational, as they will be spoken aloud. No emojis, bullet points, or lists.",
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.7,   # min speech confidence to trigger
                    stop_secs=0.2,    # silence duration before turn ends — try 0.8+ to reduce false triggers
                    min_volume=0.6,   # min audio volume to count as speech
                )
            )
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message({"role": "developer", "content": "Greet the user as a travel assistant and ask where they'd like to go."})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
