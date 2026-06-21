#
# 02-cartesia-turns.py
#
# Turn detection approach: CartesiaTurnsSTTService (Ink-2, server-driven)
# STT: Cartesia Ink-2 via WebSocket v2 turn protocol
# TTS: Cartesia (cloud)
# LLM: AWS Bedrock (Claude Haiku 4.5)
#
# The server drives turn boundaries — no local VAD or turn analyzer needed.
# Cartesia emits turn.start / turn.update / turn.end events over the WebSocket,
# which pipecat maps to UserStartedSpeakingFrame / TranscriptionFrame / UserStoppedSpeakingFrame.
#

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.cartesia.turns.stt import CartesiaTurnsSTTService
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    stt = CartesiaTurnsSTTService(api_key=os.environ["CARTESIA_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = AWSBedrockLLMService(
        aws_region=os.environ.get("AWS_REGION", "us-east-1"),
        settings=AWSBedrockLLMService.Settings(
            model="anthropic.claude-haiku-4-5-20251001",
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

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

    context.add_message({"role": "developer", "content": "Please introduce yourself to the user."})
    await worker.queue_frames([LLMRunFrame()])

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
