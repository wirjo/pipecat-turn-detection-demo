#
# 03-smart-turn.py
#
# Turn detection approach: Pipecat LocalSmartTurnAnalyzerV3 (explicit)
# STT: Cartesia Ink-Whisper (cloud)
# TTS: Cartesia (cloud)
# LLM: AWS Bedrock (Claude Haiku 4.5)
#
# This is pipecat's built-in smart turn model made explicit. LocalSmartTurnAnalyzerV3
# runs a bundled smart-turn-v3.2-cpu ONNX model locally — it understands speech
# semantics to decide if the user has finished their thought, not just stopped talking.
#
# This is actually pipecat's default under the hood; surfaced here so you can see
# exactly where to swap in CoreML, an HTTP analyzer, or your own custom turn model.
#
# Run: python 03-smart-turn.py
# Then open http://localhost:7860
#

import sys
import os

from dotenv import load_dotenv
from loguru import logger

logger.remove(0)
logger.add(sys.stderr, level="TRACE")
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3


class VerboseSmartTurnAnalyzer(LocalSmartTurnAnalyzerV3):
    def __init__(self):
        super().__init__(params=SmartTurnParams(stop_secs=1.0))

    async def analyze_end_of_turn(self):
        state, metrics = await super().analyze_end_of_turn()
        if metrics and metrics.probability is not None:
            logger.debug(f"Smart Turn: {state.name} (p={metrics.probability:.4f}, {metrics.e2e_processing_time_ms:.1f}ms)")
        return state, metrics
from pipecat.audio.vad.silero import SileroVADAnalyzer
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
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
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
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=VerboseSmartTurnAnalyzer())]
            ),
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
