import argparse
import array
import asyncio
import json
import logging
import os
import sys
import threading
import time
import aiohttp
import mlperf_loadgen as lg
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("MLPerf-Client")


class DummyDataset:

    def __init__(self, total_sample_count=1000, isl=320):
        self.total_sample_count = total_sample_count
        self.perf_count = total_sample_count
        self.input_ids = [np.ones(isl, dtype=np.int32)] * total_sample_count

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass


class SUT:

    def __init__(self, url, model_name, osl=200):
        self.url = url
        self.model_name = model_name
        self.osl = osl
        self.dataset = DummyDataset()
        self.qsl = lg.ConstructQSL(
            self.dataset.total_sample_count,
            self.dataset.perf_count,
            self.dataset.LoadSamplesToRam,
            self.dataset.UnloadSamplesFromRam,
        )
        self.loop = asyncio.new_event_loop()
        self.query_queue = None  # Will be initialized in the loop
        self.thread = threading.Thread(target=self.run_loop, daemon=True)

    def start(self):
        self.thread.start()
        # Wait for loop and queue to be ready
        while self.query_queue is None:
            time.sleep(0.1)

    def stop(self):
        future = asyncio.run_coroutine_threadsafe(
            self.query_queue.put(None), self.loop
        )
        future.result()
        self.thread.join()

    def run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.query_queue = asyncio.Queue()
        self.loop.run_until_complete(self.async_worker())

    async def async_worker(self):
        connector = aiohttp.TCPConnector(limit=5000)
        async with aiohttp.ClientSession(connector=connector) as session:
            while True:
                qitem = await self.query_queue.get()
                if qitem is None:
                    break
                # Spawn task to handle query concurrently
                asyncio.create_task(self.do_post(session, qitem))

    async def do_post(self, session, qitem):
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "prompt_token_ids": self.dataset.input_ids[qitem.index].tolist(),
            "max_tokens": self.osl,
            "ignore_eos": True,
        }

        try:
            async with session.post(
                f"{self.url}/v1/completions", headers=headers, json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    tokens_received = data.get('usage', {}).get('completion_tokens', self.osl)
                    
                    n_tokens = tokens_received
                    dummy_resp = np.zeros(n_tokens, dtype=np.int32)
                    response_array = array.array("B", dummy_resp.tobytes())
                    bi = response_array.buffer_info()
                    response = [
                        lg.QuerySampleResponse(
                            qitem.id, bi[0], bi[1], n_tokens)
                    ]
                    lg.QuerySamplesComplete(response)
                else:
                    text = await resp.text()
                    log.error(f"Request failed: {resp.status} - {text}")
                    response = [lg.QuerySampleResponse(qitem.id, 0, 0, 0)]
                    lg.QuerySamplesComplete(response)
        except Exception as e:
            log.exception("Exception in do_post")
            response = [lg.QuerySampleResponse(qitem.id, 0, 0, 0)]
            lg.QuerySamplesComplete(response)

    def issue_queries(self, query_samples):
        for q in query_samples:
            asyncio.run_coroutine_threadsafe(
                self.query_queue.put(q), self.loop
            )

    def flush_queries(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["Offline", "Server", "SingleStream"],
        default="Server",
    )
    parser.add_argument(
        "--url", type=str, default="http://localhost:8000", help="Server URL"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--user-conf", type=str, default="user.conf", help="user.conf path"
    )
    parser.add_argument(
        "--audit-conf", type=str, default="audit.conf", help="audit.conf path"
    )
    parser.add_argument(
        "--lg-model-name",
        type=str,
        default="llama3_1-8b",
        help="Model name in user.conf",
    )
    parser.add_argument(
        "--osl", type=int, default=200, help="Output sequence length"
    )
    parser.add_argument(
        "--isl", type=int, default=320, help="Input sequence length"
    )
    args = parser.parse_args()

    scenario_map = {
        "offline": lg.TestScenario.Offline,
        "server": lg.TestScenario.Server,
        "singlestream": lg.TestScenario.SingleStream,
    }

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario.lower()]
    settings.mode = lg.TestMode.PerformanceOnly

    if os.path.exists(args.user_conf):
        settings.FromConfig(args.user_conf, args.lg_model_name, args.scenario)
    else:
        log.warning(
            f"user.conf not found at {args.user_conf}, using defaults."
        )

    sut = SUT(url=args.url, model_name=args.model_name, osl=args.osl)
    sut.dataset = DummyDataset(isl=args.isl)

    sut.start()
    lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)

    log.info("Starting Benchmark run")
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    lg.StartTestWithLogSettings(lgSUT, sut.qsl, settings, log_settings)

    sut.stop()
    lg.DestroySUT(lgSUT)
    lg.DestroyQSL(sut.qsl)
    log.info("Run Completed!")


if __name__ == "__main__":
    main()
