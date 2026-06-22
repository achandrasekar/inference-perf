# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from inference_perf.datagen.replay_graph_session_datagen import (
    EventOutputRegistry,
    SessionChatCompletionAPIData,
    WorkerSessionTracker,
)
from inference_perf.datagen.replay_graph_types import InputSegment
from inference_perf.apis.chat import ChatMessage

from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType
from inference_perf.datagen.weka_trace_replay_datagen import (
    HashIdRandomGenerator,
    WekaTraceReplayDataGenerator,
    RoleSegment,
    ConversationReconstructor,
    longest_common_prefix,
    truncate_synth_buf_at_block,
    _IdleGapTimeWarp,
)


def test_longest_common_prefix() -> None:
    assert longest_common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert longest_common_prefix([1, 2], [1, 2, 3]) == 2
    assert longest_common_prefix([1], [2]) == 0
    assert longest_common_prefix([], [1]) == 0


def test_truncate_synth_buf_at_block() -> None:
    def dummy_decode(tokens: list[int]) -> str:
        return ",".join(str(t) for t in tokens)

    segments = [
        RoleSegment(role="system", block_start=0, block_count=2, tokens=[10, 11], content="10,11"),
        RoleSegment(role="user", block_start=2, block_count=3, tokens=[20, 21, 22], content="20,21,22"),
    ]

    # Truncate at block 3
    # block 3 lies inside the second segment (since system has blocks 0,1 and user has 2,3,4)
    # Target blocks = 3, so we keep 3 blocks total: system (2) + user (1)
    # The user segment should be truncated to block_count=1, tokens=[20]
    disturbed = truncate_synth_buf_at_block(segments, target_blocks=3, block_size=1, decode_tokens_to_text=dummy_decode)

    assert disturbed == 1  # Index of disturbed segment
    assert len(segments) == 2
    assert segments[0].block_count == 2
    assert segments[1].block_count == 1
    assert segments[1].tokens == [20]
    assert segments[1].content == "20"


def test_hash_id_random_generator() -> None:
    rng1 = HashIdRandomGenerator(base_seed=42)
    rng1.set_trace_id("trace_xyz")
    rng1.reseed_for_hash_id(1001)
    val1_a = rng1.randrange(1000)
    val1_b = rng1.randrange(1000)

    # Re-instantiate with same seed, trace ID and hash ID
    rng2 = HashIdRandomGenerator(base_seed=42)
    rng2.set_trace_id("trace_xyz")
    rng2.reseed_for_hash_id(1001)
    val2_a = rng2.randrange(1000)
    val2_b = rng2.randrange(1000)

    assert val1_a == val2_a
    assert val1_b == val2_b

    # A different seed, trace ID, or hash ID should produce a different result
    rng3 = HashIdRandomGenerator(base_seed=43)
    rng3.set_trace_id("trace_xyz")
    rng3.reseed_for_hash_id(1001)
    val3_a = rng3.randrange(1000)
    assert val3_a != val1_a


def test_idle_gap_time_warp() -> None:
    # Gap cap = 10s
    warp = _IdleGapTimeWarp(request_starts=[0.0, 5.0, 35.0, 45.0], cap_seconds=10.0)

    # 0.0 maps to 0.0
    assert warp.map(0.0) == 0.0
    # 5.0 maps to 5.0 (since 5.0 - 0.0 = 5.0 <= 10.0)
    assert warp.map(5.0) == 5.0

    # 35.0 has a gap from 5.0 of 30s. Cap is 10s. Excess is 20s.
    # So 35.0 shifts left by 20s to 15.0
    assert warp.map(35.0) == 15.0

    # 45.0 has a gap from 35.0 of 10s (equal to cap). So it shifts left by same 20s to 25.0
    assert warp.map(45.0) == 25.0


def test_conversation_reconstructor() -> None:
    def decode_block_tokens(hash_ids: list[int]) -> list[int]:
        # Simple dummy block decoder: block content is [hash_id]
        return [h for h in hash_ids]

    def sample_partial_tail_tokens(n: int, seed: str) -> list[int]:
        return [99] * n

    def decode_tokens_to_text(tokens: list[int]) -> str:
        return ",".join(str(t) for t in tokens)

    recon = ConversationReconstructor(
        block_size=1,
        decode_block_tokens=decode_block_tokens,
        sample_partial_tail_tokens=sample_partial_tail_tokens,
        decode_tokens_to_text=decode_tokens_to_text,
    )

    # Turn 0: input tokens = 3, hash_ids = [1, 2, 3]
    recon.init_turn_0(
        hash_ids=[1, 2, 3],
        in_tokens=3,
        tool_tokens=0,
        system_tokens=0,
        seed="seed0",
    )

    msgs = recon.snapshot_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "1,2,3"

    # Turn 1: curr_in_tokens = 5, hash_ids = [1, 2, 10, 11, 12], prev_out_tokens = 2
    # LCP of [1, 2, 3] and [1, 2, 10, 11, 12] is [1, 2] (length 2)
    # It will truncate the user segment at block 2, leaving system/user tokens [1, 2]
    # The previous assistant output size is 2. The remaining block capacity is curr_in_tokens - lcp = 3.
    # assistant blocks = min(ceil(2/1), 3) = 2. So assistant blocks = [10, 11].
    # User blocks = remaining = 1. User block = [12].
    recon.advance_turn(
        prev_hash_ids=[1, 2, 3],
        prev_in_tokens=3,
        prev_out_tokens=2,
        curr_hash_ids=[1, 2, 10, 11, 12],
        curr_in_tokens=5,
        seed="seed1",
    )

    msgs = recon.snapshot_messages()
    assert len(msgs) == 3
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "1,2"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "10,11"
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"] == "12"


def test_weka_trace_replay_generator_mock(tmp_path: Path) -> None:
    # Create a mock Weka Trace file
    trace_data = {
        "id": "mock_trace_123",
        "models": ["claude-opus-4-8"],
        "block_size": 2,
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": [
            {
                "t": 0.1,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 4,
                "out": 2,
                "hash_ids": [10, 20],
                "api_time": 0.5,
            },
            {
                "t": 1.2,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 8,
                "out": 4,
                "hash_ids": [10, 20, 30, 40],
                "api_time": 0.8,
            },
        ],
    }

    trace_file = tmp_path / "mock_trace.json"
    trace_file.write_text(json.dumps(trace_data))

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_tokenizer().encode = lambda x: [ord(c) for c in x]
    mock_tokenizer.get_tokenizer().decode = lambda x: "".join(chr(i) for i in x)

    # API config
    api_cfg = APIConfig(type=APIType.Chat, streaming=False)

    # Datagen Config
    data_cfg = DataConfig(type=DataGenType.WekaTraceReplay)
    # Setup weka_trace_replay config mock
    from inference_perf.config.datagen.replay import WekaTraceReplayConfig

    weka_cfg = WekaTraceReplayConfig(
        trace_files=[str(trace_file)],
        default_block_size=2,
    )
    data_cfg.weka_trace_replay = weka_cfg

    # Initialize generator
    gen = WekaTraceReplayDataGenerator(
        api_config=api_cfg,
        config=data_cfg,
        tokenizer=mock_tokenizer,
        num_workers=1,
    )

    assert len(gen.sessions) == 1
    session = gen.sessions[0]
    assert session.source_id == "mock_trace_123"

    # Graph should have 2 events representing the 2 parent turns
    assert len(session.graph.events) == 2
    events = sorted(session.graph.events.values(), key=lambda e: e.t_start_ms)

    assert events[0].call.total_input_tokens == 4
    assert events[0].call.expected_output_tokens == 2

    assert events[1].call.total_input_tokens == 8
    assert events[1].call.expected_output_tokens == 4

    # Predecessor edge check
    assert events[1].predecessor_event_ids == [events[0].event_id]


def test_weka_trace_replay_generator_mock_no_warp(tmp_path: Path) -> None:
    # Create a mock Weka Trace file with a huge gap (100 seconds)
    trace_data = {
        "id": "mock_trace_no_warp",
        "models": ["claude-opus-4-8"],
        "block_size": 2,
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": [
            {
                "t": 0.1,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 4,
                "out": 2,
                "hash_ids": [10, 20],
                "api_time": 0.5,
            },
            {
                "t": 100.1,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 8,
                "out": 4,
                "hash_ids": [10, 20, 30, 40],
                "api_time": 0.8,
            },
        ],
    }

    trace_file = tmp_path / "mock_trace_no_warp.json"
    trace_file.write_text(json.dumps(trace_data))

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_tokenizer().encode = lambda x: [ord(c) for c in x]
    mock_tokenizer.get_tokenizer().decode = lambda x: "".join(chr(i) for i in x)

    # API config
    api_cfg = APIConfig(type=APIType.Chat, streaming=False)

    # Datagen Config with trace_idle_gap_cap_seconds = 0 (disabled)
    data_cfg = DataConfig(type=DataGenType.WekaTraceReplay)
    from inference_perf.config.datagen.replay import WekaTraceReplayConfig

    weka_cfg = WekaTraceReplayConfig(
        trace_files=[str(trace_file)],
        default_block_size=2,
        trace_idle_gap_cap_seconds=0,
    )
    data_cfg.weka_trace_replay = weka_cfg

    # Initialize generator
    gen = WekaTraceReplayDataGenerator(
        api_config=api_cfg,
        config=data_cfg,
        tokenizer=mock_tokenizer,
        num_workers=1,
    )

    assert len(gen.sessions) == 1
    session = gen.sessions[0]
    events = sorted(session.graph.events.values(), key=lambda e: e.t_start_ms)

    # In raw seconds: t0 = 0.1s (100ms), t1 = 100.1s (100100ms)
    # The gap is exactly 100000ms. Since gap warping is disabled (<= 0),
    # the start times should reflect original timing.
    assert events[0].t_start_ms == 100
    assert events[1].t_start_ms == 100100
    assert events[1].wait_ms == 100100 - events[0].t_end_ms


def test_weka_trace_replay_warmup_snapshot(tmp_path: Path) -> None:
    # Create a mock Weka Trace file with 3 parent turns and 1 subagent spawned at turn 0
    trace_data = {
        "id": "mock_trace_warmup",
        "models": ["claude-opus-4-8"],
        "block_size": 2,
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": [
            {
                "t": 0.1,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 4,
                "out": 2,
                "hash_ids": [10, 20],
                "api_time": 0.5,
            },
            {
                "t": 0.2,
                "type": "subagent",
                "agent_id": "sa_1",
                "subagent_type": "some_type",
                "tool_tokens": 0,
                "system_tokens": 0,
                "requests": [
                    {
                        "t": 0.3,
                        "type": "n",
                        "model": "claude-opus-4-8",
                        "in": 2,
                        "out": 2,
                        "hash_ids": [99],
                        "api_time": 0.1,
                    }
                ],
            },
            {
                "t": 1.2,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 8,
                "out": 4,
                "hash_ids": [10, 20, 30, 40],
                "api_time": 0.8,
            },
            {
                "t": 2.5,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 12,
                "out": 4,
                "hash_ids": [10, 20, 30, 40, 50, 60],
                "api_time": 1.0,
            },
        ],
    }

    trace_file = tmp_path / "mock_trace_warmup.json"
    trace_file.write_text(json.dumps(trace_data))

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_tokenizer().encode = lambda x: [ord(c) for c in x]
    mock_tokenizer.get_tokenizer().decode = lambda x: "".join(chr(i) for i in x)

    # API config
    api_cfg = APIConfig(type=APIType.Chat, streaming=False)

    # Datagen Config with start_turn_index = 2
    data_cfg = DataConfig(type=DataGenType.WekaTraceReplay)
    from inference_perf.config.datagen.replay import WekaTraceReplayConfig

    weka_cfg = WekaTraceReplayConfig(
        trace_files=[str(trace_file)],
        default_block_size=2,
        start_turn_index=2,
    )
    data_cfg.weka_trace_replay = weka_cfg

    # Initialize generator
    gen = WekaTraceReplayDataGenerator(
        api_config=api_cfg,
        config=data_cfg,
        tokenizer=mock_tokenizer,
        num_workers=1,
    )

    assert len(gen.sessions) == 1
    session = gen.sessions[0]

    # Parent turn 0 and 1 are warm/pruned.
    # The subagent spawned at turn 0 is also pruned.
    # Graph should contain exactly 1 event: parent turn 2.
    assert len(session.graph.events) == 1
    event = list(session.graph.events.values())[0]
    assert event.call.call_id == "parent_turn_2"

    # Verify that warm context messages are pre-populated
    assert len(event.call.messages) > 1
    # Check roles alternate or contain system, user, assistant messages
    roles = [m.get("role") for m in event.call.messages]
    assert "system" in roles or "user" in roles or "assistant" in roles


def test_weka_trace_replay_delayed_join(tmp_path: Path) -> None:
    # Timeline:
    # 0.1s - 0.2s: Parent Turn 0 (ends at 200ms)
    # 0.22s - 0.3s: Subagent Spawn (ends at 300ms)
    # 0.32s - 0.42s: Parent Turn 1 (starts after subagent ends, but does not join subagent)
    # 0.45s: Parent Turn 2 (joins subagent)
    trace_data = {
        "id": "mock_trace_delayed_join",
        "models": ["claude-opus-4-8"],
        "block_size": 2,
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": [
            {
                "t": 0.1,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 4,
                "out": 2,
                "hash_ids": [10, 20],
                "api_time": 0.1,
            },
            {
                "t": 0.22,
                "type": "subagent",
                "agent_id": "sa_1",
                "subagent_type": "some_type",
                "tool_tokens": 0,
                "system_tokens": 0,
                "requests": [
                    {
                        "t": 0.23,
                        "type": "n",
                        "model": "claude-opus-4-8",
                        "in": 2,
                        "out": 2,
                        "hash_ids": [99],
                        "api_time": 0.07,
                    }
                ],
            },
            {
                "t": 0.32,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 8,
                "out": 2,
                "hash_ids": [10, 20, 30, 40],
                "api_time": 0.1,
            },
            {
                "t": 0.45,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 12,
                "out": 2,
                "hash_ids": [10, 20, 30, 40, 99, 100],
                "api_time": 0.1,
            },
        ],
    }

    trace_file = tmp_path / "mock_trace_delayed_join.json"
    trace_file.write_text(json.dumps(trace_data))

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_tokenizer().encode = lambda x: [ord(c) for c in x]
    mock_tokenizer.get_tokenizer().decode = lambda x: "".join(chr(i) for i in x)

    # API config
    api_cfg = APIConfig(type=APIType.Chat, streaming=False)

    # Datagen Config
    data_cfg = DataConfig(type=DataGenType.WekaTraceReplay)
    from inference_perf.config.datagen.replay import WekaTraceReplayConfig

    weka_cfg = WekaTraceReplayConfig(
        trace_files=[str(trace_file)],
        default_block_size=2,
    )
    data_cfg.weka_trace_replay = weka_cfg

    # Initialize generator
    gen = WekaTraceReplayDataGenerator(
        api_config=api_cfg,
        config=data_cfg,
        tokenizer=mock_tokenizer,
        num_workers=1,
    )

    assert len(gen.sessions) == 1
    session = gen.sessions[0]

    # Graph should contain exactly 4 events
    assert len(session.graph.events) == 4

    events_by_id = session.graph.events

    # Event IDs:
    # event_000_parent_turn_0
    # event_001_sa_sa_1_s0_turn_0
    # event_002_parent_turn_1
    # event_003_parent_turn_2

    e0_id = [k for k in events_by_id.keys() if "parent_turn_0" in k][0]
    e1_id = [k for k in events_by_id.keys() if "sa_sa_1_s0_turn_0" in k][0]
    e2_id = [k for k in events_by_id.keys() if "parent_turn_1" in k][0]
    e3_id = [k for k in events_by_id.keys() if "parent_turn_2" in k][0]

    # e2 (parent turn 1) should NOT depend on e1 (subagent). It should depend on e0 (parent turn 0).
    assert e1_id not in events_by_id[e2_id].predecessor_event_ids
    assert e0_id in events_by_id[e2_id].predecessor_event_ids

    # e3 (parent turn 2) should depend on both e2 (parent turn 1) and e1 (subagent turn 0)
    assert e2_id in events_by_id[e3_id].predecessor_event_ids
    assert e1_id in events_by_id[e3_id].predecessor_event_ids


@pytest.mark.asyncio
async def test_session_api_data_live_vs_canned() -> None:
    # Set up mock registry
    registry = EventOutputRegistry()
    registry.record("event_0", "live output text", [])

    # Set up inputs
    original_messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "canned output text"},
        {"role": "user", "content": "next prompt"},
    ]
    input_segments = [
        InputSegment(type="shared", message_count=1, token_count=1),
        InputSegment(type="output", message_count=1, token_count=1, source_event_id="event_0"),
        InputSegment(type="unique", message_count=1, token_count=1),
    ]

    # Test Case 1: use_live_responses = True
    data_live = SessionChatCompletionAPIData(
        messages=[ChatMessage(role=m["role"], content=m.get("content")) for m in original_messages],
        max_tokens=100,
        event_id="event_1",
        registry=registry,
        worker_tracker=WorkerSessionTracker(),
        completion_queue=None,
        total_events_in_session=2,
        predecessor_event_ids=["event_0"],
        input_segments=input_segments,
        original_messages=original_messages,
        use_live_responses=True,
    )

    await data_live.wait_for_predecessors_and_substitute()
    # Live output should be substituted!
    assert data_live.messages[1].content == "live output text"

    # Test Case 2: use_live_responses = False
    data_canned = SessionChatCompletionAPIData(
        messages=[ChatMessage(role=m["role"], content=m.get("content")) for m in original_messages],
        max_tokens=100,
        event_id="event_1",
        registry=registry,
        worker_tracker=WorkerSessionTracker(),
        completion_queue=None,
        total_events_in_session=2,
        predecessor_event_ids=["event_0"],
        input_segments=input_segments,
        original_messages=original_messages,
        use_live_responses=False,
    )

    await data_canned.wait_for_predecessors_and_substitute()
    # Canned output should be kept (no substitution)!
    assert data_canned.messages[1].content == "canned output text"


def test_weka_trace_replay_ratio_snapshot(tmp_path: Path) -> None:
    # Create a mock Weka Trace file with 10 parent turns
    requests = []
    for i in range(10):
        requests.append(
            {
                "t": 0.1 * (i + 1),
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 4,
                "out": 2,
                "hash_ids": [10, 20],
                "api_time": 0.5,
            }
        )
    trace_data = {
        "id": "mock_trace_ratio",
        "models": ["claude-opus-4-8"],
        "block_size": 2,
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": requests,
    }

    trace_file = tmp_path / "mock_trace_ratio.json"
    trace_file.write_text(json.dumps(trace_data))

    mock_tokenizer = MagicMock()
    mock_tokenizer.get_tokenizer().encode = lambda x: [ord(c) for c in x]
    mock_tokenizer.get_tokenizer().decode = lambda x: "".join(chr(i) for i in x)

    api_cfg = APIConfig(type=APIType.Chat, streaming=False)
    data_cfg = DataConfig(type=DataGenType.WekaTraceReplay)
    from inference_perf.config.datagen.replay import WekaTraceReplayConfig

    # Sample between 0.4 and 0.6
    weka_cfg = WekaTraceReplayConfig(
        trace_files=[str(trace_file)],
        default_block_size=2,
        warmup_snapshot_sampling=True,
        warmup_snapshot_min_ratio=0.4,
        warmup_snapshot_max_ratio=0.6,
    )
    data_cfg.weka_trace_replay = weka_cfg

    gen = WekaTraceReplayDataGenerator(
        api_config=api_cfg,
        config=data_cfg,
        tokenizer=mock_tokenizer,
        num_workers=1,
    )

    assert len(gen.sessions) == 1
    session = gen.sessions[0]

    # 10 turns total. With ratio in [0.4, 0.6], start_k should be 4 or 5.
    # Let's check which turn was chosen!
    # If start_k = 4, there are 6 events left.
    # If start_k = 5, there are 5 events left.
    num_events = len(session.graph.events)
    assert num_events in [5, 6]


def test_weka_trace_replay_warmup_cache_priming(tmp_path: Path) -> None:
    # Create a mock Weka Trace file with 5 parent turns
    requests = []
    for i in range(5):
        requests.append(
            {
                "t": 0.1 * (i + 1),
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 4,
                "out": 2,
                "hash_ids": [10, 20],
                "api_time": 0.5,
            }
        )
    trace_data = {
        "id": "mock_trace_priming",
        "models": ["claude-opus-4-8"],
        "block_size": 2,
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": requests,
    }

    trace_file = tmp_path / "mock_trace_priming.json"
    trace_file.write_text(json.dumps(trace_data))

    mock_tokenizer = MagicMock()
    mock_tokenizer.get_tokenizer().encode = lambda x: [ord(c) for c in x]
    mock_tokenizer.get_tokenizer().decode = lambda x: "".join(chr(i) for i in x)

    api_cfg = APIConfig(type=APIType.Chat, streaming=False)
    data_cfg = DataConfig(type=DataGenType.WekaTraceReplay)
    from inference_perf.config.datagen.replay import WekaTraceReplayConfig

    # Enable cache priming and force start turn to 3
    weka_cfg = WekaTraceReplayConfig(
        trace_files=[str(trace_file)],
        default_block_size=2,
        start_turn_index=3,
        warmup_cache_priming=True,
    )
    data_cfg.weka_trace_replay = weka_cfg

    gen = WekaTraceReplayDataGenerator(
        api_config=api_cfg,
        config=data_cfg,
        tokenizer=mock_tokenizer,
        num_workers=1,
    )

    assert len(gen.sessions) == 1
    session = gen.sessions[0]

    # Warmup event should be populated on the session!
    assert session.warmup_event is not None
    assert session.warmup_event.call_id == "sa_warmup_mock_trace_priming_turn_2"
    assert session.warmup_event.event_id == "warmup:mock_trace_priming"
    assert session.warmup_event.expected_output_tokens == 1
    assert session.warmup_event.max_tokens_recorded == 1
    assert len(session.warmup_event.messages) > 0

    # Materialize the event using the generator and check properties
    stage_id = 1
    lazy_data = gen.materialize_warmup_event(session.warmup_event, session.session_id, stage_id)
    assert lazy_data.session_id == session.session_id
    assert lazy_data.stage_id == stage_id
    assert lazy_data.max_tokens == 1
    assert lazy_data.expected_output_content == "warmup"
