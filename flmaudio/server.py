# Copyright (c) FLM Team, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from dataclasses import dataclass
import json
import random
from pathlib import Path
import time

import aiohttp
from aiohttp import web
import numpy as np
import sphn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import AsyncTextIteratorStreamer

from .models import loaders, MimiModel, LMGen
from .utils import log


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


@dataclass
class ServerState:
    mimi: MimiModel
    text_tokenizer: AutoTokenizer
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(
        self,
        mimi: MimiModel,
        text_tokenizer: AutoTokenizer,
        lm: AutoModelForCausalLM,
        device: str | torch.device,
    ):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.lm_gen = LMGen(lm, text_tokenizer)

        self.device = device
        self.frame_size = int(
            self.mimi.sample_rate / self.mimi.frame_rate
        )  # 24000/12.5 = 1920

        self.lock = asyncio.Lock()

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        self.settings = {}
        self.t_s = time.time()

    def warmup(self):
        for chunk in range(4):
            chunk = (
                np.random.rand(1, 1, self.frame_size).astype(np.float32) - 0.5
            ) * 0.002
            chunk = torch.from_numpy(chunk).to(device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])
        if self.device == "cuda":
            # Ensure all CUDA operations are complete before starting the server
            torch.cuda.synchronize(self.device)

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        text_streamer = None

        async def send_bytes(data: bytes):
            if not ws.closed:
                await ws.send_bytes(data)

        async def recv_loop():
            nonlocal close, text_streamer
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        opus_reader.append_bytes(payload)
                    else:
                        log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                if text_streamer:
                    text_streamer.end()
                    text_streamer = None
                log("info", "connection closed")

        async def opus_loop():
            all_pcm_data = None

            while not close:
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()

                if pcm is None:
                    continue

                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    be = time.time()
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size :]
                    await decode_step(chunk)
                    log("info", f"frame handled in {1000 * (time.time() - be):.1f}ms")
                    # log("info", f"overhead cost {1000 * (be - self.t_s):.1f}ms"); self.t_s = time.time()

        async def decode_step(chunk):
            # t_s = time.time()
            nonlocal text_streamer
            chunk = torch.from_numpy(chunk).to(torch.float32)
            chunk = chunk.to(device=self.device)[None, None]

            codes = self.mimi.encode(chunk)
            # log("info", f"[infer]mimi_encode: {1000 * (time.time() - t_s):.1f}ms"); t_s = time.time()
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                # log("info", f"[infer]lm_step: {1000 * (time.time() - t_s):.1f}ms"); t_s = time.time()
                if tokens is None:
                    continue
                assert tokens.shape[1] == 9  # self.lm_gen.lm_model.dep_q + 1
                main_pcm = self.mimi.decode(tokens[:, 1:])
                # log("info", f"[infer]mimi_decode: {1000 * (time.time() - t_s):.1f}ms"); t_s = time.time()
                main_pcm = main_pcm.cpu()
                output_pcm = main_pcm[0, 0].numpy()
                opus_writer.append_pcm(output_pcm)
                text_token = tokens[0, 0, 0].item()
                if text_token not in (0, 3):
                    if text_token != self.lm_gen.lm_model.config.mm_token_info.text_wait_token_id:
                        if not text_streamer:
                            text_streamer = AsyncTextIteratorStreamer(self.text_tokenizer)
                        text_streamer.put(torch.tensor([text_token]))
                    else:
                        if text_streamer:
                            text_streamer.end()
                            text_streamer = None
                # log("info", f"[infer]write_output: {1000 * (time.time() - t_s):.1f}ms"); t_s = time.time()

        async def audio_loop():
            while not close:
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await send_bytes(b"\x01" + msg)

        async def text_loop():
            while not close:
                await asyncio.sleep(0.001)
                if text_streamer:
                    async for text in text_streamer:
                        if text != "":
                            msg = b"\x02" + bytes(text, encoding="utf8")
                            log("info", f"text token {repr(text)}")
                            await send_bytes(msg)
                    await send_bytes(b"\x02\n")

        log("info", "accepted connection")
        close = False
        async with self.lock:
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            # Send the handshake.
            await send_bytes(b"\x00")
            await asyncio.gather(opus_loop(), recv_loop(), audio_loop(), text_loop())
        log("info", "done with connection")
        return ws

    async def handle_settings(self, request):
        try:
            settings = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        if self.lock.locked():
            return web.json_response({"error": "Server is busy, please try again later."}, status=503)

        self.settings = settings
        print("-->")
        print(json.dumps(settings, ensure_ascii=False, indent=2))
        for key, value in settings.items():
            try:
                self.lm_gen.__setattr__(key, value)
            except Exception as e:
                print(e)
        print("<--")
        return web.json_response({"status": "success"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8990, type=int)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run, defaults to 'cuda'.",
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        ),
    )
    parser.add_argument(
        "--mimi-path",
        type=str,
        default="CofeAI/FLM-Audio",
        help=(
            "the model path of mimi, default is CofeAI/FLM-Audio"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="CofeAI/FLM-Audio",
        help=(
            "the model path of FLM-Audio, default is CofeAI/FLM-Audio"
        ),
    )

    args = parser.parse_args()
    seed_all(42)

    log("info", "loading mimi")

    mimi = loaders.get_mimi(args.mimi_path, args.device)
    log("info", "mimi loaded")

    log("info", "loading model")
    ckpt_path = args.model_path
    lm = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        trust_remote_code=True,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    text_tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

    log("info", "model loaded")

    state = ServerState(mimi, text_tokenizer, lm, args.device)
    log("info", "warming up the model")
    state.warmup()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    app.router.add_post("/api/settings", state.handle_settings)
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        import ssl
        ssl_path = Path(args.ssl)

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert_file = ssl_path / "cert.pem"
        key_file = ssl_path / "key.pem"
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        protocol = "https"

    log("info", f"Access the API directly at {protocol}://{args.host}:{args.port}/api/chat")
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
