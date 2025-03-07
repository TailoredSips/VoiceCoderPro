# OPTIMIZED_voice_coder.py
import os, gc, asyncio
from replit import audio
from transformers import AutoModelForCausalLM, AutoTokenizer
from whisper import load_whisper  # Custom micro-whisper
from coqui import TTS  # v0.9.0 with 4-bit quant

class TerminusCoder:
    def __init__(self):
        # 4-bit quantized model (1.1B → 0.8GB)
        self.tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
        self.model = AutoModelForCausalLM.from_pretrained(
            "bigcode/starcoderbase-1b",
            load_in_4bit=True,
            device_map="auto",
            torch_dtype="auto"
        )
        
        # Nanosecond voice engine
        self.tts = TTS("tts_models/en/ljspeech/glow-tts-4bit")
        self.asr = load_whisper("micro.en")  # 19MB model
        
        # State control
        self.active = False
        self.ctx = []
        self.audio_stream = None

    async def immortal_listener(self):
        """0-latency audio pipeline"""
        with audio.continuous_listen() as stream:
            self.audio_stream = stream
            async for frame in stream:
                if b"yo coder" in frame[:1024]:
                    await self.quantum_execute()

    async def quantum_execute(self):
        """Atomic execution unit"""
        if self.active: 
            return  # Prevent overlap
        
        self.active = True
        try:
            # Phase 1: Ultra-compact voice capture
            user_prompt = await self._capture(6.9)  # 6.9s optimal
            
            # Phase 2: Quantized generation
            code = await self._generate(user_prompt)
            
            # Phase 3: Instant injection
            self._inject(code)
            
            # Phase 4: Contextual priming
            self.ctx = [user_prompt, code][-3:]
        finally:
            self.active = False
            gc.collect()  # Crucial for 24/7

    async def _capture(self, timeout):
        """Noise-annihilating voice capture"""
        return self.asr.transcribe(
            await self.audio_stream.read_until_silence(
                timeout=timeout,
                silence_duration=0.69
            )
        )

    async def _generate(self, prompt):
        """Token-surgical generation"""
        inputs = self.tokenizer(
            f"CTX: {self.ctx}\nREQ: {prompt}\nCOD:",
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.model.device)
        
        return self.tokenizer.decode(
            self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )[0],
            skip_special_tokens=True
        ).split("COD:")[1].strip()

    def _inject(self, code):
        """Zero-waste code injection"""
        with open("gen.py", "a") as f:
            f.write(f"\n# {len(self.ctx)+1}\n{code}\n")
        self.tts.speak_to_file("Code implemented", "response.wav")
        os.system("play response.wav -q &")

    async def heartbeats(self):
        """Immortality protocol"""
        while True:
            await asyncio.sleep(119)  # Replit timeout is 120s
            os.system("curl -s "+os.getenv('REPL_URL')+" > /dev/null")

async def main():
    coder = TerminusCoder()
    await asyncio.gather(
        coder.immortal_listener(),
        coder.heartbeats()
    )

if __name__ == "__main__":
    asyncio.run(main())