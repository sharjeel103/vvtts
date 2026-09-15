%%writefile /kaggle/working/VibeVoice/demo/gradio_demo.py
"""
RSR TTS - High-Quality Dialogue Generation Interface with Streaming Support
Optimized for NVIDIA T4 (4-bit NF4 Quantization) & Dual GPU Setups
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Union, Callable, Tuple
from datetime import datetime
import threading
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import os
import traceback
import pyrubberband as pyrb
import types
import gc
from tqdm import tqdm
import subprocess
import math
import uuid
import re

# Transformers & Model Imports
from transformers.generation import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers import BitsAndBytesConfig
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference, VibeVoiceGenerationOutput, VibeVoiceTokenConstraintProcessor
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed
from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache

# --- DeepFilterNet Imports (Hard Dependency) ---
from df.enhance import enhance, init_df, save_audio
from df.io import load_audio

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# --- GLOBAL BACKGROUND JOB STORAGE ---
BACKGROUND_JOBS = {}

class RSRTTSDemo:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 15, auto_device_map: bool = False, t4_mode: bool = False):
        self.model_path = model_path
        self.device = "cuda" 
        self.inference_steps = inference_steps
        self.auto_device_map = auto_device_map
        self.t4_mode = t4_mode 
        self.is_generating = False 
        
        # --- THREAD SAFETY ---
        self.gpu_lock = threading.Lock() # Prevents concurrent GPU access
        self.active_live_stop_event = None # Specific stop signal for the Live tab
        self.current_streamer = None 
        
        torch.backends.cudnn.benchmark = False 
        
        print("🎙️ Initializing DeepFilterNet for mandatory denoising...")
        self.df_model, self.df_state, _ = init_df()

        self.output_dir = "saved_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.load_model()
        self.setup_voice_presets()
        self.load_example_scripts()
        
    def load_model(self):
        print(f"Loading processor & model from {self.model_path}")
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        
        load_dtype = torch.float16
        attn_impl_primary = "sdpa" 
        load_kwargs = {
            "torch_dtype": load_dtype,
            "attn_implementation": attn_impl_primary,
        }

        if self.t4_mode:
            print("🟢 T4 Optimization Enabled: Loading with 4-bit NF4 Quantization...")
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    load_in_8bit=False,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=["prediction_head", "acoustic_connector", "semantic_connector", "acoustic_tokenizer", "semantic_tokenizer"]
                )
                load_kwargs["quantization_config"] = bnb_config
                load_kwargs["device_map"] = {"": 0} 
            except ImportError:
                raise ImportError("bitsandbytes not installed")

        elif self.auto_device_map and torch.cuda.device_count() >= 2:
            print("⚡ Dual GPU detected. Using Manual Split...")
            device_map = {}
            device_map["model.language_model.embed_tokens"] = 0
            for i in range(18): device_map[f"model.language_model.layers.{i}"] = 0
            for i in range(18, 28): device_map[f"model.language_model.layers.{i}"] = 1
            device_map["model.language_model.norm"] = 1
            device_map["lm_head"] = 1
            device_map["model.prediction_head"] = 1
            device_map["model.acoustic_tokenizer"] = 1
            device_map["model.semantic_tokenizer"] = 1
            device_map["model.acoustic_connector"] = 1
            device_map["model.semantic_connector"] = 1
            device_map["model.speech_bias_factor"] = 1
            device_map["model.speech_scaling_factor"] = 1
            load_kwargs["device_map"] = device_map
        else:
            load_kwargs["device_map"] = "auto"
        
        try:
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(self.model_path, **load_kwargs)
            self._apply_patches()
            self.model.eval()
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def _apply_patches(self):
        # --- Patch 1: _process_speech_inputs ---
        def patched_process_speech_inputs(self, speech_tensors, speech_masks, speech_type="audio"):
            with torch.no_grad():
                try: target_device = self.model.get_input_embeddings().weight.device
                except: target_device = self.model.device

                if speech_type == "audio":
                    tokenizer_device = self.model.acoustic_tokenizer.device
                    speech_tensors = speech_tensors.to(tokenizer_device)
                    encoder_output = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
                    acoustic_latents = encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]
                    scale = self.model.speech_scaling_factor.to(acoustic_latents.device)
                    bias = self.model.speech_bias_factor.to(acoustic_latents.device)
                    acoustic_features = (acoustic_latents + bias) * scale
                    connector_device = self.model.acoustic_connector.fc1.weight.device
                    speech_masks_device = speech_masks.to(connector_device)
                    acoustic_connected = self.model.acoustic_connector(acoustic_features.to(connector_device))[speech_masks_device]
                    return acoustic_features, acoustic_connected.to(target_device)
                else:
                      return self._original_process_speech_inputs(speech_tensors, speech_masks, speech_type)

        self.model._original_process_speech_inputs = self.model._process_speech_inputs
        self.model._process_speech_inputs = types.MethodType(patched_process_speech_inputs, self.model)

        # --- Patch 2: generate ---
        def patched_generate(
            self, inputs: Optional[torch.Tensor] = None, generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None, stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None, synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None, audio_streamer: Optional[Union[AudioStreamer, Any]] = None, 
            negative_prompt_ids: Optional[torch.Tensor] = None, negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            speech_tensors: Optional[torch.FloatTensor] = None, speech_masks: Optional[torch.BoolTensor] = None,
            speech_input_mask: Optional[torch.BoolTensor] = None, return_speech: bool = True, cfg_scale: float = 1.7,
            stop_check_fn: Optional[Callable[[], bool]] = None, **kwargs,
        ) -> Union[torch.LongTensor, VibeVoiceGenerationOutput]:
            
            tokenizer = kwargs.pop("tokenizer", None)
            parsed_scripts = kwargs.pop("parsed_scripts", None)
            all_speakers_list = kwargs.pop("all_speakers_list", None)
            max_length_times = kwargs.pop("max_length_times", 2)

            if kwargs.get('max_new_tokens', None) is None:
                kwargs['max_new_tokens'] = self.config.decoder_config.max_position_embeddings - kwargs['input_ids'].shape[-1]

            generation_config, model_kwargs, input_ids, logits_processor, stopping_criteria = self._build_generate_config_model_kwargs(
                generation_config, inputs, tokenizer, return_processors=True, **kwargs
            )
            
            negative_kwargs = {
                'input_ids': torch.full((kwargs['input_ids'].shape[0], 1), tokenizer.speech_start_id, dtype=torch.long, device=kwargs['input_ids'].device),
                'attention_mask':  torch.ones((kwargs['input_ids'].shape[0], 1), dtype=torch.long, device=kwargs['input_ids'].device),
                'max_new_tokens': kwargs.get('max_new_tokens', 100) 
            }
            negative_generation_config, negative_model_kwargs, negative_input_ids = self._build_generate_config_model_kwargs(
                None, None, tokenizer, return_processors=False, **negative_kwargs
            )

            acoustic_cache = VibeVoiceTokenizerStreamingCache()
            semantic_cache = VibeVoiceTokenizerStreamingCache()
            
            batch_size = input_ids.shape[0]
            device = input_ids.device
            finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=device)
            correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=device)
            is_prefill = True
            inputs_embeds = None
            audio_chunks = [[] for _ in range(batch_size)]
            initial_length = input_ids.shape[-1]
            initial_length_per_sample = model_kwargs['attention_mask'].sum(dim=-1)

            valid_tokens = [generation_config.speech_start_id, generation_config.speech_end_id, generation_config.speech_diffusion_id, generation_config.eos_token_id]
            if hasattr(generation_config, 'bos_token_id') and generation_config.bos_token_id is not None: valid_tokens.append(generation_config.bos_token_id)
            
            token_constraint_processor = VibeVoiceTokenConstraintProcessor(valid_tokens, device=device)
            if logits_processor is None: logits_processor = LogitsProcessorList()
            logits_processor.append(token_constraint_processor)
            
            max_steps = min(generation_config.max_length - initial_length, int(max_length_times * initial_length))
            max_step_per_sample = torch.min(generation_config.max_length - initial_length_per_sample, (max_length_times * initial_length_per_sample).long())
            reach_max_step_sample = torch.zeros(batch_size, dtype=torch.bool, device=device)

            progress_bar = tqdm(range(max_steps), desc="Generating", leave=False) if kwargs.get("show_progress_bar", True) else range(max_steps)
            
            for step in progress_bar:
                # --- CHECK STOP SIGNAL (Unique per job) ---
                if stop_check_fn is not None and stop_check_fn():
                    if audio_streamer is not None: audio_streamer.end()
                    break
                
                if audio_streamer is not None and hasattr(audio_streamer, 'finished_flags'):
                    if any(audio_streamer.finished_flags): break
                
                if finished_tags.all(): break

                if input_ids.shape[-1] >= generation_config.max_length:
                    reached_samples = torch.arange(batch_size, device=device)[~finished_tags]
                    if reached_samples.numel() > 0: reach_max_step_sample[reached_samples] = True
                    break
                
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                if is_prefill:
                    prefill_inputs = {"speech_tensors": speech_tensors.to(device=device), "speech_masks": speech_masks.to(device), "speech_input_mask": speech_input_mask.to(device)}
                    is_prefill = False
                else:
                    _ = model_inputs.pop('inputs_embeds', None)
                    prefill_inputs = {'inputs_embeds': inputs_embeds}

                outputs = self(**model_inputs, **prefill_inputs, logits_to_keep=1, return_dict=True, output_attentions=False, output_hidden_states=False)
                model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)

                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
                next_token_scores = logits_processor(input_ids, next_token_logits)
                
                if generation_config.do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                next_tokens[finished_tags] = generation_config.eos_token_id
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                
                if not kwargs.get('refresh_negative', True):
                    negative_model_inputs = self.prepare_inputs_for_generation(negative_input_ids, **negative_model_kwargs)
                    if negative_model_inputs['inputs_embeds'] is None and inputs_embeds is not None:
                        negative_model_inputs['inputs_embeds'] = inputs_embeds
                        negative_model_inputs['input_ids'] = None
                    negative_outputs = self(**negative_model_inputs, logits_to_keep=0, return_dict=True, output_attentions=False, output_hidden_states=False)
                    negative_model_kwargs = self._update_model_kwargs_for_generation(negative_outputs, negative_model_kwargs, is_encoder_decoder=False)
                    negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)

                if (next_tokens == generation_config.eos_token_id).any():
                    eos_indices = (next_tokens == generation_config.eos_token_id).nonzero(as_tuple=False).squeeze(1)
                    new_eos_indices = eos_indices[~finished_tags[eos_indices]]
                    if new_eos_indices.numel() > 0:
                        finished_tags[new_eos_indices] = True
                        if audio_streamer is not None: audio_streamer.end(new_eos_indices)

                max_length_reached = step >= max_step_per_sample
                new_max_length_indices = torch.nonzero(max_length_reached & ~finished_tags, as_tuple=False).squeeze(1)
                if new_max_length_indices.numel() > 0:
                    finished_tags[new_max_length_indices] = True
                    reach_max_step_sample[new_max_length_indices] = True
                    if audio_streamer is not None: audio_streamer.end(new_max_length_indices)

                diffusion_end_indices = (next_tokens == generation_config.speech_end_id).nonzero(as_tuple=False).squeeze(1)
                if diffusion_end_indices.numel() > 0:
                    acoustic_cache.set_to_zero(diffusion_end_indices)
                    semantic_cache.set_to_zero(diffusion_end_indices)
                
                diffusion_start_indices = torch.arange(batch_size, device=device)[~finished_tags & (next_tokens == generation_config.speech_start_id)]
                if diffusion_start_indices.numel() > 0 and kwargs.get('refresh_negative', True):
                    for i, sample_idx in enumerate(diffusion_start_indices.tolist()):
                        negative_model_kwargs['attention_mask'][sample_idx, :] = 0
                        negative_model_kwargs['attention_mask'][sample_idx, -1] = 1
                    for layer_idx, (k_cache, v_cache) in enumerate(zip(negative_model_kwargs['past_key_values'].key_cache, negative_model_kwargs['past_key_values'].value_cache)):
                        for sample_idx in diffusion_start_indices.tolist():
                            k_cache[sample_idx, :, -1, :] = k_cache[sample_idx, :, 0, :].clone()
                            v_cache[sample_idx, :, -1, :] = v_cache[sample_idx, :, 0, :].clone()
                    for sample_idx in diffusion_start_indices.tolist():
                        negative_input_ids[sample_idx, -1] = generation_config.speech_start_id
                
                next_inputs_embeds = self.model.get_input_embeddings()(next_tokens).unsqueeze(1)
                diffusion_indices = torch.arange(batch_size, device=device)[~finished_tags & (next_tokens == generation_config.speech_diffusion_id)]
                
                if diffusion_indices.numel() > 0:
                    if kwargs.get('refresh_negative', True):
                        negative_model_inputs = self.prepare_inputs_for_generation(negative_input_ids, **negative_model_kwargs)
                        if negative_model_inputs['inputs_embeds'] is None and inputs_embeds is not None:
                            negative_model_inputs['inputs_embeds'] = inputs_embeds
                            negative_model_inputs['input_ids'] = None
                        negative_outputs = self(**negative_model_inputs, logits_to_keep=0, return_dict=True, output_attentions=False, output_hidden_states=False)
                        negative_model_kwargs = self._update_model_kwargs_for_generation(negative_outputs, negative_model_kwargs, is_encoder_decoder=False)
                        negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)
                    
                    non_diffusion_mask = ~finished_tags & (next_tokens != generation_config.speech_diffusion_id)
                    if non_diffusion_mask.any():
                        non_diffusion_indices = torch.arange(batch_size, device=device)[non_diffusion_mask]
                        start_indices = correct_cnt[non_diffusion_indices]
                        seq_len = negative_model_kwargs['attention_mask'].shape[1]
                        for i, (sample_idx, start_idx) in enumerate(zip(non_diffusion_indices.tolist(), start_indices.tolist())):
                            if start_idx + 1 < seq_len - 1:
                                negative_model_kwargs['attention_mask'][sample_idx, start_idx+1:] = negative_model_kwargs['attention_mask'][sample_idx, start_idx:-1].clone()
                            negative_model_kwargs['attention_mask'][sample_idx, start_idx] = 0
                        for layer_idx, (k_cache, v_cache) in enumerate(zip(negative_model_kwargs['past_key_values'].key_cache, negative_model_kwargs['past_key_values'].value_cache)):
                            for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                                if start_idx + 1 < k_cache.shape[2] - 1:
                                    k_cache[sample_idx, :, start_idx+1:, :] = k_cache[sample_idx, :, start_idx:-1, :].clone()
                                    v_cache[sample_idx, :, start_idx+1:, :] = v_cache[sample_idx, :, start_idx:-1, :].clone()
                        for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                            if start_idx + 1 < negative_input_ids.shape[1] - 1:
                                negative_input_ids[sample_idx, start_idx+1:] = negative_input_ids[sample_idx, start_idx:-1].clone()
                        correct_cnt[non_diffusion_indices] += 1

                    positive_condition = outputs.last_hidden_state[diffusion_indices, -1, :]
                    negative_condition = negative_outputs.last_hidden_state[diffusion_indices, -1, :]
                    
                    speech_latent = self.sample_speech_tokens(positive_condition, negative_condition, cfg_scale=cfg_scale).unsqueeze(1)
                    scaled_latent = speech_latent / self.model.speech_scaling_factor.to(speech_latent.device) - self.model.speech_bias_factor.to(speech_latent.device)
                    audio_chunk = self.model.acoustic_tokenizer.decode(
                        scaled_latent.to(self.model.acoustic_tokenizer.device), cache=acoustic_cache, sample_indices=diffusion_indices.to(self.model.acoustic_tokenizer.device),
                        use_cache=True, debug=False
                    )
                    
                    for i, sample_idx in enumerate(diffusion_indices):
                        idx = sample_idx.item()
                        if not finished_tags[idx]: audio_chunks[idx].append(audio_chunk[i])

                    if audio_streamer is not None: audio_streamer.put(audio_chunk, diffusion_indices)
                        
                    semantic_features = self.model.semantic_tokenizer.encode(audio_chunk, cache=semantic_cache, sample_indices=diffusion_indices, use_cache=True, debug=False).mean
                    acoustic_embed = self.model.acoustic_connector(speech_latent)
                    semantic_embed = self.model.semantic_connector(semantic_features)
                    diffusion_embeds = acoustic_embed + semantic_embed
                    target_device_for_embeds = next_inputs_embeds.device
                    next_inputs_embeds[diffusion_indices] = diffusion_embeds.to(target_device_for_embeds)
                
                inputs_embeds = next_inputs_embeds

            if audio_streamer is not None: audio_streamer.end()

            final_audio_outputs = [torch.cat(c, dim=-1) if c else None for c in audio_chunks]

            return VibeVoiceGenerationOutput(
                sequences=input_ids, speech_outputs=final_audio_outputs if return_speech else None, reach_max_step_sample=reach_max_step_sample,
            )

        self.model.generate = types.MethodType(patched_generate, self.model)
        
        if hasattr(self.model, "model") and hasattr(self.model.model, "noise_scheduler"):
            try:
                self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
                    self.model.model.noise_scheduler.config, algorithm_type='sde-dpmsolver++', beta_schedule='squaredcos_cap_v2'
                )
                self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
            except Exception as e:
                print(f"Warning: Could not configure noise scheduler: {e}")

    # --- Utility Functions ---
    def setup_voice_presets(self):
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        if not os.path.exists(voices_dir):
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        wav_files = [f for f in os.listdir(voices_dir) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'))]
        self.voice_presets = {os.path.splitext(f)[0]: os.path.join(voices_dir, f) for f in wav_files}
        self.available_voices = dict(sorted(self.voice_presets.items()))
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1: wav = np.mean(wav, axis=1)
            if sr != target_sr: wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except: return np.array([])

    def _adjust_voice_speed(self, audio_np: np.ndarray, speed_factor: float, sample_rate: int = 24000) -> np.ndarray:
        if speed_factor == 1.0: return audio_np
        try: return pyrb.time_stretch(y=audio_np, sr=sample_rate, rate=speed_factor)
        except: return audio_np
    
    def denoise_audio(self, audio_np, sample_rate: int = 24000):
        if not self.df_model: return audio_np
        try:
            if torch.is_tensor(audio_np): audio_np = audio_np.detach().cpu().numpy()
            audio_np = audio_np.astype(np.float32)
            target_sr = 48000
            if sample_rate != target_sr: audio_48k = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=target_sr)
            else: audio_48k = audio_np

            chunk_size = 48000 * 30 
            enhanced_parts = []
            for i in range(0, len(audio_48k), chunk_size):
                chunk = audio_48k[i:i+chunk_size]
                chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
                with torch.no_grad(): enhanced = enhance(self.df_model, self.df_state, chunk_tensor)
                enhanced_parts.append(enhanced[0].cpu().numpy().squeeze())
            return np.concatenate(enhanced_parts)
        except Exception as e:
            print(f"Denoise Error: {e}")
            return audio_np

    def generate_podcast_streaming(self, num_speakers: int, script: str, speaker_1, speaker_2, speaker_3, speaker_4, 
                                 speaker_1_upload, speaker_2_upload, speaker_3_upload, speaker_4_upload, 
                                 speaker_1_speed, speaker_2_speed, speaker_3_speed, speaker_4_speed, cfg_scale,
                                 stop_event: Optional[threading.Event] = None) -> Iterator[tuple]:
        
        start_time = time.time()
        
        try:
            self.is_generating = True
            if not script.strip(): raise gr.Error("Please provide a script.")
            script = script.replace("’", "'")
            
            speaker_inputs = [
                (speaker_1, speaker_1_upload, speaker_1_speed),
                (speaker_2, speaker_2_upload, speaker_2_speed),
                (speaker_3, speaker_3_upload, speaker_3_speed),
                (speaker_4, speaker_4_upload, speaker_4_speed)
            ]
            voice_samples = []
            for i in range(num_speakers):
                dd, up, speed = speaker_inputs[i]
                path = up if (up and os.path.exists(up)) else self.available_voices.get(dd)
                if not path: raise gr.Error(f"Select voice for Speaker {i+1}")
                audio = self.read_audio(path)
                if len(audio) == 0: raise gr.Error(f"Failed to load audio for Speaker {i+1}")
                if speed != 1.0: audio = self._adjust_voice_speed(audio, speed)
                voice_samples.append(audio)

            lines = [l.strip() for l in script.split('\n') if l.strip()]
            formatted_turns = []
            auto_idx = 0
            for line in lines:
                # --- FIX: ALIGN SPEAKER TAGS ---
                # Check for explicit tag "Speaker X:"
                match = re.match(r'^Speaker\s+(\d+)\s*:(.*)', line, re.IGNORECASE)
                if match:
                    try:
                        s_id = int(match.group(1))
                        content = match.group(2).strip()
                        
                        # Correcting alignment:
                        # User writes "Speaker 1" (1-based) -> We map to Index 0 (0-based)
                        # User writes "Speaker 0" (0-based) -> We map to Index 0 (0-based)
                        # This logic ensures "Speaker 1" uses the First voice (Index 0).
                        if s_id > 0: s_id -= 1
                        
                        # Safety modulo to keep it valid
                        s_id = s_id % num_speakers
                        
                        formatted_turns.append(f"Speaker {s_id}: {content}")
                    except:
                        formatted_turns.append(line)
                elif line.startswith('Speaker ') and ':' in line:
                    # Fallback for non-numeric tags
                    formatted_turns.append(line)
                else:
                    # Auto-assign
                    formatted_turns.append(f"Speaker {auto_idx % num_speakers}: {line}")
                    auto_idx += 1
            
            full_script = '\n'.join(formatted_turns)
            log = f"🚀 Starting Generation (T4 Mode: {self.t4_mode} | Steps: {self.inference_steps})\nTurns: {len(formatted_turns)}\n"
            yield None, None, log, gr.update(visible=True), "Calculating..."

            inputs = self.processor(text=[full_script], voice_samples=[voice_samples], padding=True, return_tensors="pt", return_attention_mask=True)
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            for k, v in inputs.items():
                if torch.is_tensor(v): inputs[k] = v.to(target_device)
            
            audio_streamer = AudioStreamer(batch_size=1)
            self.current_streamer = audio_streamer
            
            thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, audio_streamer, stop_event)
            )
            thread.start()
            
            sample_rate = 24000
            all_chunks = []
            pending_chunks = []
            
            try:
                for chunk in audio_streamer.get_stream(0):
                    if stop_event and stop_event.is_set(): break
                    
                    if torch.is_tensor(chunk): chunk = chunk.float().cpu().numpy().astype(np.float32)
                    else: chunk = np.array(chunk, dtype=np.float32)
                    if chunk.ndim > 1: chunk = chunk.squeeze()
                    
                    all_chunks.append(chunk)
                    pending_chunks.append(convert_to_16_bit_wav(chunk))
                    
                    pending_size = sum(len(c) for c in pending_chunks)
                    current_duration = time.time() - start_time
                    time_display = f"{current_duration:.1f}s"
                    
                    if pending_size > sample_rate * 2:
                        yield (sample_rate, np.concatenate(pending_chunks)), None, log, gr.update(visible=True), time_display
                        pending_chunks = []
            
            finally:
                audio_streamer.end()
                thread.join()

            total_time = time.time() - start_time
            time_display = f"{total_time:.2f}s"

            if all_chunks and not (stop_event and stop_event.is_set()):
                full_audio = np.concatenate(all_chunks)
                log += "\n🧹 Denoising (DeepFilterNet)..."
                yield None, None, log, gr.update(visible=True), time_display
                
                denoised = self.denoise_audio(full_audio, sample_rate)
                denoised_int16 = convert_to_16_bit_wav(denoised)
                
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                wav_path = os.path.join(self.output_dir, f"rsr_tts_{ts}.wav")
                sf.write(wav_path, denoised, 48000)
                
                mp3_path = os.path.join(self.output_dir, f"rsr_tts_{ts}.mp3")
                try:
                    subprocess.run(["ffmpeg", "-i", wav_path, "-acodec", "libmp3lame", "-b:a", "256k", "-ar", "48000", "-y", mp3_path], 
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    saved_msg = f"Saved: {os.path.basename(mp3_path)} & .wav"
                except: saved_msg = f"Saved: {os.path.basename(wav_path)} (FFmpeg missing)"
                
                log += f"\n✅ {saved_msg}\n⏱️ Generation Time: {time_display}"
                yield None, (48000, denoised_int16), log, gr.update(visible=False), time_display
            
            elif stop_event and stop_event.is_set():
                yield None, None, "🛑 Stopped.", gr.update(visible=False), time_display
            else:
                 yield None, None, "❌ Error: No audio.", gr.update(visible=False), time_display

        except Exception as e:
            traceback.print_exc()
            yield None, None, f"Error: {e}", gr.update(visible=False), "Error"
        finally:
            self.is_generating = False

    def _generate_with_streamer(self, inputs, cfg_scale, streamer, stop_event=None):
        try:
            torch.cuda.empty_cache()
            
            # --- CRITICAL: Wait for Queue ---
            with self.gpu_lock:
                # --- CHECK STOP AGAIN AFTER WAITING ---
                if stop_event and stop_event.is_set():
                    streamer.end()
                    return

                with torch.inference_mode():
                    self.model.generate(
                        **inputs,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={'do_sample': False},
                        audio_streamer=streamer,
                        # Pass the unique stop event to the model
                        stop_check_fn=lambda: stop_event.is_set() if stop_event else False,
                        verbose=False,
                        refresh_negative=True,
                    )
        except Exception as e:
            print(f"Gen Error: {e}")
            traceback.print_exc()
            streamer.end()

    def stop_live_generation(self):
        """Signals ONLY the live job to stop."""
        if self.active_live_stop_event:
            self.active_live_stop_event.set()
            if self.current_streamer: self.current_streamer.end()

    def start_background_task(self, num_speakers, script, 
                            s1_d, s2_d, s3_d, s4_d, s1_u, s2_u, s3_u, s4_u, s1_s, s2_s, s3_s, s4_s, cfg_scale):
        
        job_id = str(uuid.uuid4())[:8]
        BACKGROUND_JOBS[job_id] = {"status": "running", "log": "Initializing...", "audio": None}
        
        # Unique stop event for this specific background job (independent of live job)
        bg_stop_event = threading.Event()

        def task_runner():
            try:
                generator = self.generate_podcast_streaming(
                    num_speakers, script, s1_d, s2_d, s3_d, s4_d, s1_u, s2_u, s3_u, s4_u, s1_s, s2_s, s3_s, s4_s, cfg_scale,
                    stop_event=bg_stop_event
                )
                final_audio = None
                last_log = ""
                
                for stream_out, final_out, log_msg, _, _ in generator:
                    BACKGROUND_JOBS[job_id]["log"] = log_msg
                    last_log = log_msg
                    if final_out is not None: final_audio = final_out

                if bg_stop_event.is_set():
                     BACKGROUND_JOBS[job_id]["status"] = "stopped"
                     BACKGROUND_JOBS[job_id]["log"] = last_log + "\n🛑 Job Stopped."
                else:
                    BACKGROUND_JOBS[job_id]["status"] = "completed"
                    BACKGROUND_JOBS[job_id]["audio"] = final_audio
                    BACKGROUND_JOBS[job_id]["log"] = last_log + "\n✅ Background Task Finished."
                
            except Exception as e:
                BACKGROUND_JOBS[job_id]["status"] = "failed"
                BACKGROUND_JOBS[job_id]["log"] += f"\n❌ Error: {str(e)}"
                traceback.print_exc()

        t = threading.Thread(target=task_runner)
        t.daemon = True 
        t.start()
        return job_id

    def check_job_status(self, job_id):
        job = BACKGROUND_JOBS.get(job_id)
        if not job: return None, "❌ Job ID not found.", gr.update(visible=False)
        
        if job["status"] == "running":
            log_msg = job['log']
            if "Calculating..." in log_msg and self.gpu_lock.locked(): log_msg += "\n(Waiting for GPU access...)"
            return None, f"⏳ Processing...\n\nLogs:\n{log_msg}", gr.update(visible=False)
        
        elif job["status"] == "failed": return None, f"❌ Failed.\n\nLogs:\n{job['log']}", gr.update(visible=False)
        elif job["status"] == "stopped": return None, f"🛑 Stopped.\n\nLogs:\n{job['log']}", gr.update(visible=False)
        elif job["status"] == "completed": return job["audio"], f"✅ Done!\n\nLogs:\n{job['log']}", gr.update(visible=True)
        return None, "Unknown Status", gr.update(visible=False)

    def get_saved_files(self):
        if not os.path.exists(self.output_dir): return []
        files = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) if f.endswith('.wav') or f.endswith('.mp3')]
        files.sort(key=os.path.getmtime, reverse=True)
        return files
    
    def load_example_scripts(self): self.example_scripts = [] 

def convert_to_16_bit_wav(data):
    if torch.is_tensor(data): data = data.detach().cpu().numpy()
    data = np.array(data)
    if np.max(np.abs(data)) > 1.0: data = data / np.max(np.abs(data))
    return (data * 32767).astype(np.int16)

def create_demo_interface(demo):
    with gr.Blocks(title="RSR TTS (T4 Optimized)", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# RSR TTS (T4 Optimized Mode)")
        
        with gr.Row():
            with gr.Column(scale=1):
                num_speakers = gr.Slider(1, 4, value=2, step=1, label="Speakers")
                spk_inputs = []
                for i in range(4):
                    with gr.Group(visible=(i<2)) as g:
                        dd = gr.Dropdown(list(demo.available_voices.keys()), label=f"Speaker {i+1}")
                        up = gr.Audio(type="filepath", label="Upload")
                        spd = gr.Slider(0.5, 2.0, value=1.0, label="Speed")
                        spk_inputs.extend([dd, up, spd])
                        def update_vis(n, idx=i, grp=g): return gr.update(visible=(idx < n))
                        num_speakers.change(update_vis, num_speakers, g)
                cfg = gr.Slider(1.0, 3.0, value=1.5, label="CFG Scale")
                gr.Markdown("### 📂 Saved History")
                refresh_btn = gr.Button("🔄 Refresh Files")
                history_files = gr.File(label="Download Generated Files", file_count="multiple", value=demo.get_saved_files())
                refresh_btn.click(lambda: demo.get_saved_files(), None, history_files)

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("🎙️ Live Streaming"):
                        script = gr.Textbox(lines=10, label="Script", placeholder="Speaker 1: Hello...")
                        with gr.Row():
                            btn = gr.Button("Generate", variant="primary")
                            stop = gr.Button("Stop", variant="stop", visible=False)
                        time_box = gr.Textbox(label="Generation Time", value="0.0s", interactive=False)
                        stream_out = gr.Audio(label="Streaming", streaming=True, autoplay=True)
                        final_out = gr.Audio(label="Final (Denoised)", type="numpy")
                        log = gr.Textbox(label="Logs")

                        def wrapper(n, scr, *args):
                            spk_data, cfg_val = args[:-1], args[-1]
                            s1_d, s1_u, s1_s = spk_data[0:3]
                            s2_d, s2_u, s2_s = spk_data[3:6]
                            s3_d, s3_u, s3_s = spk_data[6:9]
                            s4_d, s4_u, s4_s = spk_data[9:12]
                            
                            # Create a unique stop signal for this specific run
                            stop_event = threading.Event()
                            demo.active_live_stop_event = stop_event
                            
                            yield None, None, "Starting...", gr.update(visible=True), "0.0s"
                            for out in demo.generate_podcast_streaming(n, scr, s1_d, s2_d, s3_d, s4_d, s1_u, s2_u, s3_u, s4_u, s1_s, s2_s, s3_s, s4_s, cfg_val, stop_event=stop_event):
                                yield out

                        btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [btn, stop]) \
                            .then(wrapper, [num_speakers, script] + spk_inputs + [cfg], [stream_out, final_out, log, stop, time_box]) \
                            .then(lambda: (gr.update(visible=True), gr.update(visible=False)), None, [btn, stop]) \
                            .then(lambda: demo.get_saved_files(), None, history_files)
                        stop.click(demo.stop_live_generation, None, None)

                    with gr.Tab("☁️ Background Job (Disconnect Safe)"):
                        gr.Markdown("Use this mode to start a long job. You can close the tab and check back later using the **Job ID**.")
                        bg_script = gr.Textbox(lines=10, label="Script", placeholder="Paste script here...")
                        with gr.Row(): bg_btn = gr.Button("🚀 Start Background Job", variant="primary")
                        bg_output_info = gr.Textbox(label="Job ID (Save this!)", interactive=False)
                        gr.Markdown("---")
                        gr.Markdown("### Check Job Status")
                        with gr.Row():
                            job_id_input = gr.Textbox(label="Enter Job ID", placeholder="e.g. a1b2c3d4")
                            check_btn = gr.Button("Check Status")
                        bg_status_log = gr.Textbox(label="Status / Logs")
                        bg_audio_out = gr.Audio(label="Final Audio", visible=False)

                        def bg_wrapper(n, scr, *args):
                            spk_data, cfg_val = args[:-1], args[-1]
                            s1_d, s1_u, s1_s = spk_data[0:3]
                            s2_d, s2_u, s2_s = spk_data[3:6]
                            s3_d, s3_u, s3_s = spk_data[6:9]
                            s4_d, s4_u, s4_s = spk_data[9:12]
                            job_id = demo.start_background_task(n, scr, s1_d, s2_d, s3_d, s4_d, s1_u, s2_u, s3_u, s4_u, s1_s, s2_s, s3_s, s4_s, cfg_val)
                            return f"Job started! ID: {job_id}"

                        bg_btn.click(bg_wrapper, [num_speakers, bg_script] + spk_inputs + [cfg], bg_output_info)
                        check_btn.click(demo.check_job_status, inputs=[job_id_input], outputs=[bg_audio_out, bg_status_log, bg_audio_out])

    return interface

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="aoi-ot/VibeVoice-Large")
    parser.add_argument("--inference_steps", type=int, default=15, help="DDPM steps for diffusion (Default: 15)")
    parser.add_argument("--t4_mode", action="store_true", help="Force 4-bit NF4 for Single T4 GPU compatibility")
    parser.add_argument("--auto_device_map", action="store_true", help="Use Dual-GPU split (if not using t4_mode)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    set_seed(42)
    demo = RSRTTSDemo(args.model_path, inference_steps=args.inference_steps, auto_device_map=args.auto_device_map, t4_mode=args.t4_mode)
    iface = create_demo_interface(demo)
    iface.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()
