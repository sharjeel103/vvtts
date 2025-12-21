"""
RSR TTS - High-Quality Dialogue Generation Interface with Streaming Support
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

from transformers.generation import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference, VibeVoiceGenerationOutput, VibeVoiceTokenConstraintProcessor
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed
from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class RSRTTSDemo:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 8, auto_device_map: bool = False):
        """Initialize the RSR TTS demo with model loading."""
        self.model_path = model_path
        self.device = "cuda" # Enforce CUDA for T4 setup
        self.inference_steps = inference_steps
        self.auto_device_map = auto_device_map
        self.is_generating = False  # Track generation state
        self.stop_generation = False  # Flag to stop generation
        self.current_streamer = None  # Track current audio streamer
        
        # Performance tuning
        torch.backends.cudnn.benchmark = True
        
        # Ensure output directory exists immediately
        self.output_dir = "saved_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.load_model()
        self.setup_voice_presets()
        self.load_example_scripts()  # Load example scripts
        
    def load_model(self):
        """Load the TTS model and processor."""
        print(f"Loading processor & model from {self.model_path}")
        print(f"Using device: {self.device} (Dual T4 Optimized)")
        
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        
        # --- T4 SPECIFIC CONFIGURATION ---
        load_dtype = torch.float16
        attn_impl_primary = "sdpa" 
            
        print(f"Configured: torch_dtype={load_dtype}, attn_implementation={attn_impl_primary}")

        load_kwargs = {
            "torch_dtype": load_dtype,
            "attn_implementation": attn_impl_primary,
        }

        # --- DUAL T4 MANUAL SPLIT STRATEGY (OPTIMIZED BALANCE) ---
        if self.auto_device_map:
            if torch.cuda.device_count() >= 2:
                print("⚡ Dual GPU detected. Constructing RE-BALANCED manual split map...")
                device_map = {}
                
                # OPTIMIZED BALANCE: 
                # GPU 1 was heavy (11.3GB) vs GPU 0 (7GB).
                # Moving 4 more layers to GPU 0.
                # Split at Layer 18 (0-17 on GPU 0, 18-27 on GPU 1)
                
                # --- GPU 0 ---
                # Embeddings + 18 Layers
                device_map["model.language_model.embed_tokens"] = 0
                for i in range(18): 
                    device_map[f"model.language_model.layers.{i}"] = 0
                
                # --- GPU 1 ---
                # 10 Layers + Heads + Overhead
                for i in range(18, 28): 
                    device_map[f"model.language_model.layers.{i}"] = 1
                
                device_map["model.language_model.norm"] = 1
                device_map["lm_head"] = 1
                
                # Custom Components on GPU 1
                device_map["model.prediction_head"] = 1
                device_map["model.acoustic_tokenizer"] = 1
                device_map["model.semantic_tokenizer"] = 1
                device_map["model.acoustic_connector"] = 1
                device_map["model.semantic_connector"] = 1
                device_map["model.speech_bias_factor"] = 1
                device_map["model.speech_scaling_factor"] = 1
                
                print(f"🗺️  Using Explicit Device Map: Optimized Split at Layer 18")
                load_kwargs["device_map"] = device_map
            else:
                print("⚡ Single GPU or non-standard setup. Using 'auto'.")
                load_kwargs["device_map"] = "auto"
        
        # Load model
        try:
            print(f"Loading model with kwargs: {load_kwargs}")
            
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                **load_kwargs
            )
            
            
            # --- PATCH 1: _process_speech_inputs (Prefill Stage) ---
            def patched_process_speech_inputs(self, speech_tensors, speech_masks, speech_type="audio"):
                """Patched version to ensure device compatibility."""
                with torch.no_grad():
                    # Determine target device from model's input embeddings
                    target_device = self.model.get_input_embeddings().weight.device
                    
                    if speech_type == "audio":
                        tokenizer_device = self.model.acoustic_tokenizer.device
                        speech_tensors = speech_tensors.to(tokenizer_device)
                        
                        encoder_output = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
                        acoustic_latents = encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]
                        
                        acoustic_features = (acoustic_latents + self.model.speech_bias_factor.to(acoustic_latents.device)) * self.model.speech_scaling_factor.to(acoustic_latents.device)
                        
                        connector_device = self.model.acoustic_connector.fc1.weight.device
                        speech_masks_device = speech_masks.to(connector_device)
                        
                        acoustic_connected = self.model.acoustic_connector(acoustic_features.to(connector_device))[speech_masks_device]
                        
                        # MOVE TO TARGET DEVICE
                        return acoustic_features, acoustic_connected.to(target_device)
                    else:
                         return self._original_process_speech_inputs(speech_tensors, speech_masks, speech_type)

            self.model._original_process_speech_inputs = self.model._process_speech_inputs
            self.model._process_speech_inputs = types.MethodType(patched_process_speech_inputs, self.model)
            print("✅ Applied device-alignment patch to _process_speech_inputs")

            # --- PATCH 2: generate (Autoregressive Stage) ---
            def patched_generate(
                self,
                inputs: Optional[torch.Tensor] = None,
                generation_config: Optional[GenerationConfig] = None,
                logits_processor: Optional[LogitsProcessorList] = None,
                stopping_criteria: Optional[StoppingCriteriaList] = None,
                prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
                synced_gpus: Optional[bool] = None,
                assistant_model: Optional["PreTrainedModel"] = None,
                audio_streamer: Optional[Union[AudioStreamer, Any]] = None, 
                negative_prompt_ids: Optional[torch.Tensor] = None,
                negative_prompt_attention_mask: Optional[torch.Tensor] = None,
                speech_tensors: Optional[torch.FloatTensor] = None,
                speech_masks: Optional[torch.BoolTensor] = None,
                speech_input_mask: Optional[torch.BoolTensor] = None,
                return_speech: bool = True,
                cfg_scale: float = 1.0,
                stop_check_fn: Optional[Callable[[], bool]] = None,
                **kwargs,
            ) -> Union[torch.LongTensor, VibeVoiceGenerationOutput]:
                
                # --- START ORIGINAL GENERATE LOGIC COPY ---
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
                verbose = kwargs.get("verbose", False)

                audio_chunks = [[] for _ in range(batch_size)]

                initial_length = input_ids.shape[-1]
                initial_length_per_sample = model_kwargs['attention_mask'].sum(dim=-1)

                valid_tokens = [
                    generation_config.speech_start_id,
                    generation_config.speech_end_id, 
                    generation_config.speech_diffusion_id,
                    generation_config.eos_token_id
                ]
                if hasattr(generation_config, 'bos_token_id') and generation_config.bos_token_id is not None:
                    valid_tokens.append(generation_config.bos_token_id)
                
                token_constraint_processor = VibeVoiceTokenConstraintProcessor(valid_tokens, device=device)
                if logits_processor is None:
                    logits_processor = LogitsProcessorList()
                logits_processor.append(token_constraint_processor)
                
                max_steps = min(generation_config.max_length - initial_length, int(max_length_times * initial_length))
                max_step_per_sample = torch.min(generation_config.max_length - initial_length_per_sample, (max_length_times * initial_length_per_sample).long())
                reach_max_step_sample = torch.zeros(batch_size, dtype=torch.bool, device=device)

                if kwargs.get("show_progress_bar", True):
                    progress_bar = tqdm(range(max_steps), desc="Generating", leave=False)
                else:
                    progress_bar = range(max_steps)
                
                for step in progress_bar:
                    if stop_check_fn is not None and stop_check_fn():
                        if audio_streamer is not None:
                            audio_streamer.end()
                        break
                    
                    if audio_streamer is not None and hasattr(audio_streamer, 'finished_flags'):
                        if any(audio_streamer.finished_flags):
                            break
                    
                    if finished_tags.all():
                        break

                    if input_ids.shape[-1] >= generation_config.max_length:
                        reached_samples = torch.arange(batch_size, device=device)[~finished_tags]
                        if reached_samples.numel() > 0:
                            reach_max_step_sample[reached_samples] = True
                        break
                    
                    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                    if is_prefill:
                        prefill_inputs = {
                            "speech_tensors": speech_tensors.to(device=device),
                            "speech_masks": speech_masks.to(device),
                            "speech_input_mask": speech_input_mask.to(device),
                        }
                        is_prefill = False
                    else:
                        _ = model_inputs.pop('inputs_embeds', None)
                        prefill_inputs = {'inputs_embeds': inputs_embeds}

                    outputs = self(
                        **model_inputs, **prefill_inputs, logits_to_keep=1, return_dict=True, output_attentions=False, output_hidden_states=False,
                    )
                    model_kwargs = self._update_model_kwargs_for_generation(
                        outputs, model_kwargs, is_encoder_decoder=False,
                    )

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

                        negative_outputs = self(
                            **negative_model_inputs, logits_to_keep=0, return_dict=True, output_attentions=False, output_hidden_states=False,
                        )
                        negative_model_kwargs = self._update_model_kwargs_for_generation(
                            negative_outputs, negative_model_kwargs, is_encoder_decoder=False,
                        )
                        negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)

                    if (next_tokens == generation_config.eos_token_id).any():
                        eos_indices = (next_tokens == generation_config.eos_token_id).nonzero(as_tuple=False).squeeze(1)
                        new_eos_indices = eos_indices[~finished_tags[eos_indices]]
                        if new_eos_indices.numel() > 0:
                            finished_tags[new_eos_indices] = True
                            if audio_streamer is not None:
                                audio_streamer.end(new_eos_indices)

                    max_length_reached = step >= max_step_per_sample
                    new_max_length_indices = torch.nonzero(max_length_reached & ~finished_tags, as_tuple=False).squeeze(1)
                    if new_max_length_indices.numel() > 0:
                        finished_tags[new_max_length_indices] = True
                        reach_max_step_sample[new_max_length_indices] = True
                        if audio_streamer is not None:
                            audio_streamer.end(new_max_length_indices)

                    diffusion_end_indices = (next_tokens == generation_config.speech_end_id).nonzero(as_tuple=False).squeeze(1)
                    if diffusion_end_indices.numel() > 0:
                        acoustic_cache.set_to_zero(diffusion_end_indices)
                        semantic_cache.set_to_zero(diffusion_end_indices)
                    
                    diffusion_start_indices = torch.arange(batch_size, device=device)[~finished_tags & (next_tokens == generation_config.speech_start_id)]
                    if diffusion_start_indices.numel() > 0 and kwargs.get('refresh_negative', True):
                        for i, sample_idx in enumerate(diffusion_start_indices.tolist()):
                            negative_model_kwargs['attention_mask'][sample_idx, :] = 0
                            negative_model_kwargs['attention_mask'][sample_idx, -1] = 1
                        for layer_idx, (k_cache, v_cache) in enumerate(zip(negative_model_kwargs['past_key_values'].key_cache, 
                                                                                negative_model_kwargs['past_key_values'].value_cache)):
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

                            negative_outputs = self(
                                **negative_model_inputs, logits_to_keep=0, return_dict=True, output_attentions=False, output_hidden_states=False,
                            )
                            negative_model_kwargs = self._update_model_kwargs_for_generation(
                                negative_outputs, negative_model_kwargs, is_encoder_decoder=False,
                            )
                            negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)
                        
                        non_diffusion_mask = ~finished_tags & (next_tokens != generation_config.speech_diffusion_id)
                        if non_diffusion_mask.any():
                            non_diffusion_indices = torch.arange(batch_size, device=device)[non_diffusion_mask]
                            start_indices = correct_cnt[non_diffusion_indices]

                            seq_len = negative_model_kwargs['attention_mask'].shape[1]
                            for i, (sample_idx, start_idx) in enumerate(zip(non_diffusion_indices.tolist(), start_indices.tolist())):
                                if start_idx + 1 < seq_len - 1:
                                    negative_model_kwargs['attention_mask'][sample_idx, start_idx+1:] = \
                                        negative_model_kwargs['attention_mask'][sample_idx, start_idx:-1].clone()
                                negative_model_kwargs['attention_mask'][sample_idx, start_idx] = 0

                            for layer_idx, (k_cache, v_cache) in enumerate(zip(negative_model_kwargs['past_key_values'].key_cache, 
                                                                                negative_model_kwargs['past_key_values'].value_cache)):
                                for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                                    if start_idx + 1 < k_cache.shape[2] - 1:
                                        k_cache[sample_idx, :, start_idx+1:, :] = k_cache[sample_idx, :, start_idx:-1, :].clone()
                                        v_cache[sample_idx, :, start_idx+1:, :] = v_cache[sample_idx, :, start_idx:-1, :].clone()
                            
                            for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                                if start_idx + 1 < negative_input_ids.shape[1] - 1:
                                    negative_input_ids[sample_idx, start_idx+1:] = \
                                        negative_input_ids[sample_idx, start_idx:-1].clone()
                                        
                            correct_cnt[non_diffusion_indices] += 1

                        positive_condition = outputs.last_hidden_state[diffusion_indices, -1, :]
                        negative_condition = negative_outputs.last_hidden_state[diffusion_indices, -1, :]
                        
                        speech_latent = self.sample_speech_tokens(
                            positive_condition,
                            negative_condition,
                            cfg_scale=cfg_scale,
                        ).unsqueeze(1)
                                        
                        scaled_latent = speech_latent / self.model.speech_scaling_factor.to(speech_latent.device) - self.model.speech_bias_factor.to(speech_latent.device)
                        audio_chunk = self.model.acoustic_tokenizer.decode(
                            scaled_latent.to(self.model.acoustic_tokenizer.device),
                            cache=acoustic_cache,
                            sample_indices=diffusion_indices.to(self.model.acoustic_tokenizer.device),
                            use_cache=True,
                            debug=False
                        )
                        
                        for i, sample_idx in enumerate(diffusion_indices):
                            idx = sample_idx.item()
                            if not finished_tags[idx]:
                                audio_chunks[idx].append(audio_chunk[i])

                        if audio_streamer is not None:
                            audio_streamer.put(audio_chunk, diffusion_indices)
                            
                        semantic_features = self.model.semantic_tokenizer.encode(
                            audio_chunk,
                            cache=semantic_cache,
                            sample_indices=diffusion_indices,
                            use_cache=True,
                            debug=False
                        ).mean
                        
                        acoustic_embed = self.model.acoustic_connector(speech_latent)
                        semantic_embed = self.model.semantic_connector(semantic_features)
                        diffusion_embeds = acoustic_embed + semantic_embed

                        # === CRITICAL FIX: Ensure diffusion_embeds is on the same device as next_inputs_embeds ===
                        target_device_for_embeds = next_inputs_embeds.device
                        next_inputs_embeds[diffusion_indices] = diffusion_embeds.to(target_device_for_embeds)
                    
                    inputs_embeds = next_inputs_embeds

                if audio_streamer is not None:
                    audio_streamer.end()

                final_audio_outputs = []
                for sample_chunks in audio_chunks:
                    if sample_chunks:
                        concatenated_audio = torch.cat(sample_chunks, dim=-1)
                        final_audio_outputs.append(concatenated_audio)
                    else:
                        final_audio_outputs.append(None)

                return VibeVoiceGenerationOutput(
                    sequences=input_ids,
                    speech_outputs=final_audio_outputs if return_speech else None,
                    reach_max_step_sample=reach_max_step_sample,
                )
                # --- END ORIGINAL GENERATE LOGIC COPY ---

            # Apply the patch to the instance
            self.model.generate = types.MethodType(patched_generate, self.model)
            print("✅ Applied device-alignment patch to generate()")

            # Ensure model is in evaluation mode
            self.model.eval()
            print("✅ Model loaded successfully.")
                
        except Exception as e:
            print(f"[ERROR] Loading failed: {e}")
            print(traceback.format_exc())
            raise e
        
        # Configure Scheduler
        if hasattr(self.model, "model") and hasattr(self.model.model, "noise_scheduler"):
            try:
                self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
                    self.model.model.noise_scheduler.config, 
                    algorithm_type='sde-dpmsolver++',
                    beta_schedule='squaredcos_cap_v2'
                )
                self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
            except Exception as e:
                print(f"Warning: Could not configure noise scheduler: {e}")
    
    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        
        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        # Scan for all WAV files in the voices directory
        self.voice_presets = {}
        
        # Get all .wav files in the voices directory
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')) and os.path.isfile(os.path.join(voices_dir, f))]
        
        # Create dictionary with filename (without extension) as key
        for wav_file in wav_files:
            # Remove .wav extension to get the name
            name = os.path.splitext(wav_file)[0]
            # Create full path
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path
        
        # Sort the voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        
        # Filter out voices that don't exist
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        if not self.available_voices:
            print("Warning: No voice presets found in demo/voices directory.")
        
        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file."""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])

    def _adjust_voice_speed(self, audio_np: np.ndarray, speed_factor: float, sample_rate: int = 24000) -> np.ndarray:
        """Adjust voice speed using time-stretching without changing pitch."""
        if speed_factor == 1.0:
            return audio_np  # No change needed
        
        try:
            # Use pyrb.time_stretch instead of librosa
            adjusted_audio = pyrb.time_stretch(y=audio_np, sr=sample_rate, rate=speed_factor)
            original_length = len(audio_np)
            target_length = len(adjusted_audio)
            logger.info(f"Adjusted voice speed by factor {speed_factor:.2f} ({original_length} -> {target_length} samples)")
            return adjusted_audio
        except Exception as e:
            logger.error(f"Error during voice speed adjustment: {e}. Returning original audio.")
            return audio_np
    
    def generate_podcast_streaming(self, 
                                 num_speakers: int,
                                 script: str,
                                 speaker_1: str = None,
                                 speaker_2: str = None,
                                 speaker_3: str = None,
                                 speaker_4: str = None,
                                 speaker_1_upload: str = None,
                                 speaker_2_upload: str = None,
                                 speaker_3_upload: str = None,
                                 speaker_4_upload: str = None,
                                 speaker_1_speed: float = 1.0,
                                 speaker_2_speed: float = 1.0,
                                 speaker_3_speed: float = 1.0,
                                 speaker_4_speed: float = 1.0,
                                 cfg_scale: float = 1.3) -> Iterator[tuple]:
        
        # Setup output directory
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create output directory: {e}")

        try:
            # Reset stop flag and set generating state
            self.stop_generation = False
            self.is_generating = True
            
            # Validate inputs
            if not script.strip():
                self.is_generating = False
                raise gr.Error("Error: Please provide a script.")

            script = script.replace("’", "'")
            
            if num_speakers < 1 or num_speakers > 4:
                self.is_generating = False
                raise gr.Error("Error: Number of speakers must be between 1 and 4.")
            
            # --- Handle custom uploads and dropdowns ---
            speaker_dropdowns = [speaker_1, speaker_2, speaker_3, speaker_4]
            speaker_uploads = [speaker_1_upload, speaker_2_upload, speaker_3_upload, speaker_4_upload]
            speaker_speeds = [speaker_1_speed, speaker_2_speed, speaker_3_speed, speaker_4_speed]
            
            selected_audio_paths = [] 
            selected_speaker_names_for_log = []
            
            for i in range(num_speakers):
                upload_path = speaker_uploads[i]
                dropdown_name = speaker_dropdowns[i]
                
                if upload_path and os.path.exists(upload_path):
                    selected_audio_paths.append(upload_path)
                    selected_speaker_names_for_log.append(f"Custom (Speaker {i+1})")
                elif dropdown_name and dropdown_name in self.available_voices:
                    selected_audio_paths.append(self.available_voices[dropdown_name])
                    selected_speaker_names_for_log.append(dropdown_name)
                else:
                    self.is_generating = False
                    raise gr.Error(f"Error: Please select a default voice or upload a custom voice for Speaker {i+1}.")

            # Build initial log
            log = f"🎙️ Generating Audio with {num_speakers} speakers\n"
            log += f"📊 Parameters: CFG Scale={cfg_scale}, Inference Steps={self.inference_steps}\n"
            log += f"🎭 Speakers: {', '.join(selected_speaker_names_for_log)}\n"
            
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Load voice samples
            voice_samples = []
            for i, audio_path in enumerate(selected_audio_paths):
                audio_data = self.read_audio(audio_path)
                if len(audio_data) == 0:
                    self.is_generating = False
                    raise gr.Error(f"Error: Failed to load audio for {selected_speaker_names_for_log[i]}")
                
                speed_factor = speaker_speeds[i]
                if speed_factor != 1.0:
                    logger.info(f"Applying speed factor {speed_factor:.2f} to Speaker {i+1}")
                    audio_data = self._adjust_voice_speed(audio_data, speed_factor, sample_rate=24000)
                    selected_speaker_names_for_log[i] += f" ({speed_factor:.2f}x speed)"
                
                voice_samples.append(audio_data)
            
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Parse script
            lines = script.strip().split('\n')
            formatted_script_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('Speaker ') and ':' in line:
                    formatted_script_lines.append(line)
                else:
                    speaker_id = len(formatted_script_lines) % num_speakers
                    formatted_script_lines.append(f"Speaker {speaker_id}: {line}")
            
            formatted_script = '\n'.join(formatted_script_lines)
            log += f"📝 Formatted script with {len(formatted_script_lines)} turns\n\n"
            log += "🔄 Processing (streaming mode)...\n"
            
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            start_time = time.time()
            
            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            # Move inputs to device (usually cuda:0)
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(target_device)
            
            # Create audio streamer
            audio_streamer = AudioStreamer(
                batch_size=1,
                stop_signal=None,
                timeout=None
            )
            self.current_streamer = audio_streamer
            
            # Start generation thread
            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, audio_streamer)
            )
            generation_thread.start()
            
            # Wait for start
            time.sleep(1)

            if self.stop_generation:
                audio_streamer.end()
                generation_thread.join(timeout=5.0)
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            # Collect audio chunks
            sample_rate = 24000
            all_audio_chunks = []
            pending_chunks = []
            chunk_count = 0
            last_yield_time = time.time()
            min_yield_interval = 15
            min_chunk_size = sample_rate * 30
            
            audio_stream = audio_streamer.get_stream(0)
            
            has_yielded_audio = False
            has_received_chunks = False
            
            try:
                for audio_chunk in audio_stream:
                    if self.stop_generation:
                        audio_streamer.end()
                        break
                        
                    chunk_count += 1
                    has_received_chunks = True
                    
                    if torch.is_tensor(audio_chunk):
                        if audio_chunk.dtype == torch.bfloat16:
                            audio_chunk = audio_chunk.float()
                        elif audio_chunk.dtype == torch.float16:
                            audio_chunk = audio_chunk.float()
                        audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                    else:
                        audio_np = np.array(audio_chunk, dtype=np.float32)
                    
                    if len(audio_np.shape) > 1:
                        audio_np = audio_np.squeeze()
                    
                    audio_16bit = convert_to_16_bit_wav(audio_np)
                    all_audio_chunks.append(audio_16bit)
                    pending_chunks.append(audio_16bit)
                    
                    pending_audio_size = sum(len(chunk) for chunk in pending_chunks)
                    current_time = time.time()
                    time_since_last_yield = current_time - last_yield_time
                    
                    should_yield = False
                    if not has_yielded_audio and pending_audio_size >= min_chunk_size:
                        should_yield = True
                        has_yielded_audio = True
                    elif has_yielded_audio and (pending_audio_size >= min_chunk_size or time_since_last_yield >= min_yield_interval):
                        should_yield = True
                    
                    if should_yield and pending_chunks:
                        new_audio = np.concatenate(pending_chunks)
                        total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                        log_update = log + f"🎵 Streaming: {total_duration:.1f}s generated (chunk {chunk_count})\n"
                        yield (sample_rate, new_audio), None, log_update, gr.update(visible=True)
                        pending_chunks = []
                        last_yield_time = current_time
            
            except GeneratorExit:
                print("Client disconnected.")
            except Exception as e:
                print(f"Error during streaming loop: {e}")
                raise e
            finally:
                if all_audio_chunks:
                    try:
                        complete_audio = np.concatenate(all_audio_chunks)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"rsr_tts_{timestamp}.wav"
                        filepath = os.path.join(self.output_dir, filename)
                        sf.write(filepath, complete_audio, sample_rate)
                        print(f"✅ Audio saved to: {filepath}")
                    except Exception as save_error:
                        print(f"❌ Failed to save backup: {save_error}")
                
                audio_streamer.end()
                generation_thread.join(timeout=2.0)
            
            if pending_chunks and not self.stop_generation:
                final_new_audio = np.concatenate(pending_chunks)
                total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                log_update = log + f"🎵 Streaming final chunk: {total_duration:.1f}s total\n"
                yield (sample_rate, final_new_audio), None, log_update, gr.update(visible=True)
                has_yielded_audio = True

            self.current_streamer = None
            self.is_generating = False
            generation_time = time.time() - start_time
            
            if self.stop_generation:
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            if has_received_chunks and not has_yielded_audio and all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                final_log = log + f"⏱️ Completed in {generation_time:.2f}s\n🎵 Duration: {final_duration:.2f}s\n✨ Success!"
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
                return
            
            if not has_received_chunks:
                error_log = log + f"\n❌ Error: No audio chunks received. Time: {generation_time:.2f}s"
                yield None, None, error_log, gr.update(visible=False)
                return

            if all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                final_log = log + f"⏱️ Completed in {generation_time:.2f}s\n🎵 Duration: {final_duration:.2f}s\n✨ Success!"
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
            else:
                final_log = log + "❌ No audio was generated."
                yield None, None, final_log, gr.update(visible=False)

        except gr.Error as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ Input Error: {str(e)}"
            print(error_msg)
            yield None, None, error_msg, gr.update(visible=False)
            
        except Exception as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            yield None, None, error_msg, gr.update(visible=False)
    
    def _generate_with_streamer(self, inputs, cfg_scale, audio_streamer):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            # Clear CUDA cache before generation
            torch.cuda.empty_cache()
            gc.collect()
            
            if self.stop_generation:
                audio_streamer.end()
                return
                
            def check_stop_generation():
                return self.stop_generation
                
            # Use inference_mode instead of no_grad for better memory optimization
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={
                        'do_sample': False,
                    },
                    audio_streamer=audio_streamer,
                    stop_check_fn=check_stop_generation,
                    verbose=False,
                    refresh_negative=True,
                )
            
            # Clear cache after generation
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error in generation thread: {e}")
            traceback.print_exc()
            audio_streamer.end()
    
    def stop_audio_generation(self):
        """Stop the current audio generation process."""
        self.stop_generation = True
        if self.current_streamer is not None:
            try:
                self.current_streamer.end()
            except Exception as e:
                print(f"Error stopping streamer: {e}")
        print("🛑 Audio generation stop requested")
    
    def load_example_scripts(self):
        """Load example scripts from the text_examples directory."""
        examples_dir = os.path.join(os.path.dirname(__file__), "text_examples")
        self.example_scripts = []
        
        if not os.path.exists(examples_dir):
            print(f"Warning: text_examples directory not found at {examples_dir}")
            return
        
        txt_files = sorted([f for f in os.listdir(examples_dir) 
                          if f.lower().endswith('.txt') and os.path.isfile(os.path.join(examples_dir, f))])
        
        for txt_file in txt_files:
            file_path = os.path.join(examples_dir, txt_file)
            import re
            time_pattern = re.search(r'(\d+)min', txt_file.lower())
            if time_pattern:
                minutes = int(time_pattern.group(1))
                if minutes > 15:
                    print(f"Skipping {txt_file}: duration {minutes} minutes exceeds 15-minute limit")
                    continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    script_content = f.read().strip()
                
                script_content = '\n'.join(line for line in script_content.split('\n') if line.strip())
                
                if not script_content:
                    continue
                
                num_speakers = self._get_num_speakers_from_script(script_content)
                self.example_scripts.append([num_speakers, script_content])
                print(f"Loaded example: {txt_file} with {num_speakers} speakers")
                
            except Exception as e:
                print(f"Error loading example script {txt_file}: {e}")
        
        if self.example_scripts:
            print(f"Successfully loaded {len(self.example_scripts)} example scripts")
        else:
            print("No example scripts were loaded")
    
    def _get_num_speakers_from_script(self, script: str) -> int:
        """Determine the number of unique speakers in a script."""
        import re
        speakers = set()
        lines = script.strip().split('\n')
        for line in lines:
            match = re.match(r'^Speaker\s+(\d+)\s*:', line.strip(), re.IGNORECASE)
            if match:
                speaker_id = int(match.group(1))
                speakers.add(speaker_id)
        
        if not speakers:
            return 1
        
        max_speaker = max(speakers)
        min_speaker = min(speakers)
        
        if min_speaker == 0:
            return max_speaker + 1
        else:
            return len(speakers)
    
    def get_saved_files(self) -> List[str]:
        """Get list of saved audio files sorted by creation time (newest first)."""
        if not os.path.exists(self.output_dir):
            return []
        try:
            files = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) 
                    if f.lower().endswith('.wav')]
            files.sort(key=os.path.getmtime, reverse=True)
            return files
        except Exception as e:
            print(f"Error listing saved files: {e}")
            return []
    

def create_demo_interface(demo_instance: RSRTTSDemo):
    """Create the Gradio interface with streaming support."""
    
    # Custom CSS for high-end aesthetics with lighter theme
    custom_css = """
    /* Modern light theme with gradients */
    .gradio-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
    .settings-card, .generation-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Speaker selection styling */
    .speaker-grid {
        display: grid;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .speaker-item {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
        border: 1px solid rgba(148, 163, 184, 0.4);
        border-radius: 12px;
        padding: 1rem;
        color: #374151;
        font-weight: 500;
    }
    
    /* Streaming indicator */
    .streaming-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.1); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Queue status styling */
    .queue-status {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        text-align: center;
        font-size: 0.9rem;
        color: #0369a1;
    }
    
    .generate-btn {
        background: linear-gradient(135deg, #059669 0%, #0d9488 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(5, 150, 105, 0.4);
        transition: all 0.3s ease;
    }
    
    .generate-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(5, 150, 105, 0.6);
    }
    
    .stop-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
        transition: all 0.3s ease;
    }
    
    .stop-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(239, 68, 68, 0.6);
    }
    
    /* Audio player styling */
    .audio-output {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    
    .complete-audio-section {
        margin-top: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
    }
    
    /* Text areas */
    .script-input, .log-output {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(148, 163, 184, 0.4) !important;
        border-radius: 12px !important;
        color: #1e293b !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .script-input::placeholder {
        color: #64748b !important;
    }
    
    /* Sliders */
    .slider-container {
        background: rgba(248, 250, 252, 0.8);
        border: 1px solid rgba(226, 232, 240, 0.6);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Labels and text */
    .gradio-container label {
        color: #374151 !important;
        font-weight: 600 !important;
    }
    
    .gradio-container .markdown {
        color: #1f2937 !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .settings-card, .generation-card { padding: 1rem; }
    }
    
    /* Random example button styling - more subtle professional color */
    .random-btn {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 20px rgba(100, 116, 139, 0.3);
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .random-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(100, 116, 139, 0.4);
        background: linear-gradient(135deg, #475569 0%, #334155 100%);
    }

    /* --- Dropdown fixes: ensure correct stacking & positioning --- */
    /* Allow popovers to overflow card boundaries */
    .settings-card,
    .generation-card,
    .speaker-item { overflow: visible !important; }

    /* Provide local positioning context and raise stacking */
    .speaker-item { position: relative; z-index: 1000 !important; }

    /* Gradio dropdown popover wrapper is usually an .absolute sibling of input within .wrap */
    .speaker-item .wrap { overflow: visible !important; position: relative; }
    .speaker-item .wrap > .absolute {
        z-index: 10000 !important;
        top: calc(100% + 6px) !important;
        bottom: auto !important;
        transform-origin: top !important;
    }

    /* Also elevate the ARIA listbox in case it has a separate stacking context */
    .gradio-container div[role="listbox"] { z-index: 200000 !important; }

    /* General fallback for absolute popovers used by Gradio */
    .gradio-container .absolute,
    .gradio-container .z-20,
    .gradio-container .z-30,
    .gradio-container .z-40,
    .gradio-container .z-50 { z-index: 200000 !important; }

    /* Ensure portal-based menus are above everything */
    .fixed { z-index: 300000 !important; }

    /* Ensure slider does not sit above dropdown popovers */
    .slider-container { position: relative; z-index: 0 !important; overflow: visible !important; }
    .slider-container input[type="range"] { position: relative; z-index: 0 !important; }

    /* ========================= */
    /* Dark Mode          */
    /* ========================= */
    :root { color-scheme: dark; }

    .gradio-container {
        background: radial-gradient(1200px 800px at 20% 0%, #0b1220 0%, #0a0f1a 40%, #090e19 100%);
        color: #e5e7eb;
    }

    .main-header {
        background: linear-gradient(90deg, #0ea5e9 0%, #7c3aed 100%);
        box-shadow: 0 10px 40px rgba(20, 184, 166, 0.15);
    }
    .main-header h1 { color: #ffffff; }
    .main-header p { color: rgba(255,255,255,0.82); }

    .settings-card, .generation-card {
        background: rgba(2, 6, 23, 0.72);
        border: 1px solid rgba(51, 65, 85, 0.7);
        box-shadow: 0 8px 32px rgba(2, 6, 23, 0.6);
    }

    .speaker-item {
        background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
        border: 1px solid rgba(71, 85, 105, 0.55);
        color: #e2e8f0;
    }

    .generate-btn {
        background: linear-gradient(135deg, #059669 0%, #0f766e 100%);
        box-shadow: 0 4px 20px rgba(5, 150, 105, 0.35);
    }
    .generate-btn:hover {
        box-shadow: 0 6px 25px rgba(5, 150, 105, 0.55);
    }

    .stop-btn {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        box-shadow: 0 4px 20px rgba(220, 38, 38, 0.35);
    }
    .stop-btn:hover {
        box-shadow: 0 6px 25px rgba(220, 38, 38, 0.55);
    }

    .random-btn {
        background: linear-gradient(135deg, #334155 0%, #1f2937 100%);
        box-shadow: 0 4px 20px rgba(31, 41, 55, 0.35);
        color: #e5e7eb;
    }
    .random-btn:hover {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        box-shadow: 0 6px 25px rgba(31, 41, 55, 0.5);
    }

    .audio-output {
        background: linear-gradient(135deg, #0b1220 0%, #0f172a 100%);
        border: 1px solid rgba(51, 65, 85, 0.6);
    }

    .complete-audio-section {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.2) 0%, rgba(4, 120, 87, 0.15) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .script-input, .log-output {
        background: rgba(15, 23, 42, 0.92) !important;
        border: 1px solid rgba(51, 65, 85, 0.8) !important;
        color: #e2e8f0 !important;
    }
    .script-input::placeholder { color: #94a3b8 !important; }

    .slider-container {
        background: rgba(2, 6, 23, 0.6);
        border: 1px solid rgba(51, 65, 85, 0.7);
    }

    .gradio-container label { color: #e2e8f0 !important; }
    .gradio-container .markdown { color: #e5e7eb !important; }

    /* Dropdown menu dark palette */
    .gradio-container div[role="listbox"] {
        background-color: #0b1220 !important;
        border: 1px solid #334155 !important;
        color: #e5e7eb !important;
        box-shadow: 0 12px 32px rgba(2, 6, 23, 0.7);
    }
    .gradio-container [role="option"] {
        color: #e5e7eb !important;
    }
    .gradio-container [role="option"][aria-selected="true"],
    .gradio-container [role="option"]:hover {
        background-color: #0f172a !important;
    }

    /* Remove blur/backdrop filters that create problematic stacking/containing contexts */
    .settings-card, .generation-card {
        -webkit-backdrop-filter: none !important;
        backdrop-filter: none !important;
        /* position: relative;  <-- REMOVED to fix dropdown clipping */
        /* z-index: 0;          <-- REMOVED to fix dropdown clipping */
    }
    """
    
    with gr.Blocks(
        title="RSR TTS",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
        )
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>RSR TTS</h1>
        </div>
        """)
        
        with gr.Row():
            # Left column - Settings
            with gr.Column(scale=1, elem_classes="settings-card"):
                gr.Markdown("### 🎛️ **Audio Settings**")
                
                # Number of speakers
                num_speakers = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1,
                    label="Number of Speakers",
                    elem_classes="slider-container"
                )
                
                # Speaker selection
                gr.Markdown("### 🎭 **Speaker Selection**")
                
                available_speaker_names = list(demo_instance.available_voices.keys())
                default_speakers = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']

                speaker_selections = []
                speaker_uploads = []
                speaker_speed_sliders = [] # <-- NEW
                speaker_groups = []
                for i in range(4):
                    with gr.Group(visible=(i < 2), elem_classes="speaker-item") as speaker_group:
                        default_value = default_speakers[i] if i < len(default_speakers) else None
                        speaker_dd = gr.Dropdown(
                            choices=available_speaker_names,
                            value=default_value,
                            label=f"Speaker {i+1} (Default Voice)",
                        )
                        speaker_up = gr.Audio(
                            label=f"OR Upload Custom Voice for Speaker {i+1}",
                            type="filepath",  # Use filepath to get a temp path to the uploaded file
                            sources=["upload", "microphone"],
                        )
                        # --- START: Add speed slider ---
                        speaker_speed = gr.Slider(
                            minimum=0.8,
                            maximum=1.2,
                            value=1.0,
                            step=0.01,
                            label="Voice Speed",
                            info="< 1.0 = Slower, > 1.0 = Faster",
                            elem_classes="slider-container"
                        )
                        # --- END: Add speed slider ---
                    speaker_selections.append(speaker_dd)
                    speaker_uploads.append(speaker_up)
                    speaker_speed_sliders.append(speaker_speed) # <-- NEW
                    speaker_groups.append(speaker_group)
                
                # Advanced settings
                gr.Markdown("### ⚙️ **Advanced Settings**")
                
                # Sampling parameters (contains all generation settings)
                with gr.Accordion("Generation Parameters", open=False):
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=4.0,
                        value=1.3,
                        step=0.05,
                        label="CFG Scale (Guidance Strength)",
                        # info="Higher values increase adherence to text",
                        elem_classes="slider-container"
                    )
                
            # Right column - Generation
            with gr.Column(scale=2, elem_classes="generation-card"):
                gr.Markdown("### 📝 **Script Input**")
                
                script_input = gr.Textbox(
                    label="Conversation Script",
                    placeholder="""Enter your Audio script here. You can format it as:

Speaker 1: Welcome to our Show today!
Speaker 2: Thanks for having me. I'm excited to discuss...

Or paste text directly and it will auto-assign speakers.""",
                    lines=12,
                    max_lines=20,
                    elem_classes="script-input"
                )
                
                # Button row with Random Example on the left and Generate on the right
                with gr.Row():
                    # Random example button (now on the left)
                    random_example_btn = gr.Button(
                        "🎲 Random Example",
                        size="lg",
                        variant="secondary",
                        elem_classes="random-btn",
                        scale=1  # Smaller width
                    )
                    
                    # Generate button (now on the right)
                    generate_btn = gr.Button(
                        "🚀 Generate Audio",
                        size="lg",
                        variant="primary",
                        elem_classes="generate-btn",
                        scale=2  # Wider than random button
                    )
                
                # Stop button
                stop_btn = gr.Button(
                    "🛑 Stop Generation",
                    size="lg",
                    variant="stop",
                    elem_classes="stop-btn",
                    visible=False
                )
                
                # Streaming status indicator
                streaming_status = gr.HTML(
                    value="""
                    <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                                border: 1px solid rgba(34, 197, 94, 0.3); 
                                border-radius: 8px; 
                                padding: 0.75rem; 
                                margin: 0.5rem 0;
                                text-align: center;
                                font-size: 0.9rem;
                                color: #166534;">
                        <span class="streaming-indicator"></span>
                        <strong>LIVE STREAMING</strong> - Audio is being generated in real-time
                    </div>
                    """,
                    visible=False,
                    elem_id="streaming-status"
                )
                
                # Output section
                gr.Markdown("### 🎵 **Generated Audio**")
                
                # Streaming audio output (outside of tabs for simpler handling)
                audio_output = gr.Audio(
                    label="Streaming Audio (Real-time)",
                    type="numpy",
                    elem_classes="audio-output",
                    streaming=True,  # Enable streaming mode
                    autoplay=True,
                    show_download_button=False,  # Explicitly show download button
                    visible=True
                )
                
                # Complete audio output (non-streaming)
                complete_audio_output = gr.Audio(
                    label="Complete Audio (Download after generation)",
                    type="numpy",
                    elem_classes="audio-output complete-audio-section",
                    streaming=False,  # Non-streaming mode
                    autoplay=False,
                    show_download_button=True,  # Explicitly show download button
                    visible=False  # Initially hidden, shown when audio is ready
                )
                
                gr.Markdown("""
                *💡 **Streaming**: Audio plays as it's being generated (may have slight pauses)  
                *💡 **Complete Audio**: Will appear below after generation finishes*
                """)
                
                # Generation log
                log_output = gr.Textbox(
                    label="Generation Log",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    elem_classes="log-output"
                )
        
        # --- New Section: Saved Files ---
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📂 **Saved Audio Files**")
                gr.Markdown("All generated audio is automatically saved locally. Click refresh to see the latest files.")
                
                with gr.Row():
                    refresh_files_btn = gr.Button("🔄 Refresh File List", variant="secondary", size="sm", scale=0)
                
                saved_files_output = gr.File(
                    label="History (Downloadable)",
                    file_count="multiple",
                    type="filepath",
                    interactive=False,
                    value=demo_instance.get_saved_files  # Load initially
                )
                
                # Connect refresh button
                refresh_files_btn.click(
                    fn=demo_instance.get_saved_files,
                    inputs=[],
                    outputs=[saved_files_output]
                )
        
        def update_speaker_visibility(num_speakers):
            updates = []
            for i in range(4):
                updates.append(gr.update(visible=(i < num_speakers)))
            return updates
        
        num_speakers.change(
            fn=update_speaker_visibility,
            inputs=[num_speakers],
            outputs=speaker_groups # Target the groups for visibility
        )
        
        # Main generation function with streaming
        def generate_podcast_wrapper(num_speakers, script, *speakers_and_params):
            """Wrapper function to handle the streaming generation call."""
            try:
                # Extract speakers and parameters
                # 4 dropdowns + 4 uploads + 4 speed + 1 cfg_scale = 13 params
                dropdown_selections = speakers_and_params[0:4]
                upload_selections = speakers_and_params[4:8]
                speed_selections = speakers_and_params[8:12] # <-- NEW
                cfg_scale = speakers_and_params[12] # <-- Index updated
                
                # Clear outputs and reset visibility at start
                yield None, gr.update(value=None, visible=False), "🎙️ Starting generation...", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
                
                # The generator will yield multiple times
                final_log = "Starting generation..."
                
                for streaming_audio, complete_audio, log, streaming_visible in demo_instance.generate_podcast_streaming(
                    num_speakers=int(num_speakers),
                    script=script,
                    speaker_1=dropdown_selections[0],
                    speaker_2=dropdown_selections[1],
                    speaker_3=dropdown_selections[2],
                    speaker_4=dropdown_selections[3],
                    speaker_1_upload=upload_selections[0],
                    speaker_2_upload=upload_selections[1],
                    speaker_3_upload=upload_selections[2],
                    speaker_4_upload=upload_selections[3],
                    speaker_1_speed=speed_selections[0], # <-- NEW
                    speaker_2_speed=speed_selections[1], # <-- NEW
                    speaker_3_speed=speed_selections[2], # <-- NEW
                    speaker_4_speed=speed_selections[3], # <-- NEW
                    cfg_scale=cfg_scale
                ):
                    final_log = log
                    
                    # Check if we have complete audio (final yield)
                    if complete_audio is not None:
                        # Final state: clear streaming, show complete audio
                        yield None, gr.update(value=complete_audio, visible=True), log, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                    else:
                        # Streaming state: update streaming audio only
                        if streaming_audio is not None:
                            yield streaming_audio, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)
                        else:
                            # No new audio, just update status
                            yield None, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)

            except Exception as e:
                error_msg = f"❌ A critical error occurred in the wrapper: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                # Reset button states on error
                yield None, gr.update(value=None, visible=False), error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        def stop_generation_handler():
            """Handle stopping generation."""
            demo_instance.stop_audio_generation()
            # Return values for: log_output, streaming_status, generate_btn, stop_btn
            return "🛑 Generation stopped.", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        # Add a clear audio function
        def clear_audio_outputs():
            """Clear both audio outputs before starting new generation."""
            return None, gr.update(value=None, visible=False)

        # Connect generation button with streaming outputs
        generate_btn.click(
            fn=clear_audio_outputs,
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        ).then(  # Immediate UI update to hide Generate, show Stop (non-queued)
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            inputs=[],
            outputs=[generate_btn, stop_btn],
            queue=False
        ).then(
            fn=generate_podcast_wrapper,
            inputs=[num_speakers, script_input] + speaker_selections + speaker_uploads + speaker_speed_sliders + [cfg_scale], # Pass all lists
            outputs=[audio_output, complete_audio_output, log_output, streaming_status, generate_btn, stop_btn],
            queue=True  # Enable Gradio's built-in queue
        ).then( # Auto-refresh file list after generation finishes
            fn=demo_instance.get_saved_files,
            inputs=[],
            outputs=[saved_files_output],
            queue=False
        )
        
        # Connect stop button
        stop_btn.click(
            fn=stop_generation_handler,
            inputs=[],
            outputs=[log_output, streaming_status, generate_btn, stop_btn],
            queue=False  # Don't queue stop requests
        ).then(
            # Clear both audio outputs after stopping
            fn=lambda: (None, None),
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        )
        
        # Function to randomly select an example
        def load_random_example():
            """Randomly select and load an example script."""
            import random
            
            # Get available examples
            if hasattr(demo_instance, 'example_scripts') and demo_instance.example_scripts:
                example_scripts = demo_instance.example_scripts
            else:
                # Fallback to default
                example_scripts = [
                    [2, "Speaker 0: Welcome to our AI Audio demonstration!\nSpeaker 1: Thanks for having me. This is exciting!"]
                ]
            
            # Randomly select one
            if example_scripts:
                selected = random.choice(example_scripts)
                num_speakers_value = selected[0]
                script_value = selected[1]
                
                # Return the values to update the UI
                return num_speakers_value, script_value
            
            # Default values if no examples
            return 2, ""
        
        # Connect random example button
        random_example_btn.click(
            fn=load_random_example,
            inputs=[],
            outputs=[num_speakers, script_input],
            queue=False  # Don't queue this simple operation
        )
        
        # Add usage tips
        gr.Markdown("""
        ### 💡 **Usage Tips**
        
        - Click **🚀 Generate Audio** to start audio generation
        - **Live Streaming** tab shows audio as it's generated (may have slight pauses)
        - **Complete Audio** tab provides the full, uninterrupted Audio after generation
        - During generation, you can click **🛑 Stop Generation** to interrupt the process
        - The streaming indicator shows real-time generation progress
        """)
        
        # Add example scripts
        gr.Markdown("### 📚 **Example Scripts**")
        
        # Use dynamically loaded examples if available, otherwise provide a default
        if hasattr(demo_instance, 'example_scripts') and demo_instance.example_scripts:
            example_scripts = demo_instance.example_scripts
        else:
            # Fallback to a simple default example if no scripts loaded
            example_scripts = [
                [1, "Speaker 1: Welcome to our AI Audio demonstration! This is a sample script showing how the model can generate natural-sounding speech."]
            ]
        
        gr.Examples(
            examples=example_scripts,
            inputs=[num_speakers, script_input],
            label="Try these example scripts:"
        )


    return interface


def convert_to_16_bit_wav(data):
    # Check if data is a tensor and move to cpu
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    # Ensure data is numpy array
    data = np.array(data)

    # Normalize to range [-1, 1] if it's not already
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    
    # Scale to 16-bit integer range
    data = (data * 32767).astype(np.int16)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="RSR TTS Gradio Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/models/VibeVoice-large",
        help="Path to the TTS model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=10,
        help="Number of inference steps for DDPM (not exposed to users)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the demo publicly via Gradio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the demo on",
    )
    # NEW ARGUMENT
    parser.add_argument(
        "--auto_device_map",
        action="store_true",
        default=False,
        help="Use device_map='auto' for loading the model. Helps with VRAM on smaller GPUs but may need careful layer splitting.",
    )
    
    return parser.parse_args()


def main():
    """Main function to run the demo."""
    args = parse_args()
    
    set_seed(42)  # Set a fixed seed for reproducibility

    print("🎙️ Initializing RSR TTS Demo with Streaming Support...")
    
    # Initialize demo instance
    demo_instance = RSRTTSDemo(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps,
        auto_device_map=args.auto_device_map  # Pass the new argument
    )
    
    # Create interface
    interface = create_demo_interface(demo_instance)
    
    print(f"🚀 Launching demo on port {args.port}")
    print(f"📁 Model path: {args.model_path}")
    print(f"🎭 Available voices: {len(demo_instance.available_voices)}")
    print(f"🔴 Streaming mode: ENABLED")
    print(f"🔒 Session isolation: ENABLED")
    
    # Launch the interface
    try:
        interface.queue(
            max_size=20,  # Maximum queue size
            default_concurrency_limit=1  # Process one request at a time
        ).launch(
            share=True,
            # server_port=args.port,
            server_name="0.0.0.0" if args.share else "127.0.0.1",
            show_error=True,
            show_api=False  # Hide API docs for cleaner interface
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    main()