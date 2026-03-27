import os
import json
import random
from datetime import datetime
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
import pyttsx3
from moviepy.editor import *

# ====================== CONFIG ======================
OUTPUT_FOLDER = "generated_ghibli_videos"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get HF_TOKEN from environment (passed from GitHub Secrets)
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("✅ HF_TOKEN loaded from GitHub Secrets")

# Use GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# ====================== LOAD MODELS ======================
print("🚀 Loading models...")

story_generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

pipe = StableDiffusionPipeline.from_pretrained(
    "nitrosocke/Ghibli-Diffusion",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    token=HF_TOKEN if HF_TOKEN else None
)
pipe = pipe.to(device)
pipe.safety_checker = None

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 160)

# Moral themes
MORAL_THEMES = ["kindness", "honesty", "friendship", "courage", "sharing", "patience"]

# ====================== CORE FUNCTIONS ======================
def generate_story():
    moral = random.choice(MORAL_THEMES)
    print(f"📖 Generating story about {moral.upper()}...")

    prompt = f"""You are a master Ghibli storyteller. Create a gentle, heartwarming moral story for children (4-8 years) about {moral}.
Return ONLY a valid JSON array with exactly 8 scenes (for ~3 minute video).
Each scene must contain:
- "scene_number"
- "visual_description" (start with "ghibli style,")
- "narration_text" (max 25 words)"""

    response = story_generator(prompt, max_new_tokens=1800, temperature=0.85, do_sample=True)
    text = response[0]['generated_text']

    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        scenes = json.loads(text[start:end])
        return scenes, moral
    except:
        print("⚠️ Using fallback story")
        return [{"scene_number": i+1, "visual_description": f"ghibli style, beautiful {moral} scene with cute animals", "narration_text": f"Once upon a time..."} for i in range(8)], moral

def generate_image(desc, num):
    print(f"🖼️  Generating Ghibli image {num}...")
    image = pipe(f"{desc}, masterpiece, best quality", num_inference_steps=30, guidance_scale=7.5).images[0]
    path = os.path.join(OUTPUT_FOLDER, f"scene_{num:02d}.png")
    image.save(path)
    return path

def text_to_speech(text, num):
    path = os.path.join(OUTPUT_FOLDER, f"narration_{num:02d}.mp3")
    tts_engine.save_to_file(text, path)
    tts_engine.runAndWait()
    return path

def create_video(scenes, moral):
    print("🎬 Creating final video with subtitles...")
    clips = []
    subs = []

    for scene in scenes:
        img_path = os.path.join(OUTPUT_FOLDER, f"scene_{scene['scene_number']:02d}.png")
        audio_path = os.path.join(OUTPUT_FOLDER, f"narration_{scene['scene_number']:02d}.mp3")

        image_clip = ImageClip(img_path)
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration + 0.5

        clip = image_clip.set_duration(duration).set_audio(audio_clip).crossfadein(0.5).crossfadeout(0.5)
        clips.append(clip)

        subtitle = TextClip(scene["narration_text"], fontsize=55, color='white', stroke_color='black', stroke_width=2, size=(1000, None), align='center')
        subtitle = subtitle.set_position(('center', 'bottom')).set_start(sum(c.duration for c in clips)-duration).set_duration(duration)
        subs.append(subtitle)

    final_video = concatenate_videoclips(clips)
    final_video = CompositeVideoClip([final_video] + subs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = os.path.join(OUTPUT_FOLDER, f"ghibli_{moral}_{timestamp}.mp4")

    final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", preset="medium", threads=4)

    print(f"\n✅ VIDEO GENERATED: {output_path}")
    return output_path

# ====================== MAIN ======================
if __name__ == "__main__":
    scenes, moral = generate_story()

    for scene in scenes:
        generate_image(scene["visual_description"], scene["scene_number"])

    for scene in scenes:
        text_to_speech(scene["narration_text"], scene["scene_number"])

    create_video(scenes, moral)

    print("🎉 All done!")
