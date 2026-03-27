import os
import json
import random
from datetime import datetime
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from gtts import gTTS
from moviepy.editor import *
from moviepy.video.fx.all import resize

OUTPUT_FOLDER = "generated_ghibli_videos"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("✅ HF_TOKEN loaded")

print("🚀 Starting Ultra High-Quality Ghibli Video with Advanced Transitions...")

# ====================== MODELS ======================
story_generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)

pipe = StableDiffusionPipeline.from_pretrained(
    "nitrosocke/Ghibli-Diffusion",
    torch_dtype=torch.float32,
    token=HF_TOKEN
)
pipe = pipe.to("cpu")
pipe.safety_checker = None

print("✅ Models loaded!")

# ====================== FUNCTIONS ======================
def generate_story():
    moral = random.choice(["kindness", "honesty", "friendship", "courage", "sharing", "patience", "gratitude"])
    print(f"📖 Generating emotional story about {moral.upper()}...")

    prompt = f"""You are Hayao Miyazaki. Write a touching, magical Studio Ghibli-style moral story for children 4-8 years old about "{moral}".

Return ONLY valid JSON with 8 scenes.
Each scene: "scene_number", "visual_description" (start with "ghibli style,"), "narration_text" (22-30 words)."""

    response = story_generator(prompt, max_new_tokens=2400, temperature=0.78, do_sample=True)
    text = response[0]['generated_text']

    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        return json.loads(text[start:end]), moral
    except:
        print("Using premium fallback")
        return [{"scene_number": i+1, 
                 "visual_description": f"ghibli style, emotional {moral} scene with cute animals and children, soft magical lighting, highly detailed", 
                 "narration_text": f"A gentle child discovered the true beauty of {moral}."} for i in range(8)], moral

def generate_image(desc, num):
    print(f"🖼️  Generating high-quality Ghibli image {num}...")
    image = pipe(
        f"{desc}, masterpiece, best quality, intricate details, soft cinematic lighting, whimsical atmosphere",
        num_inference_steps=45,
        guidance_scale=9.0
    ).images[0]
    path = os.path.join(OUTPUT_FOLDER, f"scene_{int(num):02d}.png")
    image.save(path)
    return path

def text_to_speech(text, num):
    print(f"🗣️  Generating warm narration {num}...")
    tts = gTTS(text=text, lang='en', slow=False)
    path = os.path.join(OUTPUT_FOLDER, f"narration_{int(num):02d}.mp3")
    tts.save(path)
    return path

def create_video(scenes, moral):
    print("🎬 Creating video with advanced cinematic transitions...")
    clips = []
    subs = []
    current_time = 0.0

    for i, scene in enumerate(scenes):
        img_path = os.path.join(OUTPUT_FOLDER, f"scene_{int(scene['scene_number']):02d}.png")
        audio_path = os.path.join(OUTPUT_FOLDER, f"narration_{int(scene['scene_number']):02d}.mp3")

        image_clip = ImageClip(img_path)
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration + 1.3

        # Base clip
        clip = image_clip.set_duration(duration).set_audio(audio_clip)

        # Advanced Cinematic Transitions (Fixed Resize)
        if i == 0:
            # First scene: Slow fade in + gentle zoom
            clip = clip.crossfadein(2.0)
            clip = resize(clip, lambda t: 1 + 0.012 * t)   # Gentle zoom in
        elif i == len(scenes) - 1:
            # Last scene: Slow fade out
            clip = clip.crossfadeout(2.0)
        else:
            # Middle scenes: Soft breathing zoom + dissolve
            clip = resize(clip, lambda t: 1 + 0.018 * (t % 3))
            clip = clip.crossfadein(1.3).crossfadeout(1.3)

        clips.append(clip)

        # Refined Subtitles
        subtitle = TextClip(
            scene["narration_text"],
            fontsize=52,
            color='#F8F1E9',
            stroke_color='#1a1a1a',
            stroke_width=4.5,
            font='Arial-Bold',
            size=(1100, None),
            align='center',
            method='caption',
            kerning=1.3
        )
        subtitle = subtitle.set_position(('center', 0.79 * subtitle.h)).set_start(current_time).set_duration(duration)
        subs.append(subtitle)

        current_time += duration

    final_video = concatenate_videoclips(clips, method="compose")
    final_video = CompositeVideoClip([final_video] + subs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = os.path.join(OUTPUT_FOLDER, f"ghibli_cinematic_{moral}_{timestamp}.mp4")

    final_video.write_videofile(
        output_path, 
        fps=24, 
        codec="libx264", 
        audio_codec="aac", 
        preset="slow",
        threads=4,
        bitrate="8000k"
    )

    print(f"\n✅ CINEMATIC HIGH-QUALITY VIDEO GENERATED: {output_path}")

# ====================== MAIN ======================
if __name__ == "__main__":
    scenes, moral = generate_story()

    for scene in scenes:
        generate_image(scene["visual_description"], scene["scene_number"])

    for scene in scenes:
        text_to_speech(scene["narration_text"], scene["scene_number"])

    create_video(scenes, moral)

    print("🎉 All done!")
