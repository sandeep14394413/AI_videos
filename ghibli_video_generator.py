import os
import json
import random
from datetime import datetime
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from gtts import gTTS
from moviepy.editor import *

OUTPUT_FOLDER = "generated_ghibli_videos"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("✅ HF_TOKEN loaded")

print("Running on GitHub Hosted Runner")

# Load models
print("🚀 Loading models...")
story_generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
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

def generate_story():
    moral = random.choice(["kindness", "honesty", "friendship", "courage", "sharing", "patience"])
    print(f"📖 Generating story about {moral.upper()}...")

    prompt = f"""Create a beautiful Studio Ghibli style moral story for children about {moral}.
Return ONLY valid JSON with 8 scenes.
Each scene: "scene_number", "visual_description" (start with "ghibli style,"), "narration_text" (max 25 words)."""

    response = story_generator(prompt, max_new_tokens=1500, temperature=0.85, do_sample=True)
    text = response[0]['generated_text']

    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        return json.loads(text[start:end]), moral
    except:
        return [{"scene_number": i+1, "visual_description": f"ghibli style, beautiful {moral} scene", "narration_text": f"Scene {i+1}"} for i in range(8)], moral

def generate_image(desc, num):
    print(f"🖼️  Generating Ghibli image {num}...")
    image = pipe(f"{desc}, masterpiece, best quality", num_inference_steps=28, guidance_scale=7.5).images[0]
    path = os.path.join(OUTPUT_FOLDER, f"scene_{num:02d}.png")
    image.save(path)
    return path

def text_to_speech(text, num):
    print(f"🗣️  Generating narration {num}...")
    tts = gTTS(text=text, lang='en', slow=False)
    path = os.path.join(OUTPUT_FOLDER, f"narration_{num:02d}.mp3")
    tts.save(path)
    return path

def create_video(scenes, moral):
    print("🎬 Creating final video with subtitles...")
    clips = []
    subs = []

    current_time = 0
    for scene in scenes:
        img_path = os.path.join(OUTPUT_FOLDER, f"scene_{scene['scene_number']:02d}.png")
        audio_path = os.path.join(OUTPUT_FOLDER, f"narration_{scene['scene_number']:02d}.mp3")

        image_clip = ImageClip(img_path)
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration + 0.5

        clip = image_clip.set_duration(duration).set_audio(audio_clip).crossfadein(0.5).crossfadeout(0.5)
        clips.append(clip)

        # Subtitles using TextClip (now works with ImageMagick)
        subtitle = TextClip(
            scene["narration_text"],
            fontsize=50,
            color='white',
            stroke_color='black',
            stroke_width=2,
            font='Arial-Bold',
            size=(1000, None),
            align='center'
        )
        subtitle = subtitle.set_position(('center', 'bottom')).set_start(current_time).set_duration(duration)
        subs.append(subtitle)

        current_time += duration

    final_video = concatenate_videoclips(clips)
    final_video = CompositeVideoClip([final_video] + subs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = os.path.join(OUTPUT_FOLDER, f"ghibli_{moral}_{timestamp}.mp4")

    final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", preset="medium", threads=4)

    print(f"\n✅ VIDEO GENERATED: {output_path}")

if __name__ == "__main__":
    scenes, moral = generate_story()

    for scene in scenes:
        generate_image(scene["visual_description"], scene["scene_number"])

    for scene in scenes:
        text_to_speech(scene["narration_text"], scene["scene_number"])

    create_video(scenes, moral)

    print("🎉 All done!")
