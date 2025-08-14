# Project: Real-time Video Captioning for Soft-body Manipulation

## What my project is
I make a system to read a video of a robot handling soft objects (cloth, rope).
System writes short captions that say what the robot is doing (for example:
"A robotic arm grasps the cloth and stretches it flat").

I use a simulator dataset (SoftGym) and code that makes videos and captions.
I try to make the captioning run in real time.

## The novelty I found (simple)
- Most papers do video captioning for normal rigid objects (cars, people).
- I found less work for soft-body tasks (cloth, rope). Soft-body needs special data and more time to understand how the object moves.
- My project makes a pipeline that:
  - collects episodes from SoftGym (or a mock),
  - runs a captioning model in near real-time,
  - tracks results and logs with a small workflow (Airflow DAG / scripts).
- This means my project focuses on "real-time captions for soft-body robot tasks" which I did not find many examples of.

## What I will do next (plan)
1. Make the pipeline run end-to-end on Ubuntu (clone -> venv -> run demo).
2. If SoftGym is missing, use mock env for now and later install SoftGym.
3. Improve caption quality by fine-tuning a small model or using a better captioner.
4. Make a weekly log and show sample videos with captions.

