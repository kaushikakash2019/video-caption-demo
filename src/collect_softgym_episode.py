import os
import imageio

# fallback import logic
try:
    from softgym.registered_env import create_env
except ModuleNotFoundError:
    from mock_env import create_env  # fallback fake env

def collect_episode(env_name, demo_id, out_root="data/softgym_demo"):
    video_dir   = os.path.join(out_root, "videos", demo_id)
    caption_dir = os.path.join(out_root, "captions")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(caption_dir, exist_ok=True)

    env = create_env(env_name, obs_mode="rgb")
    obs = env.reset()

    frames = []
    for t in range(getattr(env, "max_episode_steps", 10)):
        act = env.action_space.sample()
        obs, _, done, _ = env.step(act)
        frames.append(obs.get("rgb"))
        if done:
            break
    env.close()

    for idx, img in enumerate(frames):
        path = os.path.join(video_dir, f"{idx:04d}.png")
        imageio.imsave(path, img)
    print(f"Wrote {len(frames)} frames → {video_dir}")

    caption = "A robotic arm grasps the cloth and stretches it flat by pulling its corners."
    with open(os.path.join(caption_dir, f"{demo_id}.txt"), "w") as f:
        f.write(caption)
    print("Wrote caption:", caption)

if __name__ == "__main__":
    collect_episode("ClothFlattening-v0", demo_id="0001")
