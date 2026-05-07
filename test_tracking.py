"""
Quick test script to download YOLOv11 model and run tracking.
"""

from agents.tracking_agent import TrackingAgent

# Initialize agent (downloads model if needed)
print("Initializing Tracking Agent...")
agent = TrackingAgent(model_size='n')

# Check VRAM
print(f"VRAM usage: {agent.get_vram_usage():.1f} MB")

# Process the test clip
print("\nProcessing test clip...")
tracking_data = agent.process_video(
    video_path='data/matches/test_clip.mp4',
    output_path='data/matches/tracking_output.json',
    visualize=True
)

print(f"\nDone! Processed {len(tracking_data)} frames")
print(f"Total player detections: {sum(len(f['players']) for f in tracking_data)}")
print(f"Final VRAM: {agent.get_vram_usage():.1f} MB")
