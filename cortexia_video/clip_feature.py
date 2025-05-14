import os, sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
import decord

# add path from perception_models to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "perception_models"))

def preprocess_video(video_path, num_frames=8, transform=None, return_first_frame_for_demo=True):
    # TODO: make this working with VideoContent class. It can work with out video preprocessing workflow.
    """
    Uniformly samples a specified number of frames from a video and preprocesses them.
    Parameters:
    - video_path: str, path to the video file.
    - num_frames: int, number of frames to sample. Defaults to 8.
    - transform: torchvision.transforms, a transform function to preprocess frames.
    Returns:
    - Video Tensor: a tensor of shape (num_frames, 3, H, W) where H and W are the height and width of the frames.
    """
    # Load the video
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    # Uniformly sample frame indices
    frame_indices = [int(i * (total_frames / num_frames)) for i in range(num_frames)]
    frames = vr.get_batch(frame_indices).asnumpy()
    # Preprocess frames
    preprocessed_frames = [transform(Image.fromarray(frame)) for frame in frames]

    first_frame = None
    if return_first_frame_for_demo:
        first_frame = frames[0]
    return torch.stack(preprocessed_frames, dim=0), first_frame

def calculate_score(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
    return text_probs # each for related feature
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Import modules from perception_models package
    import perception_models.core.vision_encoder.pe as pe
    import perception_models.core.vision_encoder.transforms as transforms

    model_name = "PE-Core-B16-224"

    model = pe.CLIP.from_config(model_name, pretrained=True)
    model.to(device)

    # init model

    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)

    # calculate the featurs 
    sample_image = Image.open("perception_models/apps/pe/docs/assets/cat.png")
    sample_text = ["a photo of a cat", "a photo of a dog"]


    image_inputs = preprocess(sample_image).unsqueeze(0).to(device)
    text_inputs = tokenizer(sample_text).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)
    
    text_probs = calculate_score(image_features, text_features)
    print(sample_text)
    print(text_probs)

    # calcualte video features

    sample_video_path = "perception_models/apps/pe/docs/assets/dog.mp4"
    video_frames, first_frame = preprocess_video(sample_video_path, transform=preprocess)
    video_inputs = video_frames.unsqueeze(0).to(device)

    with torch.no_grad():
        video_features = model.encode_video(video_inputs)
        text_features = model.encode_text(text_inputs)

    # calculate the score
    text_probs = calculate_score(video_features, text_features)
    print(sample_text)
    print(text_probs)