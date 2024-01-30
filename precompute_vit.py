import os
from PIL import Image

from games.vit_game import ViTGame

if __name__ == "__main__":
    n = 16
    pre_computed_images = os.listdir(os.path.join("data", "vision_transformer", str(n)))

    # read all images from "imagenet_images" folder
    for image_name in os.listdir("imagenet_images"):
        image = Image.open(os.path.join("imagenet_images", image_name))
        image_name = image_name.split(".")[0]
        if image_name + ".csv" in pre_computed_images:
            continue
        print(image_name)
        vit_game = ViTGame(image, image_name=image_name, n=n)
        vit_game.precompute()
