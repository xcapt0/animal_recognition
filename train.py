import subprocess

from data import split_into_folders


def train():
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5"])

    file_content = '''path: ../.
    train: train/images
    val: test/images
    test:  # test images (optional)

    # Classes
    nc: 80
    names: ['Spider', 'Parrot', 'Scorpion', 'Sea turtle', 'Cattle', 'Fox', 'Hedgehog', 'Turtle', 'Cheetah', 'Snake',
            'Shark', 'Horse', 'Magpie', 'Hamster', 'Woodpecker', 'Eagle', 'Penguin', 'Butterfly', 'Lion', 'Otter',
            'Raccoon', 'Hippopotamus', 'Bear', 'Chicken', 'Pig', 'Owl', 'Caterpillar', 'Koala', 'Polar bear', 'Squid',
            'Whale', 'Harbor seal', 'Raven', 'Mouse', 'Tiger', 'Lizard', 'Ladybug', 'Red panda', 'Kangaroo', 'Starfish',
            'Worm', 'Tortoise', 'Ostrich', 'Goldfish', 'Frog', 'Swan', 'Elephant', 'Sheep', 'Snail', 'Zebra',
            'Moths and butterflies', 'Shrimp', 'Fish', 'Panda', 'Lynx', 'Duck', 'Jaguar', 'Goose', 'Goat', 'Rabbit',
            'Giraffe', 'Crab', 'Tick', 'Monkey', 'Bull', 'Seahorse', 'Centipede', 'Mule', 'Rhinoceros', 'Canary', 'Camel',
            'Brown bear', 'Sparrow', 'Squirrel', 'Leopard', 'Jellyfish', 'Crocodile', 'Deer', 'Turkey', 'Sea lion']'''

    with open('yolov5/dataset.yml', 'w') as f:
        f.write(file_content)

    subprocess.run(["cd", "yolov5", "&&", "python", "train.py", "--img", "640",
                    "--batch", "32", "--epochs", "50", "--data", "dataset.yml",
                    "--weights", "../yolov5m.pt"])


if __name__ == '__main__':
    split_into_folders()
    train()