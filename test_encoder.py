from src.models.encoder import Encoder
import torchinfo

if __name__ == "__main__":

    encoder = Encoder()
    torchinfo.summary(encoder, (8, 3, 256, 256))

