from torchvision.models import vgg19, VGG19_Weights
import torchinfo

if __name__ == "__main__":

    model = vgg19(weights=VGG19_Weights.DEFAULT)
    print(model)

    torchinfo.summary(model, (8, 3, 256, 256))
