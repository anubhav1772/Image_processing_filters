from PIL import Image
import numpy as np

class R2Pixel(object):
    def __init__(self, image_name):
        self.image_name = image_name

        self.channels = None
        self.width = None
        self.height = None
        self.image = None
        # self.pixels = None
        self.load_image()

    def load_image(self):
        try:
            self.image = Image.open(self.image_name, 'r').convert('RGBA')
            # print(self.image.mode=='RGBA')
            self.channels = len(self.image.getbands())
            self.width, self.height = self.image.size
            # self.pixels = np.array(self.image)

        except IOError: 
            print("Error occured while loading image from the disk!")
            pass
    
    def luminance(self, image, i, j):
        """
        Return luminance
        """ 
        return (0.30 * image[i, j, 0] + 0.59 * image[i, j, 1] + 0.11 * image[i, j, 2])
        
    def get_pixel(self, row, col):
        if row>self.width or col>self.height:
            return None
        return self.image.getpixel((row, col))

    def show_image(self):
        self.image.show()
    
    def save_image(self, image, save_path):
        image.convert('RGB').save(save_path)
    
# x = R2Pixel("input/princeton_small.jpg")
# x.load_image()
# print(x.width, x.height)
# print(x.pixels.shape)
# print(x.get_pixel(100, 100))    