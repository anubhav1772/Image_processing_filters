import math
import numpy as np
from PIL import Image
from R2Pixel import R2Pixel
from scipy.ndimage.filters import convolve

np.seterr(divide='ignore', invalid='ignore')

class Filters(R2Pixel):
    def __init__(self, image_name):
        R2Pixel.__init__(self, image_name)              # Alternatively, use super().__init__(self, image_name)
        
    def brighten(self, factor):
        """
        Brighten the image by multiplying each pixel component by the factor,
        then clamping the result to a valid range.
        """

        img = np.array(self.image)
        for i in range(self.channels):
            try:
                img[:,:,i] = img[:,:,i].clip(0, 255.0 / factor) * factor
            except ZeroDivisionError:
                img[:,:,i] = img[:,:,i] * factor        # when factor is 0.0

        # img = img.astype(np.uint8)
        img = Image.fromarray(img)
        self.save_image(img, 'output/princeton_small_brightness_'+str(factor)+'.jpg')
    
    def changeContrast(self, factor):
        """
        Change the contrast of an image by interpolating between the image
        and a constant gray image with the average luminance.
        Interpolation reduces constrast, extrapolation boosts constrast,
        and negative factors generate inverted images.
        """
        img = np.array(self.image)
        avgLumi = 0
        for i in range(self.height):
            for j in range(self.width):
                avgLumi += self.luminance(img, i, j)
        avgLumi /= self.width * self.height
        # print(avgLumi)
        pixel = np.array([avgLumi, avgLumi, avgLumi, 255])
        
        for i in range(self.height):
            for j in range(self.width):
                img[i, j] = np.clip((1-factor)*pixel + factor*img[i, j], 0.0, 255.0)
        
        img = Image.fromarray(img)
        self.save_image(img, 'output/c_contrast_'+str(factor)+'.jpg')
    
    def blur(self, sigma):
        """
        Blur an image with a Gaussian filter with a given sigma.
        Gaussian function used: 
            G(x) = exp(-x^2/(2*sigma^2))
        Gaussian filter width:
            ceil(3*sigma)*2+1
        """
        if sigma == 0:
            return

        img = np.array(self.image)[:,:,:3]
        size = math.ceil(3*sigma)*2+1
        pad = (size-1)//2
        img = np.pad(img, [(pad, pad), (pad, pad), (0, 0)], mode='constant', constant_values=0)

        gaussianKernel = np.array([[0.0 for i in range(size)] for j in range(size)])

        total = 0
        for i in range(size):
            for j in range(size):
                gaussianKernel[i][j] = np.exp(-(i**2 + j**2) / (2.*sigma**2))
                total += gaussianKernel[i][j]

        gaussianKernel = gaussianKernel/total
        
        output = np.zeros((self.height,self.width, 3), dtype="float32")

        for y in range(pad, self.height+pad):
            for x in range(pad, self.width+pad):
                roi_r = img[(y-pad):(y+pad+1), (x-pad):(x+pad+1), 0]
                roi_g = img[(y-pad):(y+pad+1), (x-pad):(x+pad+1), 1]
                roi_b = img[(y-pad):(y+pad+1), (x-pad):(x+pad+1), 2]
                k_r = (roi_r * gaussianKernel).sum()
                k_g = (roi_g * gaussianKernel).sum()
                k_b = (roi_b * gaussianKernel).sum()

                output[y - pad, x - pad, 0], output[y - pad, x - pad, 1], output[y - pad, x - pad, 2] = k_r, k_g, k_b
        
        output = np.clip(output, 0.0, 255.0)

        return output.astype("uint8")            
        # for sharpen fuction to work correctly
        # output = Image.fromarray(output.astype("uint8"))
        # self.save_image(output, 'output/blur_'+str(sigma)+'.jpg')

    
    def sharpen(self):
        """
        Apply a linear sharpening filter to the image.
        """
        img = np.array(self.image)[:,:,:3]
        blurred_img = self.blur(2.0)
        factor = 2.0
        img[:, :] = np.clip((1-factor)*blurred_img[:, :] + factor*img[:, :], 0.0, 255.0)
        # for i in range(self.height):
        #     for j in range(self.width):
        #         print(img[i, j], blurred_img[i, j])
        #         img[i, j] = np.clip((1-factor)*blurred_img[i, j] + factor*img[i, j], 0.0, 255.0)
        
        img = Image.fromarray(img)
        self.save_image(img, 'output/sharpen.jpg')
        
    
    def detectEdge(self):
        """
        Detect edges in an image by convolving it with an edge detection kernel and taking absolute values. 
        Kernel used is:
        [-1,-1, -1], 
        [-1, 8, -1], 
        [-1, -1, -1]
        """
        img = np.array(self.image)
        img = (img[:,:,0]+img[:,:,1]+img[:,:,2])/3

        kernel = np.array([[-1,-1, -1], [-1, 8, -1], [-1, -1, -1]])

        pad = (kernel.shape[0]-1)//2 
        img = np.pad(img, [(pad, pad), (pad, pad)], mode='constant', constant_values=0)

        output = np.zeros((self.height, self.width, 3), dtype="float32")
        
        for y in range(pad, self.height+pad):
            for x in range(pad, self.width+pad):
                roi = img[(y-pad):(y+pad+1), (x-pad):(x+pad+1)]
                k = (roi * kernel).sum()
                if abs(k>255):
                    output[y - pad, x - pad] = 255.0
                else:
                    output[y - pad, x - pad] = abs(k)

        output = Image.fromarray(output.astype("uint8"))
        self.save_image(output, 'output/edgedetect.jpg')

    def point_sampling(self, sx, sy, h, w):
        """
        Point Sampling Method Implementation
        """
        height, width = h, w
        img = np.array(self.image)[:,:,:3]
        
        scaled_img = np.zeros((height, width, 3), dtype="float32")

        for i in range(0, height):
            for j in range(0, width):
                scaled_img[i, j] = img[round(i/sy), round(j/sx)]
        return scaled_img

    def bilinear_sampling(self, x0, y0, h, w): 
        """
        Bilinear Sampling Method Implementation
        """
        height, width  = h, w
        original = np.array(self.image)[:,:,:3]

        out = []

        mod_x = int(x0)
        mod_y = int(y0)
        delta_x = x0 - mod_x
        delta_y = y0 - mod_y
        
        modXiPlusOneLim = min(mod_x+1, self.width-1)
        modYiPlusOneLim = min(mod_y+1, self.height-1)
        
        for chan in range(original.shape[2]):
            bl = original[mod_y, mod_x, chan]
            br = original[mod_y, modXiPlusOneLim, chan]
            tl = original[modYiPlusOneLim, mod_x, chan]
            tr = original[modYiPlusOneLim, modXiPlusOneLim, chan]
 
            b = delta_x * br + (1. - delta_x) * bl
            t = delta_x * tr + (1. - delta_x) * tl
            pxf = delta_y * t + (1. - delta_y) * b
            out.append(int(pxf+0.5))      
        return out

    def gaussian_sampling(self, x0, y0, h, w, sigma_x, sigma_y):
        """
        Gaussian Sampling Method Implementation
        """
        height, width  = h, w
        original = np.array(self.image)[:, :, :3]

        x_orig = round(x0)
        y_orig = round(y0)

        size_x = sigma_x * 3
        if size_x<1:
            size_x = 1

        size_y = sigma_y * 3
        if size_y<1:
            size_y = 1

        p = np.array((self.height, self.width, 3), dtype="float32")
        total = 0

        xl = int(max(0, x_orig - size_x))
        xh = int(min(x_orig + size_x + 1, width))
        yl = int(max(0, y_orig - size_y))
        yh = int(min(y_orig + size_y + 1, height))

        for i in range(xl, xh):
            for j in range(yl, yh):
                g = math.exp(-1*((i-x0)**2)/(2*sigma_x**2))*math.exp(-1*((j-y0)**2)/(2*sigma_y**2))
                p += g*original[i, j]
                total += g
        p /= total
        return p    

    def scale(self, sx, sy, sampling_method):
        """
        Scales an image in x by sx, and y by sy.
        The result depends on the current sampling method:
        (point, bilinear, or Gaussian)
        """
        # image.shape :- [height, width, channels]
        img = np.array(self.image)[:,:,:3]

        height = round(sy*self.height) # no. of rows
        width = round(sx*self.width)   # no. of columns

        # print(height, width)

        rs = float(self.height) / float(height)        # row scale
        cs = float(self.width) / float(width)          # column scale
  
        if sampling_method=="bilinear":
            scaled_img = np.empty(list(map(int, [height, width, img.shape[2]])), dtype=np.uint8)
            
            for row in range(0, scaled_img.shape[0]):
                for col in range(0, scaled_img.shape[1]):
                    row_orig = row * rs
                    col_orig = col * cs
                    scaled_img[row, col] = self.bilinear_sampling(col_orig, row_orig, height, width)

        elif sampling_method=="point":
            scaled_img = self.point_sampling(sx, sy, height, width)
            scaled_img = scaled_img.astype("uint8")
              
        elif sampling_method=="gaussian":
            scaled_img = np.zeros((height, width, 3), dtype="float32")

            for row in range(0, height):
                for col in range(0, width):
                    row_orig = row * rs
                    col_orig = col * cs

                    sigma_x = 1.0/(3.0*sx)
                    sigma_y = 1.0/(3.0*sy)
                    if sigma_x>1:
                        sigma_x = 0.5
                    if sigma_y>1:
                        sigma_y = 0.5

                    scaled_img[row, col] = self.gaussian_sampling(col_orig, row_orig, height, width, sigma_x, sigma_y)
                    scaled_img = scaled_img.astype("uint8")
        else:
            print("Entered sampling method is incorrect!")
            return
               
        output = Image.fromarray(scaled_img)
        self.save_image(output, 'output/scale_'+sampling_method+'.jpg')
      
    def composite(self, image_top_path, alpha_top_path, operation):
        """
        Composites the top image over the base image, using the alpha channel of the top image as a mask.
        I = alpha*F + (1-alpha)*B
        """
        if operation=="over":
            image_top = np.array(Image.open(image_top_path, 'r'))

            alpha_channel_top = np.array(Image.open(alpha_top_path, 'r'))
            alpha_top = alpha_channel_top/255.0

            img = np.array(self.image)

            img[:, :, 0] = alpha_top[:, :, 0] * image_top[:, :, 0] + (1 - alpha_top[:, :, 0]) * img[:, :, 0]
            img[:, :, 1] = alpha_top[:, :, 1] * image_top[:, :, 1] + (1 - alpha_top[:, :, 1]) * img[:, :, 1]
            img[:, :, 2] = alpha_top[:, :, 2] * image_top[:, :, 2] + (1 - alpha_top[:, :, 2]) * img[:, :, 2]
    

            output = Image.fromarray(img.astype("uint8"))
            self.save_image(output, 'output/composite.jpg')

        else:
            print("Entered operation is incorrect!")
            return 

def main():
    # x = Filters("input/princeton_small.jpg")
    # x.brighten(0.0)
    # x.brighten(0.5)
    # x.brighten(2.0)

    # x = Filters("input/c.jpg")
    # x.changeContrast(-0.5)
    # x.changeContrast(0.0)
    # x.changeContrast(0.5)
    # x.changeContrast(2.0)

    # x = Filters("input/princeton_small.jpg")
    # x.blur(0.125)
    # x.blur(2)
    # x.blur(8)

    # x = Filters("input/princeton_small.jpg")
    # x.sharpen()

    # x = Filters("input/princeton_small.jpg")
    # x.detectEdge()

    # x = Filters("input/comp_background.jpg")
    # x.composite("input/comp_foreground.jpg", "input/comp_mask.jpg", "over")

    x = Filters("input/scaleinput.jpg")
    x.scale(0.3, 0.3, "point")
    # x.scale(0.3, 0.3, "bilinear")
    # x.scale(0.3, 0.3, "gaussian")

if __name__=="__main__":
    main()


