import time
import numpy as np
import os
import cv2

class DOM(object):
    def __init__(self):
        self.image = None
        self.Im = None
        self.edgex, self.edgey = None, None
    
    @staticmethod
    def load(img, blur=False, blurSize=(5,5)):
        if isinstance(img, str):
            if os.path.exists(img):
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            else:
                raise FileNotFoundError('Image is not found on your system')
        elif isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif len(img.shape) == 2:
                image = img
            else:
                raise ValueError('Image is not in correct shape')
        else:
            raise ValueError('Only image can be passed to the constructor')
        
        if blur:
            image = cv2.GaussianBlur(image, blurSize)
        
        Im = cv2.medianBlur(image, 3, cv2.CV_64F).astype("double")/255.0
        return image, Im

    @staticmethod
    def dom(Im):
        median_shift_up = np.pad(Im, ((0,2), (0,0)), 'constant')[2:,:]
        median_shift_down = np.pad(Im, ((2,0), (0,0)), 'constant')[:-2,:]
        domx = np.abs(median_shift_up - 2*Im + median_shift_down)
        
        median_shift_left = np.pad(Im, ((0,0), (0,2)), 'constant')[:,2:]
        median_shift_right = np.pad(Im, ((0,0), (2,0)), 'constant')[:,:-2]
        domy = np.abs(median_shift_left - 2*Im + median_shift_right)
        
        return domx, domy

    @staticmethod
    def contrast(Im):
        Cx = np.abs(Im - np.pad(Im, ((1,0), (0,0)), 'constant')[:-1, :])
        Cy = np.abs(Im - np.pad(Im, ((0,0), (1,0)), 'constant')[:, :-1])
        return Cx, Cy

    @staticmethod
    def smoothenImage(image, transpose = False, epsilon = 1e-8):
        fil = np.array([0.5, 0, -0.5])

        if transpose:
            image = image.T

        image_smoothed = np.array([np.convolve(image[i], fil, mode="same") for i in range(image.shape[0])])
        
        if transpose:
            image_smoothed = image_smoothed.T

        image_smoothed = np.abs(image_smoothed)/(np.max(image_smoothed) + epsilon)
        return image_smoothed

    def edges(self, image, edge_threshold=0.0001):
        smoothx = self.smoothenImage(image, transpose=True)
        smoothy = self.smoothenImage(image)
        self.edgex = smoothx > edge_threshold
        self.edgey = smoothy > edge_threshold

    def sharpness_matrix(self, Im, width=2, debug=False):
        domx, domy = self.dom(Im)

        Cx, Cy = self.contrast(Im)
        
        Cx = np.multiply(Cx, self.edgex)
        Cy = np.multiply(Cy, self.edgey)

        Sx = np.zeros(domx.shape)
        Sy = np.zeros(domy.shape)
        
        for i in range(width, domx.shape[0]-width):
            num = np.abs(domx[i-width:i+width, :]).sum(axis=0)
            dn = Cx[i-width:i+width, :].sum(axis=0)
            Sx[i] = [(num[k]/dn[k] if dn[k] > 1e-3 else 0) for k in range(Sx.shape[1])]
        
        for j in range(width, domy.shape[1]-width):
            num = np.abs(domy[:, j-width: j+width]).sum(axis=1)
            dn = Cy[:, j-width:j+width].sum(axis=1)
            Sy[:, j] = [(num[k]/dn[k] if dn[k] > 1e-3 else 0) for k in range(Sy.shape[0])]
            
        if debug:
            print(f"domx {domx.shape}: {[(i,round(np.quantile(domx, i/100), 2)) for i in range(0, 101, 25)]}")
            print(f"domy {domy.shape}: {[(i,round(np.quantile(domy, i/100), 2)) for i in range(0, 101, 25)]}")
            print(f"Cx {Cx.shape}: {[(i,round(np.quantile(Cx, i/100),2)) for i in range(50, 101, 10)]}")
            print(f"Cy {Cy.shape}: {[(i,round(np.quantile(Cy, i/100),2)) for i in range(50, 101, 10)]}")
            print(f"Sx {Sx.shape}: {[(i,round(np.quantile(Sx, i/100),2)) for i in range(50, 101, 10)]}")
            print(f"Sy {Sy.shape}: {[(i,round(np.quantile(Sy, i/100),2)) for i in range(50, 101, 10)]}")
            
        return Sx, Sy

    def sharpness_measure(self, Im, width, sharpness_threshold, debug=False, epsilon = 1e-8):
        Sx, Sy = self.sharpness_matrix(Im, width=width, debug=debug)
        Sx = np.multiply(Sx, self.edgex)
        Sy = np.multiply(Sy, self.edgey)
        
        n_sharpx = np.sum(Sx >= sharpness_threshold)
        n_sharpy = np.sum(Sy >= sharpness_threshold)

        n_edgex = np.sum(self.edgex)
        n_edgey = np.sum(self.edgey)
        
        Rx = n_sharpx/(n_edgex + epsilon)
        Ry = n_sharpy/(n_edgey + epsilon)

        S = np.sqrt(Rx**2 + Ry**2)
        
        if debug:
            print(f"Sharpness: {S}")
            print(f"Rx: {Rx}, Ry: {Ry}")
            print(f"Sharpx: {n_sharpx}, Sharpy: {n_sharpy}, Edges: {n_edgex, n_edgey}")
        return S

    def get_sharpness(self, img, width=2, sharpness_threshold=2, edge_threshold=0.0001, debug=False):
        start_time = time.time()
        image, Im = self.load(img)
        self.edges(image, edge_threshold=edge_threshold)
        score = self.sharpness_measure(Im, width=width, sharpness_threshold=sharpness_threshold)
        end_time = time.time()
        processing_time = end_time - start_time
        # print(f"Time Processing Score Bullry: {score} - {processing_time:.4f}")
        return score