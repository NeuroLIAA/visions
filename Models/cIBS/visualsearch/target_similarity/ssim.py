from .target_similarity import TargetSimilarity
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim
from skimage import transform

class Ssim(TargetSimilarity):
    def compute_target_similarity(self, image, target, target_bbox):
        target_size = target.shape[:2]
        # Rescale target to its size in the image
        target_size_in_image = (target_bbox[2] - target_bbox[0], target_bbox[3] - target_bbox[1])
        if target_size != target_size_in_image:
            target = transform.resize(target, target_size)
            
        target_size = np.shape(target)[:2]
        image_size  = np.shape(image)[:2]
        ssim_values = np.zeros(shape=image_size, dtype= np.dtype('float64'))
        
        off_bounds_area = self.get_image_off_bounds_area(target_bbox, target_size, image_size)
        padding_size    =  (np.tile(target_size, 2) - off_bounds_area) % np.tile(target_size, 2)
        padded_image_shape = tuple(np.array(image_size) + padding_size[0:2] + padding_size[2:4])
        #basicamente agrego como padding lo necesario para que ancho y altura sean multiplos de target_size
        #si la parte que queda fuera de la imagen es 0, no sumo nada, por eso el módulo
        
        for row in range(0, padded_image_shape[0], target_size[0]):
            for column in range(0, padded_image_shape[1], target_size[1]):
                target_to_use, row_in_image, column_in_image, end_row, end_column= self.handle_image_borders(target, off_bounds_area, row, column, target_size, image_size, padding_size)
                
                pixels_in_interval = np.array(image[row_in_image:end_row, column_in_image:end_column])
                if np.shape(pixels_in_interval)[0] >= 7 and np.shape(pixels_in_interval)[1] >= 7:
                    ssim_result = ssim(pixels_in_interval, target_to_use)
                else:
                    ssim_result = 0
                ssim_values[row_in_image:end_row, column_in_image:end_column] += ssim_result

        
        # io.imsave('test_ssim.png',ssim_values) esto es para testear
        # print('SAVED!')

        return ssim_values
    
    def get_image_off_bounds_area(self, target_bbox, target_size, image_size):
        target_starting_pixel = np.array(target_bbox[:2])
        
        target_size_as_array = np.array(target_size)
        image_size_as_array  = np.array(image_size)
        #target_starting_pixel me devuelve indices (arrancan de 0) y los tamaños son > 0
        top_and_left = target_starting_pixel % target_size_as_array
        #obtuve la cantidad de pixeles que me sobran arriba y a la izquierda
        bottom_and_right = (image_size_as_array - (target_starting_pixel + target_size_as_array)) % target_size_as_array
        #obtuve la cantidad de pixeles que me sobran a la derecha y abajo
        
        return np.concatenate((top_and_left, bottom_and_right))

    def handle_image_borders(self, target, off_bounds_area,row,column, target_size, image_size, padding):
        
        row_in_image = row - padding[0]
        column_in_image = column - padding[1]
        target_to_use = target
        end_row = row_in_image+target_size[0]
        end_column = column_in_image+target_size[1]
        if row_in_image < 0: 
            target_to_use = target_to_use[padding[0]:target_size[0],:]
            row_in_image = 0
        if column_in_image < 0:
            target_to_use = target_to_use[:,padding[1]:target_size[1]]
            column_in_image = 0
        if row_in_image >= image_size[0] - off_bounds_area[2]:
            target_to_use = target_to_use[0:target_size[0] - padding[2],:]
            end_row = end_row - padding[2]
        if column_in_image >= image_size[1] - off_bounds_area[3]:
            target_to_use = target_to_use[:,0:target_size[1] - padding[3]]
            end_column = end_column - padding[3]
        return (target_to_use, row_in_image, column_in_image, end_row, end_column)
