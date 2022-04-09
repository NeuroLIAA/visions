from .target_similarity import TargetSimilarity
import numpy as np
from skimage import transform, img_as_ubyte
from skimage.metrics import structural_similarity as ssim
from multiprocessing import Process, Queue

""" For each pixel in the image, a patch of the size of the target (starting from that pixel) is used to compute SSIM with the target image.
    The result from the SSIM computation is always stored at the centre of the patch. Therefore, the region from outside of the borders is equally distributed.
""" 

class Ssim(TargetSimilarity):
    def compute_target_similarity(self, image, target, target_bbox):
        target_size = target.shape[:2]
        # Rescale target to its size in the image
        target_size_in_image = (target_bbox[2] - target_bbox[0], target_bbox[3] - target_bbox[1])
        if target_size != target_size_in_image:
            target = img_as_ubyte(transform.resize(target, target_size_in_image))

        image_size  = image.shape[:2]
        if self.number_of_processes == 1:
            return self.compute_ssim_single_process(image, target, image_size, target_size_in_image)
        else:
            return self.parallelize_ssim_computation(image, target, image_size, target_size_in_image)
    
    
    def parallelize_ssim_computation(self, image, target, image_size, target_size):
        ssim_values     = np.zeros(shape=image_size, dtype=np.dtype('float32')) 
        off_bounds_area = self.get_image_off_bounds_area(target_size)
        
        number_of_rows   = len(range(0, image_size[0]))
        remainder        = number_of_rows % self.number_of_processes
        indexes          = list(range(number_of_rows)) 
        chunks = [indexes[i * (number_of_rows // self.number_of_processes) + min(i, remainder):(i + 1) * (number_of_rows // self.number_of_processes) + min(i + 1, remainder)] for i in range(self.number_of_processes)]

        values_queue = Queue()

        procs = []
        for chunk in chunks:
            # Each process has its own matrix with the SSIM values; when computation is finished, those matrices are stored in the queue
            proc = Process(target=self.ssim_process_chunk, args=(image, target, image_size, target_size, ssim_values, chunk, values_queue, off_bounds_area,))
            procs.append(proc)
            proc.start()
        
        number_of_tasks_completed = self.number_of_processes

        while not (number_of_tasks_completed == 0):            
            ssim_values += values_queue.get()
            number_of_tasks_completed -=1

        # Wait for each process to complete
        for proc in procs:
            proc.join()

        return ssim_values

    def ssim_process_chunk(self, image, target, image_size, target_size, ssim_values, chunk, values_queue, off_bounds_area):
        coloured = False
        if len(image.shape) >= 3:
            coloured = True
        for row in chunk: 
            for column in range(0, image_size[1]):
                target_to_use, row_in_image, column_in_image, end_row, end_column= self.handle_image_borders(target, off_bounds_area, row, column, target_size, image_size)
                pixels_in_interval = image[row_in_image:end_row, column_in_image:end_column]
                if np.shape(pixels_in_interval)[0] >= 7 and np.shape(pixels_in_interval)[1] >= 7:
                    ssim_result = ssim(pixels_in_interval, target_to_use, multichannel=coloured)
                else:
                    ssim_result = 0
                ssim_values[row, column] = ssim_result
        values_queue.put(ssim_values)

    def compute_ssim_single_process(self, image, target, image_size, target_size):
        ssim_values = np.zeros(shape=image_size, dtype= np.dtype('float32'))
        off_bounds_area = self.get_image_off_bounds_area(target_size)

        coloured = False
        if len(image.shape) >= 3:
            coloured = True

        for row in range(0, image_size[0]): 
            for column in range(0, image_size[1]):
                target_to_use, row_in_image, column_in_image, end_row, end_column= self.handle_image_borders(target, off_bounds_area, row, column, target_size, image_size)
                pixels_in_interval = image[row_in_image:end_row, column_in_image:end_column]
                if np.shape(pixels_in_interval)[0] >= 7 and np.shape(pixels_in_interval)[1] >= 7:
                    ssim_result = ssim(pixels_in_interval, target_to_use, multichannel=coloured)
                else:
                    ssim_result = 0
                ssim_values[row, column] = ssim_result
        return ssim_values

    
    def get_image_off_bounds_area(self,  target_size):        
        target_size_as_array = np.array(target_size)
        top_and_left     = target_size_as_array // 2
        bottom_and_right = target_size_as_array - top_and_left
        
        return np.concatenate((top_and_left, bottom_and_right))

    def handle_image_borders(self, target, off_bounds_area,row,column, target_size, image_size):
        # The centre of the target is mapped onto the pixel on which i am standing on
        row_in_image    = row - off_bounds_area[0]
        column_in_image = column - off_bounds_area[1] 

        target_to_use = target

        end_row    = row_in_image + target_size[0]
        end_column = column_in_image + target_size[1]
        if row_in_image < 0:
            # The part of the target that would fall outside of the image is discarded
            target_to_use = target_to_use[abs(row_in_image):target_size[0], :]
            row_in_image  = 0

        if column_in_image < 0:
            target_to_use   = target_to_use[:, abs(column_in_image):target_size[1]]
            column_in_image = 0

        if row_in_image >= image_size[0] - target_size[0]:
            pixels_remainder = row_in_image - (image_size[0] - target_size[0])
            target_to_use    = target_to_use[0:target_size[0] - pixels_remainder, :]
            end_row          = end_row - pixels_remainder

        if column_in_image >= image_size[1] - target_size[1]:
            pixels_remainder = column_in_image - (image_size[1] - target_size[1])
            target_to_use    = target_to_use[:, 0:target_size[1] - pixels_remainder]
            end_column       = end_column - pixels_remainder

        return (target_to_use, row_in_image, column_in_image, end_row, end_column)
