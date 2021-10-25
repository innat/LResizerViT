
INP_SIZE      = (512, 512) # Input size of the Image Resizer Module (IRM)
TARGET_SIZE   = (224, 224) # Output size of IRM and Input size of the Vision Transformer 
INTERPOLATION = "bilinear"
BATCH_SIZE = 8
CLASSES = 5

num_res_blocks_in_trainable_resizer = 3