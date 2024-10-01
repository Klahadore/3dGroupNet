import os
import numpy as np


def load_img(img_dir, img_list):
    images=[]
    for i, image_name in enumerate(img_list):    
        if (image_name.split('.')[1] == 'npy'):
            
            image = np.load(img_dir+image_name)
                      
            images.append(image)
    images = np.array(images)
    
    return(images)




def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):

    L = len(img_list)

    #keras needs the generator infinite, so we will use while true  
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
                       
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size   
            batch_end += batch_size

if __name__ == '__main__':
    train_img_dir = "./data/train/images/"
    train_mask_dir = "./data/train/masks/"
    train_img_list=os.listdir(train_img_dir)
    train_mask_list = os.listdir(train_mask_dir)
    train_img_datagen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, 3)

    img, msk = train_img_datagen.__next__()
    print(img.shape)
